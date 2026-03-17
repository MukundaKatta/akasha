"""Orbit determination from astrometric observations.

Implements Gauss's method for preliminary orbit determination from three
observations, followed by iterative refinement.  All equations follow
Bate, Mueller & White, *Fundamentals of Astrodynamics* (1971).

Coordinate conventions:
  - Distances in AU
  - Time in years (Julian)
  - Angles in radians
  - mu = 4*pi^2 AU^3/yr^2  (solar gravitational parameter, Gauss units)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from akasha.models import Observation, OrbitalElements

# Solar gravitational parameter in AU^3 / yr^2
MU_SUN = 4.0 * np.pi**2

# AU per km, seconds per Julian year
AU_KM = 1.496e8
SEC_PER_YEAR = 365.25 * 86400.0


def _ra_dec_to_unit_vector(ra_rad: float, dec_rad: float) -> NDArray:
    """Convert RA/Dec (radians) to a unit direction vector (equatorial)."""
    return np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


class OrbitCalculator:
    """Compute Keplerian orbital elements from observations.

    Parameters
    ----------
    mu : float
        Gravitational parameter (AU^3/yr^2).  Default is solar.
    observer_position_au : NDArray, optional
        Heliocentric position of the observer (AU).  Defaults to [1, 0, 0]
        (simplified geocentric at 1 AU on the x-axis).
    """

    def __init__(
        self,
        mu: float = MU_SUN,
        observer_position_au: Optional[NDArray] = None,
    ) -> None:
        self.mu = mu
        self.R_obs = (
            observer_position_au
            if observer_position_au is not None
            else np.array([1.0, 0.0, 0.0])
        )

    # ------------------------------------------------------------------
    # Gauss's method (three observations)
    # ------------------------------------------------------------------

    def gauss_preliminary(
        self,
        obs1: Observation,
        obs2: Observation,
        obs3: Observation,
    ) -> OrbitalElements:
        """Preliminary orbit from three observations using Gauss's method.

        Steps:
        1. Convert RA/Dec to unit line-of-sight vectors rho_hat.
        2. Compute time intervals tau1, tau3 relative to obs2.
        3. Solve the Gauss scalar equations for the slant ranges rho_i.
        4. Compute position and velocity at the middle epoch.
        5. Convert state vector (r, v) to Keplerian elements.
        """
        # Unit direction vectors
        rho1 = _ra_dec_to_unit_vector(np.radians(obs1.ra), np.radians(obs1.dec))
        rho2 = _ra_dec_to_unit_vector(np.radians(obs2.ra), np.radians(obs2.dec))
        rho3 = _ra_dec_to_unit_vector(np.radians(obs3.ra), np.radians(obs3.dec))

        # Time intervals in years
        t1 = (obs1.timestamp - obs2.timestamp).total_seconds() / SEC_PER_YEAR
        t3 = (obs3.timestamp - obs2.timestamp).total_seconds() / SEC_PER_YEAR
        tau = t3 - t1

        # Cross products for the D-matrix (Gauss's method)
        p1 = np.cross(rho2, rho3)
        p2 = np.cross(rho1, rho3)
        p3 = np.cross(rho1, rho2)

        D0 = np.dot(rho1, p1)

        # D-matrix elements  D_ij = dot(R_i, p_j)
        # Simplified: observer at same position for all three obs
        R1 = self.R_obs.copy()
        R2 = self.R_obs.copy()
        R3 = self.R_obs.copy()

        D = np.array([
            [np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)],
            [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)],
            [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)],
        ])

        # Gauss ratios (first approximation: f,g series truncated)
        A = (1.0 / D0) * (-D[0, 1] * (tau / t3) + D[1, 1] + D[2, 1] * (tau / t1))
        B = (1.0 / (6.0 * D0)) * (
            D[0, 1] * (tau**2 - t3**2) * (tau / t3)
            + D[2, 1] * (tau**2 - t1**2) * (tau / t1)
        )

        E = np.dot(R2, rho2)
        R2sq = np.dot(R2, R2)

        # Solve 8th-degree polynomial for r2 (use Newton-Raphson from initial guess)
        # Simplified: assume r2 ~ 1.5 AU as initial guess and iterate
        r2 = 1.5
        for _ in range(50):
            f_r = r2**8 - (A + E) * r2**6 - self.mu * B * r2**3 - self.mu**2 * B**2 / 4.0
            fp_r = 8 * r2**7 - 6 * (A + E) * r2**5 - 3 * self.mu * B * r2**2
            if abs(fp_r) < 1e-30:
                break
            r2_new = r2 - f_r / fp_r
            if abs(r2_new - r2) < 1e-12:
                r2 = r2_new
                break
            r2 = r2_new

        r2 = abs(r2)

        # Slant ranges
        rho_mag_1 = (1.0 / D0) * (
            (6.0 * (D[2, 0] * (t1 / t3) + D[1, 0] * (tau / t3)) * r2**3
             + self.mu * D[2, 0] * (tau**2 - t3**2) * (t1 / t3))
            / (6.0 * r2**3 + self.mu * (tau**2 - t3**2))
            - D[0, 0]
        )
        rho_mag_2 = A + self.mu * B / r2**3
        rho_mag_3 = (1.0 / D0) * (
            (6.0 * (D[0, 2] * (t3 / t1) + D[1, 2] * (tau / t1)) * r2**3
             + self.mu * D[0, 2] * (tau**2 - t1**2) * (t3 / t1))
            / (6.0 * r2**3 + self.mu * (tau**2 - t1**2))
            - D[2, 2]
        )

        # Heliocentric positions
        r1_vec = R1 + rho_mag_1 * rho1
        r2_vec = R2 + rho_mag_2 * rho2
        r3_vec = R3 + rho_mag_3 * rho3

        # Velocity at middle observation via Lagrange interpolation (f,g series)
        f1 = 1.0 - 0.5 * (self.mu / r2**3) * t1**2
        f3 = 1.0 - 0.5 * (self.mu / r2**3) * t3**2
        g1 = t1 - (1.0 / 6.0) * (self.mu / r2**3) * t1**3
        g3 = t3 - (1.0 / 6.0) * (self.mu / r2**3) * t3**3

        v2_vec = (-f3 * r1_vec + f1 * r3_vec) / (f1 * g3 - f3 * g1)

        return self.state_to_elements(r2_vec, v2_vec, obs2.timestamp)

    # ------------------------------------------------------------------
    # State vector -> Keplerian elements
    # ------------------------------------------------------------------

    def state_to_elements(
        self,
        r_vec: NDArray,
        v_vec: NDArray,
        epoch: datetime,
    ) -> OrbitalElements:
        """Convert a state vector (r, v) to classical orbital elements.

        Follows the algorithm in Bate, Mueller & White ch. 2.
        """
        r = float(np.linalg.norm(r_vec))
        v = float(np.linalg.norm(v_vec))

        # Specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = float(np.linalg.norm(h_vec))

        # Node vector
        K = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(K, h_vec)
        n = float(np.linalg.norm(n_vec))

        # Eccentricity vector: e_vec = (1/mu)*[(v^2 - mu/r)*r - (r.v)*v]
        rdotv = float(np.dot(r_vec, v_vec))
        e_vec = (1.0 / self.mu) * ((v**2 - self.mu / r) * r_vec - rdotv * v_vec)
        e = float(np.linalg.norm(e_vec))

        # Semi-major axis (vis-viva)
        energy = 0.5 * v**2 - self.mu / r
        if abs(energy) < 1e-15:
            a = 1e12  # parabolic
        else:
            a = -self.mu / (2.0 * energy)

        # Inclination
        inc = np.arccos(np.clip(h_vec[2] / h, -1, 1))

        # Longitude of ascending node
        if n > 1e-15:
            Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
            if n_vec[1] < 0:
                Omega = 2.0 * np.pi - Omega
        else:
            Omega = 0.0

        # Argument of perihelion
        if n > 1e-15 and e > 1e-15:
            omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
            if e_vec[2] < 0:
                omega = 2.0 * np.pi - omega
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-15:
            nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
            if rdotv < 0:
                nu = 2.0 * np.pi - nu
        else:
            nu = 0.0

        # Eccentric anomaly -> Mean anomaly
        if e < 1.0:
            E = 2.0 * np.arctan2(
                np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                np.sqrt(1.0 + e) * np.cos(nu / 2.0),
            )
            M = E - e * np.sin(E)
        else:
            M = 0.0  # hyperbolic case not handled

        M = M % (2.0 * np.pi)

        return OrbitalElements(
            semi_major_axis=max(a, 0.01),
            eccentricity=min(max(e, 0.0), 0.999),
            inclination=inc,
            longitude_of_ascending_node=Omega,
            argument_of_perihelion=omega,
            mean_anomaly=M,
            epoch=epoch,
            mu=self.mu,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def determine_orbit(self, observations: list[Observation]) -> OrbitalElements:
        """Determine an orbit from a list of observations.

        Uses the first, middle, and last observations for Gauss's method.
        Requires at least three observations.
        """
        if len(observations) < 3:
            raise ValueError("Need at least 3 observations for orbit determination")

        obs_sorted = sorted(observations, key=lambda o: o.timestamp)
        i_mid = len(obs_sorted) // 2
        return self.gauss_preliminary(obs_sorted[0], obs_sorted[i_mid], obs_sorted[-1])
