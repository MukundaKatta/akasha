"""Orbit propagation via Kepler's equation.

Predicts future (or past) positions of an asteroid by:
1. Advancing the mean anomaly M(t) = M_0 + n*(t - t_0).
2. Solving Kepler's equation  M = E - e*sin(E)  for the eccentric
   anomaly E using Newton-Raphson iteration.
3. Converting E to true anomaly nu and then to heliocentric Cartesian
   coordinates via the perifocal-to-inertial rotation.

All equations follow standard two-body mechanics.  Distances in AU,
time in Julian years, angles in radians.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from akasha.models import OrbitalElements

SEC_PER_YEAR = 365.25 * 86400.0


class OrbitPropagator:
    """Propagate an orbit forward or backward in time.

    Parameters
    ----------
    max_kepler_iters : int
        Maximum Newton-Raphson iterations for solving Kepler's equation.
    kepler_tol : float
        Convergence tolerance (radians) for Kepler's equation.
    """

    def __init__(
        self, max_kepler_iters: int = 100, kepler_tol: float = 1e-12
    ) -> None:
        self.max_iters = max_kepler_iters
        self.tol = kepler_tol

    # ------------------------------------------------------------------
    # Kepler's equation solver
    # ------------------------------------------------------------------

    def solve_kepler(self, M: float, e: float) -> float:
        """Solve Kepler's equation  M = E - e*sin(E)  for E.

        Uses Newton-Raphson iteration:
            E_{n+1} = E_n - (E_n - e*sin(E_n) - M) / (1 - e*cos(E_n))

        Parameters
        ----------
        M : float
            Mean anomaly (radians).
        e : float
            Eccentricity (0 <= e < 1).

        Returns
        -------
        float
            Eccentric anomaly E (radians).
        """
        # Initial guess (Markley's starter or simple M + e*sin(M))
        E = M + e * np.sin(M) if e < 0.8 else np.pi

        for _ in range(self.max_iters):
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            dE = f / fp
            E -= dE
            if abs(dE) < self.tol:
                break
        return E

    # ------------------------------------------------------------------
    # Anomaly conversions
    # ------------------------------------------------------------------

    @staticmethod
    def eccentric_to_true(E: float, e: float) -> float:
        """Convert eccentric anomaly E to true anomaly nu.

        nu = 2 * atan2(sqrt(1+e)*sin(E/2), sqrt(1-e)*cos(E/2))
        """
        return 2.0 * np.arctan2(
            np.sqrt(1.0 + e) * np.sin(E / 2.0),
            np.sqrt(1.0 - e) * np.cos(E / 2.0),
        )

    # ------------------------------------------------------------------
    # Position in the orbital plane
    # ------------------------------------------------------------------

    @staticmethod
    def position_in_perifocal(a: float, e: float, nu: float) -> NDArray:
        """Heliocentric position in the perifocal (PQW) frame.

        r = a(1 - e^2) / (1 + e*cos(nu))
        x_pf = r * cos(nu),  y_pf = r * sin(nu),  z_pf = 0
        """
        r = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))
        return np.array([r * np.cos(nu), r * np.sin(nu), 0.0])

    @staticmethod
    def velocity_in_perifocal(a: float, e: float, nu: float, mu: float) -> NDArray:
        """Heliocentric velocity in the perifocal frame.

        p = a(1-e^2)
        v_pf = sqrt(mu/p) * [-sin(nu), e+cos(nu), 0]
        """
        p = a * (1.0 - e**2)
        coeff = np.sqrt(mu / p)
        return coeff * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # ------------------------------------------------------------------
    # Rotation matrix: perifocal -> inertial (equatorial)
    # ------------------------------------------------------------------

    @staticmethod
    def perifocal_to_inertial(
        Omega: float, omega: float, inc: float
    ) -> NDArray:
        """3x3 rotation matrix from perifocal (PQW) to inertial (IJK).

        R = R3(-Omega) * R1(-i) * R3(-omega)
        """
        cO, sO = np.cos(Omega), np.sin(Omega)
        co, so = np.cos(omega), np.sin(omega)
        ci, si = np.cos(inc), np.sin(inc)

        return np.array([
            [cO * co - sO * so * ci, -cO * so - sO * co * ci,  sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si,                  co * si,                  ci     ],
        ])

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        elements: OrbitalElements,
        target_time: datetime,
    ) -> tuple[NDArray, NDArray]:
        """Propagate to a target time and return (position, velocity) in AU.

        Parameters
        ----------
        elements : OrbitalElements
            Orbital elements at reference epoch.
        target_time : datetime
            UTC datetime to propagate to.

        Returns
        -------
        (r_vec, v_vec) : tuple of NDArray
            Heliocentric equatorial position (AU) and velocity (AU/yr).
        """
        dt_years = (target_time - elements.epoch).total_seconds() / SEC_PER_YEAR

        # Advance mean anomaly
        n = elements.mean_motion  # rad/yr
        M = (elements.mean_anomaly + n * dt_years) % (2.0 * np.pi)

        # Solve Kepler's equation
        E = self.solve_kepler(M, elements.eccentricity)
        nu = self.eccentric_to_true(E, elements.eccentricity)

        # Position & velocity in perifocal frame
        r_pf = self.position_in_perifocal(
            elements.semi_major_axis, elements.eccentricity, nu
        )
        v_pf = self.velocity_in_perifocal(
            elements.semi_major_axis, elements.eccentricity, nu, elements.mu
        )

        # Rotate to inertial frame
        Q = self.perifocal_to_inertial(
            elements.longitude_of_ascending_node,
            elements.argument_of_perihelion,
            elements.inclination,
        )

        return Q @ r_pf, Q @ v_pf

    def propagate_trajectory(
        self,
        elements: OrbitalElements,
        start: datetime,
        end: datetime,
        steps: int = 100,
    ) -> list[tuple[datetime, NDArray, NDArray]]:
        """Propagate over a time range and return a list of (time, r, v)."""
        total_sec = (end - start).total_seconds()
        results = []
        for i in range(steps + 1):
            t = start + timedelta(seconds=total_sec * i / steps)
            r, v = self.propagate(elements, t)
            results.append((t, r, v))
        return results
