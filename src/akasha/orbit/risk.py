"""Impact risk assessment: MOID, Palermo scale, Torino scale.

Minimum Orbit Intersection Distance (MOID) is the closest possible
distance between two confocal orbits, computed by sampling true anomaly
on both orbits and minimising the distance.

Palermo Technical Scale:
    P = log10(p_i / f_B * dT)
where p_i is the impact probability, f_B is the annual background
impact frequency, and dT is the time window in years.

    f_B = 0.03 * E^{-4/5}   (impacts/year for energy E in MT)

Torino Scale is an integer 0-10 based on collision probability and
kinetic energy (see decision tree in the code).

Reference: Chesley et al. (2002), Binzel (2000).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from akasha.models import OrbitalElements, RiskAssessment, TorinoScale
from akasha.orbit.propagator import OrbitPropagator

SEC_PER_YEAR = 365.25 * 86400.0

# Earth's approximate orbital elements (J2000, simplified)
EARTH_ELEMENTS = OrbitalElements(
    semi_major_axis=1.000,
    eccentricity=0.0167,
    inclination=0.0,
    longitude_of_ascending_node=0.0,
    argument_of_perihelion=np.radians(102.9),
    mean_anomaly=np.radians(100.5),
    epoch=datetime(2000, 1, 1, 12, 0, 0),
)


class ImpactRiskAssessor:
    """Assess impact risk for a potentially hazardous asteroid.

    Parameters
    ----------
    earth_elements : OrbitalElements, optional
        Earth's orbital elements. Defaults to simplified J2000 values.
    moid_samples : int
        Number of true-anomaly samples per orbit for MOID computation.
    """

    def __init__(
        self,
        earth_elements: Optional[OrbitalElements] = None,
        moid_samples: int = 3600,
    ) -> None:
        self.earth = earth_elements or EARTH_ELEMENTS
        self.moid_samples = moid_samples
        self._propagator = OrbitPropagator()

    # ------------------------------------------------------------------
    # MOID computation
    # ------------------------------------------------------------------

    def _orbit_positions(
        self, elements: OrbitalElements, n_samples: int
    ) -> NDArray:
        """Sample heliocentric positions around an orbit.

        Returns shape (n_samples, 3).
        """
        a = elements.semi_major_axis
        e = elements.eccentricity
        Omega = elements.longitude_of_ascending_node
        omega = elements.argument_of_perihelion
        inc = elements.inclination

        Q = OrbitPropagator.perifocal_to_inertial(Omega, omega, inc)

        nus = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        positions = np.zeros((n_samples, 3))
        for i, nu in enumerate(nus):
            r_pf = OrbitPropagator.position_in_perifocal(a, e, nu)
            positions[i] = Q @ r_pf
        return positions

    def compute_moid(self, asteroid_elements: OrbitalElements) -> float:
        """Compute the Minimum Orbit Intersection Distance (AU).

        Samples positions on both orbits and returns the smallest
        pairwise distance.  This is an approximation; a production
        implementation would use the Gronchi (2005) algebraic method.
        """
        ast_pos = self._orbit_positions(asteroid_elements, self.moid_samples)
        earth_pos = self._orbit_positions(self.earth, self.moid_samples)

        # Brute-force minimum (vectorized for speed)
        min_dist = np.inf
        # Use a coarse-then-fine strategy
        coarse_n = min(360, self.moid_samples)
        step = max(1, self.moid_samples // coarse_n)

        ast_coarse = ast_pos[::step]
        earth_coarse = earth_pos[::step]

        for a_pt in ast_coarse:
            diffs = earth_coarse - a_pt
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            d = float(np.min(dists))
            if d < min_dist:
                min_dist = d

        return min_dist

    # ------------------------------------------------------------------
    # Kinetic energy estimate
    # ------------------------------------------------------------------

    @staticmethod
    def kinetic_energy_mt(
        diameter_km: float,
        velocity_km_s: float = 20.0,
        density_kg_m3: float = 2600.0,
    ) -> float:
        """Estimate kinetic energy in megatons of TNT.

        KE = 0.5 * m * v^2
        1 MT TNT = 4.184e15 J

        Parameters
        ----------
        diameter_km : float
            Asteroid diameter in kilometres.
        velocity_km_s : float
            Impact velocity in km/s (default 20 km/s, typical NEO).
        density_kg_m3 : float
            Bulk density in kg/m^3 (default 2600, S-type asteroid).
        """
        radius_m = diameter_km * 1e3 / 2.0
        volume_m3 = (4.0 / 3.0) * np.pi * radius_m**3
        mass_kg = density_kg_m3 * volume_m3
        v_m_s = velocity_km_s * 1e3
        ke_joules = 0.5 * mass_kg * v_m_s**2
        mt_tnt = ke_joules / 4.184e15
        return float(mt_tnt)

    # ------------------------------------------------------------------
    # Palermo scale
    # ------------------------------------------------------------------

    @staticmethod
    def palermo_scale(
        impact_probability: float,
        energy_mt: float,
        time_window_years: float,
    ) -> float:
        """Compute the Palermo Technical Impact Hazard Scale value.

        P = log10(p_i / (f_B * dT))

        where f_B = 0.03 * E^{-4/5} is the annual background impact
        frequency for an event of energy E megatons.
        """
        if energy_mt <= 0 or impact_probability <= 0:
            return -100.0  # effectively zero risk
        f_B = 0.03 * energy_mt ** (-4.0 / 5.0)
        return float(np.log10(impact_probability / (f_B * time_window_years)))

    # ------------------------------------------------------------------
    # Torino scale
    # ------------------------------------------------------------------

    @staticmethod
    def torino_scale(impact_probability: float, energy_mt: float) -> TorinoScale:
        """Compute the Torino Scale rating (0-10).

        Simplified decision tree following Binzel (2000).
        """
        if impact_probability <= 0 or energy_mt <= 0:
            return TorinoScale.NO_HAZARD

        # No chance of collision
        if impact_probability < 1e-7:
            return TorinoScale.NO_HAZARD

        # "Normal" -- meriting careful monitoring
        if impact_probability < 1e-4:
            if energy_mt < 1:
                return TorinoScale.NO_HAZARD
            elif energy_mt < 100:
                return TorinoScale.NORMAL
            else:
                return TorinoScale.MERITING_ATTENTION

        # Close encounter probabilities
        if impact_probability < 1e-2:
            if energy_mt < 1:
                return TorinoScale.NORMAL
            elif energy_mt < 100:
                return TorinoScale.CONCERNING
            elif energy_mt < 1e6:
                return TorinoScale.THREATENING
            else:
                return TorinoScale.SIGNIFICANT_THREAT

        # High probability
        if impact_probability < 0.99:
            if energy_mt < 1:
                return TorinoScale.MERITING_ATTENTION
            elif energy_mt < 100:
                return TorinoScale.CLOSE_ENCOUNTER
            elif energy_mt < 1e6:
                return TorinoScale.DANGEROUS_ENCOUNTER
            else:
                return TorinoScale.DANGEROUS_ENCOUNTER

        # Certain collision
        if energy_mt < 100:
            return TorinoScale.CERTAIN_COLLISION_LOCAL
        elif energy_mt < 1e6:
            return TorinoScale.CERTAIN_COLLISION_REGIONAL
        else:
            return TorinoScale.CERTAIN_COLLISION_GLOBAL

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        asteroid_elements: OrbitalElements,
        asteroid_id: str = "UNKNOWN",
        diameter_km: float = 0.1,
        impact_velocity_km_s: float = 20.0,
        time_window_years: float = 100.0,
    ) -> RiskAssessment:
        """Run a full impact risk assessment.

        Parameters
        ----------
        asteroid_elements : OrbitalElements
            Orbital elements of the asteroid.
        asteroid_id : str
            Identifier for the asteroid.
        diameter_km : float
            Estimated diameter in km.
        impact_velocity_km_s : float
            Estimated impact velocity in km/s.
        time_window_years : float
            Time horizon for impact probability estimate.

        Returns
        -------
        RiskAssessment
        """
        moid = self.compute_moid(asteroid_elements)

        # Simplified impact probability model:
        # p ~ cross-section / (2*pi*MOID * orbit_circumference)
        # This is a rough geometric estimate
        earth_radius_au = 4.26e-5  # ~6371 km in AU
        capture_radius = earth_radius_au * 3  # gravitational focusing
        if moid > 0.05:  # well outside Earth's sphere of influence
            impact_prob = 0.0
        else:
            # Probability scales inversely with MOID and with orbital period
            impact_prob = (capture_radius / max(moid, 1e-10)) ** 2 * 1e-6
            impact_prob = min(impact_prob, 1.0)

        energy = self.kinetic_energy_mt(diameter_km, impact_velocity_km_s)
        palermo = self.palermo_scale(impact_prob, energy, time_window_years)
        torino = self.torino_scale(impact_prob, energy)

        return RiskAssessment(
            asteroid_id=asteroid_id,
            moid_au=moid,
            impact_probability=impact_prob,
            palermo_scale=palermo,
            torino_scale=torino,
            kinetic_energy_mt=energy,
            time_to_closest_approach_years=time_window_years,
        )
