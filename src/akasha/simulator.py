"""Synthetic asteroid observation generator.

Creates realistic test data by:
1. Generating random Keplerian orbits with configurable parameter ranges.
2. Propagating each asteroid to the requested observation epochs.
3. Converting heliocentric positions to geocentric RA/Dec with optional noise.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from akasha.models import Asteroid, NEOType, Observation, OrbitalElements
from akasha.orbit.propagator import OrbitPropagator

SEC_PER_YEAR = 365.25 * 86400.0


class AsteroidSimulator:
    """Generate synthetic asteroid populations and observations.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.propagator = OrbitPropagator()

    # ------------------------------------------------------------------
    # Random orbit generation
    # ------------------------------------------------------------------

    def random_elements(
        self,
        epoch: datetime,
        a_range: tuple[float, float] = (0.5, 3.0),
        e_range: tuple[float, float] = (0.0, 0.6),
        i_range_deg: tuple[float, float] = (0.0, 45.0),
    ) -> OrbitalElements:
        """Generate random Keplerian elements.

        Parameters
        ----------
        epoch : datetime
            Reference epoch.
        a_range : tuple
            Semi-major axis range (AU).
        e_range : tuple
            Eccentricity range.
        i_range_deg : tuple
            Inclination range (degrees).
        """
        a = float(self.rng.uniform(*a_range))
        e = float(self.rng.uniform(*e_range))
        inc = float(np.radians(self.rng.uniform(*i_range_deg)))
        Omega = float(self.rng.uniform(0, 2 * np.pi))
        omega = float(self.rng.uniform(0, 2 * np.pi))
        M = float(self.rng.uniform(0, 2 * np.pi))

        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=e,
            inclination=inc,
            longitude_of_ascending_node=Omega,
            argument_of_perihelion=omega,
            mean_anomaly=M,
            epoch=epoch,
        )

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def _heliocentric_to_radec(
        self, r_helio: np.ndarray, observer_pos: np.ndarray
    ) -> tuple[float, float]:
        """Convert heliocentric position to geocentric RA/Dec (degrees)."""
        r_geo = r_helio - observer_pos
        x, y, z = r_geo
        r = np.linalg.norm(r_geo)
        dec = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        ra = np.degrees(np.arctan2(y, x)) % 360.0
        return float(ra), float(dec)

    def generate_observations(
        self,
        elements: OrbitalElements,
        asteroid_id: str,
        start: datetime,
        num_frames: int = 5,
        cadence_minutes: float = 30.0,
        noise_arcsec: float = 0.5,
        magnitude_base: float = 18.0,
        magnitude_scatter: float = 0.3,
    ) -> list[Observation]:
        """Generate synthetic observations for one asteroid.

        Parameters
        ----------
        elements : OrbitalElements
            True orbital elements.
        asteroid_id : str
            Identifier to assign.
        start : datetime
            Start time of the observing sequence.
        num_frames : int
            Number of observation epochs.
        cadence_minutes : float
            Time between frames in minutes.
        noise_arcsec : float
            Gaussian astrometric noise (1-sigma) in arcseconds.
        magnitude_base : float
            Base apparent magnitude.
        magnitude_scatter : float
            Magnitude scatter (1-sigma).
        """
        observations: list[Observation] = []
        # Simplified observer at 1 AU on x-axis (no Earth orbit propagation)
        observer_pos = np.array([1.0, 0.0, 0.0])

        for i in range(num_frames):
            t = start + timedelta(minutes=cadence_minutes * i)
            r_vec, _ = self.propagator.propagate(elements, t)
            ra, dec = self._heliocentric_to_radec(r_vec, observer_pos)

            # Add noise
            noise_deg = noise_arcsec / 3600.0
            ra += float(self.rng.normal(0, noise_deg / np.cos(np.radians(dec))))
            dec += float(self.rng.normal(0, noise_deg))
            ra = ra % 360.0
            dec = float(np.clip(dec, -90, 90))

            mag = magnitude_base + float(self.rng.normal(0, magnitude_scatter))

            observations.append(
                Observation(
                    obs_id=f"{asteroid_id}-{i:03d}",
                    timestamp=t,
                    ra=ra,
                    dec=dec,
                    magnitude=mag,
                    ra_uncertainty=noise_arcsec,
                    dec_uncertainty=noise_arcsec,
                    frame_id=f"frame-{i:03d}",
                    asteroid_id=asteroid_id,
                )
            )
        return observations

    # ------------------------------------------------------------------
    # Population generation
    # ------------------------------------------------------------------

    def generate_population(
        self,
        num_asteroids: int = 10,
        num_frames: int = 5,
        start: Optional[datetime] = None,
        cadence_minutes: float = 30.0,
        noise_arcsec: float = 0.5,
    ) -> list[Asteroid]:
        """Generate a synthetic asteroid population with observations.

        Returns a list of Asteroid objects, each with generated observations
        and known true orbital elements.
        """
        if start is None:
            start = datetime(2026, 1, 15, 3, 0, 0)

        asteroids: list[Asteroid] = []
        for idx in range(num_asteroids):
            aid = f"AST-{idx:03d}"
            elements = self.random_elements(start)
            obs = self.generate_observations(
                elements,
                aid,
                start,
                num_frames=num_frames,
                cadence_minutes=cadence_minutes,
                noise_arcsec=noise_arcsec,
            )
            diameter = float(self.rng.lognormal(np.log(0.1), 1.0))
            H = 18.0 + float(self.rng.normal(0, 2))
            asteroids.append(
                Asteroid(
                    asteroid_id=aid,
                    name=f"Synthetic-{idx:03d}",
                    observations=obs,
                    orbital_elements=elements,
                    absolute_magnitude=H,
                    diameter_km=diameter,
                )
            )
        return asteroids

    @staticmethod
    def to_json(asteroids: list[Asteroid]) -> str:
        """Serialize a list of asteroids to JSON."""
        data = [ast.model_dump(mode="json") for ast in asteroids]
        return json.dumps(data, indent=2, default=str)
