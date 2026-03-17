"""Pydantic data models for AKASHA."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class NEOType(str, enum.Enum):
    """Near-Earth object orbit classification.

    Classification follows the standard scheme based on semi-major axis (a),
    perihelion distance (q), and aphelion distance (Q) relative to Earth's
    orbit (a_E ~ 1.0 AU, q_E ~ 0.983 AU, Q_E ~ 1.017 AU).
    """

    ATIRA = "Atira"      # a < 1.0 AU, Q < q_E  (orbit entirely inside Earth's)
    ATEN = "Aten"        # a < 1.0 AU, Q >= q_E  (Earth-crossing, a < 1)
    APOLLO = "Apollo"    # a >= 1.0 AU, q <= Q_E  (Earth-crossing, a >= 1)
    AMOR = "Amor"        # a >= 1.0 AU, q > Q_E, q < 1.3 AU  (Mars-crossing, near-Earth)
    UNKNOWN = "Unknown"


class TorinoScale(int, enum.Enum):
    """Torino Impact Hazard Scale (0-10)."""

    NO_HAZARD = 0
    NORMAL = 1
    MERITING_ATTENTION = 2
    CONCERNING = 3
    CLOSE_ENCOUNTER = 4
    THREATENING = 5
    SIGNIFICANT_THREAT = 6
    DANGEROUS_ENCOUNTER = 7
    CERTAIN_COLLISION_LOCAL = 8
    CERTAIN_COLLISION_REGIONAL = 9
    CERTAIN_COLLISION_GLOBAL = 10


class Observation(BaseModel):
    """A single astrometric observation of a celestial object."""

    obs_id: str = Field(description="Unique observation identifier")
    timestamp: datetime = Field(description="UTC time of observation")
    ra: float = Field(description="Right ascension in degrees [0, 360)")
    dec: float = Field(description="Declination in degrees [-90, 90]")
    magnitude: float = Field(description="Apparent visual magnitude")
    ra_uncertainty: float = Field(default=0.5, description="RA uncertainty in arcseconds")
    dec_uncertainty: float = Field(default=0.5, description="Dec uncertainty in arcseconds")
    frame_id: Optional[str] = Field(default=None, description="Source image frame ID")
    asteroid_id: Optional[str] = Field(default=None, description="Linked asteroid ID, if known")

    class Config:
        arbitrary_types_allowed = True


class OrbitalElements(BaseModel):
    """Classical (Keplerian) orbital elements.

    All angles are in radians unless otherwise noted.  Distances are in AU,
    and the gravitational parameter mu is in AU^3 / yr^2.
    """

    semi_major_axis: float = Field(description="Semi-major axis a (AU)")
    eccentricity: float = Field(description="Eccentricity e [0, 1)")
    inclination: float = Field(description="Inclination i (radians)")
    longitude_of_ascending_node: float = Field(description="Longitude of ascending node Omega (radians)")
    argument_of_perihelion: float = Field(description="Argument of perihelion omega (radians)")
    mean_anomaly: float = Field(description="Mean anomaly M at epoch (radians)")
    epoch: datetime = Field(description="Reference epoch for mean anomaly")
    mu: float = Field(
        default=4 * np.pi**2,
        description="Gravitational parameter (AU^3/yr^2); default is solar mu",
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def perihelion(self) -> float:
        """Perihelion distance q = a(1 - e)."""
        return self.semi_major_axis * (1.0 - self.eccentricity)

    @property
    def aphelion(self) -> float:
        """Aphelion distance Q = a(1 + e)."""
        return self.semi_major_axis * (1.0 + self.eccentricity)

    @property
    def period(self) -> float:
        """Orbital period in years: T = 2*pi * sqrt(a^3 / mu)."""
        return 2.0 * np.pi * np.sqrt(self.semi_major_axis**3 / self.mu)

    @property
    def mean_motion(self) -> float:
        """Mean motion n = 2*pi / T  (radians / year)."""
        return 2.0 * np.pi / self.period


class Asteroid(BaseModel):
    """An asteroid with observations and, optionally, a computed orbit."""

    asteroid_id: str = Field(description="Unique asteroid identifier")
    name: Optional[str] = Field(default=None, description="Human-readable name")
    observations: list[Observation] = Field(default_factory=list)
    orbital_elements: Optional[OrbitalElements] = Field(default=None)
    neo_type: NEOType = Field(default=NEOType.UNKNOWN)
    absolute_magnitude: Optional[float] = Field(
        default=None, description="Absolute magnitude H"
    )
    diameter_km: Optional[float] = Field(
        default=None, description="Estimated diameter in km"
    )

    class Config:
        arbitrary_types_allowed = True


class RiskAssessment(BaseModel):
    """Impact-risk assessment for a single asteroid."""

    asteroid_id: str
    moid_au: float = Field(description="Minimum Orbit Intersection Distance (AU)")
    impact_probability: float = Field(description="Estimated impact probability [0,1]")
    palermo_scale: float = Field(description="Palermo Technical Impact Hazard Scale value")
    torino_scale: TorinoScale = Field(description="Torino Scale rating (0-10)")
    kinetic_energy_mt: float = Field(
        description="Estimated kinetic energy of impact in megatons of TNT"
    )
    time_to_closest_approach_years: float = Field(
        description="Years until closest predicted approach"
    )
    notes: str = Field(default="")
