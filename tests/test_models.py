"""Tests for Pydantic data models."""

from datetime import datetime

import numpy as np
import pytest

from akasha.models import (
    Asteroid,
    NEOType,
    Observation,
    OrbitalElements,
    RiskAssessment,
    TorinoScale,
)


@pytest.fixture
def sample_elements():
    return OrbitalElements(
        semi_major_axis=1.5,
        eccentricity=0.3,
        inclination=np.radians(10.0),
        longitude_of_ascending_node=np.radians(45.0),
        argument_of_perihelion=np.radians(90.0),
        mean_anomaly=np.radians(30.0),
        epoch=datetime(2026, 1, 1),
    )


@pytest.fixture
def sample_observation():
    return Observation(
        obs_id="OBS-001",
        timestamp=datetime(2026, 1, 15, 3, 0, 0),
        ra=120.0,
        dec=25.0,
        magnitude=19.5,
    )


class TestOrbitalElements:
    def test_perihelion(self, sample_elements):
        expected = 1.5 * (1.0 - 0.3)
        assert abs(sample_elements.perihelion - expected) < 1e-10

    def test_aphelion(self, sample_elements):
        expected = 1.5 * (1.0 + 0.3)
        assert abs(sample_elements.aphelion - expected) < 1e-10

    def test_period(self, sample_elements):
        # T = 2*pi*sqrt(a^3/mu), mu = 4*pi^2
        # T = 2*pi*sqrt(1.5^3 / (4*pi^2)) = sqrt(1.5^3) = 1.5^1.5
        expected = 1.5**1.5
        assert abs(sample_elements.period - expected) < 1e-10

    def test_mean_motion(self, sample_elements):
        expected = 2 * np.pi / sample_elements.period
        assert abs(sample_elements.mean_motion - expected) < 1e-10


class TestObservation:
    def test_creation(self, sample_observation):
        assert sample_observation.ra == 120.0
        assert sample_observation.dec == 25.0
        assert sample_observation.magnitude == 19.5

    def test_defaults(self, sample_observation):
        assert sample_observation.ra_uncertainty == 0.5
        assert sample_observation.dec_uncertainty == 0.5
        assert sample_observation.frame_id is None


class TestAsteroid:
    def test_creation(self, sample_elements, sample_observation):
        ast = Asteroid(
            asteroid_id="AST-001",
            name="TestAsteroid",
            observations=[sample_observation],
            orbital_elements=sample_elements,
            neo_type=NEOType.APOLLO,
        )
        assert ast.asteroid_id == "AST-001"
        assert ast.neo_type == NEOType.APOLLO
        assert len(ast.observations) == 1

    def test_defaults(self):
        ast = Asteroid(asteroid_id="AST-002")
        assert ast.neo_type == NEOType.UNKNOWN
        assert ast.orbital_elements is None
        assert ast.observations == []


class TestRiskAssessment:
    def test_creation(self):
        ra = RiskAssessment(
            asteroid_id="AST-001",
            moid_au=0.002,
            impact_probability=1e-5,
            palermo_scale=-2.5,
            torino_scale=TorinoScale.NORMAL,
            kinetic_energy_mt=50.0,
            time_to_closest_approach_years=25.0,
        )
        assert ra.torino_scale == TorinoScale.NORMAL
        assert ra.moid_au == 0.002
