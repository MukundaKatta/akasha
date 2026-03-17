"""Tests for orbit determination, propagation, and risk assessment."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from akasha.models import OrbitalElements, TorinoScale
from akasha.orbit.calculator import OrbitCalculator
from akasha.orbit.propagator import OrbitPropagator
from akasha.orbit.risk import ImpactRiskAssessor


@pytest.fixture
def circular_orbit():
    """Circular orbit at 1.5 AU, zero inclination."""
    return OrbitalElements(
        semi_major_axis=1.5,
        eccentricity=0.0,
        inclination=0.0,
        longitude_of_ascending_node=0.0,
        argument_of_perihelion=0.0,
        mean_anomaly=0.0,
        epoch=datetime(2026, 1, 1),
    )


@pytest.fixture
def eccentric_orbit():
    """Moderately eccentric orbit."""
    return OrbitalElements(
        semi_major_axis=1.2,
        eccentricity=0.4,
        inclination=np.radians(15.0),
        longitude_of_ascending_node=np.radians(60.0),
        argument_of_perihelion=np.radians(45.0),
        mean_anomaly=np.radians(0.0),
        epoch=datetime(2026, 1, 1),
    )


class TestOrbitPropagator:
    def test_solve_kepler_circular(self):
        prop = OrbitPropagator()
        # For e=0, E should equal M
        for M in [0.0, 0.5, 1.0, np.pi, 5.0]:
            E = prop.solve_kepler(M, 0.0)
            assert abs(E - M) < 1e-10

    def test_solve_kepler_eccentric(self):
        prop = OrbitPropagator()
        e = 0.5
        M = 1.0
        E = prop.solve_kepler(M, e)
        # Verify: M = E - e*sin(E)
        assert abs(E - e * np.sin(E) - M) < 1e-10

    def test_solve_kepler_high_eccentricity(self):
        prop = OrbitPropagator()
        e = 0.95
        M = 0.1
        E = prop.solve_kepler(M, e)
        assert abs(E - e * np.sin(E) - M) < 1e-10

    def test_propagate_circular_returns_to_start(self, circular_orbit):
        """After one full period, position should return to start."""
        prop = OrbitPropagator()
        r0, v0 = prop.propagate(circular_orbit, circular_orbit.epoch)
        period_days = circular_orbit.period * 365.25
        t_end = circular_orbit.epoch + timedelta(days=period_days)
        r1, v1 = prop.propagate(circular_orbit, t_end)
        np.testing.assert_allclose(r0, r1, atol=1e-8)
        np.testing.assert_allclose(v0, v1, atol=1e-8)

    def test_propagate_position_at_epoch(self, circular_orbit):
        """At epoch with M=0, e=0, should be at perihelion on the x-axis."""
        prop = OrbitPropagator()
        r, v = prop.propagate(circular_orbit, circular_orbit.epoch)
        # For circular orbit with M=0: true anomaly=0, position = (a, 0, 0)
        assert abs(r[0] - 1.5) < 1e-8
        assert abs(r[1]) < 1e-8
        assert abs(r[2]) < 1e-8

    def test_propagate_conserves_distance_circular(self, circular_orbit):
        """Distance from Sun should be constant for circular orbit."""
        prop = OrbitPropagator()
        for day in range(0, 365, 30):
            t = circular_orbit.epoch + timedelta(days=day)
            r, _ = prop.propagate(circular_orbit, t)
            dist = float(np.linalg.norm(r))
            assert abs(dist - 1.5) < 1e-6

    def test_eccentric_anomaly_to_true(self):
        # At E=0, true anomaly should also be 0
        nu = OrbitPropagator.eccentric_to_true(0.0, 0.5)
        assert abs(nu) < 1e-10

        # At E=pi, true anomaly should be pi
        nu = OrbitPropagator.eccentric_to_true(np.pi, 0.5)
        assert abs(nu - np.pi) < 1e-10


class TestOrbitCalculator:
    def test_state_to_elements_circular(self):
        """Convert a known circular-orbit state back to elements."""
        calc = OrbitCalculator()
        # Circular orbit at 1 AU in the xy-plane
        r = np.array([1.0, 0.0, 0.0])
        # v = sqrt(mu/a) in y-direction for circular orbit
        v_mag = np.sqrt(calc.mu / 1.0)
        v = np.array([0.0, v_mag, 0.0])
        epoch = datetime(2026, 1, 1)
        elements = calc.state_to_elements(r, v, epoch)
        assert abs(elements.semi_major_axis - 1.0) < 0.01
        assert elements.eccentricity < 0.01
        assert abs(elements.inclination) < 0.01

    def test_state_to_elements_eccentric(self):
        """Convert an eccentric orbit state and check consistency."""
        calc = OrbitCalculator()
        # At perihelion of an e=0.3, a=1.5 orbit: r = a(1-e) = 1.05
        a, e = 1.5, 0.3
        r_peri = a * (1 - e)
        r = np.array([r_peri, 0.0, 0.0])
        # At perihelion: v = sqrt(mu * (2/r - 1/a))
        v_mag = np.sqrt(calc.mu * (2.0 / r_peri - 1.0 / a))
        v = np.array([0.0, v_mag, 0.0])
        epoch = datetime(2026, 1, 1)
        elements = calc.state_to_elements(r, v, epoch)
        assert abs(elements.semi_major_axis - a) < 0.01
        assert abs(elements.eccentricity - e) < 0.01


class TestImpactRiskAssessor:
    def test_kinetic_energy(self):
        # 1 km asteroid at 20 km/s
        ke = ImpactRiskAssessor.kinetic_energy_mt(1.0, 20.0)
        # Should be on the order of ~10^4 MT for a 1km asteroid
        assert ke > 1000
        assert ke < 1e6

    def test_palermo_scale_zero_prob(self):
        p = ImpactRiskAssessor.palermo_scale(0.0, 100.0, 100.0)
        assert p == -100.0

    def test_palermo_scale_value(self):
        # Known calculation: p_i=1e-4, E=100 MT, dT=100 yr
        # f_B = 0.03 * 100^(-0.8) = 0.03 * 0.01585 = 4.756e-4
        # P = log10(1e-4 / (4.756e-4 * 100)) = log10(1e-4 / 0.04756) = log10(2.103e-3) ~ -2.677
        p = ImpactRiskAssessor.palermo_scale(1e-4, 100.0, 100.0)
        assert -3.0 < p < -2.0

    def test_torino_no_hazard(self):
        t = ImpactRiskAssessor.torino_scale(1e-9, 10.0)
        assert t == TorinoScale.NO_HAZARD

    def test_torino_certain_global(self):
        t = ImpactRiskAssessor.torino_scale(1.0, 1e7)
        assert t == TorinoScale.CERTAIN_COLLISION_GLOBAL

    def test_moid_distant_orbit(self):
        """An orbit far from Earth should have large MOID."""
        assessor = ImpactRiskAssessor(moid_samples=360)
        elements = OrbitalElements(
            semi_major_axis=5.0,
            eccentricity=0.05,
            inclination=0.0,
            longitude_of_ascending_node=0.0,
            argument_of_perihelion=0.0,
            mean_anomaly=0.0,
            epoch=datetime(2026, 1, 1),
        )
        moid = assessor.compute_moid(elements)
        # Perihelion of this orbit is 4.75 AU, far from Earth at ~1 AU
        assert moid > 3.0

    def test_full_assessment(self):
        assessor = ImpactRiskAssessor(moid_samples=360)
        elements = OrbitalElements(
            semi_major_axis=1.1,
            eccentricity=0.3,
            inclination=np.radians(5.0),
            longitude_of_ascending_node=0.0,
            argument_of_perihelion=0.0,
            mean_anomaly=0.0,
            epoch=datetime(2026, 1, 1),
        )
        result = assessor.assess(elements, asteroid_id="TEST-001", diameter_km=0.5)
        assert result.asteroid_id == "TEST-001"
        assert result.moid_au >= 0
        assert result.kinetic_energy_mt > 0
