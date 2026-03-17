"""Tests for the synthetic observation generator."""

import json
from datetime import datetime

import pytest

from akasha.simulator import AsteroidSimulator


class TestAsteroidSimulator:
    def test_random_elements(self):
        sim = AsteroidSimulator(seed=42)
        epoch = datetime(2026, 1, 1)
        elements = sim.random_elements(epoch)
        assert 0.5 <= elements.semi_major_axis <= 3.0
        assert 0.0 <= elements.eccentricity <= 0.6
        assert elements.epoch == epoch

    def test_generate_observations(self):
        sim = AsteroidSimulator(seed=42)
        epoch = datetime(2026, 1, 1)
        elements = sim.random_elements(epoch)
        obs = sim.generate_observations(
            elements, "AST-TEST", epoch, num_frames=5
        )
        assert len(obs) == 5
        for o in obs:
            assert o.asteroid_id == "AST-TEST"
            assert 0 <= o.ra < 360
            assert -90 <= o.dec <= 90

    def test_generate_population(self):
        sim = AsteroidSimulator(seed=123)
        pop = sim.generate_population(num_asteroids=5, num_frames=3)
        assert len(pop) == 5
        for ast in pop:
            assert len(ast.observations) == 3
            assert ast.orbital_elements is not None
            assert ast.diameter_km is not None

    def test_to_json(self):
        sim = AsteroidSimulator(seed=99)
        pop = sim.generate_population(num_asteroids=2, num_frames=3)
        j = sim.to_json(pop)
        data = json.loads(j)
        assert len(data) == 2
        assert "asteroid_id" in data[0]

    def test_reproducibility(self):
        sim1 = AsteroidSimulator(seed=42)
        sim2 = AsteroidSimulator(seed=42)
        pop1 = sim1.generate_population(num_asteroids=3, num_frames=2)
        pop2 = sim2.generate_population(num_asteroids=3, num_frames=2)
        for a1, a2 in zip(pop1, pop2):
            assert a1.asteroid_id == a2.asteroid_id
            assert a1.orbital_elements.semi_major_axis == a2.orbital_elements.semi_major_axis
