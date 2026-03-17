"""Tests for the detection pipeline."""

import numpy as np
import pytest

from akasha.detection.scanner import AsteroidScanner, Detection
from akasha.detection.classifier import NEOClassifier
from akasha.detection.tracker import TrajectoryTracker
from akasha.models import NEOType, OrbitalElements

from datetime import datetime


class TestAsteroidScanner:
    def test_estimate_background(self):
        rng = np.random.default_rng(42)
        image = rng.normal(100, 10, (64, 64))
        scanner = AsteroidScanner()
        med, std = scanner.estimate_background(image)
        assert abs(med - 100) < 3
        assert abs(std - 10) < 3

    def test_extract_sources(self):
        # Create an image with a bright point source
        image = np.ones((32, 32)) * 100.0
        image[16, 16] = 200.0  # bright source
        scanner = AsteroidScanner(sigma_threshold=3.0)
        dets = scanner.extract_sources(image)
        assert len(dets) >= 1
        # The brightest detection should be near (16, 16)
        bright = max(dets, key=lambda d: d.flux)
        assert abs(bright.x - 16) <= 1
        assert abs(bright.y - 16) <= 1

    def test_find_movers(self):
        scanner = AsteroidScanner(
            min_motion_arcsec=1.0,
            max_motion_arcsec=20.0,
            plate_scale=1.0,
        )
        det_a = [Detection(x=10.0, y=10.0, flux=100, snr=10, frame_index=0)]
        det_b = [Detection(x=15.0, y=10.0, flux=100, snr=10, frame_index=1)]
        pairs = scanner.find_movers(det_a, det_b, dt_seconds=60.0)
        assert len(pairs) == 1

    def test_find_movers_too_fast(self):
        scanner = AsteroidScanner(
            min_motion_arcsec=1.0,
            max_motion_arcsec=5.0,
            plate_scale=1.0,
        )
        det_a = [Detection(x=10.0, y=10.0, flux=100, snr=10, frame_index=0)]
        det_b = [Detection(x=30.0, y=10.0, flux=100, snr=10, frame_index=1)]
        pairs = scanner.find_movers(det_a, det_b, dt_seconds=60.0)
        assert len(pairs) == 0  # 20 pix > max_motion_arcsec=5


class TestNEOClassifier:
    @pytest.fixture
    def classifier(self):
        return NEOClassifier()

    def _make_elements(self, a, e):
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=e,
            inclination=0.1,
            longitude_of_ascending_node=0.0,
            argument_of_perihelion=0.0,
            mean_anomaly=0.0,
            epoch=datetime(2026, 1, 1),
        )

    def test_atira(self, classifier):
        # a=0.7, e=0.2 -> Q=0.84, q_E=0.983 -> Q < q_E -> Atira
        result = classifier.classify(self._make_elements(0.7, 0.2))
        assert result == NEOType.ATIRA

    def test_aten(self, classifier):
        # a=0.9, e=0.3 -> Q=1.17 >= q_E, a < 1 -> Aten
        result = classifier.classify(self._make_elements(0.9, 0.3))
        assert result == NEOType.ATEN

    def test_apollo(self, classifier):
        # a=1.2, e=0.3 -> q=0.84 <= Q_E=1.017 -> Apollo
        result = classifier.classify(self._make_elements(1.2, 0.3))
        assert result == NEOType.APOLLO

    def test_amor(self, classifier):
        # a=1.2, e=0.1 -> q=1.08 > Q_E=1.017 and < 1.3 -> Amor
        result = classifier.classify(self._make_elements(1.2, 0.1))
        assert result == NEOType.AMOR

    def test_unknown(self, classifier):
        # a=3.0, e=0.1 -> q=2.7 > 1.3 -> Unknown
        result = classifier.classify(self._make_elements(3.0, 0.1))
        assert result == NEOType.UNKNOWN


class TestTrajectoryTracker:
    def test_link_simple(self):
        """Three frames with a single object moving linearly."""
        tracker = TrajectoryTracker(
            search_radius_pix=3.0,
            max_velocity_pix_per_sec=1.0,
            min_tracklet_length=3,
        )
        frames = [
            [Detection(x=10.0, y=10.0, flux=100, snr=10, frame_index=0)],
            [Detection(x=12.0, y=10.0, flux=100, snr=10, frame_index=1)],
            [Detection(x=14.0, y=10.0, flux=100, snr=10, frame_index=2)],
        ]
        dt = [10.0, 10.0]  # 10 seconds between frames -> v=0.2 pix/s
        tracklets = tracker.link(frames, dt)
        assert len(tracklets) >= 1
        assert tracklets[0].num_points == 3

    def test_link_no_match(self):
        """Object disappears in third frame."""
        tracker = TrajectoryTracker(
            search_radius_pix=2.0,
            max_velocity_pix_per_sec=1.0,
            min_tracklet_length=3,
        )
        frames = [
            [Detection(x=10.0, y=10.0, flux=100, snr=10, frame_index=0)],
            [Detection(x=12.0, y=10.0, flux=100, snr=10, frame_index=1)],
            [Detection(x=50.0, y=50.0, flux=100, snr=10, frame_index=2)],  # far away
        ]
        dt = [10.0, 10.0]
        tracklets = tracker.link(frames, dt)
        assert len(tracklets) == 0  # can't form a 3-point tracklet
