"""Asteroid detection in telescope image sequences.

Detects moving objects by comparing consecutive frames and identifying sources
whose positions shift between exposures at rates consistent with solar-system
bodies (typically 0.5-50 arcsec/min for NEOs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from akasha.models import Observation


@dataclass
class Detection:
    """A candidate moving-object detection in a single frame."""

    x: float               # pixel column
    y: float               # pixel row
    flux: float             # integrated flux (ADU)
    snr: float              # signal-to-noise ratio
    frame_index: int        # index into the image sequence
    ra: Optional[float] = None   # WCS-derived RA (degrees)
    dec: Optional[float] = None  # WCS-derived Dec (degrees)


class AsteroidScanner:
    """Detect moving objects in a sequence of telescope images.

    The algorithm:
    1. For each frame, estimate and subtract the background (sigma-clipped median).
    2. Identify point sources above ``sigma_threshold`` x background RMS.
    3. Match sources across consecutive frame pairs; flag those whose
       displacement exceeds the stellar-motion limit as candidate asteroids.

    Parameters
    ----------
    sigma_threshold : float
        Detection threshold in units of background standard deviation.
    min_motion_arcsec : float
        Minimum displacement (arcsec) between consecutive frames to be
        considered a moving object rather than a star.
    max_motion_arcsec : float
        Maximum displacement (arcsec) per inter-frame interval.
    plate_scale : float
        Arcseconds per pixel (default 1.0 for unit plate scale).
    """

    def __init__(
        self,
        sigma_threshold: float = 3.0,
        min_motion_arcsec: float = 0.5,
        max_motion_arcsec: float = 50.0,
        plate_scale: float = 1.0,
    ) -> None:
        self.sigma_threshold = sigma_threshold
        self.min_motion_arcsec = min_motion_arcsec
        self.max_motion_arcsec = max_motion_arcsec
        self.plate_scale = plate_scale

    # ------------------------------------------------------------------
    # Background estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_background(
        image: NDArray[np.floating],
        sigma_clip: float = 3.0,
        max_iters: int = 5,
    ) -> tuple[float, float]:
        """Iterative sigma-clipped background estimation.

        Returns (median, std) of the background after clipping outliers.
        """
        data = image.ravel().copy()
        for _ in range(max_iters):
            med = float(np.median(data))
            std = float(np.std(data))
            if std == 0:
                break
            mask = np.abs(data - med) < sigma_clip * std
            if mask.sum() == len(data):
                break
            data = data[mask]
        return float(np.median(data)), float(np.std(data))

    # ------------------------------------------------------------------
    # Source extraction
    # ------------------------------------------------------------------

    def extract_sources(
        self, image: NDArray[np.floating], frame_index: int = 0
    ) -> list[Detection]:
        """Find point sources above the detection threshold.

        Uses a simple local-maximum search after background subtraction.
        """
        bg_med, bg_std = self.estimate_background(image)
        if bg_std == 0:
            return []

        threshold = bg_med + self.sigma_threshold * bg_std
        # Label connected regions above threshold (simple row-scan)
        binary = image > threshold
        detections: list[Detection] = []

        # Find local peaks: pixel brighter than its 4-connected neighbours
        padded = np.pad(image, 1, mode="constant", constant_values=0.0)
        rows, cols = image.shape
        for r in range(rows):
            for c in range(cols):
                if not binary[r, c]:
                    continue
                val = padded[r + 1, c + 1]
                neighbours = [
                    padded[r, c + 1],
                    padded[r + 2, c + 1],
                    padded[r + 1, c],
                    padded[r + 1, c + 2],
                ]
                if val >= max(neighbours):
                    flux = float(val - bg_med)
                    snr = flux / bg_std if bg_std > 0 else 0.0
                    detections.append(
                        Detection(
                            x=float(c),
                            y=float(r),
                            flux=flux,
                            snr=snr,
                            frame_index=frame_index,
                        )
                    )
        return detections

    # ------------------------------------------------------------------
    # Motion detection
    # ------------------------------------------------------------------

    def find_movers(
        self,
        detections_a: list[Detection],
        detections_b: list[Detection],
        dt_seconds: float,
    ) -> list[tuple[Detection, Detection]]:
        """Match detections between two frames and return moving-object pairs.

        Parameters
        ----------
        detections_a, detections_b : list[Detection]
            Source lists for two consecutive frames.
        dt_seconds : float
            Time interval between frames in seconds.

        Returns
        -------
        list of (det_a, det_b) pairs whose inter-frame motion falls within
        [min_motion_arcsec, max_motion_arcsec].
        """
        if dt_seconds <= 0:
            return []

        min_pix = self.min_motion_arcsec / self.plate_scale
        max_pix = self.max_motion_arcsec / self.plate_scale

        pairs: list[tuple[Detection, Detection]] = []
        for da in detections_a:
            for db in detections_b:
                dx = db.x - da.x
                dy = db.y - da.y
                dist = np.hypot(dx, dy)
                if min_pix <= dist <= max_pix:
                    pairs.append((da, db))
        return pairs

    # ------------------------------------------------------------------
    # Full-sequence scan
    # ------------------------------------------------------------------

    def scan_sequence(
        self,
        images: list[NDArray[np.floating]],
        timestamps_sec: list[float],
        wcs_transform=None,
    ) -> list[list[tuple[Detection, Detection]]]:
        """Scan a full image sequence for moving objects.

        Parameters
        ----------
        images : list of 2-D arrays
            Ordered telescope frames.
        timestamps_sec : list of float
            Timestamps (seconds since reference epoch) for each frame.
        wcs_transform : callable, optional
            Function (x, y) -> (ra_deg, dec_deg) for coordinate conversion.

        Returns
        -------
        list of pair-lists, one per consecutive frame pair.
        """
        all_detections: list[list[Detection]] = []
        for idx, img in enumerate(images):
            dets = self.extract_sources(img, frame_index=idx)
            if wcs_transform is not None:
                for d in dets:
                    d.ra, d.dec = wcs_transform(d.x, d.y)
            all_detections.append(dets)

        results: list[list[tuple[Detection, Detection]]] = []
        for i in range(len(images) - 1):
            dt = timestamps_sec[i + 1] - timestamps_sec[i]
            movers = self.find_movers(all_detections[i], all_detections[i + 1], dt)
            results.append(movers)
        return results
