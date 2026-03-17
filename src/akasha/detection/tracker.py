"""Multi-frame detection linker (trajectory tracker).

Links detections of the same moving object across three or more frames,
building consistent *tracklets* that can be fed into orbit determination.

The linking uses a simple nearest-neighbour approach with velocity
consistency checks: if a candidate pair (frame i -> i+1) implies a velocity
vector that, when extrapolated to frame i+2, lands within a search radius
of an actual detection, the three-point tracklet is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from akasha.detection.scanner import Detection


@dataclass
class Tracklet:
    """A linked set of detections believed to be the same object."""

    tracklet_id: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_points(self) -> int:
        return len(self.detections)

    @property
    def mean_snr(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.snr for d in self.detections]))

    def velocity_pixels_per_sec(self, dt_seconds: list[float]) -> Optional[tuple[float, float]]:
        """Average velocity (vx, vy) in pixels/sec from the tracklet endpoints."""
        if len(self.detections) < 2 or not dt_seconds:
            return None
        total_dt = sum(dt_seconds[: len(self.detections) - 1])
        if total_dt == 0:
            return None
        dx = self.detections[-1].x - self.detections[0].x
        dy = self.detections[-1].y - self.detections[0].y
        return (dx / total_dt, dy / total_dt)


class TrajectoryTracker:
    """Link detections across multiple frames into tracklets.

    Parameters
    ----------
    search_radius_pix : float
        Maximum residual (pixels) when extrapolating a pair's velocity to
        the next frame.
    max_velocity_pix_per_sec : float
        Hard upper bound on apparent velocity to reject cosmic rays and
        detector artefacts.
    min_tracklet_length : int
        Minimum number of frames a tracklet must span to be accepted.
    """

    def __init__(
        self,
        search_radius_pix: float = 5.0,
        max_velocity_pix_per_sec: float = 2.0,
        min_tracklet_length: int = 3,
    ) -> None:
        self.search_radius = search_radius_pix
        self.max_vel = max_velocity_pix_per_sec
        self.min_length = min_tracklet_length
        self._next_id = 0

    def _make_id(self) -> str:
        tid = f"TRK-{self._next_id:04d}"
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------
    # Core linking
    # ------------------------------------------------------------------

    def link(
        self,
        frame_detections: list[list[Detection]],
        dt_seconds: list[float],
    ) -> list[Tracklet]:
        """Link detections across frames into tracklets.

        Parameters
        ----------
        frame_detections : list of list[Detection]
            Detections per frame, ordered chronologically.
        dt_seconds : list of float
            Time gaps between consecutive frames (length = n_frames - 1).

        Returns
        -------
        list[Tracklet]
            Accepted tracklets with >= ``min_tracklet_length`` points.
        """
        n_frames = len(frame_detections)
        if n_frames < 2:
            return []

        # Build candidate pairs from frames 0->1
        active_tracklets: list[Tracklet] = []
        for da in frame_detections[0]:
            for db in frame_detections[1]:
                dt = dt_seconds[0]
                if dt <= 0:
                    continue
                vx = (db.x - da.x) / dt
                vy = (db.y - da.y) / dt
                speed = np.hypot(vx, vy)
                if speed > self.max_vel:
                    continue
                t = Tracklet(tracklet_id=self._make_id(), detections=[da, db])
                active_tracklets.append(t)

        # Extend tracklets through subsequent frames
        for fi in range(2, n_frames):
            dt = dt_seconds[fi - 1]
            if dt <= 0:
                continue
            next_dets = frame_detections[fi]
            for trk in active_tracklets:
                if trk.num_points < fi:
                    continue  # already lost
                # Predict position from last two points
                d_prev = trk.detections[-2]
                d_last = trk.detections[-1]
                dt_prev = dt_seconds[fi - 2] if fi >= 2 else dt
                if dt_prev == 0:
                    continue
                vx = (d_last.x - d_prev.x) / dt_prev
                vy = (d_last.y - d_prev.y) / dt_prev
                pred_x = d_last.x + vx * dt
                pred_y = d_last.y + vy * dt

                best_det = None
                best_dist = self.search_radius + 1
                for nd in next_dets:
                    dist = np.hypot(nd.x - pred_x, nd.y - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = nd

                if best_det is not None and best_dist <= self.search_radius:
                    trk.detections.append(best_det)

        # Filter by minimum length
        return [t for t in active_tracklets if t.num_points >= self.min_length]
