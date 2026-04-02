"""
Trash Detector — only flags objects that:
1. Appear in the lower portion of the frame (near ground)
2. Have stopped moving (stationary for N frames)
3. Are NOT overlapping with a person
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from .config import TRASH_MIN_AREA, TRASH_LABEL


@dataclass
class TrashDetection:
    bbox: np.ndarray
    class_name: str = TRASH_LABEL
    confidence: float = 1.0
    class_id: int = -1


@dataclass
class TrackedBlob:
    bbox: np.ndarray        # current bbox
    stationary_frames: int  # how many frames it hasn't moved
    confirmed: bool = False # True once it's been still long enough


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


class TrashDetector:
    """
    Two-stage trash detection:
    Stage 1 — MOG2 finds foreground blobs
    Stage 2 — Only confirm blobs that are:
               - In the bottom 60% of the frame (ground level)
               - Stationary for STATIONARY_FRAMES_REQUIRED frames
               - Not overlapping with any person
    """

    STATIONARY_FRAMES_REQUIRED = 15   # ~0.5s at 30fps — tune this
    MOVE_THRESHOLD_PX           = 20  # pixels of centroid movement = "still moving"
    GROUND_ZONE_FRACTION        = 0.4 # only look in bottom 40% of frame

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=60,        # higher = less sensitive, fewer false positives
            detectShadows=False,
        )
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.tracked_blobs: List[TrackedBlob] = []

    # ─────────────────────────────────────────
    def detect(self, frame: np.ndarray, person_bboxes: List[np.ndarray]) -> List[TrashDetection]:
        h, w = frame.shape[:2]
        ground_y = int(h * (1.0 - self.GROUND_ZONE_FRACTION))  # e.g. y > 60% height

        # ── Stage 1: get foreground mask ──────
        fg = self.bg_subtractor.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open,  iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        # Mask out person regions + upper frame
        fg[:ground_y, :] = 0                          # ignore upper portion
        for bbox in person_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            pad = 25
            fg[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)] = 0

        # ── Stage 2: find blobs ───────────────
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_blobs: List[np.ndarray] = []

        for cnt in contours:
            if cv2.contourArea(cnt) < TRASH_MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Skip tall blobs (likely partial person)
            if bh / max(bw, 1) > 2.5:
                continue

            # Skip blobs overlapping any person
            bbox = np.array([x, y, x+bw, y+bh], dtype=float)
            overlap = any(_iou(bbox, pb) > 0.1 for pb in person_bboxes)
            if overlap:
                continue

            current_blobs.append(bbox)

        # ── Stage 3: track stationarity ───────
        updated: List[TrackedBlob] = []
        used = set()

        for blob in self.tracked_blobs:
            cx_old = (blob.bbox[0] + blob.bbox[2]) / 2
            cy_old = (blob.bbox[1] + blob.bbox[3]) / 2

            # Find best matching current blob
            best_idx, best_dist = -1, float("inf")
            for i, cb in enumerate(current_blobs):
                if i in used:
                    continue
                cx = (cb[0] + cb[2]) / 2
                cy = (cb[1] + cb[3]) / 2
                dist = np.hypot(cx - cx_old, cy - cy_old)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0 and best_dist < 60:  # matched
                used.add(best_idx)
                moved = best_dist > self.MOVE_THRESHOLD_PX
                updated.append(TrackedBlob(
                    bbox              = current_blobs[best_idx],
                    stationary_frames = 0 if moved else blob.stationary_frames + 1,
                    confirmed         = blob.confirmed or (
                        blob.stationary_frames + 1 >= self.STATIONARY_FRAMES_REQUIRED
                    )
                ))
            # else: blob disappeared — drop it

        # Add new unmatched blobs
        for i, cb in enumerate(current_blobs):
            if i not in used:
                updated.append(TrackedBlob(bbox=cb, stationary_frames=0))

        self.tracked_blobs = updated

        # ── Return only confirmed stationary trash ─
        return [
            TrashDetection(bbox=b.bbox)
            for b in self.tracked_blobs
            if b.confirmed
        ]