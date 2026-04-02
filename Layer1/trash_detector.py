"""
Trash Detector v4 — Drop Event Detection

Logic:
  Phase 1 (HELD):   Object bbox overlaps with person bbox
  Phase 2 (DROPPED): Object was previously held, now no longer
                     overlapping person AND is in ground zone
                     → immediately flag as TRASH

This catches the exact moment of dropping, not after a long wait.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .detector import Detection
from enum import Enum


class ObjectState(Enum):
    APPEARING  = "appearing"   # just detected, not yet held
    HELD       = "held"        # overlapping with a person
    DROPPED    = "dropped"     # was held, now separated from person on ground
    GROUNDED   = "grounded"    # was near ground and person walked away


@dataclass
class TrashDetection:
    bbox:       np.ndarray
    class_name: str   = "trash"
    confidence: float = 1.0
    class_id:   int   = -1
    label:      str   = ""


@dataclass
class _TrackedObject:
    det:          Detection
    state:        ObjectState = ObjectState.APPEARING
    held_frames:  int = 0          # how many frames was it held
    frames_seen:  int = 0
    confirmed:    bool = False     # True = show red trash box


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter)


def _centroid(bbox):
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


def _is_near_person(obj_bbox: np.ndarray, persons: List[Detection], iou_thresh=0.08) -> bool:
    """True if object is overlapping or very close to any person."""
    for p in persons:
        if _iou(obj_bbox, p.bbox) > iou_thresh:
            return True
        # Also check if object centroid is inside person bbox (for carried bags)
        cx, cy = _centroid(obj_bbox)
        px1, py1, px2, py2 = p.bbox
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
    return False


def _is_on_ground(obj_bbox: np.ndarray, frame_h: int, ground_fraction=0.35) -> bool:
    """True if bottom of object is in the lower ground_fraction of frame."""
    ground_y = frame_h * (1.0 - ground_fraction)
    return obj_bbox[3] >= ground_y   # y2 (bottom edge) in ground zone


class TrashDetector:
    MATCH_DIST_PX     = 80    # px to match same object across frames
    MIN_HELD_FRAMES   = 3     # must be held for at least this many frames to count as "carried"
    GROUND_FRACTION   = 0.40  # bottom 40% = ground zone

    def __init__(self):
        self._tracked: List[_TrackedObject] = []

    def detect(
        self,
        frame_shape: tuple,
        all_detections: List[Detection],
    ) -> List[TrashDetection]:

        H = frame_shape[0]
        persons = [d for d in all_detections if d.class_name == "person"]
        objects = [d for d in all_detections if d.class_name != "person"]

        # ── Match current detections → existing tracked objects ──
        updated: List[_TrackedObject] = []
        used = set()

        for tracked in self._tracked:
            cx, cy = _centroid(tracked.det.bbox)
            best_i, best_d = -1, float("inf")

            for i, obj in enumerate(objects):
                if i in used:
                    continue
                ox, oy = _centroid(obj.bbox)
                d = np.hypot(cx-ox, cy-oy)
                if d < best_d:
                    best_d, best_i = d, i

            if best_i >= 0 and best_d < self.MATCH_DIST_PX:
                used.add(best_i)
                obj = objects[best_i]
                near_person = _is_near_person(obj.bbox, persons)
                on_ground   = _is_on_ground(obj.bbox, H, self.GROUND_FRACTION)

                # ── State machine ──────────────────────────
                new_state     = tracked.state
                new_held      = tracked.held_frames
                new_confirmed = tracked.confirmed

                if near_person:
                    # Object is with a person → HELD
                    new_state  = ObjectState.HELD
                    new_held   = tracked.held_frames + 1
                    new_confirmed = False  # reset — not trash while held

                elif tracked.state == ObjectState.HELD and not near_person:
                    # WAS held, now separated → DROPPED
                    if tracked.held_frames >= self.MIN_HELD_FRAMES:
                        new_state     = ObjectState.DROPPED
                        new_confirmed = True   # 🚨 flag immediately on drop
                    else:
                        new_state = ObjectState.APPEARING

                elif tracked.state == ObjectState.APPEARING and on_ground and not near_person:
                    # Object appeared on ground without being held
                    # (e.g. rolled into frame) — flag after brief check
                    new_state     = ObjectState.GROUNDED
                    new_confirmed = tracked.frames_seen >= 8

                elif tracked.state in (ObjectState.DROPPED, ObjectState.GROUNDED):
                    # Keep confirmed once flagged, stays as trash
                    new_confirmed = True

                updated.append(_TrackedObject(
                    det          = obj,
                    state        = new_state,
                    held_frames  = new_held,
                    frames_seen  = tracked.frames_seen + 1,
                    confirmed    = new_confirmed,
                ))
            # else: object left frame → drop it

        # ── New objects not yet tracked ──
        for i, obj in enumerate(objects):
            if i not in used:
                near_person = _is_near_person(obj.bbox, persons)
                updated.append(_TrackedObject(
                    det         = obj,
                    state       = ObjectState.HELD if near_person else ObjectState.APPEARING,
                    held_frames = 1 if near_person else 0,
                    frames_seen = 1,
                ))

        self._tracked = updated

        # ── Return confirmed trash ──
        return [
            TrashDetection(
                bbox       = t.det.bbox,
                label      = t.det.class_name,
                confidence = t.det.confidence,
            )
            for t in self._tracked
            if t.confirmed
        ]