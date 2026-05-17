"""
Trash Detector v5 — Universal Dumping Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNCHANGED from baseline. Receives List[Detection] from detector.py —
the Detection dataclass structure is identical, so this file requires
zero modifications after the Layer 1 upgrade.

Two parallel pipelines:

Pipeline A (SLOW DROP):
  Object held → separated from person → stays stationary without person
  → Works for: walking drop, leaving bag, fly-tipping

Pipeline B (FAST THROW):
  Object held → suddenly disappears while person arm is still extended
  → Works for: car window throw, cyclist litter, fast toss
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .detector import Detection
from enum import Enum
try:
    from Layer2.config import MAX_POSSESSION_DISTANCE_PX, POSSESSION_HISTORY_FRAMES
except ImportError:
    MAX_POSSESSION_DISTANCE_PX = 150
    POSSESSION_HISTORY_FRAMES  = 20


class ObjectState(Enum):
    APPEARING = "appearing"
    HELD      = "held"
    RELEASED  = "released"
    DROPPED   = "dropped"
    THROWN    = "thrown"


@dataclass
class TrashDetection:
    bbox:       np.ndarray
    class_name: str   = "trash"
    confidence: float = 1.0
    class_id:   int   = -1
    label:      str   = ""
    how:        str   = ""


@dataclass
class _TrackedObject:
    det:             Detection
    state:           ObjectState = ObjectState.APPEARING
    held_frames:     int  = 0
    released_frames: int  = 0
    frames_seen:     int  = 0
    confirmed:       bool = False
    how:             str  = ""


@dataclass
class _PersonState:
    bbox:            np.ndarray
    prev_bbox:       Optional[np.ndarray] = None
    held_object_ids: List[int] = field(default_factory=list)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter)


def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


def _near_person(obj_bbox: np.ndarray, persons: List[Detection],
                 iou_thresh=0.08, dist_px=MAX_POSSESSION_DISTANCE_PX) -> bool:
    """Current-frame proximity check (used for ongoing HELD tracking)."""
    for p in persons:
        if _iou(obj_bbox, p.bbox) > iou_thresh:
            return True
        cx, cy = _centroid(obj_bbox)
        px1, py1, px2, py2 = p.bbox
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
        # Distance-based fallback — catches held bag with low/zero IoU
        pcx, pcy = _centroid(p.bbox)
        if np.hypot(cx - pcx, cy - pcy) <= dist_px:
            return True
    return False

def _near_person_historical(
    obj_bbox: np.ndarray,
    person_history: List[dict],          # list of {"centroid": (x,y), "velocity": (vx,vy)}
    max_dist_px: float = MAX_POSSESSION_DISTANCE_PX,
    max_frames_back: int = POSSESSION_HISTORY_FRAMES,
) -> bool:
    """
    Projected Proximity Search — Fix for Root Causes 2 & 3.

    When a bag reappears after occlusion, the person may have already
    moved away. This checks the bag's centroid against the person's
    HISTORICAL centroids (+ optional velocity extrapolation) up to
    max_frames_back frames ago.

    Returns True if the bag was plausibly near the person recently.
    """
    if not person_history:
        return False

    obj_cx, obj_cy = _centroid(obj_bbox)

    for i, entry in enumerate(person_history[-max_frames_back:]):
        hist_cx, hist_cy = entry.get("centroid", (None, None))
        if hist_cx is None:
            continue

        # Optional: extrapolate person position forward using velocity
        frames_ago = max_frames_back - i
        vx, vy = entry.get("velocity", (0.0, 0.0))
        projected_cx = hist_cx + vx * frames_ago
        projected_cy = hist_cy + vy * frames_ago

        dist = np.hypot(obj_cx - projected_cx, obj_cy - projected_cy)
        if dist <= max_dist_px:
            return True

    return False


def _person_arm_extended(persons: List[Detection], frame_w: int) -> bool:
    for p in persons:
        x1, x2 = p.bbox[0], p.bbox[2]
        if x1 <= 15 or x2 >= frame_w - 15:
            return True
    return False


class TrashDetector:

    MIN_HELD_FRAMES     = 5
    RELEASE_CONFIRM     = 8
    MATCH_DIST_PX       = 80

    THROW_MIN_HELD      = 3
    THROW_VANISH_FRAMES = 4

    def __init__(self):
        self._tracked:       List[_TrackedObject] = []
        self._ghosts:        List[dict]           = []
        # Person centroid history for projected proximity search
        # List of {"centroid": (x,y), "velocity": (vx,vy)} newest-last
        self._person_history: List[dict]          = []

    def detect(
        self,
        frame_shape:    tuple,
        all_detections: List[Detection],
    ) -> List[TrashDetection]:

        H, W = frame_shape[:2]
        persons = [d for d in all_detections if d.class_name == "person"]
        objects = [d for d in all_detections if d.class_name != "person"]

        # ── Update person centroid history for projected proximity search ──
        if persons:
            # Average centroid across all persons in frame
            centroids = [_centroid(p.bbox) for p in persons]
            avg_cx = float(np.mean([c[0] for c in centroids]))
            avg_cy = float(np.mean([c[1] for c in centroids]))
            # Compute velocity vs last known position
            vx, vy = 0.0, 0.0
            if self._person_history:
                prev_cx, prev_cy = self._person_history[-1]["centroid"]
                vx = avg_cx - prev_cx
                vy = avg_cy - prev_cy
            self._person_history.append({"centroid": (avg_cx, avg_cy),
                                          "velocity": (vx, vy)})
            # Keep only last POSSESSION_HISTORY_FRAMES entries
            if len(self._person_history) > POSSESSION_HISTORY_FRAMES:
                self._person_history.pop(0)

        # ── Pipeline A (slow drop) ────────────────────────────────────
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
                obj         = objects[best_i]
                with_person = _near_person(obj.bbox, persons)

                new_held      = tracked.held_frames
                new_released  = tracked.released_frames
                new_state     = tracked.state
                new_confirmed = tracked.confirmed
                new_how       = tracked.how

                if with_person:
                    new_state     = ObjectState.HELD
                    new_held      = tracked.held_frames + 1
                    new_released  = 0
                    new_confirmed = False

                elif tracked.state == ObjectState.HELD:
                    new_state    = ObjectState.RELEASED
                    new_released = 1

                elif tracked.state == ObjectState.RELEASED:
                    new_released = tracked.released_frames + 1
                    if (tracked.held_frames >= self.MIN_HELD_FRAMES
                            and new_released >= self.RELEASE_CONFIRM):
                        new_state     = ObjectState.DROPPED
                        new_confirmed = True
                        new_how       = "dropped"

                elif tracked.state in (ObjectState.DROPPED, ObjectState.THROWN):
                    new_confirmed = True

                updated.append(_TrackedObject(
                    det             = obj,
                    state           = new_state,
                    held_frames     = new_held,
                    released_frames = new_released,
                    frames_seen     = tracked.frames_seen + 1,
                    confirmed       = new_confirmed,
                    how             = new_how,
                ))
            else:
                if (tracked.state == ObjectState.HELD
                        and tracked.held_frames >= self.THROW_MIN_HELD):
                    self._ghosts.append({
                        "last_bbox":   tracked.det.bbox.copy(),
                        "held_frames": tracked.held_frames,
                        "frames_gone": 0,
                        "label":       tracked.det.class_name,
                        "conf":        tracked.det.confidence,
                    })

        # New unmatched objects
        for i, obj in enumerate(objects):
            if i not in used:
                with_person = _near_person(obj.bbox, persons)

                # Fix Root Causes 2 & 3: if not near person NOW,
                # check if it was near a person historically (bag reappearing after occlusion)
                if not with_person and self._person_history:
                    was_held_historically = _near_person_historical(
                        obj.bbox, self._person_history
                    )
                else:
                    was_held_historically = False

                if with_person:
                    init_state   = ObjectState.HELD
                    init_held    = 1
                    init_confirm = False
                elif was_held_historically:
                    init_state   = ObjectState.HELD
                    init_held    = self.MIN_HELD_FRAMES
                    init_confirm = False
                else:
                    init_state   = ObjectState.APPEARING
                    init_held    = 0
                    init_confirm = False

                updated.append(_TrackedObject(
                    det         = obj,
                    state       = init_state,
                    held_frames = init_held,
                    frames_seen = 1,
                    confirmed   = init_confirm,
                    how         = "thrown" if was_held_historically else "",
                ))

        self._tracked = updated

        # ── Pipeline B (fast throw) ───────────────────────────────────
        arm_extended = _person_arm_extended(persons, W)
        throw_detections: List[TrashDetection] = []
        live_ghosts = []

        for ghost in self._ghosts:
            ghost["frames_gone"] += 1
            if ghost["frames_gone"] <= self.THROW_VANISH_FRAMES:
                throw_detections.append(TrashDetection(
                    bbox       = ghost["last_bbox"],
                    label      = ghost["label"],
                    confidence = ghost["conf"],
                    how        = "thrown",
                ))
                live_ghosts.append(ghost)

        self._ghosts = live_ghosts

        # ── Combine both pipelines ────────────────────────────────────
        pipeline_a = [
            TrashDetection(
                bbox       = t.det.bbox,
                label      = t.det.class_name,
                confidence = t.det.confidence,
                how        = t.how,
            )
            for t in self._tracked if t.confirmed
        ]

        return pipeline_a + throw_detections