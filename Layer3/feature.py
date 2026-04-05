"""
Layer 3/feature.py — Feature Extractor

Computes a fixed-length feature vector for a (person, object) pair
at a SINGLE frame.  The Memory engine calls this every frame and
appends the result to the pair's sequence buffer.

Feature vector  (dimension = 9):
  0  distance_norm       — centroid distance, normalised
  1  delta_distance_norm — how much closer/farther vs previous frame
  2  obj_velocity_norm   — object centroid speed (px/frame), normalised
  3  person_velocity_norm— person centroid speed (px/frame), normalised
  4  direction_x         — unit-vector x component of obj→person direction
  5  direction_y         — unit-vector y component of obj→person direction
  6  is_holding          — 1.0 if distance < HOLD_DISTANCE_PX, else 0.0
  7  obj_area_norm       — object bbox area, normalised
  8  visibility_score    — 1.0 if both tracks visible, <1 if either is lost

Why these features:
  - distance + delta_distance capture "approach then separation" (dumping pattern)
  - velocities catch rapid throws
  - direction is useful for throws toward specific locations (e.g. away from person)
  - is_holding provides a clean binary signal for the model
  - obj_area helps distinguish small litter vs large bags
  - visibility_score lets the model weight uncertain frames lower
"""

import numpy as np
from typing import Optional, Tuple
from .config import NORM_DISTANCE, NORM_VELOCITY, NORM_AREA, HOLD_DISTANCE_PX

# ── Public constant — Layer 4 needs to know this ──────────
FEATURE_DIM = 9


def _centroid(bbox: np.ndarray) -> np.ndarray:
    """Returns [cx, cy] as float32 array."""
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        dtype=np.float32,
    )


def _bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def extract_features(
    person_bbox:      np.ndarray,
    object_bbox:      np.ndarray,
    prev_person_bbox: Optional[np.ndarray],
    prev_object_bbox: Optional[np.ndarray],
    prev_distance:    Optional[float],
    both_visible:     bool = True,
) -> np.ndarray:
    """
    Extract a single feature vector for one (person, object) pair.

    Args:
        person_bbox:      [x1,y1,x2,y2] — current person bbox
        object_bbox:      [x1,y1,x2,y2] — current object bbox
        prev_person_bbox: bbox from previous frame (None on first frame)
        prev_object_bbox: bbox from previous frame (None on first frame)
        prev_distance:    centroid distance from previous frame (None on first)
        both_visible:     False if either track was missing this frame

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float32
    """
    pc = _centroid(person_bbox)
    oc = _centroid(object_bbox)

    # ── 0. Distance ───────────────────────────────────────
    diff       = pc - oc
    distance   = float(np.linalg.norm(diff))
    dist_norm  = np.clip(distance / NORM_DISTANCE, 0.0, 1.0)

    # ── 1. Delta distance ─────────────────────────────────
    if prev_distance is not None:
        delta_dist = (distance - prev_distance) / NORM_DISTANCE
        delta_dist = np.clip(delta_dist, -1.0, 1.0)
    else:
        delta_dist = 0.0

    # ── 2 & 3. Velocities ────────────────────────────────
    if prev_object_bbox is not None:
        prev_oc      = _centroid(prev_object_bbox)
        obj_vel      = float(np.linalg.norm(oc - prev_oc))
        obj_vel_norm = np.clip(obj_vel / NORM_VELOCITY, 0.0, 1.0)
    else:
        obj_vel_norm = 0.0

    if prev_person_bbox is not None:
        prev_pc         = _centroid(prev_person_bbox)
        person_vel      = float(np.linalg.norm(pc - prev_pc))
        person_vel_norm = np.clip(person_vel / NORM_VELOCITY, 0.0, 1.0)
    else:
        person_vel_norm = 0.0

    # ── 4 & 5. Direction (object → person unit vector) ───
    if distance > 1e-3:
        dir_vec = diff / distance          # unit vector pointing from obj to person
        dir_x   = float(np.clip(dir_vec[0], -1.0, 1.0))
        dir_y   = float(np.clip(dir_vec[1], -1.0, 1.0))
    else:
        dir_x, dir_y = 0.0, 0.0

    # ── 6. Is holding ─────────────────────────────────────
    is_holding = 1.0 if distance < HOLD_DISTANCE_PX else 0.0

    # ── 7. Object area ────────────────────────────────────
    obj_area_norm = np.clip(_bbox_area(object_bbox) / NORM_AREA, 0.0, 1.0)

    # ── 8. Visibility score ───────────────────────────────
    visibility = 1.0 if both_visible else 0.5

    return np.array([
        dist_norm,
        delta_dist,
        obj_vel_norm,
        person_vel_norm,
        dir_x,
        dir_y,
        is_holding,
        obj_area_norm,
        visibility,
    ], dtype=np.float32)