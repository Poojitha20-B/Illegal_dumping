"""
Layer 3/memory.py — Memory / Temporal State Engine

Sits between the tracker (Layer 2) and the behaviour model (Layer 4).

Responsibilities:
  1. PAIRING     — for every person track, find the nearest object track
                   within MAX_PAIR_DISTANCE and form a (person_id, object_id) pair
  2. FEATURE EXT — call Layer3.features.extract_features() each frame per pair
  3. BUFFERING   — maintain a rolling deque of SEQUENCE_LENGTH feature vectors
                   per pair (the PairState.sequence)
  4. PERSISTENCE — when a track disappears briefly, do NOT delete its history;
                   hold it for MAX_MISSING_FRAMES then clean up
  5. OUTPUT      — return a list of PairState objects each frame so Layer 4
                   can run inference on any pair whose .ready() returns True
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from Layer2.track_state import TrackedObject
from Layer3.pair_state  import PairState
from Layer3.feature    import extract_features, FEATURE_DIM
from Layer3.config      import (
    MAX_PAIR_DISTANCE,
    MAX_MISSING_FRAMES,
    HOLD_DISTANCE_PX,
)


def _centroid(bbox: np.ndarray) -> np.ndarray:
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        dtype=np.float32,
    )


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(_centroid(a) - _centroid(b)))


class MemoryEngine:
    """
    Temporal State Engine — Layer 3.

    Usage (inside run_pipeline.py):
        memory = MemoryEngine()
        ...
        pairs = memory.update(tracked_objects)
        # pairs is a List[PairState]
        # Layer 4 filters: ready_pairs = [p for p in pairs if p.ready()]
    """

    def __init__(self):
        # Active pairs keyed by (person_id, object_id)
        self._pairs: Dict[Tuple[int, int], PairState] = {}

    # ─────────────────────────────────────────────────────
    def update(self, tracks: List[TrackedObject]) -> List[PairState]:
        """
        Main entry point. Call once per frame with the full list of
        TrackedObject from Layer 2.

        Returns:
            List[PairState] — all currently active pairs (including those
            temporarily missing). Layer 4 filters with pair.ready().
        """
        persons = [t for t in tracks if t.class_name == "person"]
        objects = [t for t in tracks if t.class_name != "person"]

        # ── Step 1: Build this frame's (person, object) pairings ──────
        # Greedy nearest-neighbour: assign each person to its closest object.
        # Each object can only be assigned to one person.
        frame_pairs:     Dict[Tuple[int, int], Tuple[TrackedObject, TrackedObject]] = {}
        used_object_ids: set = set()

        # Sort by confidence so most-confident persons get first pick
        for person in sorted(persons, key=lambda t: t.confidence, reverse=True):
            best_dist = MAX_PAIR_DISTANCE
            best_obj  = None

            for obj in objects:
                if obj.track_id in used_object_ids:
                    continue
                d = _distance(person.bbox, obj.bbox)
                if d < best_dist:
                    best_dist = d
                    best_obj  = obj

            if best_obj is not None:
                key = (person.track_id, best_obj.track_id)
                frame_pairs[key] = (person, best_obj)
                used_object_ids.add(best_obj.track_id)

        # ── Step 2: Update existing and create new pairs ──────────────
        updated_keys: set = set()

        for key, (person_track, obj_track) in frame_pairs.items():
            updated_keys.add(key)

            if key not in self._pairs:
                self._pairs[key] = PairState(
                    person_id=person_track.track_id,
                    object_id=obj_track.track_id,
                )

            pair = self._pairs[key]

            # Distance this frame
            dist = _distance(person_track.bbox, obj_track.bbox)

            # Update hold state
            if dist < HOLD_DISTANCE_PX:
                pair.held_frames     += 1
                pair.released_frames  = 0
                pair.ever_held        = True
            else:
                if pair.ever_held:
                    pair.released_frames += 1

            # Extract features
            feat = extract_features(
                person_bbox      = person_track.bbox,
                object_bbox      = obj_track.bbox,
                prev_person_bbox = pair.last_person_bbox,
                prev_object_bbox = pair.last_object_bbox,
                prev_distance    = pair.last_distance,
                both_visible     = True,
            )

            pair.push(feat)

            # Save state for next frame's delta calculations
            pair.last_person_bbox = person_track.bbox.copy()
            pair.last_object_bbox = obj_track.bbox.copy()
            pair.last_distance    = dist

        # ── Step 3: Handle pairs whose tracks disappeared this frame ──
        for key, pair in self._pairs.items():
            if key in updated_keys:
                continue
            if not pair.is_active:
                continue
            pair.mark_missing()

        # ── Step 4: Garbage-collect dead pairs ────────────────────────
        self._pairs = {k: v for k, v in self._pairs.items() if v.is_active}

        return list(self._pairs.values())

    # ─────────────────────────────────────────────────────
    def get_ready_pairs(self, min_frames: int = 8) -> List[PairState]:
        """Returns only pairs ready for Layer 4 inference."""
        return [p for p in self._pairs.values() if p.ready(min_frames)]

    def get_all_pairs(self) -> List[PairState]:
        """Returns ALL active pairs including those not yet ready."""
        return list(self._pairs.values())

    @property
    def pair_count(self) -> int:
        return len(self._pairs)