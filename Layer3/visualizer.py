"""
Layer 3/visualizer.py — Visualizer

Draws on top of the Layer 2 frame output:
  - Lines connecting each (person, object) pair
  - Sequence fill bar showing how much history has accumulated
  - Live feature readout panel for the most recent pair
  - Colour coding:
      orange  — pair forming (not enough history yet)
      teal    — pair ready for Layer 4 inference
      red     — pair whose object is flagged as trash
"""
# Add these at the top of Layer3/visualizer.py (module level, after imports)
_seen_pair_keys = set()
_peak_ready = 0
import cv2
import numpy as np
from typing import List

from Layer2.track_state import TrackedObject
from Layer3.pair_state  import PairState
from Layer3.feature    import FEATURE_DIM
from Layer3.config      import (
    SHOW_PAIRS,
    SHOW_SEQUENCES,
    PAIR_LINE_COLOR,
    ACTIVE_PAIR_COLOR,
    SEQUENCE_TEXT_COLOR,
    SEQUENCE_LENGTH,
)

# Feature names for the live readout panel
_FEAT_NAMES = [
    "dist_norm",
    "delta_dist",
    "obj_vel",
    "person_vel",
    "dir_x",
    "dir_y",
    "holding",
    "area_norm",
    "visibility",
]


def _centroid(bbox: np.ndarray):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def draw_memory(
    frame:  np.ndarray,
    pairs:  List[PairState],
    tracks: List[TrackedObject],
) -> np.ndarray:
    """
    Overlay Layer 3 memory information on a frame.

    Args:
        frame:   BGR frame (already annotated by Layer 2 visualizer)
        pairs:   all active PairState objects from MemoryEngine.update()
        tracks:  all TrackedObject from Layer 2 (needed to get bboxes)

    Returns:
        Annotated frame (modified in-place, also returned for chaining).
    """
    if not SHOW_PAIRS:
        return frame

    H, W = frame.shape[:2]

    # Fast lookup: track_id → TrackedObject
    track_map = {t.track_id: t for t in tracks}

    for pair in pairs:
        person_track = track_map.get(pair.person_id)
        obj_track    = track_map.get(pair.object_id)

        # Only draw if BOTH tracks are currently visible
        if person_track is None or obj_track is None:
            continue

        pc = _centroid(person_track.bbox)
        oc = _centroid(obj_track.bbox)

        # Choose line colour
        if obj_track.is_trash:
            line_color = (0, 0, 220)        # red — confirmed illegal dump
        elif pair.ready():
            line_color = ACTIVE_PAIR_COLOR  # teal — ready for inference
        else:
            line_color = PAIR_LINE_COLOR    # orange — still accumulating

        # Draw pair connection line
        cv2.line(frame, pc, oc, line_color, 1, cv2.LINE_AA)

        # Sequence fill bar (sits just above the object bbox)
        if SHOW_SEQUENCES:
            seq_len = len(pair.sequence)
            bar_x   = oc[0] - 20
            bar_y   = int(obj_track.bbox[1]) - 18
            bar_w   = 40
            bar_h   = 6
            fill_w  = int(bar_w * seq_len / max(SEQUENCE_LENGTH, 1))

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (60, 60, 60), -1)
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          line_color, -1)

            seq_label = f"{seq_len}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, seq_label,
                        (bar_x, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        SEQUENCE_TEXT_COLOR, 1, cv2.LINE_AA)

        # HELD / RELEASED indicator next to object
        if pair.ever_held and pair.released_frames == 0:
            cv2.putText(frame, "HELD",
                        (oc[0] + 6, oc[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 220, 255), 1, cv2.LINE_AA)
        elif pair.ever_held and pair.released_frames > 0:
            cv2.putText(frame, f"REL:{pair.released_frames}f",
                        (oc[0] + 6, oc[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180, 180, 255), 1, cv2.LINE_AA)

    # Live feature panel (top-right corner)
    _draw_feature_panel(frame, pairs, track_map, W)

    # Pair count (bottom-left)
    # AFTER
    global _seen_pair_keys, _peak_ready

    # Accumulate every unique (person_id, object_id) pair ever seen
    for p in pairs:
        _seen_pair_keys.add(p.pair_key)

    ready_count = sum(1 for p in pairs if p.ready())
    _peak_ready = max(_peak_ready, ready_count)

    cv2.putText(frame,
                f"Pairs: {len(_seen_pair_keys)}  Ready: {_peak_ready}",
                (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def _draw_feature_panel(
    frame:     np.ndarray,
    pairs:     List[PairState],
    track_map: dict,
    frame_w:   int,
):
    """
    Compact feature readout in the top-right corner.
    Shows the most recent feature vector of the first ready pair.
    """
    if not SHOW_SEQUENCES:
        return

    # Pick best pair to display
    display_pair = None
    for p in pairs:
        if p.ready():
            display_pair = p
            break
    if display_pair is None and pairs:
        display_pair = pairs[0]
    if display_pair is None or len(display_pair.sequence) == 0:
        return

    latest_feat = display_pair.sequence[-1]   # shape (FEATURE_DIM,)

    panel_x = frame_w - 195
    panel_y = 50
    line_h  = 17
    panel_h = FEATURE_DIM * line_h + 20
    panel_w = 185

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 5, panel_y - 15),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Header
    header = f"P{display_pair.person_id} <-> O{display_pair.object_id}"
    cv2.putText(frame, header,
                (panel_x, panel_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 180), 1, cv2.LINE_AA)

    # One row per feature
    for i, (name, val) in enumerate(zip(_FEAT_NAMES, latest_feat)):
        y = panel_y + i * line_h + line_h

        bar_val   = float(np.clip(abs(val), 0, 1))
        bar_color = (0, 200, 100) if val >= 0 else (0, 100, 200)
        bar_w     = int(bar_val * 80)
        cv2.rectangle(frame,
                      (panel_x + 85, y - 9),
                      (panel_x + 85 + bar_w, y - 2),
                      bar_color, -1)

        txt = f"{name:<13} {val:+.3f}"
        cv2.putText(frame, txt,
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (220, 220, 220), 1, cv2.LINE_AA)