"""
Layer 2 — Visualizer

Fix: HUD now shows PEAK (cumulative max) counts for persons and objects,
so "Tracked persons: 1" and "Tracked objects: 2" stay visible permanently
even after the person and objects leave the frame — matching the behaviour
of "Trash events: 2" which also never resets.
"""

import cv2
import numpy as np
from typing import List
from .track_state import TrackedObject
from .config import SHOW_TRAILS


def _id_color(tid: int):
    np.random.seed(abs(tid) * 7 + 13)
    return tuple(int(x) for x in np.random.randint(80, 230, 3))


def draw_tracks(
    frame:              np.ndarray,
    tracks:             List[TrackedObject],
    total_trash_events: int = 0,
    max_persons_seen:   int = 0,   # ← NEW: peak persons, passed from tracker
    max_objects_seen:   int = 0,   # ← NEW: peak objects, passed from tracker
) -> np.ndarray:

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        is_ghost = t.track_id < 0

        # ── Color ────────────────────────────────────────────────────
        if t.is_trash:
            color = (0, 0, 220)           # red for all trash (real + ghost)
        elif t.class_name == "person":
            color = (0, 200, 0)           # green for persons
        else:
            color = _id_color(t.track_id) # unique color per object ID

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Label ────────────────────────────────────────────────────
        if is_ghost:
            label = f"GHOST TRASH({t.trash_how}):{t.trash_label}"
        elif t.is_trash:
            label = f"ID:{t.track_id} TRASH({t.trash_how}):{t.trash_label}"
        else:
            label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Trail (skip ghosts — they have no history) ───────────────
        if SHOW_TRAILS and not is_ghost and len(t.trail) > 1:
            pts = list(t.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(x * alpha) for x in color)
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, c, -1)

    # ── Stats HUD ────────────────────────────────────────────────────
    # All three lines now use CUMULATIVE PEAK values so they never drop
    # back to 0 once a person/object has been seen — consistent with
    # how "Trash events" already behaves.
    for i, txt in enumerate([
        f"Tracked persons : {max_persons_seen}",
        f"Tracked objects : {max_objects_seen}",
        f"Trash events    : {total_trash_events}",
    ]):
        cv2.putText(frame, txt, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return frame