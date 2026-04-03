"""
Layer 2 — Visualizer

HUD shows PEAK (cumulative max) counts passed in from the tracker,
so counts never drop to 0 after objects leave frame — consistent
with how total_trash_events already behaves.
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
    max_persons_seen:   int = 0,   # peak value maintained by tracker
    max_objects_seen:   int = 0,   # peak value maintained by tracker
    max_trash_tagged:   int = 0,   # ← add
) -> np.ndarray:

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        is_ghost = t.track_id < 0

        # ── Color ────────────────────────────────────────────────────
        if t.is_trash:
            color = (0, 0, 220)
        elif t.class_name == "person":
            color = (0, 200, 0)
        else:
            color = _id_color(t.track_id)

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

        # ── Trail (skip ghosts — no history) ─────────────────────────
        if SHOW_TRAILS and not is_ghost and len(t.trail) > 1:
            pts = list(t.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(x * alpha) for x in color)
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, c, -1)

    # ── Stats HUD — use PEAK values from tracker, never compute here ──
    #trash_tagged = sum(1 for t in tracks if t.is_trash and t.track_id >= 0)

    for i, txt in enumerate([
        f"Tracked persons : {max_persons_seen}",
        f"Tracked objects : {max_objects_seen}",
        f"Trash tagged    : {max_trash_tagged}",   # ← peak, never drops
        f"Trash events    : {total_trash_events}",
    ]):
        cv2.putText(frame, txt, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return frame