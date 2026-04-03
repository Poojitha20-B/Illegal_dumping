"""
Layer 2 — Visualizer (fixed)
"""

import cv2
import numpy as np
from typing import List
from .track_state import TrackedObject
from .config import SHOW_TRAILS

def _id_color(tid: int):
    np.random.seed(tid * 7 + 13)
    return tuple(int(x) for x in np.random.randint(80, 230, 3))


def draw_tracks(frame: np.ndarray, tracks: List[TrackedObject],
                total_trash_events: int = 0) -> np.ndarray:

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)

        # ── Color: red ONLY for trash, unique color per ID for everything else
        if t.is_trash:
            color = (0, 0, 220)
        else:
            color = _id_color(t.track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        if t.is_trash:
            label = f"ID:{t.track_id} TRASH({t.trash_how}):{t.trash_label}"
        else:
            label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Trail
        if SHOW_TRAILS and len(t.trail) > 1:
            pts = list(t.trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(x * alpha) for x in color)
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, c, -1)

    # ── Stats — fixed counts ──────────────────────────────
    # Only count non-trash tracks for persons/objects
    persons = sum(1 for t in tracks if t.class_name == "person" and not t.is_trash)
    objects = sum(1 for t in tracks if t.class_name != "person" and not t.is_trash)
    # Use cumulative trash count passed in, not current frame count
    trash   = total_trash_events

    for i, txt in enumerate([
        f"Tracked persons : {persons}",
        f"Tracked objects : {objects}",
        f"Trash events    : {trash}",
    ]):
        cv2.putText(frame, txt, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return frame