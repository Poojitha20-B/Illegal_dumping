"""
Layer 1 — Visualizer
Draws bounding boxes, labels, and a live stats overlay.
"""
from .trash_detector import TrashDetection
from .config import TRASH_COLOR
import cv2
import numpy as np
from typing import List
from .detector import Detection
from .config import (
    PERSON_COLOR, OBJECT_COLOR, TEXT_COLOR,
    BOX_THICKNESS, FONT_SCALE
)


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw all detections on the frame. Returns annotated copy."""
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = PERSON_COLOR if det.class_name == "person" else OBJECT_COLOR

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label background
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            out, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            TEXT_COLOR, 1, cv2.LINE_AA
        )

    # ── Stats overlay ─────────────────────
    persons = sum(1 for d in detections if d.class_name == "person")
    objects = len(detections) - persons

    stats = [
        f"Persons : {persons}",
        f"Objects : {objects}",
        f"Total   : {len(detections)}",
    ]
    for i, txt in enumerate(stats):
        cv2.putText(
            out, txt,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 1, cv2.LINE_AA
        )

    return out
def draw_trash(frame: np.ndarray, trash_detections) -> np.ndarray:
    """Draw red boxes around detected trash blobs."""
    for det in trash_detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), TRASH_COLOR, 2)
        cv2.putText(frame, "trash", (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame