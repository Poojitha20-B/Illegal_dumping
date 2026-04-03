"""
Dataclass representing a single active track.
This is the clean interface Layer 3 (Memory) will consume.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class TrackedObject:
    """
    A single object being tracked across frames.
    
    Produced by Layer 2, consumed by Layer 3.
    """
    track_id:    int              # stable ID across frames
    bbox:        np.ndarray       # [x1, y1, x2, y2] current position
    class_name:  str              # "person", "bottle", "trash", etc.
    confidence:  float            # detection confidence this frame
    is_trash:    bool = False     # True if Layer 1 flagged this as dropped trash
    trash_label: str  = ""        # original class name if trash e.g. "bottle"
    trash_how:   str  = ""        # "dropped" or "thrown"

    # Trajectory history — Layer 3 will use this
    trail: deque = field(
        default_factory=lambda: deque(maxlen=30),
        repr=False
    )

    def centroid(self):
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def update_trail(self):
        """Call once per frame after bbox is updated."""
        self.trail.append(self.centroid())