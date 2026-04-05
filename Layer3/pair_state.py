"""
Layer 3/pair_state.py — PairState

One PairState object exists for every (person_id, object_id) pair
that Layer 3 is actively tracking.

It owns:
  - the rolling sequence buffer of feature vectors
  - raw bbox history (needed for delta calculations)
  - metadata (how long held, how long missing, etc.)
  - a ready() method Layer 4 calls to know if the sequence is usable

Layer 4 reads pair.get_sequence() to get a (T, FEATURE_DIM) numpy array
ready to feed into the TimeSformer.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque

from Layer3.config import SEQUENCE_LENGTH, MAX_MISSING_FRAMES
from Layer3.feature import FEATURE_DIM


@dataclass
class PairState:
    """
    Tracks the relationship between one person and one nearby object
    across time.

    Attributes:
        person_id:          track_id of the person
        object_id:          track_id of the object
        sequence:           rolling deque of feature vectors, len <= SEQUENCE_LENGTH
        frames_seen:        total frames this pair has been observed
        frames_missing:     consecutive frames either track was absent
        held_frames:        how many frames the object was "held" (close to person)
        released_frames:    how many frames since the object was last held
        ever_held:          True once the object was held at least once
        is_active:          False when the pair should be garbage-collected
        last_person_bbox:   previous frame's person bbox (for velocity calc)
        last_object_bbox:   previous frame's object bbox (for velocity calc)
        last_distance:      previous frame's centroid distance (for delta calc)
    """

    person_id: int
    object_id: int

    # Rolling feature buffer — this is what Layer 4 reads
    sequence: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH),
        repr=False,
    )

    frames_seen:     int   = 0
    frames_missing:  int   = 0
    held_frames:     int   = 0
    released_frames: int   = 0
    ever_held:       bool  = False
    is_active:       bool  = True

    # Previous-frame state for delta features
    last_person_bbox: Optional[np.ndarray] = field(default=None, repr=False)
    last_object_bbox: Optional[np.ndarray] = field(default=None, repr=False)
    last_distance:    Optional[float]      = None

    def push(self, feature_vec: np.ndarray):
        """Append one frame's feature vector to the buffer."""
        self.sequence.append(feature_vec.copy())
        self.frames_seen    += 1
        self.frames_missing  = 0   # reset — we saw both tracks this frame

    def mark_missing(self):
        """Call when one or both tracks were absent this frame."""
        self.frames_missing += 1
        if self.frames_missing > MAX_MISSING_FRAMES:
            self.is_active = False

        # Push a padded repeat of the last known feature so the sequence
        # buffer doesn't have gaps — model needs consistent-length input
        if len(self.sequence) > 0:
            last      = self.sequence[-1].copy()
            last[8]   = 0.5   # index 8 = visibility_score (see features.py)
            self.sequence.append(last)

    def ready(self, min_frames: int = 8) -> bool:
        """
        True if this pair has enough history for Layer 4 to run inference.

        Args:
            min_frames: minimum sequence length required (default 8).
        """
        return (
            self.is_active
            and self.ever_held                  # only pairs that had contact
            and len(self.sequence) >= min_frames
        )

    def get_sequence(self) -> np.ndarray:
        """
        Returns the current sequence as a numpy array of shape (T, FEATURE_DIM)
        where T = SEQUENCE_LENGTH.

        If the buffer has fewer than SEQUENCE_LENGTH frames, pre-pad with zeros
        so recent frames are always at the end (standard transformer convention).
        """
        seq = np.array(list(self.sequence), dtype=np.float32)   # (T, FEATURE_DIM)
        T   = seq.shape[0]

        if T < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - T, FEATURE_DIM), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)            # pre-pad

        return seq   # shape: (SEQUENCE_LENGTH, FEATURE_DIM)

    @property
    def pair_key(self) -> tuple:
        return (self.person_id, self.object_id)

    def __repr__(self):
        return (
            f"PairState(person={self.person_id}, obj={self.object_id}, "
            f"frames={self.frames_seen}, held={self.held_frames}, "
            f"seq_len={len(self.sequence)}, active={self.is_active})"
        )