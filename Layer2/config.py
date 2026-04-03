# ─────────────────────────────────────────
#  Layer 2 — ByteTrack Configuration
# ─────────────────────────────────────────

# ByteTrack thresholds
TRACK_HIGH_THRESH   = 0.35   # detections above this → high confidence pool
TRACK_LOW_THRESH    = 0.15   # detections above this → low confidence pool (still tracked)
TRACK_MATCH_THRESH  = 0.30   # was 0.8 (too strict) → 0.3 (standard ByteTrack)
NEW_TRACK_THRESH    = 0.3    # min confidence to START a new track

# How many frames a track survives without a matching detection
MAX_TIME_LOST       = 40     # ~1s at 25fps — keeps IDs stable through occlusions

# Minimum frames a PERSON track must exist before being shown
# Objects always appear on frame 1 (hardcoded in tracker Step 7)
MIN_TRACK_FRAMES    = 2      # persons: suppress false positives from brief detections

# Visualisation
TRACK_ID_COLOR      = (255, 255, 0)
TRAIL_COLOR         = (200, 200, 0)
TRAIL_LENGTH        = 30
SHOW_TRAILS         = True