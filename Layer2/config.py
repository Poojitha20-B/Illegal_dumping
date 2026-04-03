# ─────────────────────────────────────────
#  Layer 2 — ByteTrack Configuration
# ─────────────────────────────────────────

# ByteTrack thresholds
TRACK_HIGH_THRESH   = 0.35   # detections above this → high confidence pool
TRACK_LOW_THRESH    = 0.15   # detections above this → low confidence pool (still tracked)
TRACK_MATCH_THRESH  = 0.30   # ✅ FIXED: was 0.8 (too strict) → 0.3 (standard ByteTrack)
NEW_TRACK_THRESH    = 0.3    # min confidence to START a new track

# How many frames a track survives without a matching detection
MAX_TIME_LOST       = 40     # ~1s at 25fps — keeps IDs stable through occlusions

# Minimum frames a track must exist before being shown
MIN_TRACK_FRAMES    = 1      # ✅ FIXED: was 2 → 1 (show from first confirmed frame)
                             #    With brief appearances (arm through window), 2 was too slow

# Visualisation
TRACK_ID_COLOR      = (255, 255, 0)    # Cyan — track ID label
TRAIL_COLOR         = (200, 200, 0)    # Trail dots color
TRAIL_LENGTH        = 30               # How many past positions to draw
SHOW_TRAILS         = True