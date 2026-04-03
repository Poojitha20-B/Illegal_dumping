# ─────────────────────────────────────────
#  Layer 2 — ByteTrack Configuration
# ─────────────────────────────────────────

TRACK_HIGH_THRESH   = 0.25   # was 0.35 — handbag at 0.37 was borderline, kept dying
TRACK_LOW_THRESH    = 0.10   # was 0.15 — catch more low-conf frames
TRACK_MATCH_THRESH  = 0.20   # was 0.30 — handbag moves fast, IoU drops between frames
NEW_TRACK_THRESH    = 0.25   # was 0.30 — consistent with new high thresh
MAX_TIME_LOST       = 60     # was 40 — ~2.4s survival at 25fps
MIN_TRACK_FRAMES    = 2      # persons stay at 2, objects use 1 (handled in tracker)

# Visualisation
TRACK_ID_COLOR      = (255, 255, 0)
TRAIL_COLOR         = (200, 200, 0)
TRAIL_LENGTH        = 30
SHOW_TRAILS         = True