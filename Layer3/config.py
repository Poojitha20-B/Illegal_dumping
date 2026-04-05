# ─────────────────────────────────────────────────────────
#  Layer 3/config.py — Memory / Temporal State Engine Configuration
# ─────────────────────────────────────────────────────────

# ── Sequence buffer ───────────────────────────────────────
SEQUENCE_LENGTH     = 24    # frames of history kept per pair (T)
                            # must be >= MIN_SEQUENCE_FOR_MODEL used in Layer 4

# ── Pairing rules ─────────────────────────────────────────
MAX_PAIR_DISTANCE   = 400   # FIX 1 (was 300): increased to handle car-window
                            # scenarios where person and object centroids are
                            # farther apart due to arm extension / throw distance

# ── Persistence — keep data alive when a track briefly disappears ────
MAX_MISSING_FRAMES  = 30    # FIX 2 (was 20): more tolerance for brief occlusions
                            # thrown objects disappear quickly — give tracker
                            # more frames to re-acquire before wiping memory

# ── Feature normalisation ─────────────────────────────────
# Normalise raw pixel values to roughly [0, 1] so the model sees stable inputs
NORM_DISTANCE       = 640.0    # divide raw pixel distances by this
NORM_VELOCITY       = 50.0     # divide pixel/frame velocity by this
NORM_AREA           = 100000.0 # divide bbox area (px²) by this

# ── Interaction thresholds ────────────────────────────────
# FIX 3 (was 90px): the original threshold assumed hand-to-hand proximity.
# For car-window dumping, the person's centroid is in the middle of the
# visible body/window area while the object centroid is at arm's tip —
# measured distance in this video is ~291px even when clearly "holding".
# 300px covers arm-length + typical bbox centroid offsets.
HOLD_DISTANCE_PX    = 300

# ── Visualisation ─────────────────────────────────────────
SHOW_PAIRS          = True
SHOW_SEQUENCES      = True
PAIR_LINE_COLOR     = (255, 180, 0)    # orange — pair line
ACTIVE_PAIR_COLOR   = (0, 255, 180)    # teal   — pair with full sequence buffer
SEQUENCE_TEXT_COLOR = (255, 255, 255)