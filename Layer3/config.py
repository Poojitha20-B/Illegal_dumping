# ─────────────────────────────────────────────────────────
#  Layer 3/config.py — Memory / Temporal State Engine Configuration
# ─────────────────────────────────────────────────────────

# ── Sequence buffer ───────────────────────────────────────
SEQUENCE_LENGTH     = 24    # frames of history kept per pair (T)
                            # must be >= MIN_SEQUENCE_FOR_MODEL used in Layer 4

# ── Pairing rules ─────────────────────────────────────────
MAX_PAIR_DISTANCE   = 300   # max pixel distance (centroid-to-centroid) to form a pair
                            # pairs farther than this are not tracked

# ── Persistence — keep data alive when a track briefly disappears ────
MAX_MISSING_FRAMES  = 20    # frames a track can be absent before its memory is wiped
                            # short occlusions must NOT break a sequence

# ── Feature normalisation ─────────────────────────────────
# Normalise raw pixel values to roughly [0, 1] so the model sees stable inputs
NORM_DISTANCE       = 640.0    # divide raw pixel distances by this
NORM_VELOCITY       = 50.0     # divide pixel/frame velocity by this
NORM_AREA           = 100000.0 # divide bbox area (px²) by this

# ── Interaction thresholds ────────────────────────────────
HOLD_DISTANCE_PX    = 90   # if person-object centroid distance < this → "holding"

# ── Visualisation ─────────────────────────────────────────
SHOW_PAIRS          = True
SHOW_SEQUENCES      = True
PAIR_LINE_COLOR     = (255, 180, 0)    # orange — pair line
ACTIVE_PAIR_COLOR   = (0, 255, 180)    # teal   — pair with full sequence buffer
SEQUENCE_TEXT_COLOR = (255, 255, 255)