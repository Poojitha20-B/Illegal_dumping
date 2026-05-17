# ─────────────────────────────────────────
#  Layer 2 — ByteTrack Configuration
# ─────────────────────────────────────────

# ByteTrack thresholds
TRACK_HIGH_THRESH   = 0.35
TRACK_LOW_THRESH    = 0.15
MATCH_THRESH        = 0.35
CENTROID_FALLBACK_PX = 80

NEW_TRACK_THRESH    = 0.3
MAX_TIME_LOST       = 40
MIN_TRACK_FRAMES    = 2

# ── Aliases for original tracker.py ──────────────────────────────────────────
TRACK_MATCH_THRESH  = MATCH_THRESH
TRACK_SECOND_THRESH = TRACK_LOW_THRESH
MIN_CONFIRM_FRAMES  = MIN_TRACK_FRAMES

# ── Visualisation ─────────────────────────────────────────────────────────────
TRACK_ID_COLOR      = (255, 255, 0)
TRAIL_COLOR         = (200, 200, 0)
TRAIL_LENGTH        = 60
TRAIL_MAXLEN        = TRAIL_LENGTH
SHOW_TRAILS         = True

# ── Velocity EMA ──────────────────────────────────────────────────────────────
VELOCITY_ALPHA      = 0.4

# ── Class locking ─────────────────────────────────────────────────────────────
LOCK_CLASS_ON_CONFIRM = True

# ── Kalman filter noise ───────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE     = 0.01
KALMAN_MEASUREMENT_NOISE = 0.1

# ── Uncertainty scoring weights ───────────────────────────────────────────────
MOTION_ERROR_WEIGHT  = 0.4
MISSED_FRAMES_WEIGHT = 0.4
SUDDEN_LOSS_WEIGHT   = 0.2

MOTION_ERROR_THRESH_PX = 50
MISSED_FRAMES_SPIKE    = 10
UNCERTAINTY_THRESHOLD  = 0.6

# ── ROI Recovery ──────────────────────────────────────────────────────────────
ROI_EXPANSION_FACTOR  = 2.0
ROI_VELOCITY_SCALE    = 3.0
ROI_MIN_SIZE          = 32
ROI_MAX_SIZE          = 640
ROI_MATCH_IOU_THRESH  = 0.15
ROI_MATCH_DIST_THRESH = 80

# ── Visualizer toggles ────────────────────────────────────────────────────────
SHOW_VELOCITY     = False
SHOW_TRACK_STATUS = False

# ── Additional aliases for GitHub tracker.py ──────────────────────────────────
PREDICT_FRAMES          = 5       # frames to predict ahead when track is lost
MAX_MATCH_DISTANCE_PX   = 200     # max centroid distance for matching
GHOST_COOLDOWN_FRAMES   = 20      # frames between ghost throw counts
MAX_RECOVERABLE_FRAMES  = 10      # max frames a lost track can be recovered

# ReID settings
REID_ENABLED            = True
REID_WEIGHT             = 0.4
MOTION_WEIGHT           = 0.4
IOU_WEIGHT              = 0.2
REID_MAX_COSINE_DIST    = 0.4
REID_FALLBACK_IOU       = 0.15
REID_MIN_CROP_SIZE      = 32
# Raise these two values to stabilize person tracking
# Raise these two values to stabilize person tracking
MAX_TIME_LOST    = 60    # keep lost person IDs alive longer
MIN_TRACK_FRAMES = 10    # hard floor — ghost tracks die before this threshold

# ── Hysteresis thresholding ──────────────────────────────────────────────────
# A new track ID can only be BORN if confidence >= INIT_TRACK_THRESH.
# Active tracks are kept alive if confidence >= TRACK_HIGH_THRESH (0.35).
INIT_TRACK_THRESH = 0.50

# ── Possession search radius ─────────────────────────────────────────────────
MAX_POSSESSION_DISTANCE_PX = 150   # max px for historical centroid match
POSSESSION_HISTORY_FRAMES  = 20    # how many frames back to search