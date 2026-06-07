"""
Layer 5 — Runtime-Tunable Config
All values here may be patched by Layer1.calibrator.apply()
"""

# Ghost filter
GHOST_MIN_FRAMES         = 15
GHOST_MIN_MOVEMENT       = 20.0

# Motion coupling
COUPLING_WINDOW          = 8
COUPLING_COS_THRESH      = 0.60
COUPLING_SPEED_RATIO     = 3.0
MIN_COUPLING_FRAMES      = 5
MIN_MOVE_PX_FOR_COUPLING = 3.0

# Release detection
DIVERGE_COS_THRESH       = 0.20
DIVERGE_DIST_GROW        = True
DIVERGE_CONFIRM_FRAMES   = 3

# Rest confirmation
REST_VEL_PX              = 4.0
REST_CONFIRM_FRAMES      = 3
REST_MAX_WAIT            = 20

# Trajectory intent
TRAJ_WINDOW              = 25
TRAJ_PERSON_WEIGHT       = 0.55
TRAJ_OBJECT_WEIGHT       = 0.45
TRAJ_LEGAL_THRESH        = 0.60

# Bin logic
BIN_LEGAL_RADIUS_PX      = 210
BIN_PERSON_RADIUS_PX     = 350
BIN_APPROACH_THRESH      = 0.35
BIN_APPROACH_CORROBORATED = 0.20
BIN_ENTRY_MIN_PEAK_COS   = 0.70

# Confidence scoring
CONF_COUPLING_W          = 0.30
CONF_DIVERGE_W           = 0.25
CONF_REST_W              = 0.20
CONF_BIN_PROX_W          = 0.25
MIN_CONFIDENCE_TO_ACT    = 0.45

# Case management
MAX_CASE_AGE_FRAMES      = 500
OFFSCREEN_RELEASE_FRAMES = 8