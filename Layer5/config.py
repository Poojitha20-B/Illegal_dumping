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
MIN_MOVE_PX_FOR_COUPLING = 3.0
# MIN_COUPLING_FRAMES — removed. State-machine threshold, no longer used
# now that the LLM reasons over the raw coupling-over-time timeline instead
# of a frame-count threshold. (_is_ghost() uses peak_coupling >= 0.7 instead.)

# Release detection — removed (state-machine only, no longer used):
# DIVERGE_COS_THRESH, DIVERGE_DIST_GROW, DIVERGE_CONFIRM_FRAMES

# Rest confirmation — removed (state-machine only, no longer used):
# REST_VEL_PX, REST_CONFIRM_FRAMES, REST_MAX_WAIT

# Trajectory intent — removed (heuristic-only, no longer used):
# TRAJ_WINDOW, TRAJ_PERSON_WEIGHT, TRAJ_OBJECT_WEIGHT, TRAJ_LEGAL_THRESH

# Bin logic — removed (heuristic-only, no longer used; bin distance is now
# passed to the LLM as bin_context and it judges proximity itself):
# BIN_LEGAL_RADIUS_PX, BIN_PERSON_RADIUS_PX, BIN_APPROACH_THRESH,
# BIN_APPROACH_CORROBORATED, BIN_ENTRY_MIN_PEAK_COS

# Confidence scoring — removed (heuristic-blend only, no longer used;
# the LLM returns confidence directly):
# CONF_COUPLING_W, CONF_DIVERGE_W, CONF_REST_W, CONF_BIN_PROX_W,
# MIN_CONFIDENCE_TO_ACT

# Case management
MAX_CASE_AGE_FRAMES      = 500
# OFFSCREEN_RELEASE_FRAMES — removed (state-machine only, no longer used).

# ── Case closure thresholds ─────────────────────────────────────────
# These replace the old state-machine timers. The monitoring window for a
# pair closes — triggering the single final LLM call — when one of these
# fires, rather than when a rest/divergence counter crosses a threshold.

# How many consecutive frames the person must be gone before the
# monitoring window closes and the LLM is called.
PERSON_GONE_CLOSE_FRAMES = 30  # ~1 second at 30fps

# Trajectory intent — TRAJ_WINDOW is still used (maxlen for
# _PersonHistory.trail in agent.py). The rest of the old intent-scoring
# heuristic is genuinely dead and stays removed:
# TRAJ_PERSON_WEIGHT, TRAJ_OBJECT_WEIGHT, TRAJ_LEGAL_THRESH
TRAJ_WINDOW = 30
# How many consecutive frames the object must be missing (after at
# least some coupling was seen) before closing the window.
OBJECT_MISSING_CLOSE_FRAMES = 45  # ~1.5 seconds at 30fps

# Ghost filter
GHOST_MIN_FRAMES         = 15
GHOST_MIN_MOVEMENT       = 20.0

# ── Agentic Attention Controller — single final LLM call per case ────────
# Reads GROQ_API_KEY from env. Swap LLM_MODEL / call_agent_final() in
# llm_controller.py if you'd rather use a different provider.
LLM_MODEL                = "llama-3.3-70b-versatile"
LLM_MAX_RETRIES          = 2          # retry on malformed JSON (Llama tool-call flakiness)
LLM_MAX_RE_EXAMINE_ROUNDS = 1         # unused now (no intermediate calls) — kept for import safety
# LLM_FLAG_CONFIDENCE_MIN — removed, unused by the new single-call path.
# LLM_CALL_DEBOUNCE_FRAMES — removed. There's exactly one LLM call per
# case now (at window closure), so there's nothing to debounce.

# Fix 2 — below this object-detection confidence, a bbox centroid isn't
# trusted for trail updates / velocity computation. A jittery low-confidence
# detection can produce a fake spike indistinguishable from a genuine
# thrown object. Still used by the confidence gate in the main per-case loop.
MIN_OBJECT_CONFIDENCE     = 0.15

# Fix 2 — below this object-detection confidence, a bbox centroid isn't
# trusted for velocity-spike computation. A jittery low-confidence detection
# can produce a fake spike indistinguishable from a genuine thrown object.
MIN_OBJECT_CONFIDENCE     = 0.15