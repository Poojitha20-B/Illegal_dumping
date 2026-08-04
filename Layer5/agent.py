"""
Layer 5 — Agentic Perception Controller
=========================================

FIXES vs previous version
--------------------------

FIX 1 — Cup wrongly marked LEGAL (no_l4_event path)
FIX 2 — MIN_COUPLING_FRAMES too strict
FIX 3 — intent=0.00 with no bins pushed toward LEGAL
FIX 4 — L4-confirmed violations had confidence penalised by no_coupling
FIX 5 — rest_frames=0 tanked evidence confidence

FIX 7 (NEW) — L4 wrongly flags LEGAL disposal as VIOLATION due to
               bin-distance heuristic failure.
  ROOT CAUSE:
    Layer 4's _decide() measures the distance between the *final tracked
    position of the thrown object* and the bin's bottom-center. When a person
    throws or drops an item INTO a bin, the tracker loses the object the moment
    it enters the bin — so final_obj_pos is wherever the object was *last seen*
    (near the person, 700+ px away), not where it landed. L4 therefore sees a
    huge distance and flags VIOLATION.

  L5 OVERRIDE SIGNALS (all must converge):
    a) rest_via_timeout=True AND rest_frames==0
       → object vanished suddenly (not settled on ground = likely entered bin)
    b) bins_present=True
       → at least one bin is in the scene
    c) strong coupling (coupling_frames >= MIN_COUPLING_FRAMES, cos near 1.0)
       → object was genuinely being carried/held before it vanished
    d) person approached a bin (bin_approach_score >= BIN_APPROACH_THRESH)
       → person's trajectory was toward the bin

  When all four signals converge, L5 overrides to LEGAL regardless of L4.
  This is tracked in the reasoning log as "l5_bin_entry_override".

  SECONDARY SIGNAL — person proximity at disappearance:
    If the person's last known position was within BIN_PERSON_RADIUS_PX of
    the bin, treat that as an additional corroboration (lowers the approach
    threshold needed).

FIX 8 (NEW) — Confidence-blind state advancement / conclusions handed to LLM
               as settled fact instead of raw evidence.
  ROOT CAUSE:
    Two separate problems compounded:
      (a) case.obj_trail (and therefore case.final_obj_pos) was fed from
          the tracker's centroid EVERY frame an object was found, with no
          check on obj.confidence. A flickering low-confidence detection
          (e.g. clothing/shadow misclassified as the tracked object) still
          produced jittery positions that fed diverge_frames (via motion
          coupling's cosine-similarity check) and rest_frames (via obj_trail
          velocity in _advance()) — so a case could reach RELEASED/RESTING
          purely from noise, before the object was ever actually released.
      (b) Once a trigger fired, TriggerDetector handed the LLM a note like
          "object came to rest (via_timeout=...)" — a stated CONCLUSION —
          rather than the raw disputable numbers (position, confidence,
          distance) a human supervisor would actually look at before
          agreeing the object had been released.

  FIX:
    (a) The confidence gate now sits at the point obj_trail gets appended —
        both in _update_motion_coupling (coupling/divergence) and in the
        main per-case loop (rest/position tracking) — not only inside
        TriggerDetector's own redundant velocity-spike check. A detection
        below cfg.MIN_OBJECT_CONFIDENCE no longer updates the trail; it's
        treated the same as "object not seen this frame" (obj_missing_frames
        increments), so it can't silently drive diverge_frames/rest_frames.
    (b) Trigger notes (divergence_onset, rest_onset, possession_confirmed)
        now report raw position/confidence/distance evidence instead of
        asserting "released"/"came to rest" as fact — the LLM is expected
        to judge for itself whether the evidence actually supports release,
        the same way FIX text in llm_controller.py's system prompt now says
        explicitly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from Layer2.track_state import TrackedObject
from Layer2.bin_tracker import TrackedBin
from Layer4.dumping_inference import DumpingEvent  # kept only for the update() type hint
import Layer5.config as cfg
from Layer5.belief_state import BeliefStateManager, PipelineState, KinematicSnapshot
from Layer5 import llm_controller



# ══════════════════════════════════════════════════════════════════════════════
#  Velocity / motion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2))

def _bottom_center(bbox: np.ndarray) -> Tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2), float(bbox[3]))

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _vel(trail: deque, n: int = 3) -> Tuple[float, float]:
    pts = list(trail)
    if len(pts) < 2:
        return (0.0, 0.0)
    tail = pts[-min(n, len(pts)):]
    vx = (tail[-1][0] - tail[0][0]) / max(len(tail) - 1, 1)
    vy = (tail[-1][1] - tail[0][1]) / max(len(tail) - 1, 1)
    return (vx, vy)

def _speed(vel: Tuple[float, float]) -> float:
    return math.hypot(vel[0], vel[1])

def _cosine_sim(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return dot / (mag1 * mag2)

def _nearest_bin(
    pt: Tuple[float, float], bins: List[TrackedBin]
) -> Tuple[float, Optional[int]]:
    if not bins:
        return float("inf"), None
    best_d, best_id = float("inf"), None
    for b in bins:
        d = _dist(pt, _bottom_center(b.bbox))
        if d < best_d:
            best_d, best_id = d, b.bin_id
    return best_d, best_id

def _point_in_bbox(
    pt:   Tuple[float, float],
    bbox: np.ndarray,
) -> bool:
    """Phase 1: Check if point is inside bbox."""
    return (bbox[0] <= pt[0] <= bbox[2] and
            bbox[1] <= pt[1] <= bbox[3])


def _iot_score(
    trash_bbox: np.ndarray,
    bin_bbox:   np.ndarray,
) -> float:
    """Phase 2: Intersection-over-Trash score."""
    ix1 = max(trash_bbox[0], bin_bbox[0])
    iy1 = max(trash_bbox[1], bin_bbox[1])
    ix2 = min(trash_bbox[2], bin_bbox[2])
    iy2 = min(trash_bbox[3], bin_bbox[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    trash_area = (trash_bbox[2]-trash_bbox[0]) * (trash_bbox[3]-trash_bbox[1])
    return inter / trash_area if trash_area > 0 else 0.0


def _perimeter_dist(
    pt:       Tuple[float, float],
    bin_bbox: np.ndarray,
) -> float:
    """Phase 3: Distance from point to nearest edge of bbox."""
    cx, cy = pt
    x1, y1, x2, y2 = bin_bbox
    dx = max(x1 - cx, 0.0, cx - x2)
    dy = max(y1 - cy, 0.0, cy - y2)
    return math.hypot(dx, dy)


def _best_bin_hierarchical(
    trash_bbox: np.ndarray,
    pt:         Tuple[float, float],
    bins:       List[TrackedBin],
) -> Tuple[float, Optional[TrackedBin]]:
    """
    Hierarchical bin assignment:
      Phase 1 — centroid inside bin bbox (containment)
      Phase 2 — highest IoT score
      Phase 3 — nearest perimeter distance
    """
    if not bins:
        return float("inf"), None

    # Phase 1: containment
    for tb in bins:
        if _point_in_bbox(pt, tb.bbox):
            return 0.0, tb

    # Phase 2: IoT overlap
    best_iot, best_tb = 0.0, None
    for tb in bins:
        score = _iot_score(trash_bbox, tb.bbox)
        if score > best_iot:
            best_iot, best_tb = score, tb
    if best_tb is not None and best_iot > 0:
        # Return a pseudo-distance inversely proportional to overlap
        return (1.0 - best_iot) * 100, best_tb

    # Phase 3: perimeter distance fallback
    best_d, best_tb = float("inf"), None
    for tb in bins:
        d = _perimeter_dist(pt, tb.bbox)
        if d < best_d:
            best_d, best_tb = d, tb
    return best_d, best_tb

# ══════════════════════════════════════════════════════════════════════════════
#  Per-person history
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PersonHistory:
    frames:   int   = 0
    movement: float = 0.0
    last_pos: Optional[Tuple[float, float]] = None
    trail:    deque = field(default_factory=lambda: deque(maxlen=cfg.TRAJ_WINDOW))

    def update(self, pos: Tuple[float, float]) -> None:
        self.frames += 1
        if self.last_pos:
            self.movement += _dist(pos, self.last_pos)
        self.last_pos = pos
        self.trail.append(pos)

    @property
    def is_ghost(self) -> bool:
        return self.frames < cfg.GHOST_MIN_FRAMES or self.movement < cfg.GHOST_MIN_MOVEMENT

    def velocity(self) -> Tuple[float, float]:
        return _vel(self.trail, n=4)

    def bin_approach_score(
        self, bins: List[TrackedBin]
    ) -> Tuple[float, Optional[int]]:
        if not bins or len(self.trail) < 4:
            return 0.0, None
        trail    = list(self.trail)
        # Pick the bin the person is moving TOWARD (closest to trail END, not start)
        best_bin = min(bins, key=lambda b: _dist(trail[-1], _bottom_center(b.bbox)))
        bin_pos  = _bottom_center(best_bin.bbox)
        converge = sum(
            1 for i in range(1, len(trail))
            if _dist(trail[i], bin_pos) < _dist(trail[i-1], bin_pos)
        )
        return converge / max(len(trail) - 1, 1), best_bin.bin_id

    def nearest_bin_dist(self, bins: List[TrackedBin]) -> float:
        """Distance from person's last known position to nearest bin."""
        if not bins or self.last_pos is None:
            return float("inf")
        return min(_dist(self.last_pos, _bottom_center(b.bbox)) for b in bins)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-pair case
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Case:
    pair_id:     str
    person_id:   int
    trash_id:    int
    start_frame: int

    # Per-frame data — updated every frame this pair is tracked.
    obj_trail:         deque = field(default_factory=lambda: deque(maxlen=40))
    person_trail_snap: deque = field(default_factory=lambda: deque(maxlen=40))
    coupling_scores:   List[float] = field(default_factory=list)
    peak_coupling:     float = 0.0

    # Object tracking
    obj_missing_frames: int = 0
    final_obj_pos:     Optional[Tuple[float, float]] = None
    # FIX 8: last object-detection confidence actually used to update the
    # trail (None while the object hasn't been seen at trustworthy
    # confidence yet). Kept alongside final_obj_pos so the kinematic
    # snapshots can report "how sure were we about this position" rather
    # than only the position itself.
    final_obj_confidence: Optional[float] = None

    # Closure tracking — replaces the old state machine. The window closes
    # (and the single final LLM call fires) when person_gone_frames or
    # obj_missing_frames crosses the thresholds in config.py.
    person_gone_frames: int = 0
    closed:  bool = False
    result:  Optional[dict] = None

    # Whether a bin was visible in ANY frame during this case's lifetime —
    # tracked_bins passed to _finalise_with_llm() only reflects the FINAL
    # frame, so a bin seen earlier and later occluded/out of view would
    # otherwise be reported to the LLM as "no bin ever present" when one
    # genuinely was.
    bins_were_present: bool = False
    reasoning: List[str] = field(default_factory=list)
    frames_since_update: int = 0

    def log(self, msg: str) -> None:
        self.reasoning.append(msg)

    def last_reason(self, n: int = 3) -> str:
        return " | ".join(self.reasoning[-n:]) if self.reasoning else ""

    def last_reason_str(self, n: int = 2) -> str:
        return " | ".join(self.reasoning[-n:]) if self.reasoning else ""

# ══════════════════════════════════════════════════════════════════════════════
#  Main Agent
# ══════════════════════════════════════════════════════════════════════════════

class DumpingAgent:
    """
    Layer 5 — Agentic Perception Controller.
    Call update() once per frame.
    """

    def __init__(self):
        self._cases:   Dict[str, _Case]          = {}
        self._persons: Dict[int, _PersonHistory] = {}
        self._used_trash_ids: set = set()
        self.active_cases:  List[_Case]    = []
        self.frame_signals: Dict[str, str] = {}

        # LLM is the sole decision-maker now — one call per case, made
        # from _finalise_with_llm() when the monitoring window closes.
        # No more TriggerDetector (no intermediate calls to trigger) and
        # no more _llm_disabled fallback flag (no heuristic path to fall
        # back to — see call_agent_final()'s own error handling instead).
        self._beliefs  = BeliefStateManager()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
    self,
    frame_idx:    int,
    tracked_objs: List[TrackedObject],
    tracked_bins: List[TrackedBin],
    l4_events:    List[DumpingEvent],
) -> List[dict]:
        # `l4_events` kept in the signature for call-site compatibility
        # (run_pipeline.py still passes it) but is no longer consumed anywhere
        # in this method. L4 is fully dropped from Layer 5 — the LLM is the
        # sole decision-maker now, reasoning over raw kinematics only.

        self._update_person_histories(tracked_objs)
        self._update_motion_coupling(tracked_objs, frame_idx)
        self._age_cases()
        self.frame_signals = {}

        # Frame's currently-visible persons — used below to drive
        # person_gone_frames per case (replaces the old per-frame presence
        # check that used to live inside TriggerDetector.check()).
        visible_person_ids = {
            o.track_id for o in tracked_objs if o.class_name == "person"
        }

        new_verdicts: List[dict] = []

        # NOTE: case creation now happens ONLY inside _update_motion_coupling
        # (via its own _get_or_create call) — there is no more l4_events loop
        # seeding cases, since L4 is dropped.

        for pair_id, case in list(self._cases.items()):
            if case.closed:
                continue

            ph = self._persons.get(case.person_id)

            if self._is_ghost(case, ph):
                continue

            obj = self._find_obj(case.trash_id, tracked_objs)

            # FIX 8a: confidence gate before trusting a detection's centroid
            # for trail/velocity purposes — unchanged from before.
            obj_confidence_ok = obj is not None and obj.confidence >= cfg.MIN_OBJECT_CONFIDENCE
            if obj_confidence_ok:
                pos = _centroid(obj.bbox)
                case.obj_trail.append(pos)
                case.final_obj_pos = pos
                case.final_obj_confidence = obj.confidence
                case.obj_missing_frames = 0
                if ph and ph.last_pos:
                    case.person_trail_snap.append(ph.last_pos)
            else:
                if obj is not None:
                    case.log(
                        f"L5_low_conf_position_ignored obj_conf={obj.confidence:.2f} "
                        f"< min={cfg.MIN_OBJECT_CONFIDENCE:.2f} (not trusted for trail)"
                    )
                case.obj_missing_frames += 1
                # No more _State.POSSESSED / offscreen_release transition —
                # obj_missing_frames itself is now one of the two closure
                # signals checked below, so nothing further to do here.

            # ── Closure signal 1: person gone ────────────────────────────────
            if case.person_id in visible_person_ids:
                case.person_gone_frames = 0
            else:
                case.person_gone_frames += 1

            # ── Track whether a bin was EVER visible during this case's
            # lifetime, not just in the final frame at closure time.
            if tracked_bins:
                case.bins_were_present = True

            # ── Per-frame kinematic snapshot (unchanged purpose — dense
            # history for the LLM's own re-examination / final-call prompt) ──
            # ── Per-frame kinematic snapshot (unchanged purpose — dense
            # history for the LLM's own re-examination / final-call prompt) ──
            # obj_pos/obj_confidence report THIS FRAME's trustworthy reading
            # only — None when the object wasn't confidently seen this
            # frame — rather than case.final_obj_pos/final_obj_confidence,
            # which hold the last confident value indefinitely and would
            # otherwise make a long-gone object look "currently tracked at
            # low confidence" for every frame after it actually disappeared.
            obj_pos            = case.final_obj_pos if obj_confidence_ok else None
            obj_conf_for_frame  = case.final_obj_confidence if obj_confidence_ok else None
            person_pos = ph.last_pos if ph else None
            obj_speed    = _speed(_vel(case.obj_trail, n=3)) if len(case.obj_trail) >= 2 else 0.0
            person_speed = _speed(ph.velocity()) if ph else 0.0
            distance     = _dist(obj_pos, person_pos) if (obj_pos and person_pos) else float("inf")

            # coupling_score: cosine similarity of this frame's object/person
            # velocity, when both have enough trail history to compute one —
            # same formula _update_motion_coupling uses internally, recomputed
            # here rather than threaded through _Case, to keep this snapshot
            # self-contained and avoid adding a new per-frame field to _Case
            # just for this. None when either trail is too short.
            coupling_score = None
            if ph and len(ph.trail) >= 3 and len(case.obj_trail) >= 3:
                p_vel = ph.velocity()
                o_vel = _vel(case.obj_trail, n=3)
                if _speed(p_vel) > 1e-3 and _speed(o_vel) > 1e-3:
                    coupling_score = _cosine_sim(p_vel, o_vel)

            belief = self._beliefs.get_or_create(pair_id)
            belief.record_snapshot(KinematicSnapshot(
                frame_idx=frame_idx, obj_pos=obj_pos, person_pos=person_pos,
                obj_speed=obj_speed, person_speed=person_speed, distance=distance,
                obj_confidence=obj_conf_for_frame,
                coupling_score=coupling_score,
            ))

            # ── Closure signal 2: object missing (only counts once some
            # coupling was actually seen — an object that was never coupled
            # in the first place shouldn't trigger this path at all; that
            # case just ages out via _purge()'s MAX_CASE_AGE_FRAMES instead) ──
            # ── Closure signal: only close once the PERSON is gone — either
            # they've fully left, or they've been gone a short while AND the
            # object has also been missing a long time (truly stale). A solo
            # obj_missing trigger fired too early for static dumping: the
            # object commonly drops out of tracking (occlusion / low
            # confidence on a small stationary item) the moment it's set
            # down — well before the person actually walks off — so closing
            # on that alone truncated the timeline before showing what the
            # person does next.
            window_closed = (
                case.person_gone_frames >= cfg.PERSON_GONE_CLOSE_FRAMES
                or (
                    case.person_gone_frames >= 15
                    and case.obj_missing_frames >= cfg.OBJECT_MISSING_CLOSE_FRAMES
                )
            )

            if window_closed:
                trigger = (
                    "person" if case.person_gone_frames >= cfg.PERSON_GONE_CLOSE_FRAMES
                    else "person+object_stale"
                )
                case.log(
                    f"window_closed person_gone={case.person_gone_frames}f "
                    f"obj_missing={case.obj_missing_frames}f "
                    f"trigger={trigger}"
                )
                verdict = self._finalise_with_llm(case, tracked_bins, frame_idx, ph)
                if verdict:
                    new_verdicts.append(verdict)

            # ── Rebuild frame_signals for the visualizer (Gap 4) ─────────────
            coupling_now = coupling_score if coupling_score is not None else 0.0
            self.frame_signals[pair_id] = (
                f"coupling={coupling_now:.2f} "
                f"peak={case.peak_coupling:.2f} | "
                f"obj_missing={case.obj_missing_frames}f "
                f"person_gone={case.person_gone_frames}f | "
                + case.last_reason_str()
            )

        self.active_cases = [c for c in self._cases.values() if not c.closed]
        self._purge()
        return new_verdicts
    
    def get_all_results(self, min_confidence: float = 0.30) -> List[dict]:
        return [c.result for c in self._cases.values()
                if c.closed and c.result is not None
                and not c.result.get("skipped", False)
                and c.result.get("confidence", 0.0) >= min_confidence]
    def finalize_all(self, frame_idx: int, tracked_bins: List[TrackedBin]) -> List[dict]:
        """
        Force-close every still-open case — called once when the video ends.
        Needed because the in-loop closure condition only fires on
        person_gone_frames (or person_gone+obj_missing-stale); a person who
        never leaves frame before the clip ends would otherwise leave every
        case open forever and produce zero verdicts.
        """
        verdicts = []
        for pair_id, case in list(self._cases.items()):
            if case.closed:
                continue
            ph = self._persons.get(case.person_id)
            case.log(f"video_ended_forcing_closure frame={frame_idx}")
            verdict = self._finalise_with_llm(case, tracked_bins, frame_idx, ph)
            if verdict:
                verdicts.append(verdict)
        return verdicts

    def get_active_beliefs(self) -> List[dict]:
        """
        Snapshot of all non-locked BeliefStates, for debug overlay / testing.
        Phase 2 only — confidence here is NOT the final verdict confidence
        (that still comes from _finalise's weighted formula); it's whatever
        the LLM belief loop will eventually write here in Phase 3.
        """
        return [b.to_prompt_dict() for b in self._beliefs.all_active()]

    # ── Motion coupling ───────────────────────────────────────────────────────

    def _update_motion_coupling(
        self, tracked_objs: List[TrackedObject], frame_idx: int
    ) -> None:
        persons     = [o for o in tracked_objs if o.class_name == "person"]
        non_persons = [o for o in tracked_objs if o.class_name != "person"]

        for obj in non_persons:
            obj_c = _centroid(obj.bbox)

            closest_p, closest_d = None, float("inf")
            for p in persons:
                d = _dist(obj_c, _centroid(p.bbox))
                if d < closest_d:
                    closest_d, closest_p = d, p

            if closest_p is None or closest_d > 300:
                continue

            def _bbox_area(bbox):
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            best_area = _bbox_area(closest_p.bbox)
            for p in persons:
                if p.track_id == closest_p.track_id:
                    continue
                d = _dist(obj_c, _centroid(p.bbox))
                if d <= closest_d * 1.5:   # within 50% further
                    area = _bbox_area(p.bbox)
                    if area > best_area * 1.8:  # at least 80% bigger
                        closest_p = p
                        closest_d = d
                        best_area = area

            # Trajectory lookback: if object just appeared, check who was
            # historically near its position rather than who is nearest now
            obj_pos = _centroid(obj.bbox)

            # Step 1: proximity lookback (held/slow drop)
            historical_pid = self._nearest_historical_person(
                obj_pos, radius_px=160.0
            )

            # Step 2: ballistic projection (far throw)
            if historical_pid is None:
                historical_pid = self._nearest_person_ballistic(
                    obj_pos, radius_px=400.0
                )

            if historical_pid is not None and historical_pid != closest_p.track_id:
                closest_p = type('_P', (), {'track_id': historical_pid})()

            pair_id = f"person_{closest_p.track_id}_trash_{obj.track_id}"
            case    = self._get_or_create(
                pair_id, closest_p.track_id, obj.track_id, frame_idx
            )

            if case.closed:
                continue

            # FIX 8a: confidence gate BEFORE this frame's centroid can feed
            # obj_trail / coupling_frames / diverge_frames. Previously this
            # loop had no confidence check at all — a flickering
            # low-confidence detection (e.g. clothing/shadow misclassified)
            # could still trip cosine-similarity divergence checks below
            # and silently advance a case toward RELEASED. Skip this
            # object entirely for THIS frame's coupling computation when
            # its detection confidence isn't trustworthy; existing
            # coupling/diverge counters simply don't get touched this
            # frame (no spurious reset either) rather than being corrupted
            # by noise.
            if obj.confidence < cfg.MIN_OBJECT_CONFIDENCE:
                case.log(
                    f"L5_low_conf_coupling_skip obj_conf={obj.confidence:.2f} "
                    f"< min={cfg.MIN_OBJECT_CONFIDENCE:.2f}"
                )
                continue

            ph = self._persons.get(closest_p.track_id)
            if ph is None or len(ph.trail) < 3:
                continue
            p_vel   = ph.velocity()
            p_speed = _speed(p_vel)
            if len(case.obj_trail) < 3:
                case.obj_trail.append(obj_c)
                continue
            case.obj_trail.append(obj_c)
            o_vel   = _vel(case.obj_trail, n=3)
            o_speed = _speed(o_vel)
            if p_speed < cfg.MIN_MOVE_PX_FOR_COUPLING and o_speed < cfg.MIN_MOVE_PX_FOR_COUPLING:
                continue
            cos_sim = _cosine_sim(p_vel, o_vel)
            if p_speed > 1e-3 and o_speed > 1e-3:
                ratio = max(p_speed, o_speed) / min(p_speed, o_speed)
                if ratio > cfg.COUPLING_SPEED_RATIO:
                    cos_sim *= 0.3
            if cos_sim >= cfg.COUPLING_COS_THRESH:
                case.coupling_scores.append(cos_sim)
                case.peak_coupling = max(case.peak_coupling, cos_sim)


    # ── Finalise verdict ──────────────────────────────────────────────────────

    def _finalise_with_llm(
    self,
    case:         _Case,
    tracked_bins: List[TrackedBin],
    frame_idx:    int,
    ph:           Optional[_PersonHistory],
) -> dict:

        # ── Signal-quality gate: a case where coupling never crossed the
        # threshold that _update_motion_coupling() itself uses to record a
        # coupling score at all means the person and object were never
        # observed moving together, even once. That's not weak evidence of
        # dumping — it's the absence of evidence for possession, which
        # dumping logically requires. Skip the LLM call entirely; there is
        # nothing for it to reason about.
        if case.peak_coupling < cfg.COUPLING_COS_THRESH or len(case.coupling_scores) < 3:
            case.log(
                f"L5_skipped_no_coupling_evidence peak={case.peak_coupling:.2f} "
                f"frames={len(case.coupling_scores)} — no LLM call made"
            )
            result = {
                "violation":       False,
                "confidence":      0.0,
                "event":           "legal_disposal",
                "skipped":         True,
                "person_id":       case.person_id,
                "object_id":       case.trash_id,
                "pair_id":         case.pair_id,
                "reason":          "L5_no_coupling_evidence: object never observed moving with a person",
                "frames":          [case.start_frame, frame_idx],
                "reasoning_log":   list(case.reasoning),
                "coupling_frames": len(case.coupling_scores),
                "peak_coupling":   round(case.peak_coupling, 2),
                "release_clarity": 0.0,
                "rest_frames":     case.obj_missing_frames,
                "intent_score":    0.0,
                "obj_approach":    0.0,
            }
            case.result = result
            case.closed = True
            return result

        bins_present = len(tracked_bins) > 0 or case.bins_were_present
        belief = self._beliefs.get_or_create(case.pair_id)
        case.log(
            f"DEBUG_FINALIZE: tracked_bins_final_frame={len(tracked_bins)} "
            f"bins_were_present_during_case={case.bins_were_present}"
        )

        # ── Last known person position from the recorded timeline — NOT the
        # live `ph`, which can be stale or missing if the person's track was
        # briefly lost/reacquired under a different id right before closure.
        # Walk the kinematic history backwards to find the last frame where
        # a person position was actually recorded.
        last_person_pos = None
        for s in reversed(belief.kinematic_history):
            if s.person_pos is not None:
                last_person_pos = s.person_pos
                break

        # ── Bin context — computed once, passed raw to the LLM instead of the
        # old BIN_LEGAL_RADIUS_PX / BIN_APPROACH_THRESH heuristic thresholds.
        # The LLM judges proximity/intent itself now.
        ref_pos = last_person_pos or case.final_obj_pos
        bin_d, nearest_bin_obj = (float("inf"), None)
        if ref_pos and tracked_bins:
            ref_bbox = np.array([ref_pos[0]-5, ref_pos[1]-5,
                                ref_pos[0]+5, ref_pos[1]+5])
            bin_d, nearest_bin_obj = _best_bin_hierarchical(ref_bbox, ref_pos, tracked_bins)
        nearest_bin_id = nearest_bin_obj.bin_id if nearest_bin_obj else None

        person_approach, approach_bin_id = (
            ph.bin_approach_score(tracked_bins) if ph else (0.0, None)
        )

        coupling_vals = case.coupling_scores
        peak_coupling = case.peak_coupling
        avg_coupling  = sum(coupling_vals) / len(coupling_vals) if coupling_vals else 0.0

        # "Sustained drop" — coupling was once strong but the most recent
        # scores are much weaker, a loose proxy for "they were together, then
        # weren't" without reviving the old diverge_frames state machine.
        sustained_drop = (
            len(coupling_vals) >= 4
            and peak_coupling >= 0.5
            and (sum(coupling_vals[-3:]) / 3) < peak_coupling * 0.5
        )

        bin_context = {
            "bins_present":          bins_present,
            "nearest_bin_distance_px": round(bin_d, 0) if bin_d < float("inf") else None,
            "nearest_bin_id":        nearest_bin_id,
            "person_approach_score": round(person_approach, 2),
            # Calibrated threshold — gives the LLM a reference frame for what
            # "near" means in this specific video. Computed by the calibrator
            # per-video from scene geometry (falls back to cfg default if
            # calibration didn't run). This is a reference value, not a
            # decision rule — the LLM still judges what the evidence means.
            "calibrated_near_bin_threshold_px": cfg.BIN_LEGAL_RADIUS_PX,
            "near_threshold_meaning": (
            f"Distances below {cfg.BIN_LEGAL_RADIUS_PX}px mean the person "
            f"could reasonably reach or interact with the bin. Distances "
            f"above 2x this threshold mean the bin was clearly out of reach. "
            f"Between these values is a grey zone — use other evidence "
            f"(approach direction, coupling pattern) to judge intent.\n"
            f"Physical context: object trackers frequently lose a thrown "
            f"or placed object the instant it enters, is occluded by, or "
            f"passes behind a bin — a sudden disappearance while within "
            f"the near-bin threshold is physically consistent with the "
            f"object having entered the bin, even without a clean, "
            f"sustained coupling drop beforehand (dropping something into "
            f"a bin from close range is a small motion, not a big one).\n"
            f"person_approach_score is computed from a short recent "
            f"movement trail and can read low or noisy even when the "
            f"person is standing right at the bin — e.g. they approached "
            f"from an angle, paused, or the trail window is short. Once "
            f"a person is already within the near-bin threshold, direct "
            f"proximity is generally a stronger signal than approach "
            f"score; a low approach score alone should not outweigh "
            f"proximity plus a disappearance consistent with entering "
            f"the bin."
        ),
        }

        final_briefing = {
            "coupling_frames_observed": len(coupling_vals),
            "peak_coupling":          round(peak_coupling, 2),
            "avg_coupling":           round(avg_coupling, 2),
            "sustained_coupling_drop": sustained_drop,
            "object_missing_frames":  case.obj_missing_frames,
            "person_gone_frames":     case.person_gone_frames,
            "object_final_position":  case.final_obj_pos,
            "object_final_confidence": case.final_obj_confidence,
            "bin_context":            bin_context,
            # No heuristic_verdict / heuristic_confidence — there is no
            # heuristic verdict anymore. The old prompt's "reference, not a
            # constraint" framing doesn't apply; llm_controller.py's final
            # prompt builder needs updating to not expect those keys (flagging
            # for the llm_controller.py chunk).
        }


        try:
            final_decision = llm_controller.call_agent_final(
                belief, final_briefing,
                kinematic_timeline=list(belief.kinematic_history),
                frame_idx=frame_idx,
            )
            is_violation = final_decision.should_flag
            final_conf   = round(final_decision.confidence, 3)
            reasoning    = final_decision.new_reasoning
            case.log(
                f"L5_llm_final_verdict flag={is_violation} conf={final_conf:.2f}"
            )
        except llm_controller.LLMResponseError as e:
            # LLM unavailable (rate limit / outage) — fall back to the same
            # raw kinematic signals we already built for the prompt (bin_context,
            # sustained_drop, peak/avg coupling) instead of silently defaulting
            # to legal_disposal conf=0.00. Silently failing open here means a
            # real dumping case with strong coupling and a long-vanished object
            # never reaches plate/FaceID at all — worse than a lower-confidence
            # flag a human can review. Kept deliberately conservative: capped
            # confidence, and explicitly tagged as heuristic so it's never
            # confused with an LLM-confirmed verdict downstream.
            case.log(f"L5_llm_final_failed: {e} — using heuristic fallback")

            near_bin = (
                bin_context["bins_present"]
                and bin_context["nearest_bin_distance_px"] is not None
                and bin_context["nearest_bin_distance_px"] <= cfg.BIN_LEGAL_RADIUS_PX
            )
            object_vanished = case.obj_missing_frames >= cfg.OBJECT_MISSING_CLOSE_FRAMES
            strong_possession = peak_coupling >= 0.85 and len(coupling_vals) >= 5

            if near_bin and object_vanished:
                # Disappeared right at a bin — consistent with legal disposal.
                is_violation = False
                final_conf   = 0.30
                reasoning = (
                    "heuristic_fallback: object vanished near a bin "
                    f"(dist={bin_context['nearest_bin_distance_px']}px) — "
                    "treated as likely legal disposal, LLM unavailable to confirm"
                )
            elif strong_possession and object_vanished and not bin_context["bins_present"]:
                # Carried with strong coupling, then vanished, no bin anywhere
                # in the scene — the pattern L4/L5 exist to catch.
                is_violation = True
                final_conf   = 0.50
                reasoning = (
                    f"heuristic_fallback: peak_coupling={peak_coupling:.2f} "
                    f"obj_missing={case.obj_missing_frames}f, no bin present in scene — "
                    "flagged for human/plate review, LLM unavailable to confirm"
                )
            else:
                # Ambiguous without the LLM's judgment — don't guess either way.
                is_violation = False
                final_conf   = 0.0
                reasoning = (
                    f"heuristic_fallback: insufficient signal to call it without "
                    f"the LLM (near_bin={near_bin}, vanished={object_vanished}, "
                    f"strong_possession={strong_possession})"
                )

        reasons = [f"L5_llm_final: {reasoning}"]

        # Low-confidence non-flag → uncertain, not a confident legal call.
        # is_violation itself is untouched (still gates the plate/FaceID/
        # challan pipeline in run_pipeline.py) — this only affects labeling
        # for display and batch_eval.py's metrics.
        needs_review = (not is_violation) and (final_conf < cfg.LOW_CONFIDENCE_LEGAL_FLOOR)

        result = {
            "violation":       is_violation,
            "confidence":      final_conf,
            "event":           "illegal_dumping" if is_violation else "legal_disposal",
            "needs_review":    needs_review,
            "person_id":       case.person_id,
            "object_id":       case.trash_id,
            "pair_id":         case.pair_id,
            "reason":          " | ".join(reasons),
            "kinematic_frames_analyzed": len(belief.kinematic_history),
            "person_approach": round(person_approach, 2),
            "frames":          [case.start_frame, frame_idx],
            "reasoning_log":   list(case.reasoning),
            # ── Backward-compatible aliases for visualizer.py (Gap 4) ────────
            "coupling_frames": len(coupling_vals),
            "peak_coupling":   round(peak_coupling, 2),
            "release_clarity": round(1.0 - avg_coupling, 2) if sustained_drop else 0.0,
            "rest_frames":     case.obj_missing_frames,
            "intent_score":    0.0,
            "obj_approach":    0.0,
        }

        # ── Guard: trash object already claimed by a prior confirmed violation.
        # Carried over from the old heuristic _finalise() at your instruction —
        # now runs AFTER the LLM verdict rather than blending into a heuristic
        # confidence score, since there's no heuristic score to blend into.
        if is_violation and case.trash_id in self._used_trash_ids:
            is_violation = False
            result["violation"] = False
            result["event"] = "legal_disposal"
            reasons.append(f"L5_trash_already_claimed T{case.trash_id}")
            result["reason"] = " | ".join(reasons)
            case.log("trash_claimed_by_prior_violation")
            final_conf = max(0.0, final_conf - 0.20)
            result["confidence"] = round(final_conf, 3)

        if is_violation:
            self._used_trash_ids.add(case.trash_id)

        case.result = result
        case.closed = True

        belief.phase = PipelineState.FINALIZED
        belief.confidence = final_conf

        tag = "🚨 VIOLATION" if is_violation else ("❓ UNCERTAIN" if needs_review else "✅ LEGAL")
        print(
            f"[Layer5] {tag} | {result['event']} | conf={final_conf:.2f} | "
            f"P{case.person_id} T{case.trash_id} | "
            f"coupling_peak={peak_coupling:.2f} avg={avg_coupling:.2f} | "
            f"obj_missing={case.obj_missing_frames}f person_gone={case.person_gone_frames}f | "
            f"frames={result['frames']}"
        )
        print(f"         reason: {result['reason']}")
        return result

    # ── Ghost filter ──────────────────────────────────────────────────────────

    def _is_ghost(self, case: _Case, ph: Optional[_PersonHistory]) -> bool:
        if ph is None:
            return True
        if ph.frames < cfg.GHOST_MIN_FRAMES or ph.movement < cfg.GHOST_MIN_MOVEMENT:
            if case.peak_coupling >= 0.7:
                return False
            return True
        return False

    # ── Person histories ──────────────────────────────────────────────────────

    def _update_person_histories(self, tracked_objs: List[TrackedObject]) -> None:
        for obj in tracked_objs:
            if obj.class_name != "person":
                continue
            if obj.track_id not in self._persons:
                self._persons[obj.track_id] = _PersonHistory()
            self._persons[obj.track_id].update(_centroid(obj.bbox))

    def _nearest_historical_person(
        self,
        point: Tuple[float, float],
        radius_px: float = 160.0,
        max_frames_back: int = 25,
    ) -> Optional[int]:
        """
        Search person trail histories for whoever was closest to `point`
        up to max_frames_back frames ago. Returns person track_id or None.
        """
        best_pid  = None
        best_dist = float("inf")
        for pid, ph in self._persons.items():
            trail = list(ph.trail)
            # Only look at last max_frames_back entries
            recent = trail[-max_frames_back:] if len(trail) > max_frames_back else trail
            for pos in recent:
                d = _dist(pos, point)
                if d < best_dist:
                    best_dist = d
                    best_pid  = pid
        if best_dist <= radius_px:
            return best_pid
        return None
    
    def _nearest_person_ballistic(
        self,
        point: Tuple[float, float],
        radius_px: float = 400.0,
        max_frames_back: int = 25,
    ) -> Optional[int]:
        """
        Ballistic trajectory check — projects each person's historical
        velocity forward and checks if the projected path passes near `point`.
        Handles far throws where proximity alone fails.
        """
        best_pid  = None
        best_dist = float("inf")

        for pid, ph in self._persons.items():
            trail = list(ph.trail)
            if len(trail) < 3:
                continue

            recent = trail[-max_frames_back:] if len(trail) > max_frames_back else trail

            for i in range(len(recent) - 2):
                # Velocity at this historical moment
                vx = recent[i+1][0] - recent[i][0]
                vy = recent[i+1][1] - recent[i][1]
                speed = math.hypot(vx, vy)
                if speed < 1.0:
                    continue

                # Project forward up to max_frames_back steps
                px, py = recent[i]
                for step in range(1, max_frames_back):
                    proj_x = px + vx * step
                    proj_y = py + vy * step
                    d = _dist((proj_x, proj_y), point)
                    if d < best_dist:
                        best_dist = d
                        best_pid  = pid

        if best_dist <= radius_px:
            return best_pid
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_obj(
        self, tid: int, tracked_objs: List[TrackedObject]
    ) -> Optional[TrackedObject]:
        for o in tracked_objs:
            if o.track_id == tid:
                return o
        return None

    def _get_or_create(
        self, pair_id: str, pid: int, tid: int, frame_idx: int
    ) -> _Case:
        if pair_id not in self._cases:
            self._cases[pair_id] = _Case(
                pair_id=pair_id, person_id=pid,
                trash_id=tid, start_frame=frame_idx,
            )
        return self._cases[pair_id]

    def _age_cases(self) -> None:
        for c in self._cases.values():
            if not c.closed:
                c.frames_since_update += 1

    def _purge(self) -> None:
        stale = [
            k for k, c in self._cases.items()
            if not c.closed and c.frames_since_update > cfg.MAX_CASE_AGE_FRAMES
        ]
        for k in stale:
            del self._cases[k]
            self._beliefs.purge(k)