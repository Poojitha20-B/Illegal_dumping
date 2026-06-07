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
from Layer4.dumping_inference import DumpingEvent
import Layer5.config as cfg


# ══════════════════════════════════════════════════════════════════════════════
#  State machine
# ══════════════════════════════════════════════════════════════════════════════

class _State(Enum):
    WATCHING     = auto()
    POSSESSED    = auto()
    DIVERGING    = auto()
    RELEASED     = auto()
    RESTING      = auto()
    LOCKED       = auto()


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

def _parse_pair_id(pair_id: str) -> Tuple[int, int]:
    parts = pair_id.split("_")
    return int(parts[1]), int(parts[3])

def _parse_held_frames(reason: str) -> int:
    try:
        for token in reason.split():
            if token.startswith("held=") and token.endswith("f"):
                return int(token[5:-1])
    except Exception:
        pass
    return 0


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

    state:   _State = _State.WATCHING
    locked:  bool   = False
    result:  Optional[dict] = None

    coupling_frames:   int   = 0
    coupling_scores:   List[float] = field(default_factory=list)
    diverge_frames:    int   = 0
    diverge_scores:    List[float] = field(default_factory=list)

    obj_trail:         deque = field(default_factory=lambda: deque(maxlen=40))
    person_trail_snap: deque = field(default_factory=lambda: deque(maxlen=40))
    rest_frames:       int   = 0
    post_release_frames: int = 0

    rest_via_timeout:  bool  = False

    obj_missing_frames: int  = 0

    peak_coupling:     float = 0.0
    release_clarity:   float = 0.0
    final_obj_pos:     Optional[Tuple[float, float]] = None

    stored_l4_event:   Optional[DumpingEvent] = None

    reasoning: List[str] = field(default_factory=list)
    frames_since_update: int = 0

    def log(self, msg: str) -> None:
        self.reasoning.append(msg)

    def last_reason(self, n: int = 3) -> str:
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

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx:    int,
        tracked_objs: List[TrackedObject],
        tracked_bins: List[TrackedBin],
        l4_events:    List[DumpingEvent],
    ) -> List[dict]:

        self._update_person_histories(tracked_objs)
        self._update_motion_coupling(tracked_objs, frame_idx)
        self._age_cases()
        self.frame_signals = {}

        new_verdicts: List[dict] = []

        for ev in l4_events:
            pid, tid = _parse_pair_id(ev.pair_id)
            case     = self._get_or_create(ev.pair_id, pid, tid, frame_idx)
            case.frames_since_update = 0
            if ev.event != "pending" and case.stored_l4_event is None:
                case.stored_l4_event = ev
                case.log(f"l4_stored: {ev.event} conf={ev.confidence:.2f}")

        for pair_id, case in list(self._cases.items()):
            if case.locked:
                continue

            ph = self._persons.get(case.person_id)

            if self._is_ghost(case, ph):
                info = (f"frames={ph.frames} move={ph.movement:.0f}px "
                        f"coupling={case.coupling_frames}f") if ph else "unseen"
                continue

            obj = self._find_obj(case.trash_id, tracked_objs)
            if obj is not None:
                pos = _centroid(obj.bbox)
                case.obj_trail.append(pos)
                case.final_obj_pos = pos
                case.obj_missing_frames = 0
                if ph and ph.last_pos:
                    case.person_trail_snap.append(ph.last_pos)
            else:
                case.obj_missing_frames += 1
                if (case.state == _State.POSSESSED
                        and case.obj_missing_frames >= cfg.OFFSCREEN_RELEASE_FRAMES):
                    case.state = _State.RELEASED
                    case.post_release_frames = 0
                    case.rest_via_timeout = True
                    case.log(
                        f"offscreen_release missing={case.obj_missing_frames}f "
                        f"coupling={case.coupling_frames}f"
                    )

            verdict = self._advance(case, tracked_bins, frame_idx, ph)
            if verdict:
                new_verdicts.append(verdict)

            self.frame_signals[pair_id] = (
                f"{case.state.name} | "
                f"coupling={case.coupling_frames}f "
                f"cos={case.peak_coupling:.2f} | "
                + case.last_reason(2)
            )

        self.active_cases = [c for c in self._cases.values() if not c.locked]
        self._purge()
        return new_verdicts

    def get_all_results(self) -> List[dict]:
        return [c.result for c in self._cases.values()
                if c.locked and c.result is not None]

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

            if case.locked or case.state in (_State.RELEASED, _State.RESTING, _State.LOCKED):
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
                if closest_d < 100:
                    case.coupling_frames += 1
                continue

            cos_sim = _cosine_sim(p_vel, o_vel)

            if p_speed > 1e-3 and o_speed > 1e-3:
                ratio = max(p_speed, o_speed) / min(p_speed, o_speed)
                if ratio > cfg.COUPLING_SPEED_RATIO:
                    cos_sim *= 0.3

            if case.state in (_State.WATCHING, _State.POSSESSED):
                if cos_sim >= cfg.COUPLING_COS_THRESH:
                    case.coupling_frames += 1
                    case.coupling_scores.append(cos_sim)
                    case.peak_coupling = max(case.peak_coupling, cos_sim)
                    case.diverge_frames = 0

                    if (case.state == _State.WATCHING
                            and case.coupling_frames >= cfg.MIN_COUPLING_FRAMES):
                        case.state = _State.POSSESSED
                        case.log(
                            f"possessed confirmed coupling={case.coupling_frames}f "
                            f"peak_cos={case.peak_coupling:.2f}"
                        )
                else:
                    if case.state == _State.POSSESSED:
                        case.diverge_frames += 1
                        case.diverge_scores.append(cos_sim)
                        case.release_clarity = cos_sim

                        if case.diverge_frames >= cfg.DIVERGE_CONFIRM_FRAMES:
                            case.state = _State.RELEASED
                            case.post_release_frames = 0
                            case.log(
                                f"L5_release_detected diverge={case.diverge_frames}f "
                                f"cos={cos_sim:.2f}"
                            )

    # ── State machine ─────────────────────────────────────────────────────────

    def _advance(
        self,
        case:         _Case,
        tracked_bins: List[TrackedBin],
        frame_idx:    int,
        ph:           Optional[_PersonHistory],
    ) -> Optional[dict]:

        # WATCHING: not yet confirmed possession
        # Do NOT jump to RELEASED from here — L4 firing early while the bag
        # is still held causes the overlay to show RELEASED prematurely.
        # Instead, wait for L5 coupling to confirm POSSESSED first.
        # WATCHING: not yet confirmed possession.
        # Never jump straight to RELEASED from WATCHING — always go through POSSESSED.
        # Jumping to RELEASED here is what causes the overlay to show "RELEASED"
        # while the bag is still in the person's hand.
        if case.state == _State.WATCHING:
            if case.stored_l4_event and case.stored_l4_event.event != "pending":
                if case.coupling_frames >= cfg.MIN_COUPLING_FRAMES:
                    # L5 confirmed possession AND L4 says released — safe to go to RELEASED
                    case.state = _State.RELEASED
                    case.post_release_frames = 0
                    case.log(f"watching_to_released l4+l5 coupling={case.coupling_frames}f")
                else:
                    # L4 fired but possession not yet confirmed by L5 — advance to POSSESSED
                    # and wait for divergence signal before going to RELEASED
                    case.state = _State.POSSESSED
                    case.log(f"watching_to_possessed l4_early coupling={case.coupling_frames}f")
            return None

        if case.state == _State.POSSESSED:
            if case.stored_l4_event and case.stored_l4_event.event != "pending":
                if case.diverge_frames == 0:
                    case.log(f"l4_release_backup coupling={case.coupling_frames}f")
                    case.state = _State.RELEASED
                    case.post_release_frames = 0
                # Fast path: if L4 has a firm verdict AND L5 has confirmed
                # possession, skip RELEASED+RESTING and finalise immediately.
                # This prevents the verdict from arriving after the video ends.
                if (case.coupling_frames >= cfg.MIN_COUPLING_FRAMES
                        and case.stored_l4_event.confidence >= 0.45):
                    case.state = _State.RESTING
                    case.rest_via_timeout = True
                    case.log(
                        f"l4_fast_path conf={case.stored_l4_event.confidence:.2f} "
                        f"coupling={case.coupling_frames}f"
                    )
                    return self._finalise(case, tracked_bins, frame_idx, ph)
            return None

        if case.state == _State.RELEASED:
            case.post_release_frames += 1

            # Fast path: L4 has a firm verdict and object has been
            # released — no need to wait for full rest confirmation.
            if (case.stored_l4_event is not None
                    and case.stored_l4_event.event != "pending"
                    and case.stored_l4_event.confidence >= 0.45
                    and case.post_release_frames >= 3):
                case.state = _State.RESTING
                case.rest_via_timeout = True
                case.log(
                    f"l4_release_fast_path frames={case.post_release_frames} "
                    f"conf={case.stored_l4_event.confidence:.2f}"
                )
                return self._finalise(case, tracked_bins, frame_idx, ph)

            if len(case.obj_trail) >= 3:
                o_vel  = _vel(case.obj_trail, n=3)
                o_spd  = _speed(o_vel)

                if o_spd < cfg.REST_VEL_PX:
                    case.rest_frames += 1
                else:
                    case.rest_frames = 0

                if case.rest_frames >= cfg.REST_CONFIRM_FRAMES:
                    case.state = _State.RESTING
                    case.log(f"object_at_rest vel={o_spd:.1f}px")
                    return self._finalise(case, tracked_bins, frame_idx, ph)

            # Also count missing frames as rest — bag on ground may flicker
            if case.obj_missing_frames >= 2:
                case.rest_frames += 1
                if case.rest_frames >= cfg.REST_CONFIRM_FRAMES:
                    case.state = _State.RESTING
                    case.rest_via_timeout = True
                    case.log(f"rest_via_missing_frames missing={case.obj_missing_frames}f")
                    return self._finalise(case, tracked_bins, frame_idx, ph)

            if case.post_release_frames >= cfg.REST_MAX_WAIT:
                case.state = _State.RESTING
                case.rest_via_timeout = True
                case.log(f"rest_timeout after {cfg.REST_MAX_WAIT}f")
                return self._finalise(case, tracked_bins, frame_idx, ph)

            return None

    # ── FIX 7: Bin-entry detection ────────────────────────────────────────────

    def _check_bin_entry(
        self,
        case:         _Case,
        tracked_bins: List[TrackedBin],
        ph:           Optional[_PersonHistory],
    ) -> Tuple[bool, str]:
        """
        Detects the "object entered bin" scenario that Layer 4 cannot handle.

        Returns (override_to_legal: bool, reason_string: str).

        The pattern we look for:
          1. Object disappeared suddenly (rest_via_timeout=True, rest_frames==0)
             → object never settled on ground; it vanished mid-air or on impact
          2. A bin is present in the scene
          3. Strong possession confirmed (coupling >= MIN_COUPLING_FRAMES,
             peak_cos >= BIN_ENTRY_MIN_PEAK_COS)
             → the object was genuinely being carried, not incidentally nearby
          4. Person was approaching the bin OR person was near the bin
             when the object disappeared
             → trajectory corroborates intentional disposal into bin

        Any combination where 1+2+3 are true and 4 is partially true will
        trigger the override.  The threshold for (4) is loosened when the
        person was spatially close to the bin (BIN_PERSON_RADIUS_PX).
        """
        # Condition 1: object vanished (timeout with zero natural rest frames)
        obj_vanished = case.rest_via_timeout and case.rest_frames == 0

        if not obj_vanished:
            return False, ""

        # Condition 2: bin present
        if not tracked_bins:
            return False, ""

        # Condition 3: strong coupling (object was genuinely being carried)
        strong_possession = (
            case.coupling_frames >= cfg.MIN_COUPLING_FRAMES
            and case.peak_coupling >= cfg.BIN_ENTRY_MIN_PEAK_COS
        )
        if not strong_possession:
            return False, ""

        # Condition 4: person trajectory toward bin
        person_approach, approach_bin_id = (
            ph.bin_approach_score(tracked_bins) if ph else (0.0, None)
        )

        # Secondary: was person near bin when object vanished?
        person_near_bin = False
        person_bin_dist = float("inf")
        if ph:
            person_bin_dist = ph.nearest_bin_dist(tracked_bins)
            person_near_bin = person_bin_dist <= cfg.BIN_PERSON_RADIUS_PX

        # Determine effective approach threshold
        effective_thresh = (
            cfg.BIN_APPROACH_CORROBORATED if person_near_bin else cfg.BIN_APPROACH_THRESH
        )

        if person_approach < effective_thresh:
            # Not enough trajectory evidence — don't override
            return False, ""

        # All signals converge → object entered bin
        reason = (
            f"l5_bin_entry_override: "
            f"obj_vanished=True "
            f"coupling={case.coupling_frames}f "
            f"peak_cos={case.peak_coupling:.2f} "
            f"person_approach={person_approach:.2f} "
            f"person_bin_dist={person_bin_dist:.0f}px "
            f"bin#{approach_bin_id}"
        )
        return True, reason

    # ── Finalise verdict ──────────────────────────────────────────────────────

    def _finalise(self, case, tracked_bins, frame_idx, ph) -> dict:

        ev           = case.stored_l4_event
        l4_verdict   = ev.event if ev else None
        bins_present = len(tracked_bins) > 0

        l5_confirmed_possession = case.coupling_frames >= cfg.MIN_COUPLING_FRAMES
        l5_confirmed_release    = (case.diverge_frames >= cfg.DIVERGE_CONFIRM_FRAMES
                                   or case.rest_via_timeout
                                   or case.state == _State.RESTING)

        if l4_verdict == "illegal_dumping":
            is_violation = True
        elif l4_verdict == "legal_disposal":
            is_violation = False
        elif l5_confirmed_possession and l5_confirmed_release and not bins_present:
            is_violation = True
        elif l5_confirmed_possession and l5_confirmed_release and bins_present:
            is_violation = True
        else:
            is_violation = False

        # Strip L4 bin label — L5 will report the correct bin from its own spatial check
        import re
        if ev:
            if ph and ph.last_pos and tracked_bins:
                ref_bbox = np.array([ph.last_pos[0]-5, ph.last_pos[1]-5,
                                     ph.last_pos[0]+5, ph.last_pos[1]+5])
                _, correct_bin_obj = _best_bin_hierarchical(ref_bbox, ph.last_pos, tracked_bins)
                correct_bin_id = correct_bin_obj.bin_id if correct_bin_obj else None
                reasons = [re.sub(r'bin#\d+', f'bin#{correct_bin_id}', ev.reason)]
            else:
                reasons = [ev.reason]
        else:
            reasons = ["l5_independent_detection"]
        # ── FIX 7: Bin-entry override (runs BEFORE other spatial checks) ──────
        # This specifically handles the case where L4 said VIOLATION because
        # the object's final tracked position was far from the bin — but the
        # object actually entered the bin (tracker lost it on entry).
        bin_entry_legal, bin_entry_reason = self._check_bin_entry(
            case, tracked_bins, ph
        )
        if bin_entry_legal:
            is_violation = False
            reasons.append(bin_entry_reason)
            case.log(bin_entry_reason)

        # ── Signal 1: Multi-bin spatial check ────────────────────────────────
        # Only run if bin-entry override did NOT already flip to legal.
        # (If it did, we trust the bin-entry logic over raw distance.)
        final_pos = case.final_obj_pos
        # Use person's last position instead of trash's last position
        # — when bag enters bin it disappears near person's hand, not near bin center
        ref_pos = ph.last_pos if (ph and ph.last_pos) else final_pos
        if ref_pos and tracked_bins and not bin_entry_legal:
            # Use hierarchical containment check with a dummy trash bbox
            # centered on ref_pos when actual trash bbox is unavailable
            ref_bbox = np.array([ref_pos[0]-5, ref_pos[1]-5,
                                 ref_pos[0]+5, ref_pos[1]+5])
            best_d, best_bin_obj = _best_bin_hierarchical(ref_bbox, ref_pos, tracked_bins)
            best_bin_id = best_bin_obj.bin_id if best_bin_obj else None
            if best_d <= cfg.BIN_LEGAL_RADIUS_PX:
                is_violation = False
                reasons.append(f"L5_bin_near dist={best_d:.0f}px bin#{best_bin_id}")
                case.log(f"bin_override {best_d:.0f}px")

        # ── Signal 2: Two-signal trajectory intent ────────────────────────────
        person_approach, approach_bin_id = (
            ph.bin_approach_score(tracked_bins) if ph else (0.0, None)
        )

        obj_approach = 0.0
        if tracked_bins and len(case.obj_trail) >= 4:
            trail      = list(case.obj_trail)
            target_bin = min(tracked_bins, key=lambda b: _dist(trail[-1], _bottom_center(b.bbox)))
            bin_pos    = _bottom_center(target_bin.bbox)
            converge   = sum(
                1 for i in range(max(0, len(trail)-10), len(trail)-1)
                if _dist(trail[i+1], bin_pos) < _dist(trail[i], bin_pos)
            )
            obj_approach = converge / max(min(10, len(trail)-1), 1)

        intent_score = (
            cfg.TRAJ_PERSON_WEIGHT * person_approach +
            cfg.TRAJ_OBJECT_WEIGHT * obj_approach
        )

        # FIX 3: Only allow intent to override to LEGAL when bins exist
        if bins_present and intent_score >= cfg.TRAJ_LEGAL_THRESH and is_violation:
            is_violation = False
            reasons.append(
                f"L5_traj_intent person={person_approach:.2f} "
                f"obj={obj_approach:.2f} combined={intent_score:.2f}"
            )
            case.log(f"traj_override intent={intent_score:.2f}")
        else:
            case.log(f"traj_intent={intent_score:.2f} bins={bins_present}")

        # ── Signal 3: Evidence-weighted confidence ────────────────────────────
        avg_coupling = (
            sum(case.coupling_scores) / len(case.coupling_scores)
            if case.coupling_scores else 0.0
        )
        coupling_conf = min(avg_coupling, 1.0)

        diverge_conf = 1.0 - max(case.release_clarity, 0.0)

        # FIX 5: rest_timeout path gets neutral rest_conf (0.5) not 0.0
        if case.rest_via_timeout:
            rest_conf = 0.5
        else:
            rest_conf = min(case.rest_frames / max(cfg.REST_CONFIRM_FRAMES, 1), 1.0)

        if ref_pos and tracked_bins:
            ref_bbox = np.array([ref_pos[0]-5, ref_pos[1]-5,
                                 ref_pos[0]+5, ref_pos[1]+5])
            bin_d, _ = _best_bin_hierarchical(ref_bbox, ref_pos, tracked_bins)
            bin_d = bin_d if bin_d < float("inf") else float("inf")
        else:
            bin_d = float("inf")
        bin_prox  = max(0.0, 1.0 - bin_d / 500.0) if bin_d < float("inf") else 0.0

        l4_conf = ev.confidence if ev else 0.5

        evidence_conf = (
            cfg.CONF_COUPLING_W * coupling_conf +
            cfg.CONF_DIVERGE_W  * diverge_conf  +
            cfg.CONF_REST_W     * rest_conf      +
            cfg.CONF_BIN_PROX_W * bin_prox
        )
        final_conf = round(0.50 * l4_conf + 0.50 * evidence_conf, 3)

        if final_conf < cfg.MIN_CONFIDENCE_TO_ACT and is_violation:
            is_violation = False
            reasons.append(f"L5_low_evidence conf={final_conf:.2f}")
            case.log("low_evidence_blocked")

        # FIX 4: Suppress no_coupling penalty when L4 independently confirms violation
        l4_confirms_violation = (l4_verdict == "illegal_dumping")
        if (case.coupling_frames < cfg.MIN_COUPLING_FRAMES
                and is_violation
                and not l4_confirms_violation):
            final_conf = max(0.0, final_conf - 0.15)
            reasons.append(f"L5_no_coupling coupling={case.coupling_frames}f")
            case.log("no_coupling_penalty")
        elif case.coupling_frames < cfg.MIN_COUPLING_FRAMES and is_violation:
            reasons.append(f"L5_weak_coupling coupling={case.coupling_frames}f (l4_confirmed)")
            case.log("weak_coupling_noted_l4_confirmed")

        result = {
            "violation":       is_violation,
            "confidence":      round(final_conf, 3),
            "event":           "illegal_dumping" if is_violation else "legal_disposal",
            "person_id":       case.person_id,
            "object_id":       case.trash_id,
            "pair_id":         case.pair_id,
            "reason":          " | ".join(reasons),
            "coupling_frames": case.coupling_frames,
            "peak_coupling":   round(case.peak_coupling, 2),
            "release_clarity": round(1.0 - case.release_clarity, 2),
            "rest_frames":     case.rest_frames,
            "person_approach": round(person_approach, 2),
            "obj_approach":    round(obj_approach, 2),
            "intent_score":    round(intent_score, 2),
            "l4_held":         _parse_held_frames(ev.reason) if ev else 0,
            "frames":          [case.start_frame, frame_idx],
            "reasoning_log":   list(case.reasoning),
        }
        # ── Guard: trash object already claimed by a prior confirmed violation ──
        if is_violation and case.trash_id in self._used_trash_ids:
            is_violation = False
            reasons.append(f"L5_trash_already_claimed T{case.trash_id}")
            case.log("trash_claimed_by_prior_violation")
            final_conf = max(0.0, final_conf - 0.20)

        if is_violation:
            self._used_trash_ids.add(case.trash_id)   # claim it

        case.result = result
        case.locked = True
        case.state  = _State.LOCKED

        tag = "🚨 VIOLATION" if is_violation else "✅ LEGAL"
        print(
            f"[Layer5] {tag} | {result['event']} | conf={final_conf:.2f} | "
            f"P{case.person_id} T{case.trash_id} | "
            f"coupling={case.coupling_frames}f cos={case.peak_coupling:.2f} | "
            f"intent={intent_score:.2f} rest={case.rest_frames}f | "
            f"frames={result['frames']}"
        )
        print(f"         evidence: coupling={coupling_conf:.2f} "
              f"diverge={diverge_conf:.2f} rest={rest_conf:.2f} bin={bin_prox:.2f}")
        print(f"         reasons:  {result['reason']}")
        return result

    # ── Ghost filter ──────────────────────────────────────────────────────────

    def _is_ghost(self, case: _Case, ph: Optional[_PersonHistory]) -> bool:
        if ph is None:
            return True
        if ph.frames < cfg.GHOST_MIN_FRAMES or ph.movement < cfg.GHOST_MIN_MOVEMENT:
            if case.coupling_frames >= cfg.MIN_COUPLING_FRAMES:
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
            if not c.locked:
                c.frames_since_update += 1

    def _purge(self) -> None:
        stale = [
            k for k, c in self._cases.items()
            if not c.locked and (
                c.frames_since_update > cfg.MAX_CASE_AGE_FRAMES
                or (
                    # Kill unconfirmed cases quickly if the object has disappeared —
                    # these are stale/phantom pairs (e.g. id2 that never really existed)
                    c.frames_since_update > 10
                    and c.state == _State.WATCHING
                    and c.coupling_frames == 0
                )
            )
        ]
        for k in stale:
            del self._cases[k]