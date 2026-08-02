"""
Layer 5 — Agentic Belief State & Trigger Detection
====================================================

Phase 1 of the Agentic Attention Controller redesign.

This module is intentionally standalone: it does NOT call an LLM and does
NOT modify agent.py / run_pipeline.py. It defines the memory object the
agent will read/write (BeliefState) and the deterministic salience checks
that decide WHEN the agent should be invoked at all (TriggerDetector).

Design note — where the trigger signals actually come from
------------------------------------------------------------
Layer3 (BinInteractionFeatureExtractor) is trash<->bin relative only
(distance_to_bin_center, entry_event_score, etc.) — it has no concept of
person-object coupling.

The coupling / diverging / rest signals this module keys off of live in
Layer5/agent.py's own `_Case` state machine, already computed every frame:

  _Case.coupling_frames   — consecutive frames person+object moved together
  _Case.diverge_frames    — consecutive frames person+object motion decoupled
  _Case.peak_coupling     — highest cosine similarity seen while POSSESSED
  _Case.rest_frames       — consecutive frames the object has been still
  _Case.rest_via_timeout  — object vanished/settled via timeout, not motion
  _Case.obj_trail         — rolling centroid trail (deque) used for velocity
  _Case.state             — _State.WATCHING/POSSESSED/DIVERGING*/RELEASED/RESTING/LOCKED

  (*DIVERGING isn't a distinct _State value today — divergence is tracked
   via diverge_frames while state == POSSESSED, and the transition to
   RELEASED is the edge we treat as "diverging confirmed".)

TriggerDetector below watches these fields for STATE-TRANSITION EDGES
(not raw thresholds re-fired every frame) so the agent is only invoked
when something actually changed — matching the event-triggered design
decision (see conversation): the LLM should stay dormant during steady
walking/holding and only engage on salient transitions.

FIX 8 — evidence, not conclusions
-----------------------------------
Previously, trigger notes stated the heuristic's own conclusion as settled
fact — e.g. rest_onset said "object came to rest", divergence_onset said
"person-object motion decoupled" — phrased as observed reality rather than
a signal crossing that still needs interpreting. A human supervisor watching
the footage wouldn't take "the tracker's counter incremented" as proof the
object left the person's hand; they'd look at the actual position and how
sure the detector was before agreeing.

As of this fix:
  - agent.py now gates obj_trail updates on detection confidence, so these
    triggers fire on genuinely-tracked motion far more often than on bbox
    jitter (see agent.py FIX 8a).
  - the notes below report the raw disputable numbers behind each trigger
    (position, confidence, distance) instead of asserting a conclusion, and
    KinematicSnapshot (used for re-examination replay) now also carries
    obj_confidence so a follow-up LLM call can see whether the positions
    it's replaying were ever trustworthy in the first place.
  - the LLM is expected to do its own judging of what the numbers mean —
    see the strengthened system prompt in llm_controller.py.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
#  Belief state — the agent's persistent memory for one (person, object) pair
# ══════════════════════════════════════════════════════════════════════════

class PipelineState(str, Enum):
    """
    Pipeline lifecycle for a belief — set EXCLUSIVELY by agent.py's own
    code (get_or_create, first LLM call, _finalise()). Never set from the
    LLM's response. This is the fix for the phase-collision bug: the LLM's
    own narrative ("locked", "confirming", "uncertain", whatever words it
    picks) must never be able to write into this field, because doing so
    let the LLM's casual "I'm confident" ("locked") get read by the frame
    loop as "pipeline says this case is closed, stop calling the LLM" —
    which silently discarded every trigger and piece of evidence that
    arrived after that single premature call.
    """
    MONITORING = "monitoring"   # belief exists, no LLM call made yet
    ACTIVE     = "active"       # LLM has been invoked at least once, still open
    FINALIZED  = "finalized"    # _finalise() has run, verdict issued, no more updates


# Kept as an alias so any code/tests still importing `Phase` don't hard-break;
# new code should use PipelineState directly.
Phase = PipelineState


@dataclass
class EvidenceEntry:
    frame_idx: int
    signal:    str     # e.g. "velocity_spike", "divergence_onset", "rest_onset", "person_exit"
    note:      str      # short human/LLM-readable description


@dataclass
class KinematicSnapshot:
    """
    ...
    """
    frame_idx:    int
    obj_pos:      Optional[Tuple[float, float]]
    person_pos:   Optional[Tuple[float, float]]
    obj_speed:    float
    person_speed: float
    distance:     float   # person-object distance this frame
    obj_confidence: Optional[float] = None  # None = object not seen this frame at all
    coupling_score: Optional[float] = None  # cosine similarity of person/object velocity this frame

    def as_line(self) -> str:
        op = f"({self.obj_pos[0]:.0f},{self.obj_pos[1]:.0f})" if self.obj_pos else "?"
        pp = f"({self.person_pos[0]:.0f},{self.person_pos[1]:.0f})" if self.person_pos else "?"
        conf = f"{self.obj_confidence:.2f}" if self.obj_confidence is not None else "n/a (not seen)"
        coup = f"{self.coupling_score:.3f}" if self.coupling_score is not None else "n/a"
        return (
            f"frame {self.frame_idx}: obj_pos={op} obj_conf={conf} "
            f"obj_speed={self.obj_speed:.1f}px/f "
            f"person_pos={pp} person_speed={self.person_speed:.1f}px/f "
            f"distance={self.distance:.0f}px coupling={coup}"
        )

@dataclass
class BeliefState:
    """
    Persistent per-pair memory the agent reads and updates across calls.
    One instance per pair_id ("person_<pid>_trash_<tid>"), mirroring how
    _Case is keyed in agent.py so the two can be joined 1:1 in Phase 2.
    """
    pair_id:    str
    track_id:   str            # alias of pair_id, kept for readability in prompts
    phase:      PipelineState = PipelineState.MONITORING
    confidence: float = 0.0    # running belief, 0.0-1.0 (NOT the final L4/L5 conf)

    # The LLM's own free-form self-assessment label (e.g. "confirming",
    # "locked", "uncertain" — whatever word it picks). Purely narrative:
    # never read by any gating logic, only surfaced back to the LLM itself
    # (Fix 3 — persistence) and into logs/prompts for readability. This is
    # the field that used to be conflated with `phase` above.
    llm_assessment: str = ""

    evidence_log:        List[EvidenceEntry] = field(default_factory=list)
    last_llm_call_frame:  Optional[int]       = None
    re_examined_frames:   List[int]           = field(default_factory=list)
    llm_reasoning:        str                 = ""
    llm_should_flag:      bool                = False
    re_examine_rounds_used: int               = 0

    # Rolling dense history for re-examination — bounded so memory doesn't
    # grow unbounded on long-running pairs. 150 frames at ~25fps is 6s of
    # history, comfortably covering the lookback a re-examine request needs.
    kinematic_history: Deque[KinematicSnapshot] = field(
        default_factory=lambda: deque(maxlen=600)
    )

    def record_snapshot(self, snap: KinematicSnapshot) -> None:
        self.kinematic_history.append(snap)

    def get_snapshots(self, frame_indices: List[int]) -> List[KinematicSnapshot]:
        """Look up stored snapshots matching the requested frame indices."""
        wanted = set(frame_indices)
        return [s for s in self.kinematic_history if s.frame_idx in wanted]

    def add_evidence(self, frame_idx: int, signal: str, note: str) -> None:
        self.evidence_log.append(EvidenceEntry(frame_idx, signal, note))

    def pending_evidence_since(self, frame_idx: int) -> List[EvidenceEntry]:
        """Evidence accumulated since the last LLM call — what gets sent up."""
        since = self.last_llm_call_frame if self.last_llm_call_frame is not None else -1
        return [e for e in self.evidence_log if since < e.frame_idx <= frame_idx]

    def to_prompt_dict(self) -> dict:
        """
        Compact JSON-able snapshot for the LLM prompt. Includes the LLM's
        own previous self-assessment + reasoning (Fix 3) so each call sees
        its own evolving read of the case, not just raw evidence.
        """
        return {
            "pair_id":             self.pair_id,
            "pipeline_state":      self.phase.value,
            "your_previous_assessment": self.llm_assessment or "(none yet — first call)",
            "your_previous_confidence": round(self.confidence, 3),
            "your_previous_reasoning":  self.llm_reasoning or "(none yet)",
            "re_examined_frames": self.re_examined_frames,
            "previously_flagged": self.llm_should_flag,
        }

# ══════════════════════════════════════════════════════════════════════════
#  Belief state manager — dict wrapper, mirrors agent.py's self._cases
# ══════════════════════════════════════════════════════════════════════════

class BeliefStateManager:
    def __init__(self) -> None:
        self._beliefs: Dict[str, BeliefState] = {}

    def get_or_create(self, pair_id: str) -> BeliefState:
        if pair_id not in self._beliefs:
            self._beliefs[pair_id] = BeliefState(pair_id=pair_id, track_id=pair_id)
        return self._beliefs[pair_id]

    def get(self, pair_id: str) -> Optional[BeliefState]:
        return self._beliefs.get(pair_id)

    def all_active(self) -> List[BeliefState]:
        return [b for b in self._beliefs.values() if b.phase != PipelineState.FINALIZED]

    def purge(self, pair_id: str) -> None:
        self._beliefs.pop(pair_id, None)