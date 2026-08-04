"""
Layer 5 — LLM Belief Controller
=================================

Phase 3 of the Agentic Attention Controller redesign.

Takes a BeliefState + newly-triggered evidence, asks an LLM to update the
belief (confidence, phase, reasoning), optionally request re-examination
of specific buffered frames, or flag the case for a challan.

This module is deliberately decoupled from agent.py's frame loop: it takes
plain data in (BeliefState.to_prompt_dict() + evidence dicts) and returns
plain data out (a validated dict). Wiring it into DumpingAgent.update() so
it actually fires on trigger events is Phase 4 — not done here.

Provider: Groq (Llama), matching CausalMed / AI Legal Assistant. Swap
`_call_groq()` for a different provider if needed — everything else
(prompt building, response validation, retry loop) is provider-agnostic.

FIX 8 — system prompt now explicitly tells the model that evidence notes
and kinematic snapshots describe raw signal crossings (position, detector
confidence, distance), not confirmed events. This mirrors the upstream
change in agent.py (confidence gate on obj_trail updates) and belief_state.py
(trigger notes rewritten to report numbers instead of conclusions like
"object came to rest"). Without this instruction, a model that's simply
told "rest_onset fired" tends to treat the trigger name itself as the
verdict; the added paragraph pushes it to actually read the numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import Layer5.config as cfg
from Layer5.belief_state import BeliefState, EvidenceEntry, KinematicSnapshot

# Load GROQ_API_KEY (and anything else) from a .env file in the repo root,
# if present. Falls back silently to whatever's already in the shell env —
# so this doesn't break if you're exporting the key manually or running in
# Colab where you set os.environ directly in a cell.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════
#  Prompt construction
# ══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an evidence-evaluating agent monitoring a surveillance feed for \
illegal dumping. You are NOT making a final violation decision — you are \
maintaining a running belief state for one person-object pair across a video.

You will be given your own previous assessment and confidence (if any), plus \
new evidence collected since your last call (motion/trajectory signals, not \
raw pixels). Re-evaluate in light of the new evidence — your confidence may \
increase, decrease, or stay the same, and your assessment may change freely. \
Do not treat your own previous read as final; you will keep being called as \
more evidence arrives, so there is no need to declare certainty prematurely. \
If a trajectory is genuinely ambiguous (e.g. a velocity spike without a clear \
divergence or rest follow-up), you may request re-examination of specific \
earlier frames — but only when you have a real reason to doubt your current \
read, not by default.

IMPORTANT — evidence vs. conclusions: every evidence line you receive (e.g. \
"divergence_onset", "rest_onset", "velocity_spike") names a raw SIGNAL \
CROSSING detected by simple thresholds on tracked position/velocity — it is \
NOT a confirmed observation that the object was released, dumped, or came to \
rest. Treat the trigger's name as a prompt to look closer, not as a fact. In \
particular:
  - "rest_onset" with via_timeout=True means the object simply stopped being \
    tracked reliably (went off-screen, occluded, or confidence dropped) — it \
    does NOT mean anyone observed the object actually settle on the ground. \
    Do not conclude "released" or "dumped" purely from a timeout.
  - Every evidence line includes obj_conf (the detector's confidence for that \
    frame's position reading). Low obj_conf (below ~0.5) means the position \
    numbers themselves are unreliable — flickering low-confidence detections \
    (e.g. on clothing or shadows) can produce fake motion that looks like a \
    real release or velocity spike. Weight low-confidence evidence accordingly.
  - You must see the object separate from the person, then genuinely come to \
    rest, before concluding a release happened. A single ambiguous signal is \
    grounds for requesting re-examination, not for flagging.
  - When you request re-examination, the denser kinematic replay you get back \
    also includes obj_conf per frame for the same reason — use it to judge \
    whether the replayed positions were ever trustworthy.

Respond with ONLY a JSON object, no other text, matching this schema exactly:
{
  "updated_belief": {"assessment": "<a short free-form label for your own current read, e.g. watching / suspicious / confirming / confident>", "confidence": <float 0.0-1.0>},
  "re_examine_request": [<frame indices as integers>] or [],
  "should_flag": true|false,
  "new_reasoning": "<one or two sentences explaining your interpretation of the new evidence>"
}

Note: "assessment" is just your own label for your current thinking — it has \
no fixed vocabulary and does not end the monitoring. Only "should_flag" and \
your confidence matter for the actual verdict."""


def _format_evidence(entries: List[EvidenceEntry]) -> str:
    if not entries:
        return "(none)"
    lines = [f"- Frame {e.frame_idx}: {e.signal} — {e.note}" for e in entries]
    return "\n".join(lines)


def _format_snapshots(snapshots: List[KinematicSnapshot]) -> str:
    if not snapshots:
        return "(none — requested frames not found in kinematic history, likely aged out)"
    ordered = sorted(snapshots, key=lambda s: s.frame_idx)
    return "\n".join(f"- {s.as_line()}" for s in ordered)


def build_prompt(
    belief: BeliefState,
    new_evidence: List[EvidenceEntry],
    re_examined_evidence: Optional[List[KinematicSnapshot]] = None,
) -> str:
    """
    Build the user-turn prompt for one belief-update call.

    `new_evidence` — evidence accumulated since belief.last_llm_call_frame
                      (use belief.pending_evidence_since(frame_idx)).
    `re_examined_evidence` — if this call is a follow-up after the agent
                      requested re-examination, the dense per-frame
                      kinematic snapshots for those specific frames
                      (use belief.get_snapshots(frame_indices)).
    """
    parts = [
        f"CURRENT BELIEF:\n{json.dumps(belief.to_prompt_dict(), indent=2)}",
        f"\nNEW EVIDENCE (since frame {belief.last_llm_call_frame}):\n"
        f"{_format_evidence(new_evidence)}",
    ]
    if re_examined_evidence:
        parts.append(
            f"\nRE-EXAMINED EVIDENCE (dense per-frame kinematics for the "
            f"frames you asked to look at more closely):\n"
            f"{_format_snapshots(re_examined_evidence)}"
        )
    parts.append(
        "\nUpdate your belief and respond with the JSON schema described above."
    )
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  Response validation
# ══════════════════════════════════════════════════════════════════════════

class LLMResponseError(ValueError):
    """Raised when the LLM's response doesn't match the expected schema."""


@dataclass
class AgentDecision:
    assessment:          str    # LLM's own free-form label — narrative only, never gates control flow
    confidence:           float
    re_examine_request:   List[int]
    should_flag:          bool
    new_reasoning:        str


def parse_response(raw_text: str) -> AgentDecision:
    """
    Parse + validate the LLM's raw text into an AgentDecision.
    Raises LLMResponseError on any schema violation — caller is
    responsible for retrying (see call_agent's retry loop).
    """
    text = raw_text.strip()
    # Llama models sometimes wrap JSON in ```json fences despite instructions —
    # strip them defensively (same pattern used in the Legal Assistant project).
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise LLMResponseError("response is not a JSON object")

    for key in ("updated_belief", "re_examine_request", "should_flag", "new_reasoning"):
        if key not in data:
            raise LLMResponseError(f"missing required key: {key}")

    ub = data["updated_belief"]
    if not isinstance(ub, dict) or "assessment" not in ub or "confidence" not in ub:
        raise LLMResponseError("updated_belief must have 'assessment' and 'confidence'")

    assessment = ub["assessment"]
    if not isinstance(assessment, str) or not assessment.strip():
        raise LLMResponseError("assessment must be a non-empty string")
    # Deliberately NOT validated against a fixed vocabulary — this field is
    # narrative only (Fix 1). It never controls whether the LLM gets called
    # again; that's PipelineState, which only agent.py's own code sets.

    try:
        confidence = float(ub["confidence"])
    except (TypeError, ValueError) as e:
        raise LLMResponseError(f"confidence not a number: {ub['confidence']}") from e
    if not (0.0 <= confidence <= 1.0):
        raise LLMResponseError(f"confidence out of range [0,1]: {confidence}")

    re_examine = data["re_examine_request"]
    if not isinstance(re_examine, list) or not all(isinstance(i, int) for i in re_examine):
        raise LLMResponseError("re_examine_request must be a list of ints")

    should_flag = data["should_flag"]
    if not isinstance(should_flag, bool):
        raise LLMResponseError("should_flag must be a boolean")

    reasoning = data["new_reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise LLMResponseError("new_reasoning must be a non-empty string")

    return AgentDecision(
        assessment=assessment.strip(),
        confidence=confidence,
        re_examine_request=re_examine,
        should_flag=should_flag,
        new_reasoning=reasoning,
    )
# ══════════════════════════════════════════════════════════════════════════
#  Final determination — called once from agent.py's _finalise(), bypasses
#  debounce, presents the complete heuristic evidence brief for a verdict.
# ══════════════════════════════════════════════════════════════════════════

_FINAL_SYSTEM_PROMPT = """You are an evidence-evaluating agent making a FINAL determination about \
whether illegal dumping occurred in a surveillance video clip.

You will receive:
1. A complete kinematic timeline — per-frame data for every frame from when the \
   person-object pair was first detected until the monitoring window closed.
2. Summary statistics computed from that timeline.
3. Bin context — whether bins were present, their positions, and (when bins are \
   present) a calibrated 'near bin' threshold in pixels, specific to this video's \
   scene geometry. This threshold tells you what pixel distance counts as "within \
   reach of a bin" here — use it to interpret the nearest-bin-distance number \
   rather than guessing from the raw pixel value alone. It is a reference scale, \
   not a decision rule: distances above ~2x the threshold are clearly out of \
   reach, distances near or below it mean the person could reasonably interact \
   with the bin, and values in between are a grey zone where you should weigh \
   approach direction and coupling pattern more heavily.

The timeline includes these values per frame:
  - coupling: cosine similarity between person and object velocity vectors.
    HIGH (>0.7): person and object moving together (carrying/holding).
    LOW (<0.3): person and object moving independently or diverging.
  - distance: person-object centroid distance in pixels.
  - obj_spd: object velocity in pixels/frame.
  - person_spd: person velocity in pixels/frame.
  - obj_conf: detector confidence for the object detection (0-1).
    LOW (<0.5): detection is unreliable, position may be noisy.

HOW TO INTERPRET THE TIMELINE:

A GENUINE DUMPING sequence looks like:
  1. HIGH coupling (0.7+) for many frames — person carrying object
  2. Coupling DROPS sharply and STAYS low — object released
  3. Distance INCREASES — person moves away from object
  4. Object speed drops to near zero while coupling stays low — object settled
  5. Person continues moving or exits frame

A STATIC DUMPING sequence (placing an object down, not throwing it) looks
DIFFERENT from the above — do not expect a sharp coupling drop here:
  1. HIGH coupling (0.7+) while the person carries the object toward a spot
  2. The person STOPS moving (person speed drops near zero); coupling MAY
     stay moderate-to-high right up to the moment of release, because
     placing an object down is a slow, deliberate motion, not a sudden
     divergence in velocity the way throwing is
  3. The object then DISAPPEARS from tracking (obj_conf drops, or the
     object is no longer detected at all) while the person is stationary
     at that spot — small stationary objects often fall below detection
     confidence once set down and no longer moving
  4. The person RESUMES moving and leaves the area
  5. The object never reappears anywhere in the remaining timeline

  Do not require a clean coupling drop before concluding static dumping.
  The key pattern is: reliable coupling during approach, object vanishing
  while the person is stationary, then the person leaving without the
  object ever reappearing.

A PERSON JUST STANDING/PAUSING looks like:
  1. HIGH coupling throughout
  2. Object speed drops to near zero (person stopped walking)
  3. BUT coupling REMAINS HIGH — person is still holding the object

CRITICAL: "Object speed near zero" only means dumping if coupling is ALSO low.
If coupling is high when speed drops, the person simply stopped walking while \
still holding the object. That is NOT dumping.

DETECTION NOISE looks like:
  - Brief coupling drops that immediately recover
  - Accompanied by LOW obj_conf (<0.5)
  - No sustained pattern

IMPORTANT — how to weigh bin absence: bins_present=False does NOT mean there is \
insufficient evidence to judge — it means there was no legal disposal receptacle \
available in the scene at all. If the object was demonstrably carried (strong, \
sustained coupling with the person) and then disappeared or was left behind while \
the person moved away or the object went untracked, the absence of a bin is \
corroborating evidence of illegal dumping, not a reason for leniency. Likewise, if \
a bin WAS present but the object's final position and the person's trajectory do \
not converge on it, that is evidence the object was discarded near — but not into \
— the bin, which still counts as illegal dumping. Do not default to "legal" simply \
because no bin was visible, or because a bin existed somewhere in the scene; weigh \
the coupling history, the person's continued presence, the object's disappearance, \
and where it ended up relative to any bin as the primary evidence.

Evaluate the COMPLETE temporal pattern and deliver your final verdict.

Respond with ONLY a JSON object:
{
  "should_flag": true|false,
  "confidence": <float 0.0-1.0>,
  "new_reasoning": "<your detailed explanation, referencing specific frame ranges and signal values>"
}"""

def _format_final_briefing(briefing: Dict[str, Any]) -> str:
    bin_ctx = briefing.get("bin_context") or {}

    bin_lines = [f"- Bins present: {bin_ctx.get('bins_present')}"]

    if bin_ctx.get("bins_present") and bin_ctx.get("nearest_bin_distance_px") is not None:
        bin_lines.append(
            f"- Nearest bin distance: {bin_ctx.get('nearest_bin_distance_px')}px "
            f"(bin id: {bin_ctx.get('nearest_bin_id')})"
        )
        bin_lines.append(
            f"- Person approach score toward bin: {bin_ctx.get('person_approach_score')}"
        )
        threshold = bin_ctx.get("calibrated_near_bin_threshold_px")
        meaning = bin_ctx.get("near_threshold_meaning", "")
        if threshold is not None:
            bin_lines.append(f"- Calibrated 'near bin' threshold: {threshold}px")
            if meaning:
                bin_lines.append(f"  Meaning: {meaning}")

    return (
        f"- Coupling frames observed: {briefing.get('coupling_frames_observed')} "
        f"(peak cosine similarity: {briefing.get('peak_coupling')}, "
        f"avg: {briefing.get('avg_coupling')})\n"
        f"- Sustained coupling drop (was strong, now weak): {briefing.get('sustained_coupling_drop')}\n"
        f"- Object missing frames: {briefing.get('object_missing_frames')}\n"
        f"- Person gone frames: {briefing.get('person_gone_frames')}\n"
        f"- Object final position: {briefing.get('object_final_position')}, "
        f"confidence: {briefing.get('object_final_confidence')}\n"
        + "\n".join(bin_lines)
    )
def _format_timeline(snapshots: List[KinematicSnapshot]) -> str:
    """Format the full kinematic timeline as a compact table."""
    if not snapshots:
        return "(no kinematic data)"

    ordered = sorted(snapshots, key=lambda s: s.frame_idx)

    # Sample if >100 frames to keep prompt size reasonable
    if len(ordered) > 100:
        step = len(ordered) // 100
        sampled = ordered[::step]
        header = f"(sampled {len(sampled)} frames from {len(ordered)} total, every {step}th frame)\n"
    else:
        sampled = ordered
        header = f"({len(sampled)} frames)\n"

    lines = [header]
    lines.append("frame  coupling  distance  obj_spd  person_spd  obj_conf")
    lines.append("-" * 65)

    for s in sampled:
        coup = f"{s.coupling_score:.2f}" if s.coupling_score is not None else "  n/a"
        dist = f"{s.distance:.0f}px" if s.distance and s.distance < float("inf") else "n/a"
        ospd = f"{s.obj_speed:.1f}"
        pspd = f"{s.person_speed:.1f}"
        conf = f"{s.obj_confidence:.2f}" if s.obj_confidence is not None else "n/a"
        lines.append(f"{s.frame_idx:5d}  {coup:>8}  {dist:>8}  {ospd:>7}  {pspd:>10}  {conf:>8}")

    return "\n".join(lines)

def build_final_prompt(
    belief: BeliefState,
    final_briefing: Dict[str, Any],
    kinematic_timeline: List[KinematicSnapshot],
) -> str:
    """
    Build the user-turn prompt for the one-shot final determination call,
    made from _finalise_with_llm() once the monitoring window has closed.
    Bypasses debounce/triggers entirely — this is a separate call path from
    build_prompt()/call_agent(). Includes the full per-frame kinematic
    timeline, not just summary stats, so the model can actually see the
    temporal pattern (coupling high → drop → object settled) rather than
    inferring it from aggregates alone.
    """
    timeline_str = _format_timeline(kinematic_timeline)
    return (
        "FINAL DETERMINATION — EVIDENCE GATHERING COMPLETE\n"
        "=================================================\n\n"
        f"YOUR PREVIOUS ASSESSMENT:\n{belief.llm_assessment or '(none — LLM never gave an intermediate read)'}\n"
        f"Your previous confidence: {round(belief.confidence, 3)}\n\n"
        f"SUMMARY STATISTICS:\n"
        f"{_format_final_briefing(final_briefing)}\n\n"
        f"COMPLETE KINEMATIC TIMELINE:\n"
        f"{timeline_str}\n\n"
        "You MUST now deliver your final verdict. There will be no further evidence. "
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "should_flag": true|false,\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "new_reasoning": "<your final explanation, referencing specific frame ranges and signal values>"\n'
        "}"
    )


def parse_final_response(raw_text: str) -> AgentDecision:
    """
    Parse + validate the final-call response. Schema is a strict subset of
    the intermediate one (no assessment/re_examine_request) — validated
    separately rather than reusing parse_response() so a malformed
    intermediate-style response doesn't accidentally pass here or vice versa.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise LLMResponseError("response is not a JSON object")

    for key in ("should_flag", "confidence", "new_reasoning"):
        if key not in data:
            raise LLMResponseError(f"missing required key: {key}")

    should_flag = data["should_flag"]
    if not isinstance(should_flag, bool):
        raise LLMResponseError("should_flag must be a boolean")

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as e:
        raise LLMResponseError(f"confidence not a number: {data['confidence']}") from e
    if not (0.0 <= confidence <= 1.0):
        raise LLMResponseError(f"confidence out of range [0,1]: {confidence}")

    reasoning = data["new_reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise LLMResponseError("new_reasoning must be a non-empty string")

    return AgentDecision(
        assessment=belief_assessment_placeholder(),
        confidence=confidence,
        re_examine_request=[],
        should_flag=should_flag,
        new_reasoning=reasoning,
    )


def belief_assessment_placeholder() -> str:
    # The final call doesn't produce a new narrative "assessment" label —
    # it's a verdict, not an intermediate belief update. Keep AgentDecision's
    # shape uniform (agent.py only reads .should_flag/.confidence/.new_reasoning
    # from this path) without inventing a fake label.
    return "final_verdict"


def call_agent_final(
    belief: BeliefState,
    final_briefing: Dict[str, Any],
    kinematic_timeline: List[KinematicSnapshot],
    frame_idx: Optional[int] = None,
    _provider: Optional[Any] = None,
) -> AgentDecision:
    """
    One-shot final determination call, made from agent.py's
    _finalise_with_llm() once the monitoring window has closed. Bypasses
    LLM_CALL_DEBOUNCE_FRAMES entirely — this is not routed through
    TriggerDetector, so debounce logic never applies to it. Retries on
    malformed JSON same as call_agent().

    `kinematic_timeline` — the full per-frame snapshot history for this
    case (belief.kinematic_history), so the model can see the actual
    temporal pattern rather than only summary statistics.

    Does NOT mutate `belief` beyond what the caller does with the returned
    AgentDecision — same contract as call_agent().
    """
    provider = _provider or _call_groq
    prompt = build_final_prompt(belief, final_briefing, kinematic_timeline)

    last_error: Optional[Exception] = None
    for attempt in range(cfg.LLM_MAX_RETRIES + 1):
        try:
            raw = provider(_FINAL_SYSTEM_PROMPT, prompt)
            decision = parse_final_response(raw)
            if frame_idx is not None:
                belief.last_llm_call_frame = frame_idx
            return decision
        except (LLMResponseError, Exception) as e:  # noqa: BLE001 — broad on purpose, retried
            last_error = e
            continue

    raise LLMResponseError(
        f"LLM final call failed after {cfg.LLM_MAX_RETRIES + 1} attempts: {last_error}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  Provider call — Groq / Llama
# ══════════════════════════════════════════════════════════════════════════

def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """
    Raw call to Groq's chat completion endpoint. Reads GROQ_API_KEY from env.
    Kept isolated so swapping providers means editing only this function.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set — export it before running the pipeline "
            "with the LLM agent enabled."
        )
    try:
        from groq import Groq
    except ImportError as e:
        raise RuntimeError(
            "groq package not installed — `pip install groq`"
        ) from e

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,   # low — this is evidence evaluation, not creative writing
        max_tokens=400,
    )
    return resp.choices[0].message.content


def call_agent(
    belief: BeliefState,
    new_evidence: List[EvidenceEntry],
    re_examined_evidence: Optional[List[KinematicSnapshot]] = None,
    frame_idx: Optional[int] = None,
    _provider: Optional[Any] = None,
) -> AgentDecision:
    """
    Full belief-update call: build prompt -> call LLM -> parse/validate
    response, with retries on malformed JSON (mirrors the num_retries
    pattern already used for Llama tool-call flakiness in CausalMed).

    Does NOT mutate `belief` — caller applies the returned AgentDecision
    (this keeps the function testable without a live BeliefStateManager).

    `_provider` — optional callable(system_prompt, user_prompt) -> str,
    overriding _call_groq. Used for offline testing; production callers
    should leave this as None.
    """
    provider = _provider or _call_groq
    prompt = build_prompt(belief, new_evidence, re_examined_evidence)

    last_error: Optional[Exception] = None
    for attempt in range(cfg.LLM_MAX_RETRIES + 1):
        try:
            raw = provider(_SYSTEM_PROMPT, prompt)
            decision = parse_response(raw)
            if frame_idx is not None:
                belief.last_llm_call_frame = frame_idx
            return decision
        except (LLMResponseError, Exception) as e:  # noqa: BLE001 — broad on purpose, retried
            last_error = e
            continue

    raise LLMResponseError(
        f"LLM call failed after {cfg.LLM_MAX_RETRIES + 1} attempts: {last_error}"
    )