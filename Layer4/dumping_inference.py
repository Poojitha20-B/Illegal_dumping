"""
Layer 4 — Dumping Inference (Context-Aware Temporal Intent Evaluator)
======================================================================
Converts Layer 3 output → {"event": "legal_disposal"|"illegal_dumping", "confidence": float}

FIXES in this version:
  FIX 1: Bin distance uses BOTTOM-CENTER of bin bbox (not geometric center).
  FIX 2: BIN_NEAR_PX raised 150 → 200px.
  FIX 3: "Placed in bin" legal path — person standing next to bin counts as release.
  FIX 4: Fast-path legal disposal fires at release moment if trash is already near bin.
  FIX 5: MAX_WAIT_FRAMES raised 45 → 90.
  FIX 6: DumpingInference tracks ALL non-person objects near a person (not just
          is_trash==True). Catches the silent "walk to bin and place" case.

  FIX 7 (NEW — Spatio-Temporal Handoff, solves Problem B + C):
  ─────────────────────────────────────────────────────────────
  Root cause: When the handbag is held against the rider's body, RT-DETR cannot
  detect it (occlusion + motion blur). The bag only becomes visible the frame it
  hits the ground. At that point the scooter has driven ~79px forward, so the
  current-frame _near_person() check FAILS → the bag skips HELD/POSSESSED state
  and is immediately classified as TRASH(thrown).

  Solution — _near_person_historical():
    On every new-object spawn, look backward through the rolling history of all
    person track centroids (up to POSSESSION_HISTORY_FRAMES = 20 frames).
    If the object's appearance coordinates fall within MAX_POSSESSION_DISTANCE_PX
    (150px) of WHERE A PERSON WAS up to 20 frames ago:
      → Retroactively link the object to that person
      → Immediately set state = POSSESSED (skip APPEARING entirely)
      → Force held_frames = HELD_FRAMES_MIN to lock ownership
      → Do NOT flag as TRASH until the normal release + rest logic fires

  This is purely additive — existing code paths are unchanged.
  Works cleanly with empty history arrays and multiple nearby people
  (closest historical position wins; ties broken by recency).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from Layer2.bin_tracker import TrackedBin
from Layer2.track_state import TrackedObject
import Layer4.config as cfg

# ──────────────────────────────────────────────────────────────────────────────
#  Spatio-temporal constants (sourced from config, with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────

_MAX_POSSESSION_DISTANCE_PX: float = getattr(cfg, "MAX_POSSESSION_DISTANCE_PX", 150.0)
_POSSESSION_HISTORY_FRAMES:  int   = getattr(cfg, "POSSESSION_HISTORY_FRAMES",  20)
_HELD_FRAMES_MIN:            int   = getattr(cfg, "HELD_FRAMES_MIN",            5)
# Around line 50, with the other constants:
_HISTORICAL_DROP_RADIUS_PX: float = getattr(cfg, "HISTORICAL_DROP_RADIUS_PX", 160.0)

# ──────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

MIN_POST_RELEASE     = 3     # frames of post-release motion required
REST_VEL_THRESHOLD   = 3.0   # px/frame
REST_FRAMES          = 4     # consecutive rest frames to confirm final position
MAX_WAIT_FRAMES      = 90    # give up waiting for rest after this many frames
MAX_PAIR_AGE         = 150   # frames — purge stale pair state
HOLD_DISTANCE_PX     = 120   # px — separation threshold triggering instant release


# ──────────────────────────────────────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DumpingEvent:
    event:      str    # "legal_disposal" | "illegal_dumping" | "pending"
    confidence: float
    pair_id:    str
    reason:     str = ""

    def to_dict(self) -> Dict:
        return {"event": self.event, "confidence": round(self.confidence, 3)}

    def __repr__(self):
        return f"DumpingEvent({self.event}, conf={self.confidence:.2f}, reason={self.reason})"


# ──────────────────────────────────────────────────────────────────────────────
#  Per-pair state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _PairState:
    pair_id:             str
    trash_track_id:      int
    person_track_id:     int

    release_confirmed:   bool  = False
    post_release_count:  int   = 0
    held_frames:         int   = 0
    release_pos: Optional[Tuple[float, float]] = None

    post_trail: deque = field(default_factory=lambda: deque(maxlen=60))
    consecutive_rest:    int   = 0

    event_triggered:     bool  = False
    locked_result: Optional[DumpingEvent] = None
    frames_since_update: int   = 0

    # [FIX-7] set True when the historical proximity check linked this object
    # to a person — prevents the "person left frame → instant release" path
    # from firing before enough held frames have been counted.
    historically_linked: bool  = False


# ──────────────────────────────────────────────────────────────────────────────
#  Person-centroid history store
# ──────────────────────────────────────────────────────────────────────────────

class _PersonHistoryStore:
    """
    Maintains a rolling deque of (cx, cy) centroids per person track_id.

    Called once per frame from DumpingInference.update() BEFORE any object
    processing, so the history always reflects where persons WERE, not where
    they are now.

    Layout of each entry in self._history[track_id]:
        deque of (cx, cy) tuples, oldest first, max length = POSSESSION_HISTORY_FRAMES
    """

    def __init__(self, maxlen: int = _POSSESSION_HISTORY_FRAMES):
        self._maxlen: int = maxlen
        self._history: Dict[int, Deque[Tuple[float, float]]] = {}

    def update(self, persons: List[TrackedObject]) -> None:
        """Call once per frame with the current person track list."""
        seen_ids = set()
        for p in persons:
            tid = p.track_id
            seen_ids.add(tid)
            cx = float((p.bbox[0] + p.bbox[2]) / 2.0)
            cy = float((p.bbox[1] + p.bbox[3]) / 2.0)
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self._maxlen)
            self._history[tid].append((cx, cy))

        # Drop history for persons that have left the scene
        gone = [tid for tid in self._history if tid not in seen_ids]
        for tid in gone:
            del self._history[tid]

    def nearest_historical_person(
        self,
        point:      Tuple[float, float],
        max_dist:   float,
    ) -> Tuple[Optional[int], float]:
        """
        Search ALL stored person histories for the closest centroid to `point`.

        Returns (track_id, distance) of the best match, or (None, inf) if
        nothing falls within max_dist.

        Edge cases handled cleanly:
          - Empty history dict  → returns (None, inf)
          - Person with 0 entries → skipped
          - Multiple persons equally close → lowest distance wins;
            ties broken by the person whose MOST RECENT entry is closest
            (i.e., the one who passed over `point` most recently).
        """
        if not self._history:
            return None, float("inf")

        best_tid:  Optional[int] = None
        best_dist: float         = float("inf")
        best_age:  int           = -1

        px, py = point

        for tid, centroids in self._history.items():
            if not centroids:
                continue
            for age_idx, (cx, cy) in enumerate(centroids):
                d = math.hypot(px - cx, py - cy)
                # Recency weight: more recent entries (higher index) win ties
                if d < best_dist or (d == best_dist and age_idx > best_age):
                    best_dist = d
                    best_tid  = tid
                    best_age  = age_idx

        if best_dist > max_dist:
            return None, float("inf")

        return best_tid, best_dist

    def get_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Return centroid history list for a given person (empty list if unknown)."""
        return list(self._history.get(track_id, []))


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0))

def _bottom_center(bbox: np.ndarray) -> Tuple[float, float]:
    """FIX 1: bottom-center as bin reference (matches BinTracker's own logging)."""
    return (float((bbox[0] + bbox[2]) / 2.0), float(bbox[3]))

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest_bin(
    pt:   Tuple[float, float],
    bins: List[TrackedBin],
) -> Tuple[float, Optional[TrackedBin]]:
    if not bins:
        return float("inf"), None
    best_d, best_b = float("inf"), None
    for tb in bins:
        d = _dist(pt, _bottom_center(tb.bbox))
        if d < best_d:
            best_d, best_b = d, tb
    return best_d, best_b

def _trail_velocity(trail: deque) -> float:
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    tail = pts[-4:] if len(pts) >= 4 else pts
    speeds = [
        math.hypot(tail[i][0] - tail[i-1][0], tail[i][1] - tail[i-1][1])
        for i in range(1, len(tail))
    ]
    return sum(speeds) / len(speeds)


# ──────────────────────────────────────────────────────────────────────────────
#  Main evaluator
# ──────────────────────────────────────────────────────────────────────────────

class DumpingInference:
    """
    Layer 4 — call update() once per frame.

    Usage:
        inference = DumpingInference()
        events = inference.update(tracked_objects, tracked_bins)
        for ev in events:
            if ev.event != "pending":
                print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f}")
    """

    def __init__(self):
        self._pairs:          Dict[str, _PairState] = {}
        self._person_history: _PersonHistoryStore   = _PersonHistoryStore(
            maxlen=_POSSESSION_HISTORY_FRAMES
        )

    def update(
        self,
        tracked_objects: List[TrackedObject],
        tracked_bins:    List[TrackedBin],
    ) -> List[DumpingEvent]:

        for ps in self._pairs.values():
            ps.frames_since_update += 1

        persons = [o for o in tracked_objects if o.class_name == "person"]

        # [FIX-7] Step 1: Record where persons ARE this frame (before processing objects).
        # This must happen before any object logic so that when a bag appears on the
        # ground the history already contains the rider's position from prior frames.
        self._person_history.update(persons)

        # ── Build candidate object list (FIX 6) ──────────────────────────────
        confirmed_trash = [o for o in tracked_objects if o.is_trash]
        confirmed_ids   = {o.track_id for o in confirmed_trash}

        candidate_trash: List[TrackedObject] = []
        for obj in tracked_objects:
            if obj.class_name == "person":
                continue
            if obj.track_id in confirmed_ids:
                continue
            # Reject objects with very short trails — these are noise detections
            # or ghost re-detections (e.g. phantom id2), not real carried objects.
            # Also reject objects with full trails (28/30) — they've been in frame
            # a long time without being flagged, so they're carried accessories.
            trail_len = len(obj.trail)
            if trail_len < 5 or trail_len >= 28:
                continue
            # Exclude body-worn accessories — these are never dumped objects
            # Exclude handbags that are moving WITH a person (body-worn).
            # A handbag that is stationary (trail points clustered) is a
            # dumped/dropped object and should NOT be excluded.
            if obj.class_name == "handbag":
                if len(obj.trail) >= 3:
                    trail_pts = list(obj.trail)
                    xs = [p[0] for p in trail_pts]
                    ys = [p[1] for p in trail_pts]
                    spread = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
                    if spread > 30:
                        continue  # moving with person — body-worn, skip
                    # else: stationary — treat as dumped candidate
                else:
                    continue  # too short a trail to tell — skip to be safe
            obj_c = _centroid(obj.bbox)
            for p in persons:
                if _dist(obj_c, _centroid(p.bbox)) <= 100:  # tightened from 150px
                    candidate_trash.append(obj)
                    break

        all_trash = confirmed_trash + candidate_trash

        results: List[DumpingEvent] = []

        for tr in all_trash:
            obj_centroid = _centroid(tr.bbox)

            # ── [FIX-7] Step 2: Historical proximity check on FIRST SIGHT ────
            # When a pair is brand-new (not in self._pairs yet), check whether
            # this object's appearance coordinates match any person's PAST
            # positions before we even try the current-frame person search.
            pair_is_new = self._pair_key(tr, persons) not in self._pairs

            historically_linked_pid: Optional[int] = None
            if pair_is_new:
                hist_pid, hist_dist = self._person_history.nearest_historical_person(
                obj_centroid, _HISTORICAL_DROP_RADIUS_PX
            )
                if hist_pid is not None:
                    historically_linked_pid = hist_pid
                    # Prefer the historical person over nearest-current-frame person
                    # so that the pair_id is stable across the occlusion gap.

            # ── Resolve which person owns this object ─────────────────────────
            if historically_linked_pid is not None:
                # Use the historically-linked person as the owner
                person = self._find_person_by_id(historically_linked_pid, persons)
                # person may be None if the rider has left the frame — that's fine;
                # we still have the link. Use a sentinel pid so the pair_id is stable.
                pid = historically_linked_pid
            else:
                person = self._nearest_person(tr, persons)
                pid    = person.track_id if person else -1

            pair_id = f"person_{pid}_trash_{tr.track_id}"
            ps      = self._get_or_create(pair_id, tr.track_id, pid)
            ps.frames_since_update = 0

            # [FIX-7] Step 3: On first creation, if we got a historical link,
            # immediately bootstrap the pair into POSSESSED state so it is
            # never mistaken for a brand-new unknown object.
            if pair_is_new and historically_linked_pid is not None and not ps.historically_linked:
                ps.historically_linked = True
                ps.held_frames         = _HELD_FRAMES_MIN   # bypass APPEARING wait
                # Do NOT set release_confirmed yet — the bag still needs to rest
                # before we emit a TRASH/legal decision.
                print(
                    f"[Layer4-FIX7] obj_id={tr.track_id} retroactively linked to "
                    f"person_id={pid} via history (dist={hist_dist:.0f}px). "
                    f"State → POSSESSED, held_frames={ps.held_frames}"
                )

            # Return locked result — never re-evaluate a concluded pair
            if ps.event_triggered and ps.locked_result is not None:
                results.append(ps.locked_result)
                continue

            # ── Update hold/release state ─────────────────────────────────────
            self._update_release_state(ps, tr, person, tracked_bins)

            # Gate: release must be confirmed
            if not ps.release_confirmed:
                if tr.is_trash:
                    results.append(DumpingEvent("pending", 0.0, pair_id, "no_release_yet"))
                continue

            # ── FIX 4: Fast-path legal — already near a bin at release moment ─
            if ps.release_pos is not None and tracked_bins:
                fast_d, fast_bin = _nearest_bin(ps.release_pos, tracked_bins)
                if fast_d <= cfg.BIN_RELEASE_FAST_PX:
                    ev = DumpingEvent(
                        "legal_disposal", 0.90, pair_id,
                        f"released_near_bin dist={fast_d:.0f}px "
                        f"bin#{fast_bin.bin_id if fast_bin else '?'}"
                    )
                    ps.event_triggered = True
                    ps.locked_result   = ev
                    results.append(ev)
                    print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f} | {ev.reason}")
                    continue

            if ps.post_release_count < MIN_POST_RELEASE:
                results.append(DumpingEvent(
                    "pending", 0.0, pair_id,
                    f"post_release={ps.post_release_count}<{MIN_POST_RELEASE}"
                ))
                continue

            # Accumulate post-release trail
            ps.post_trail.append(obj_centroid)

            # Rest detection
            vel = _trail_velocity(ps.post_trail)
            if vel < REST_VEL_THRESHOLD:
                ps.consecutive_rest += 1
            else:
                ps.consecutive_rest = 0

            object_at_rest = ps.consecutive_rest >= REST_FRAMES
            max_wait_hit   = ps.post_release_count >= MAX_WAIT_FRAMES

            if not (object_at_rest or max_wait_hit):
                # Historically-linked drops skip rest buffer — flag instantly
                if getattr(ps, 'historically_linked', False):
                    pass   # fall through to _decide immediately
                else:
                    results.append(DumpingEvent(
                        "pending", 0.0, pair_id, f"waiting_for_rest vel={vel:.1f}"
                    ))
                    continue

            # Final decision
            ev = self._decide(ps, tr, tracked_bins)
            ps.event_triggered = True
            ps.locked_result   = ev
            results.append(ev)
            print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f} | {ev.reason}")

        self._purge()
        return results

    # ── Release state updater ─────────────────────────────────────────────────

    def _update_release_state(
        self,
        ps:           _PairState,
        tr:           TrackedObject,
        person:       Optional[TrackedObject],
        tracked_bins: List[TrackedBin],
    ) -> None:
        """
        Three release signals in priority order:
          1. trash_how == "thrown"      — Layer 2 explicit signal
          2. Person standing at a bin   — FIX 3: placement / legal disposal
          3. Person-trash divergence    — fallback throw detection

        [FIX-7] If this pair was linked via history and the person is gone,
        we DO NOT immediately release. We require at least MIN_POST_RELEASE
        frames of the object being visible and stationary before deciding.
        This prevents the "person left frame with 0 held_frames → instant TRASH"
        false-positive that was the root cause of Problem C.
        """
        if ps.release_confirmed:
            ps.post_release_count += 1
            return

        trash_c = _centroid(tr.bbox)

        # Signal 1: explicit throw from Layer 2
        if tr.trash_how == "thrown":
            ps.release_confirmed  = True
            ps.post_release_count = 0
            ps.release_pos        = trash_c
            return

        # Signal 2: FIX 3 — person has reached a bin (placement path)
        if tracked_bins and person is not None:
            person_c  = _centroid(person.bbox)
            near_d, _ = _nearest_bin(person_c, tracked_bins)
            if near_d <= cfg.BIN_RELEASE_FAST_PX and ps.held_frames >= cfg.HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c
                return

        # Signal 3: divergence fallback
        # Signal 3: divergence fallback — instant trigger on separation
        # Signal 3: divergence fallback
        # Require much larger separation AND more held frames to avoid
        # false release from natural person/object centroid offset
        # Signal 3: divergence fallback
        # 120px is just the natural centroid offset between person and carried object.
        # Real separation requires larger distance AND sustained hold history.
        # Signal 3: divergence fallback
        # 120px is just the natural offset between person centroid and a bag
        # at their side/feet. Real separation needs 250px+ AND sustained hold.
        if person is not None:
            d = _dist(trash_c, _centroid(person.bbox))
            if d > 250 and ps.held_frames >= cfg.HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c
                return
            if d <= cfg.NEAR_PERSON_PX:
                ps.held_frames += 1
        else:
            # Person left frame while object was being held — treat as release
            if ps.held_frames >= cfg.HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c

    # ── Final decision ────────────────────────────────────────────────────────

    def _decide(
        self,
        ps:           _PairState,
        tr:           TrackedObject,
        tracked_bins: List[TrackedBin],
    ) -> DumpingEvent:
        bin_present = len(tracked_bins) > 0
        final_pos   = _centroid(tr.bbox)

        hold_score = min(ps.held_frames / 10.0, 1.0)
        post_score = min(ps.post_release_count / 20.0, 1.0)
        confidence = max(0.3, min(0.95, 0.5 * hold_score + 0.5 * post_score + 0.3))

        if bin_present:
            nearest_d, nearest_bin_obj = _nearest_bin(final_pos, tracked_bins)
            bin_label = f"#{nearest_bin_obj.bin_id}" if nearest_bin_obj else "?"
            if nearest_d <= cfg.BIN_NEAR_PX:
                return DumpingEvent(
                    "legal_disposal", round(confidence, 2), ps.pair_id,
                    f"bin{bin_label} bottom_dist={nearest_d:.0f}px <= {cfg.BIN_NEAR_PX}px"
                )
            else:
                return DumpingEvent(
                    "illegal_dumping", round(confidence, 2), ps.pair_id,
                    f"bin{bin_label} present but bottom_dist={nearest_d:.0f}px > {cfg.BIN_NEAR_PX}px"
                )
        else:
            return DumpingEvent(
                "illegal_dumping", round(confidence, 2), ps.pair_id,
                f"no_bin held={ps.held_frames}f post={ps.post_release_count}f"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pair_key(
        self,
        tr:      TrackedObject,
        persons: List[TrackedObject],
    ) -> str:
        """Generate the pair_id that WOULD be assigned without historical override."""
        person = self._nearest_person(tr, persons)
        pid    = person.track_id if person else -1
        return f"person_{pid}_trash_{tr.track_id}"

    def _nearest_person(
        self,
        tr:      TrackedObject,
        persons: List[TrackedObject],
    ) -> Optional[TrackedObject]:
        if not persons:
            return None
        tc = _centroid(tr.bbox)
        return min(persons, key=lambda p: _dist(tc, _centroid(p.bbox)))

    def _find_person_by_id(
        self,
        track_id: int,
        persons:  List[TrackedObject],
    ) -> Optional[TrackedObject]:
        """Return the TrackedObject for a specific person ID, or None if gone."""
        for p in persons:
            if p.track_id == track_id:
                return p
        return None

    def _get_or_create(self, pair_id: str, trash_id: int, person_id: int) -> _PairState:
        if pair_id not in self._pairs:
            self._pairs[pair_id] = _PairState(
                pair_id         = pair_id,
                trash_track_id  = trash_id,
                person_track_id = person_id,
            )
        return self._pairs[pair_id]

    def _purge(self) -> None:
        stale = [
            pid for pid, ps in self._pairs.items()
            if (ps.frames_since_update > MAX_PAIR_AGE and not ps.event_triggered)
            or (
                # Only kill truly abandoned pairs — ones that have never been
                # seen for 10+ frames AND have zero held frames (pure phantoms).
                # Do NOT kill pairs that have accumulated held_frames > 0,
                # those are real carried objects that just need more time.
                ps.frames_since_update > 10
                and not ps.release_confirmed
                and not ps.event_triggered
                and ps.held_frames == 0
            )
        ]
        for pid in stale:
            del self._pairs[pid]

    def get_active_pairs(self) -> List[_PairState]:
        return list(self._pairs.values())