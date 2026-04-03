"""
Layer 2 — ByteTrack (pure numpy implementation)

Fixes in this version:
  - TRACK_MATCH_THRESH lowered to 0.30 (was 0.8 — was killing all matches)
  - Objects appear on frame 1; persons need MIN_TRACK_FRAMES to suppress false positives
  - min_frames computed INSIDE the loop (was outside — caused UnboundLocalError on t)
  - Ghost throw bbox drawn as red TrackedObject with negative ID
  - Dead code removed: any_overlap was computed but never used
  - max_persons_seen / max_objects_seen: cumulative peak counters so
    HUD keeps showing counts even after objects leave frame
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from Layer1.detector import Detection
from Layer1.trash_detector import TrashDetection
from .track_state import TrackedObject
from .config import (
    TRACK_HIGH_THRESH, TRACK_LOW_THRESH, TRACK_MATCH_THRESH,
    NEW_TRACK_THRESH, MAX_TIME_LOST, MIN_TRACK_FRAMES,
)

GHOST_COOLDOWN_FRAMES = 20


# ── IoU utilities ─────────────────────────────────────────

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    ax1, ay1, ax2, ay2 = bboxes_a[:,0], bboxes_a[:,1], bboxes_a[:,2], bboxes_a[:,3]
    bx1, by1, bx2, by2 = bboxes_b[:,0], bboxes_b[:,1], bboxes_b[:,2], bboxes_b[:,3]

    ix1 = np.maximum(ax1[:,None], bx1[None,:])
    iy1 = np.maximum(ay1[:,None], by1[None,:])
    ix2 = np.minimum(ax2[:,None], bx2[None,:])
    iy2 = np.minimum(ay2[:,None], by2[None,:])

    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union  = area_a[:,None] + area_b[None,:] - inter

    return np.where(union > 0, inter / union, 0.0)


def _single_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


# ── Internal track representation ────────────────────────

class _Track:
    _next_id = 1

    def __init__(self, bbox: np.ndarray, conf: float, class_name: str):
        self.track_id     = _Track._next_id
        _Track._next_id  += 1
        self.bbox         = bbox.copy()
        self.conf         = conf
        self.class_name   = class_name
        self.frames_seen  = 1
        self.frames_lost  = 0
        self.is_active    = True

    def update(self, bbox: np.ndarray, conf: float, class_name: str):
        self.bbox         = bbox.copy()
        self.conf         = conf
        self.class_name   = class_name
        self.frames_seen += 1
        self.frames_lost  = 0
        self.is_active    = True

    def mark_lost(self):
        self.frames_lost += 1
        self.is_active    = False


# ── Main tracker ─────────────────────────────────────────

class ByteTrackWrapper:
    """
    Pure numpy ByteTrack.

    Public attributes (all cumulative — never reset to 0 once set):
        total_trash_events  (int) — unique trash/dump events detected
        max_persons_seen    (int) — peak number of persons in frame at once
        max_objects_seen    (int) — peak number of non-person objects in frame at once
    """

    def __init__(self):
        self._tracks:             List[_Track]             = []
        self._active_objects:     Dict[int, TrackedObject] = {}
        self._ghost_draw_objects: List[TrackedObject]      = []
        self._ghost_cooldown:     int                      = 0
        self._frame_count:        int                      = 0
        self._trash_track_ids:    set                      = set()

        # Cumulative counters — never decrease, shown permanently in HUD
        self.total_trash_events: int = 0
        self.max_persons_seen:   int = 0
        self.max_objects_seen:   int = 0
        self.max_trash_tagged:   int = 0

    # ─────────────────────────────────────────
    def update(
        self,
        detections:       List[Detection],
        trash_detections: List[TrashDetection],
        frame_shape:      Tuple[int, int],
    ) -> List[TrackedObject]:

        self._frame_count += 1

        if self._ghost_cooldown > 0:
            self._ghost_cooldown -= 1

        # Clear ghost draw objects from last frame
        self._ghost_draw_objects = []

        # Split detections into high / low confidence pools
        high = [d for d in detections if d.confidence >= TRACK_HIGH_THRESH]
        low  = [d for d in detections if TRACK_LOW_THRESH <= d.confidence < TRACK_HIGH_THRESH]

        active_tracks = [t for t in self._tracks if t.frames_lost == 0]
        lost_tracks   = [t for t in self._tracks if t.frames_lost  > 0]

        # ── Step 1: high-conf dets → active tracks ────────────────────
        matched_h, unmatched_t, unmatched_h = self._match(active_tracks, high)
        for ti, di in matched_h:
            active_tracks[ti].update(high[di].bbox, high[di].confidence, high[di].class_name)

        # ── Step 2: low-conf dets → unmatched active tracks ───────────
        remaining_tracks = [active_tracks[i] for i in unmatched_t]
        matched_l, still_unmatched_t, _ = self._match(remaining_tracks, low)
        for ti, di in matched_l:
            remaining_tracks[ti].update(low[di].bbox, low[di].confidence, low[di].class_name)

        # ── Step 3: remaining high-conf dets → lost tracks ────────────
        unmatched_high_dets = [high[i] for i in unmatched_h]
        matched_r, _, unmatched_new = self._match(lost_tracks, unmatched_high_dets)
        for ti, di in matched_r:
            lost_tracks[ti].update(
                unmatched_high_dets[di].bbox,
                unmatched_high_dets[di].confidence,
                unmatched_high_dets[di].class_name,
            )

        # ── Step 4: create new tracks for truly unmatched high-conf ───
        for i in unmatched_new:
            d = unmatched_high_dets[i]
            if d.confidence >= NEW_TRACK_THRESH:
                self._tracks.append(_Track(d.bbox, d.confidence, d.class_name))

        # ── Step 5: mark unmatched active tracks as lost ──────────────
        for i in still_unmatched_t:
            remaining_tracks[i].mark_lost()

        # ── Step 6: remove tracks lost too long ───────────────────────
        self._tracks = [t for t in self._tracks if t.frames_lost <= MAX_TIME_LOST]

        # ── Step 7: build TrackedObject output ────────────────────────
        # Persons need MIN_TRACK_FRAMES to suppress false positives.
        # Objects use 1 — appear on their very first detected frame.
        # CRITICAL: min_frames must be computed INSIDE the loop per track.
        output:   List[TrackedObject] = []
        seen_ids: set                 = set()

        for t in self._tracks:
            if t.frames_lost > 0:
                continue

            # Per-class threshold — objects always show on frame 1
            min_frames = MIN_TRACK_FRAMES if t.class_name == "person" else 1
            if t.frames_seen < min_frames:
                continue

            tid = t.track_id
            seen_ids.add(tid)

            if tid in self._active_objects:
                obj            = self._active_objects[tid]
                obj.bbox       = t.bbox.copy()
                obj.confidence = t.conf
                obj.class_name = t.class_name
            else:
                obj = TrackedObject(
                    track_id   = tid,
                    bbox       = t.bbox.copy(),
                    class_name = t.class_name,
                    confidence = t.conf,
                )
                self._active_objects[tid] = obj

            obj.update_trail()
            output.append(obj)

        # Clean up TrackedObjects for dead tracks
        for tid in list(self._active_objects):
            if tid not in seen_ids:
                del self._active_objects[tid]

        # ── Step 8: tag trash + update cumulative event counter ───────
        self._tag_trash(output, trash_detections)

        # Append ghost draw objects so their red boxes render this frame
        output.extend(self._ghost_draw_objects)

        # ── Step 9: update peak HUD counters ──────────────────────────
        # Count only real tracks (positive IDs) — exclude ghost placeholders
        current_persons = sum(
            1 for t in output
            if t.class_name == "person" and t.track_id > 0
        )
        current_objects = sum(
            1 for t in output
            if t.class_name != "person" and t.track_id > 0
        )
        current_trash_tagged = sum(
            1 for t in output
            if t.is_trash and t.track_id > 0
        )
        self.max_persons_seen = max(self.max_persons_seen, current_persons)
        self.max_objects_seen = max(self.max_objects_seen, current_objects)
        self.max_trash_tagged = max(self.max_trash_tagged, current_trash_tagged)

        return output

    # ─────────────────────────────────────────
    def _match(
        self,
        tracks: List[_Track],
        dets:   List[Detection],
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """Hungarian matching on IoU. Returns (matched pairs, unmatched_t, unmatched_d)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.array([t.bbox for t in tracks])
        det_boxes   = np.array([d.bbox for d in dets])
        iou_mat     = _iou_batch(track_boxes, det_boxes)
        cost_mat    = 1.0 - iou_mat

        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matched_r, matched_c = set(), set()
        matched = []
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= TRACK_MATCH_THRESH:
                matched.append((r, c))
                matched_r.add(r)
                matched_c.add(c)

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_r]
        unmatched_d = [j for j in range(len(dets))   if j not in matched_c]
        return matched, unmatched_t, unmatched_d

    # ─────────────────────────────────────────
    def _tag_trash(
        self,
        tracks:           List[TrackedObject],
        trash_detections: List[TrashDetection],
    ):
        """
        Tag tracked objects as trash and maintain cumulative event count.

        Matching rules:
          1. Never tag a person track as trash — skip all person tracks
          2. Match trash bbox to closest non-person track by IoU then centroid distance
          3. Ghost throw (no matching track): one event per GHOST_COOLDOWN_FRAMES window,
             rendered as a negative-ID TrackedObject so a red box appears on screen
        """
        for trash in trash_detections:
            best_iou:   float                   = 0.0
            best_dist:  float                   = float("inf")
            best_track: Optional[TrackedObject] = None

            tcx = (trash.bbox[0] + trash.bbox[2]) / 2
            tcy = (trash.bbox[1] + trash.bbox[3]) / 2

            for t in tracks:
                if t.class_name == "person":
                    continue
                iou  = _single_iou(trash.bbox, t.bbox)
                cx   = (t.bbox[0] + t.bbox[2]) / 2
                cy   = (t.bbox[1] + t.bbox[3]) / 2
                dist = np.hypot(tcx - cx, tcy - cy)

                # Prefer highest IoU; fall back to nearest centroid when IoU = 0
                if iou > best_iou or (best_iou == 0 and dist < best_dist):
                    best_iou   = iou
                    best_dist  = dist
                    best_track = t

            if best_track is not None and (best_iou > 0.10 or best_dist < 80):
                # ── Matched to a real tracked object ─────────────────
                best_track.is_trash    = True
                best_track.trash_label = trash.label
                best_track.trash_how   = trash.how

                # Increment only the first time this track ID is flagged
                if best_track.track_id not in self._trash_track_ids:
                    self._trash_track_ids.add(best_track.track_id)
                    self.total_trash_events += 1

            else:
                # ── Ghost throw — object left frame before being tracked ──
                # Cooldown prevents one physical throw being counted across
                # multiple frames. One cooldown window = exactly one event.
                if self._ghost_cooldown == 0:
                    self.total_trash_events += 1
                    self._ghost_cooldown = GHOST_COOLDOWN_FRAMES

                    ghost_obj = TrackedObject(
                        track_id    = -(self.total_trash_events),
                        bbox        = trash.bbox.copy(),
                        class_name  = trash.label,
                        confidence  = trash.confidence,
                        is_trash    = True,
                        trash_label = trash.label,
                        trash_how   = trash.how,
                    )
                    self._ghost_draw_objects.append(ghost_obj)