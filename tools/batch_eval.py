"""
tools/batch_eval.py
====================
Headless evaluation harness for VidTrace (Illegal Dumping Detection).

Runs ONLY Layers 1-5 (perception -> tracking -> memory -> inference -> agent)
on a video and returns the predicted verdict + timestamp. Skips everything
enforcement-related (OCR, challan PDF, email, FaceID, cv2.imshow/writer) so
it can run unattended over hundreds of videos.

Run from the project root:
    python tools/batch_eval.py --videos videos/ --labels labels/ --out report.json

Dataset layout expected (matches your Drive folders):
    videos/vid0011.mp4
    labels/vid0011.json   # {"ID": "vid0011.mp4", "Dumping": 1, "DumpingDetails": {"Timestamp": 11.0, ...}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from Layer1.detector          import RTDETRDetector
from Layer1.trash_detector    import TrashDetector
from Layer1.bin_detector      import BinDetector
from Layer1.calibrator        import calibrate, apply, defaults
from Layer2.tracker           import ByteTrackWrapper
from Layer2.bin_tracker       import BinTracker
from Layer3.feature_extractor import BinInteractionFeatureExtractor
from Layer4.dumping_inference import DumpingInference
from Layer5.agent             import DumpingAgent

# Feature columns pulled from each Layer5 event, used both for the
# per-event CSV export (training data for a learned scorer) and for
# consistent ordering when writing rows.
FEATURE_COLUMNS = [
    "coupling_conf", "diverge_conf", "rest_conf", "bin_prox", "l4_conf", "evidence_conf",
]


class ModelBundle:
    """
    Loads the three heavy, weight-carrying models ONCE. Reuse the same
    instance across every video and every Optuna/grid trial — only
    Layer4/Layer5 config constants change between trials, not these weights.
    Re-constructing these per video was the actual cause of the multi-hour
    runtime: every construction re-reads rtdetr-l.pt, trash_bin_detector.pt,
    and the OSNet ReID checkpoint from disk.
    """
    def __init__(self):
        print("[ModelBundle] loading detector/tracker models once …")
        self.detector       = RTDETRDetector()
        self.trash_detector = TrashDetector()
        self.bin_detector   = BinDetector()
        print("[ModelBundle] ready.")


def run_headless(source: str, models: ModelBundle, do_calibrate: bool = True,
                  max_frames: int | None = None, verbose: bool = False) -> dict:
    """
    Run Layers 1-5 on a single video, no display / OCR / challan / email.
    `models` is a shared ModelBundle — only per-video state (tracker,
    extractor, inference, agent) is constructed fresh here.

    Returns:
        {
          "source": str,
          "fps": float,
          "predicted_dumping": bool,
          "predicted_timestamp": float | None,   # seconds, first confirmed violation
          "events": [ {event, confidence, person_id, object_id, start_frame, end_frame,
                       reason, coupling_conf, diverge_conf, rest_conf, bin_prox,
                       l4_conf, evidence_conf}, ... ],
          "frames_processed": int,
          "error": str | None,
        }
    """
    result = {
        "source": source, "fps": None, "predicted_dumping": False,
        "predicted_timestamp": None, "events": [], "frames_processed": 0, "error": None,
    }
    try:
        detector       = models.detector
        trash_detector = models.trash_detector
        bin_detector   = models.bin_detector

        if do_calibrate:
            calib = calibrate(source, detector, verbose=verbose)
            cfg = calib.cfg
            apply(cfg)
        # else use whatever Layer1.config currently holds

        tracker         = ByteTrackWrapper()
        bin_tracker     = BinTracker()
        extractor       = BinInteractionFeatureExtractor(debug=False)
        inference       = DumpingInference()
        agent           = DumpingAgent()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            result["error"] = f"cannot open video: {source}"
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        result["fps"] = fps

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            dets         = detector.detect(frame)
            trash_dets   = trash_detector.detect(frame.shape, dets)
            bin_dets     = bin_detector.detect(frame)
            tracked_objs = tracker.update(dets, trash_dets, frame.shape[:2])
            tracked_bins = bin_tracker.update(bin_dets)
            ts           = time.time()
            extractor.update(tracked_objs, tracked_bins, ts)
            l4_events    = inference.update(tracked_objs, tracked_bins)
            agent.update(frame_idx, tracked_objs, tracked_bins, l4_events)

        cap.release()
        result["frames_processed"] = frame_idx

        all_results = agent.get_all_results()
        first_violation_frame = None
        for r in all_results:
            start_f, end_f = r.get("frames", [None, None])
            ev = {
                "event": r["event"], "confidence": r["confidence"],
                "violation": r.get("violation", r["event"] == "illegal_dumping"),
                "person_id": r.get("person_id"), "object_id": r.get("object_id"),
                "start_frame": start_f, "end_frame": end_f, "reason": r.get("reason", ""),
                # Evidence components — see Layer5/agent.py's _finalise().
                # Persisted here (rather than only printed) so they can be
                # exported to a training CSV for a learned scorer.
                "coupling_conf": r.get("coupling_conf"),
                "diverge_conf":  r.get("diverge_conf"),
                "rest_conf":     r.get("rest_conf"),
                "bin_prox":      r.get("bin_prox"),
                "l4_conf":       r.get("l4_conf"),
                "evidence_conf": r.get("evidence_conf"),
            }
            result["events"].append(ev)
            if ev["violation"]:
                result["predicted_dumping"] = True
                if end_f is not None and (first_violation_frame is None or end_f < first_violation_frame):
                    first_violation_frame = end_f

        if first_violation_frame is not None and fps > 0:
            result["predicted_timestamp"] = round(first_violation_frame / fps, 2)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()

    return result


def evaluate_dataset(videos_dir: str, labels_dir: str, do_calibrate: bool = True,
                      limit: int | None = None, verbose: bool = False,
                      models: "ModelBundle | None" = None) -> list[dict]:
    """
    Runs run_headless() over every video that has a matching label JSON.
    Returns a list of per-video comparison dicts. Each record also carries
    a "events" list (with evidence features) for the CSV export step.

    Pass `models=` a pre-built ModelBundle when calling this repeatedly
    (e.g. from tune_optuna.py across many trials) so the heavy weights are
    loaded ONCE for the whole tuning run, not once per trial.
    """
    videos_dir = Path(videos_dir)
    labels_dir = Path(labels_dir)

    if models is None:
        models = ModelBundle()

    label_files = sorted(labels_dir.glob("*.json"))
    if limit:
        label_files = label_files[:limit]

    out = []
    for i, lf in enumerate(label_files):
        with open(lf) as f:
            label = json.load(f)
        vid_name = label["ID"]
        vid_path = videos_dir / vid_name
        if not vid_path.exists():
            print(f"[{i+1}/{len(label_files)}] SKIP (video missing): {vid_name}")
            continue

        print(f"[{i+1}/{len(label_files)}] Running: {vid_name}")
        pred = run_headless(str(vid_path), models, do_calibrate=do_calibrate, verbose=verbose)

        gt_dumping   = bool(label.get("Dumping", 0))
        gt_timestamp = label.get("DumpingDetails", {}).get("Timestamp")
        gt_type      = label.get("DumpingDetails", {}).get("Type of Dumping")
        time_of_day  = label.get("Video Info", {}).get("Time of Day")

        record = {
            "video": vid_name,
            "gt_dumping": gt_dumping,
            "gt_timestamp": gt_timestamp,
            "gt_type": gt_type,
            "time_of_day": time_of_day,
            "fps": pred["fps"],
            "pred_dumping": pred["predicted_dumping"],
            "pred_timestamp": pred["predicted_timestamp"],
            "correct": pred["predicted_dumping"] == gt_dumping,
            "error": pred["error"],
            "events": pred["events"],
        }
        if gt_dumping and pred["predicted_dumping"] and gt_timestamp is not None and pred["predicted_timestamp"] is not None:
            record["timestamp_error_s"] = abs(pred["predicted_timestamp"] - gt_timestamp)
        out.append(record)

    return out


def summarize(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0}
    tp = sum(1 for r in records if r["gt_dumping"] and r["pred_dumping"])
    tn = sum(1 for r in records if not r["gt_dumping"] and not r["pred_dumping"])
    fp = sum(1 for r in records if not r["gt_dumping"] and r["pred_dumping"])
    fn = sum(1 for r in records if r["gt_dumping"] and not r["pred_dumping"])
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ts_errors = [r["timestamp_error_s"] for r in records if "timestamp_error_s" in r]

    by_time_of_day = {}
    for tod in set(r.get("time_of_day") for r in records):
        subset = [r for r in records if r.get("time_of_day") == tod]
        by_time_of_day[tod] = round(sum(r["correct"] for r in subset) / len(subset), 3)

    by_type = {}
    for t in set(r.get("gt_type") for r in records if r.get("gt_type")):
        subset = [r for r in records if r.get("gt_type") == t]
        by_type[t] = round(sum(r["correct"] for r in subset) / len(subset), 3)

    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 3), "precision": round(precision, 3),
        "recall": round(recall, 3), "f1": round(f1, 3),
        "mean_timestamp_error_s": round(sum(ts_errors) / len(ts_errors), 2) if ts_errors else None,
        "accuracy_by_time_of_day": by_time_of_day,
        "accuracy_by_dumping_type": by_type,
    }


def export_feature_csv(records: list[dict], out_path: str, timestamp_tolerance_s: float = 3.0) -> int:
    """
    Flattens every Layer5 event across every video into one row per event:
        video, person_id, object_id, coupling_conf, diverge_conf, rest_conf,
        bin_prox, l4_conf, evidence_conf, pred_violation, event_label

    Label assignment (event_label):
      - Negative videos (gt_dumping=False): every event is labeled 0. The
        video has no recorded violation, so no candidate pair in it is one.
      - Positive videos (gt_dumping=True) with a usable gt_timestamp and fps:
        only the event whose end_frame/fps falls within
        `timestamp_tolerance_s` seconds of gt_timestamp is labeled 1; every
        other event in that same video is labeled 0. This matters because a
        positive video's other tracked person/object pairs (e.g. someone
        holding a bag but not the one who dumped) are NOT the labeled
        violation, and blanket-labeling them 1 teaches the model that things
        like zero coupling can still mean "violation".
      - Positive videos where no event matches within tolerance (or
        gt_timestamp/fps is missing): falls back to labeling all events 1,
        since we can't disambiguate — but this is printed as a warning, as
        it re-introduces the same label-noise risk described above.

    Returns the number of rows written.
    """
    rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video", "person_id", "object_id"] + FEATURE_COLUMNS + ["pred_violation", "event_label"]
        )
        for rec in records:
            events = [ev for ev in rec.get("events", [])
                      if not any(ev.get(col) is None for col in FEATURE_COLUMNS)]
            if not events:
                continue

            if not rec["gt_dumping"]:
                labels = {id(ev): 0 for ev in events}
            else:
                gt_ts = rec.get("gt_timestamp")
                fps = rec.get("fps")
                matched = None
                if gt_ts is not None and fps:
                    best_ev, best_diff = None, float("inf")
                    for ev in events:
                        if ev.get("end_frame") is None:
                            continue
                        event_ts = ev["end_frame"] / fps
                        diff = abs(event_ts - gt_ts)
                        if diff < best_diff:
                            best_diff, best_ev = diff, ev
                    if best_ev is not None and best_diff <= timestamp_tolerance_s:
                        matched = best_ev

                if matched is not None:
                    labels = {id(ev): (1 if ev is matched else 0) for ev in events}
                else:
                    if len(events) > 1:
                        print(f"[export_feature_csv] WARNING: {rec['video']} is a positive "
                              f"video with {len(events)} candidate events but none matched "
                              f"gt_timestamp={gt_ts} within {timestamp_tolerance_s}s — "
                              f"labeling all events 1 (label-noise risk).")
                    labels = {id(ev): 1 for ev in events}

            for ev in events:
                writer.writerow(
                    [rec["video"], ev.get("person_id"), ev.get("object_id")]
                    + [ev.get(col) for col in FEATURE_COLUMNS]
                    + [int(bool(ev.get("violation"))), labels[id(ev)]]
                )
                rows_written += 1
    return rows_written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="Path to videos/ folder")
    p.add_argument("--labels", required=True, help="Path to labels/ folder")
    p.add_argument("--out", default="eval_report.json")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate first N videos (for a quick pass)")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--features-out", default=None,
                    help="If set, also write a per-event feature CSV to this path "
                         "(training data for a learned scorer, e.g. features.csv)")
    args = p.parse_args()

    records = evaluate_dataset(
        args.videos, args.labels,
        do_calibrate=not args.no_calibrate,
        limit=args.limit,
        verbose=args.verbose,
    )
    summary = summarize(records)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report written to {args.out}")

    if args.features_out:
        n = export_feature_csv(records, args.features_out)
        print(f"Feature CSV written to {args.features_out} ({n} rows)")