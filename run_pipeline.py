"""
Full Pipeline Runner — Layer 1 + Layer 2

Usage:
    python run_pipeline.py --source test2.mp4
    python run_pipeline.py --source 0
    python run_pipeline.py --source test2.mp4 --save
"""

import argparse
import time
import cv2

from Layer1.detector       import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer2.tracker        import ByteTrackWrapper
from Layer2.visualizer     import draw_tracks


def _open_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def run(source: str, save: bool = False):
    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    tracker        = ByteTrackWrapper()

    cap = cv2.VideoCapture(_open_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if save:
        writer = cv2.VideoWriter(
            "layer2_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )
        print("[Pipeline] Saving → layer2_output.mp4")

    print("[Pipeline] Running Layer 1 + Layer 2 — press Q to quit")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Layer 1 ───────────────────────
        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)

        # ── Layer 2 ───────────────────────
        tracked = tracker.update(detections, trash_detections, (H, W))

        # ── Visualise ─────────────────────
        vis = draw_tracks(
            frame.copy(),
            tracked,
            tracker.total_trash_events,
            tracker.max_persons_seen,
            tracker.max_objects_seen,
            tracker.max_trash_tagged,
        )

        now = time.time()
        cv2.putText(
            vis, f"FPS: {1/(now - prev + 1e-9):.1f}",
            (W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 1, cv2.LINE_AA,
        )
        prev = now

        cv2.imshow("Layer 1+2 — Detection + Tracking", vis)
        if writer:
            writer.write(vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(
        f"[Pipeline] Done. "
        f"persons={tracker.max_persons_seen}  "
        f"objects={tracker.max_objects_seen}  "
        f"trash_tagged={tracker.max_trash_tagged}  "
        f"trash_events={tracker.total_trash_events}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    run(args.source, args.save)