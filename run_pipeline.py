"""
Full Pipeline Runner — Layer 1 + Layer 2 + Layer 3

Layer 3 sits between the tracker and the (future) behaviour model.
It maintains per-pair sequence buffers of temporal features.

Usage:
    python run_pipeline.py --source test2.mp4
    python run_pipeline.py --source 0
    python run_pipeline.py --source test2.mp4 --save
    python run_pipeline.py --source test2.mp4 --save --no-layer3-vis
"""

import argparse
import time
import cv2

from Layer1.detector        import RTDETRDetector
from Layer1.trash_detector  import TrashDetector
from Layer2.tracker         import ByteTrackWrapper
from Layer2.visualizer      import draw_tracks
from Layer3.memory          import MemoryEngine
from Layer3.visualizer      import draw_memory


def run(source: str, save: bool = False, show_layer3: bool = True):
    # ── Init all layers ───────────────────────────────────────────
    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    tracker        = ByteTrackWrapper()
    memory         = MemoryEngine()          # Layer 3

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if save:
        writer = cv2.VideoWriter(
            "layer3_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )
        print("[Pipeline] Saving → layer3_output.mp4")

    print("[Pipeline] Running Layer 1 + 2 + 3 — press Q to quit")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Layer 1: Detect ───────────────────────────────────────
        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)
        # ── Layer 2: Track ────────────────────────────────────────
        # If tracker.update() or draw_tracks() uses bin_contours, pass [] instead
        tracked = tracker.update(
            detections,
            trash_detections,
            (H, W),
        )

        # ── Layer 3: Memory ───────────────────────────────────────
        pairs       = memory.update(tracked)
        ready_pairs = memory.get_ready_pairs(min_frames=8)

        # Debug log every 30 frames
        if tracker._frame_count % 30 == 0 and ready_pairs:
            print(f"[Layer3] frame={tracker._frame_count}  "
                  f"active_pairs={memory.pair_count}  "
                  f"ready={len(ready_pairs)}")
            for p in ready_pairs[:3]:
                print(f"  {p}  seq_shape={p.get_sequence().shape}")

        # ── Visualise ─────────────────────────────────────────────
        vis = draw_tracks(
            frame.copy(),
            tracked,
            tracker.total_trash_events,
        )

        if show_layer3:
            vis = draw_memory(vis, pairs, tracked)

        now = time.time()
        cv2.putText(
            vis, f"FPS: {1/(now-prev+1e-9):.1f}",
            (W - 110, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 1, cv2.LINE_AA,
        )
        prev = now

        cv2.imshow("Pipeline  L1 + L2 + L3", vis)
        if writer:
            writer.write(vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[Pipeline] Done.")
    print(f"  Total trash events : {tracker.total_trash_events}")
    print(f"  Active pairs       : {memory.pair_count}")
    print(f"  Ready pairs        : {len(memory.get_ready_pairs())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        default="0")
    parser.add_argument("--save",          action="store_true")
    parser.add_argument("--no-layer3-vis", action="store_true",
                        help="disable Layer 3 overlay (cleaner view)")
    args = parser.parse_args()
    run(args.source, args.save, show_layer3=not args.no_layer3_vis)