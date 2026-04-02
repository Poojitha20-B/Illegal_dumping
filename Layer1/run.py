"""Layer 1 — Entry Point with Trash Detection"""

import argparse
import time
import cv2
from .detector import RTDETRDetector
from .trash_detector import TrashDetector
from .visualizer import draw_detections, draw_trash
from .config import TRASH_ENABLE


def run(source, save: bool = False):
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        out_path = "layer1_output.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("[Layer1] Running — press Q to quit")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # RT-DETR detections (persons + known objects)
        detections = detector.detect(frame)

        # Trash detection via background subtraction
        trash_detections = []
        if trash_detector:
            person_bboxes = [d.bbox for d in detections if d.class_name == "person"]
            trash_detections = trash_detector.detect(frame, person_bboxes)

        # Visualise
        vis = draw_detections(frame, detections)
        vis = draw_trash(vis, trash_detections)

        now = time.time()
        fps_live = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(vis, f"FPS: {fps_live:.1f}", (w - 110, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Layer 1 — Perception", vis)
        if writer:
            writer.write(vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    run(args.source, args.save)