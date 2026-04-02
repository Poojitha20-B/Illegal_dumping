"""
Layer 1 — RT-DETR Detector
Wraps Ultralytics RT-DETR and returns clean, filtered detections.
"""
import torchvision
from ultralytics import RTDETR
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List
from .config import (
    MODEL_NAME, IMGSZ, CONF_THRESH, IOU_THRESH, DEVICE, KEEP_CLASSES
)


@dataclass
class Detection:
    """Single detection output."""
    bbox: np.ndarray        # [x1, y1, x2, y2]  (pixel coords)
    class_name: str
    confidence: float
    class_id: int


class RTDETRDetector:
    """
    Wraps RT-DETR for clean per-frame detection.
    
    Usage:
        detector = RTDETRDetector()
        detections = detector.detect(frame)   # frame = BGR numpy array
    """

    def __init__(self):
        print(f"[Layer1] Loading {MODEL_NAME} on {DEVICE} ...")
        self.model  = RTDETR(MODEL_NAME)
        self.device = DEVICE
        self.names  = self.model.names          # {id: class_name}
        self._keep_ids = self._resolve_keep_ids()
        print(f"[Layer1] Tracking classes: {sorted(KEEP_CLASSES)}")

    def _nms_filter(self, detections: List[Detection], iou_thresh: float = 0.4) -> List[Detection]:
        """Extra NMS pass to kill duplicate boxes RT-DETR sometimes produces."""
        if len(detections) < 2:
            return detections

        boxes  = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        keep   = torchvision.ops.nms(boxes, scores, iou_thresh)
        return [detections[i] for i in keep.tolist()]

    # ──────────────────────────────────────
    def _resolve_keep_ids(self) -> set:
        """Map class names → class IDs for fast filtering."""
        return {
            cid for cid, name in self.names.items()
            if name in KEEP_CLASSES
        }

    # ──────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run RT-DETR on a single BGR frame.

        Returns:
            List[Detection] — filtered, cleaned detections only.
        """
        results = self.model.predict(
            source   = frame,
            imgsz    = IMGSZ,
            conf     = CONF_THRESH,
            iou      = IOU_THRESH,
            device   = self.device,
            verbose  = False,
        )

        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cid  = int(box.cls[0])
                if cid not in self._keep_ids:
                    continue  # skip irrelevant classes

                detections.append(Detection(
                    bbox       = box.xyxy[0].cpu().numpy(),   # [x1,y1,x2,y2]
                    class_name = self.names[cid],
                    confidence = float(box.conf[0]),
                    class_id   = cid,
                ))

        # NMS should be applied AFTER collecting all detections
        detections = self._nms_filter(detections)

        return detections