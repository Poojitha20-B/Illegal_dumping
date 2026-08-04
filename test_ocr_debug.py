"""
Standalone OCR debug harness.

Loads a saved RAW plate crop (plate_crop_full_0_0.jpg, NOT the
preprocessed one) and runs it through the same ensemble the real
pipeline uses (_ocr_tight_crop -> ViT + CCT + EasyOCR + two-line
split + merge), without re-running detection/tracking/L4/L5/
calibration. Use this to iterate on OCR/preprocessing changes in
seconds instead of ~20s per full run_pipeline.py pass.

Usage:
    python3 test_ocr_debug.py evidence/plate_crop_full_0_0.jpg
"""

import sys
import cv2

from enhancer import _ocr_tight_crop, _best_result

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_ocr_debug.py <path_to_raw_plate_crop.jpg>")
        sys.exit(1)

    crop_path = sys.argv[1]
    crop = cv2.imread(crop_path)
    if crop is None:
        print(f"Could not load image: {crop_path}")
        sys.exit(1)

    print(f"[Test] Loaded {crop_path}  shape={crop.shape}")

    # alpr_conf normally comes from fast-alpr's own full-frame detection.
    # We don't have that here (skipping detection entirely), so use a
    # neutral placeholder — it only affects the fallback conf when a
    # per-model confidence isn't available.
    placeholder_alpr_conf = 0.5

    candidates = _ocr_tight_crop(crop, "debug", placeholder_alpr_conf)

    print("\n[Test] All candidates:")
    for text, conf in candidates:
        print(f"  '{text}'  conf={conf:.2f}")

    plate_text, plate_conf = _best_result(candidates)
    print(f"\n[Test] _best_result() picks: '{plate_text}'  conf={plate_conf:.2f}")


if __name__ == "__main__":
    main()