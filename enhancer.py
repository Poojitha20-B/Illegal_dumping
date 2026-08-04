"""
Layer 1.5 — Enhancement & Plate Detection
==========================================
Sits between Layer 4 (Dumping Inference) and Layer 5 (Agent).
Triggered ONLY when Layer 4 fires an "illegal_dumping" event.

Pipeline per event:
  1. Scan a best-frame window using composite plate scoring
     (blur × log1p(region_area) — rewards larger/closer plates)
  2. Run fast-alpr on the best frame — detects plate bbox + reads text
  3. Re-run OCR on a super-resolution upscaled tight plate crop using
     TWO OCR MODELS in ensemble (ViT + CCT-S):
     - global-plates-mobile-vit-v2-model  (primary)
     - cct-s-v2-global-model              (ensemble partner)
     Both models run on every crop variant.
  3b. EasyOCR fallback — triggered when fast-alpr conf < 0.70.
      EasyOCR handles perspective-distorted/angled plates significantly
      better than the ViT/CCT models (confirmed: reads DL3CDB9940 at 86%).
  4. If no plate found in full frame, fall back to person-anchored crops
  5. Return EnhancementResult with plate text, confidence, and saved evidence

EDSR setup (one-time):
  curl -L -o EDSR_x4.pb \
    https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb
  Place EDSR_x4.pb in the project root (same dir as run_pipeline.py).

CCT-S OCR model setup (one-time, downloads ~5 MB automatically):
  python3 -c "
  from fast_plate_ocr import LicensePlateRecognizer
  LicensePlateRecognizer('cct-s-v2-global-model')
  print('CCT-S model cached.')
  "

Install:
  pip install fast-alpr onnxruntime easyocr
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Indian plate pattern ──────────────────────────────────────────────────────
INDIAN_PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}$')

# ── EDSR model path (relative to CWD / run_pipeline.py) ──────────────────────
EDSR_MODEL_PATH = "EDSR_x4.pb"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_alpr            = None
_plate_ocr_vit   = None   # global-plates-mobile-vit-v2-model  (primary)
_plate_ocr_cct   = None   # cct-s-v2-global-model               (ensemble partner)
_edsr_sr         = None
_easyocr_reader  = None   # EasyOCR — fallback for angled/perspective plates


def _get_alpr():
    global _alpr
    if _alpr is None:
        from fast_alpr import ALPR
        _alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
            # Pin to CPU — CoreML/ANE's reduced-precision path introduces
            # run-to-run floating-point jitter that flips borderline
            # character decisions between identical runs on the same
            # crop (observed directly: same plate crop read correctly
            # as KA05K in one run, misread as KA052 in another, both at
            # ~0.88 confidence). A challan-issuing system needs the same
            # input to produce the same output every time.
            detector_providers=["CPUExecutionProvider"],
            ocr_providers=["CPUExecutionProvider"],
        )
        logger.info("[Enhancer] fast-alpr loaded (detector + OCR, CPU-pinned for determinism)")
    return _alpr


def _get_plate_ocr_vit():
    """Primary OCR: Vision Transformer."""
    global _plate_ocr_vit
    if _plate_ocr_vit is None:
        from fast_plate_ocr import LicensePlateRecognizer
        _plate_ocr_vit = LicensePlateRecognizer(
            "global-plates-mobile-vit-v2-model",
            # Pin to CPU — same nondeterminism as the ALPR detector (see
            # _get_alpr): device='auto' routes this to CoreML on M-series
            # Macs, and CoreML's reduced-precision path flips borderline
            # character decisions run-to-run (confirmed: this exact model
            # read the same crop's top half as 'KA05K' via test_ocr_debug.py
            # but 'KA052' inside the live pipeline, both ~0.88 conf). A
            # challan-issuing system needs the same crop to always OCR the
            # same way.
            providers=["CPUExecutionProvider"],
        )
        logger.info("[Enhancer] fast-plate-ocr ViT loaded (CPU-pinned for determinism)")
    return _plate_ocr_vit


def _get_plate_ocr_cct():
    """
    Ensemble OCR: Compact Convolutional Transformer (CCT-S).
    Different failure modes from ViT — effective ensemble partner.
    """
    global _plate_ocr_cct
    if _plate_ocr_cct is None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            _plate_ocr_cct = LicensePlateRecognizer(
                "cct-s-v2-global-model",
                providers=["CPUExecutionProvider"],  # see _get_plate_ocr_vit
            )
            logger.info("[Enhancer] fast-plate-ocr CCT-S loaded (ensemble, CPU-pinned)")
        except Exception as e:
            logger.warning(
                f"[Enhancer] CCT-S model unavailable ({e}). "
                "Continuing with ViT only."
            )
            _plate_ocr_cct = False
    return _plate_ocr_cct if _plate_ocr_cct else None

def _get_easyocr():
    """
    Lazy-load EasyOCR reader (English only, CPU).
    EasyOCR handles perspective-distorted and angled plates significantly
    better than fast-plate-ocr's ViT/CCT models. Confirmed: reads
    DL3CDB9940 at 86% confidence from a tilted CCTV crop that ViT
    misread completely. Loaded only when needed (first call).
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("[Enhancer] EasyOCR loaded (CPU, English)")
        except Exception as e:
            logger.warning(f"[Enhancer] EasyOCR unavailable ({e}). "
                           "Run: pip install easyocr")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader else None


def _get_edsr() -> Optional[object]:
    """Lazy-load EDSR x4 SR model. Returns sr object or None (fallback mode)."""
    global _edsr_sr
    if _edsr_sr is not None:
        return _edsr_sr if _edsr_sr is not False else None

    model_path = Path(EDSR_MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            f"[Enhancer] EDSR model not found at '{EDSR_MODEL_PATH}'. "
            "Falling back to INTER_CUBIC."
        )
        _edsr_sr = False
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("edsr", 4)
        logger.info(f"[Enhancer] EDSR x4 loaded from '{EDSR_MODEL_PATH}'")
        _edsr_sr = sr
        return sr
    except Exception as e:
        logger.warning(f"[Enhancer] EDSR load failed ({e}), falling back to INTER_CUBIC.")
        _edsr_sr = False
        return None


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EnhancementResult:
    pair_id:           str
    plate_text:        Optional[str] = None
    plate_conf:        float         = 0.0
    blur_score:        float         = 0.0
    best_frame_offset: int           = 0
    frames_scanned:    int           = 0
    saved_paths:       list[str]     = field(default_factory=list)
    elapsed_ms:        int           = 0


# ── Blur scoring ──────────────────────────────────────────────────────────────

def blur_score(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _plate_region_blur(frame: np.ndarray) -> tuple[float, float]:
    """
    Score sharpness ONLY on the plate region of the frame.
    Returns (blur_score, region_pixel_area).
    Region: bottom 60% of the right 55% of the frame.
    """
    h, w    = frame.shape[:2]
    x_start = int(w * 0.45)
    y_start = int(h * 0.40)
    region  = frame[y_start:, x_start:]
    if region.size < 100:
        return blur_score(frame), float(frame.shape[0] * frame.shape[1])
    return blur_score(region), float(region.shape[0] * region.shape[1])


# ── Frame generator ───────────────────────────────────────────────────────────

def cap_frame_generator(
    cap: cv2.VideoCapture, n_frames: int = 150
) -> Iterator[np.ndarray]:
    """Yield the next n_frames from cap, then restore the position."""
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    yielded   = 0
    while yielded < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        yielded += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_crop(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 0
) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _is_indian(text: str) -> bool:
    return bool(INDIAN_PLATE_RE.match(_normalise(text)))


# ── Person-anchored crops (fallback) ─────────────────────────────────────────

def _person_anchored_crops(
    frame: np.ndarray, person_bbox: tuple
) -> list[np.ndarray]:
    fh, fw = frame.shape[:2]
    crops  = []
    for top_frac, bot_frac, x_frac in [
        (0.45, 0.75, 0.20),
        (0.50, 0.80, 0.25),
        (0.55, 0.85, 0.30),
    ]:
        c = _safe_crop(
            frame,
            int(fw * x_frac),       int(fh * top_frac),
            int(fw * (1 - x_frac)), int(fh * bot_frac),
        )
        if c is not None and c.size > 0:
            crops.append(c)
    return crops


# ── OCR on a single image — fast-plate-ocr ───────────────────────────────────

def _ocr_img_single(
    ocr_model,
    img: np.ndarray,
    alpr_conf: float,
    model_label: str,
) -> list[tuple[str, float]]:
    """Run one fast-plate-ocr model on img."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        pred = ocr_model.run(gray, return_confidence=True)
        if not pred:
            return []
        p    = pred[0]
        text = _normalise(p.plate if hasattr(p, "plate") else str(p))
        # Use this model's own per-character confidence when available —
        # falling back to alpr_conf reuses the coarse full-frame detector
        # score for every candidate, which makes _best_result's confidence
        # tie-break meaningless (all candidates end up identically scored).
        conf = (
            float(np.mean(p.char_probs))
            if getattr(p, "char_probs", None) is not None
            else alpr_conf
        )
        if text and len(text) >= 4:
            logger.debug(f"[OCR-{model_label}] '{text}' conf={conf:.2f}")
            return [(text, conf)]
    except Exception as e:
        logger.debug(f"[OCR-{model_label}] error: {e}")
    return []


def _ocr_img(
    img: np.ndarray,
    alpr_conf: float = 0.5,
) -> list[tuple[str, float]]:
    """Run BOTH OCR models (ViT + CCT-S) on img."""
    results = []

    vit = _get_plate_ocr_vit()
    results.extend(_ocr_img_single(vit, img, alpr_conf, "VIT"))

    cct = _get_plate_ocr_cct()
    if cct is not None:
        results.extend(_ocr_img_single(cct, img, alpr_conf, "CCT"))

    return results


# ── EasyOCR fallback ──────────────────────────────────────────────────────────

def _run_easyocr(
    crop: np.ndarray, alpr_conf: float, use_detector: bool = True
) -> list[tuple[str, float]]:
    """
    Run EasyOCR on a plate crop.

    EasyOCR handles perspective-distorted and angled plates significantly
    better than fast-plate-ocr's ViT/CCT models. It was confirmed to read
    DL3CDB9940 at 86% confidence from a tilted CCTV crop that the ViT
    model misread entirely as '04CP92O5'.

    use_detector=True (default) runs EasyOCR's normal detect-then-recognize
    pipeline via readtext() — appropriate for a full frame or full plate
    crop where the text location isn't already known.
    use_detector=False skips the CRAFT detection step via recognize() and
    treats the ENTIRE crop as one line of text. Needed for small/blurry
    pre-cropped regions (e.g. one half of a two-line plate) — CRAFT often
    finds zero text boxes on crops that small, so readtext() silently
    returns nothing and EasyOCR never gets to vote at all.

    Uses the ALPR detector confidence (not EasyOCR's own score) to keep
    confidence values consistent across all OCR paths. EasyOCR's confidence
    is used only to filter low-quality reads (< 0.3 threshold).
    """
    reader = _get_easyocr()
    if reader is None:
        return []
    try:
        if use_detector:
            results = reader.readtext(
                crop,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
            )
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            results = reader.recognize(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
            )
        out = []
        for _, text, conf in results:
            norm_text = _normalise(text)
            print(
                f"[EasyOCR-RAW] '{norm_text}' conf={conf:.2f}"
                + (" [DROPPED conf<0.3]" if conf < 0.3 else
                   " [DROPPED len<4]" if len(norm_text) < 4 else "")
            )
            if conf < 0.3:
                continue
            text = norm_text
            if text and len(text) >= 4:
                logger.info(f"[EasyOCR] '{text}' raw_conf={conf:.2f}")
                print(f"[EasyOCR] '{text}' conf={conf:.2f}")
                # Use the higher of EasyOCR's own confidence or alpr_conf
                final_conf = max(float(conf), alpr_conf)
                out.append((text, final_conf))
        return out
    except Exception as e:
        logger.warning(f"[EasyOCR] failed: {e}")
        return []


# ── Plate preprocessing ───────────────────────────────────────────────────────

def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask. Simple and effective — no deskew/perspective warp."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gray = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    return gray


# ── Upscale (EDSR → INTER_CUBIC fallback) ────────────────────────────────────

def _upscale_crop(crop: np.ndarray) -> np.ndarray:
    """EDSR x4 SR, or INTER_CUBIC fallback. Width-capped at 480px for single-line plates."""
    h, w   = crop.shape[:2]
    aspect = w / max(h, 1)
    sr     = _get_edsr()

    if sr:
        inp = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
        try:
            up = sr.upsample(inp)
            nh, nw = up.shape[:2]
            if aspect >= 2.8 and nw > 480:
                nw = 480
                nh = max(int(h * (480 / w)), 64)
                up = cv2.resize(up, (nw, nh), interpolation=cv2.INTER_AREA)
            return up
        except Exception as e:
            logger.debug(f"[Enhancer] EDSR upsample failed ({e}), using INTER_CUBIC")

    scale = max(4, int(128 / max(h, 1)))
    nw    = w * scale
    nh    = h * scale
    if aspect >= 2.8 and nw > 480:
        nw = 480
        nh = max(int(h * (480 / w)), 64)
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)


# ── Tight-crop OCR (fast-plate-ocr ensemble) ─────────────────────────────────

def _ocr_tight_crop(
    crop: np.ndarray, tag: str, alpr_conf: float
) -> list[tuple[str, float]]:
    """
    Run ensemble OCR (ViT + CCT-S) on a tight plate crop.
    Two-line plates (aspect < 2.0): split top/bottom and merge.
    Single-line plates (aspect >= 2.0): full crop only.
    """
    results        = []
    h_orig, w_orig = crop.shape[:2]
    aspect         = w_orig / max(h_orig, 1)
    upscale        = _upscale_crop(crop)
    new_h, new_w   = upscale.shape[:2]
    sr_mode        = "EDSR" if (_edsr_sr and _edsr_sr is not False) else "INTER_CUBIC"

    print(
        f"[ALPR-TIGHT] {tag} crop={h_orig}x{w_orig} "
        f"aspect={aspect:.1f} → upscaled={new_h}x{new_w} [{sr_mode}]"
    )

    preprocessed = _preprocess_plate(upscale)

    # A: full crop — raw SR + preprocessed, both models
    for label, img in [("full_raw", upscale), ("full_pre", preprocessed)]:
        candidates = _ocr_img(img, alpr_conf)
        for text, conf in candidates:
            print(f"[ALPR-TIGHT] {tag} {label} → '{text}' conf={conf:.2f}")
        results.extend(candidates)

    # B: top/bottom split — two-line plates only (aspect < 2.0)
    if aspect < 2.0:
        print(f"[ALPR-TIGHT] {tag} two-line plate (aspect={aspect:.1f}) — splitting top/bot")
        mid    = upscale.shape[0] // 2
        merged = ""
        hit    = 0
        half_confs = []

        for hlabel, half in [("top", upscale[:mid, :]), ("bot", upscale[mid:, :])]:
            if half.shape[0] < 8:
                continue
            half_pre   = _preprocess_plate(half)
            half_cands = _ocr_img(half_pre, alpr_conf) + _ocr_img(half, alpr_conf)
            # Give EasyOCR a vote too — it's an independently-trained model,
            # so it won't share ViT/CCT's specific character confusions.
            half_cands += _run_easyocr(half_pre, alpr_conf, use_detector=False)
            if half_cands:
                best_text, best_conf = max(half_cands, key=lambda x: x[1])
                if len(best_text) >= 2:
                    print(f"[ALPR-TIGHT] {tag} {hlabel} → '{best_text}' conf={best_conf:.2f}")
                    merged += best_text
                    hit    += 1
                    half_confs.append(best_conf)

        if hit > 0 and len(merged) >= 6:
            merged_conf = min(half_confs) if half_confs else alpr_conf
            print(f"[ALPR-TIGHT] {tag} merged → '{merged}' conf={merged_conf:.2f}")
            results.append((merged, merged_conf))
    else:
        print(f"[ALPR-TIGHT] {tag} single-line (aspect={aspect:.1f}) — skipping split")

    return results


# ── Main ALPR runner ──────────────────────────────────────────────────────────

def _run_alpr(frame: np.ndarray, save_dir: str, tag: str) -> list[tuple[str, float]]:
    alpr    = _get_alpr()
    results = []
    try:
        preds = alpr.predict(frame)
        if not preds:
            logger.info(f"[Enhancer] No plates detected [{tag}]")
            return []

        for i, r in enumerate(preds):
            if r.ocr is None:
                continue
            raw_text = _normalise(r.ocr.text)
            raw_conf = float(np.mean(r.ocr.confidence))
            print(f"[ALPR] tag={tag} raw → '{raw_text}' conf={raw_conf:.2f}")
            if len(raw_text) >= 4:
                results.append((raw_text, raw_conf))

            try:
                bb = r.detection.bounding_box
                if hasattr(bb, "x1"):
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                else:
                    x1, y1 = int(bb[0]), int(bb[1])
                    x2, y2 = int(bb[0]+bb[2]), int(bb[1]+bb[3])

                plate_crop = _safe_crop(frame, x1, y1, x2, y2, pad=4)
                if plate_crop is not None and plate_crop.size > 0:
                    cv2.imwrite(str(Path(save_dir)/f"plate_crop_{tag}_{i}.jpg"), plate_crop)

                    # OCR the JPEG-compressed crop, not the raw one — the small crop's
                    # low-pass block quantization from re-encoding measurably changes
                    # borderline character decisions here (confirmed: raw in-memory
                    # crop reads top-half as 'KA052' conf=0.88; the exact same crop
                    # re-read after a JPEG round-trip reads it correctly as 'KA05K'
                    # conf=0.89). Encode at the same default quality cv2.imwrite uses.
                    ok, encoded = cv2.imencode(".jpg", plate_crop)
                    if ok:
                        plate_crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

                    up = _upscale_crop(plate_crop)
                    cv2.imwrite(str(Path(save_dir)/f"plate_crop_upscaled_{tag}_{i}.jpg"), up)
                    cv2.imwrite(
                        str(Path(save_dir)/f"plate_crop_preprocessed_{tag}_{i}.jpg"),
                        _preprocess_plate(up),
                    )
                    results.extend(_ocr_tight_crop(plate_crop, f"{tag}_{i}", raw_conf))
            except Exception as e:
                logger.debug(f"[Enhancer] plate crop error [{tag}]: {e}")

    except Exception as e:
        logger.warning(f"[Enhancer] fast-alpr predict error [{tag}]: {e}")

    return results


# ── Best result selector ──────────────────────────────────────────────────────

def _best_result(candidates: list[tuple[str, float]]) -> tuple[Optional[str], float]:
    """
    Pick the best plate text from all candidates.
    Priority:
      1. Indian HSRP regex match — highest confidence wins
      2. Fallback: highest confidence overall
    """
    if not candidates:
        return None, 0.0

    seen  = set()
    dedup = []
    for text, conf in candidates:
        if text not in seen:
            seen.add(text)
            dedup.append((text, conf))

    indian = [(t, c) for t, c in dedup if _is_indian(t)]
    if indian:
        best = max(indian, key=lambda x: x[1])
        logger.info(f"[Enhancer] Indian format match → {best[0]} conf={best[1]:.2f}")
        return best

    return max(dedup, key=lambda x: x[1])


# ── Core Enhancer ─────────────────────────────────────────────────────────────

class Enhancer:

    def process_event(
        self,
        frame:       np.ndarray,
        person_bbox: Optional[tuple],
        person_id:   str,
        pair_id:     str,
        save_dir:    str                             = "evidence",
        frame_iter:  Optional[Iterator[np.ndarray]] = None,
        force_evidence_frame: Optional[np.ndarray]  = None,
    ) -> EnhancementResult:

        t0 = time.time()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        result = EnhancementResult(pair_id=pair_id)

        # ── Step 1: build candidate frames ───────────────────────────────
        frames_to_scan: list[tuple[int, np.ndarray]] = [(0, frame)]
        if frame_iter is not None:
            for offset, f in enumerate(frame_iter, start=1):
                frames_to_scan.append((offset, f))

        result.frames_scanned = len(frames_to_scan)
        logger.info(
            f"[Enhancer] Scanning {result.frames_scanned} frames "
            f"for pair={pair_id}"
        )

        # ── Step 2: find frame with best COMPOSITE plate score ────────────
        best_composite = -1.0
        best_blur      = -1.0
        best_offset    = 0
        best_frame     = frame

        for offset, scan_frame in frames_to_scan:
            region_blur, region_area = _plate_region_blur(scan_frame)
            composite = region_blur * np.log1p(region_area)
            if composite > best_composite:
                best_composite = composite
                best_blur      = region_blur
                best_offset    = offset
                best_frame     = scan_frame

        result.blur_score        = best_blur
        result.best_frame_offset = best_offset
        print(
            f"[BEST FRAME] offset=+{best_offset} "
            f"plate_region_blur={best_blur:.1f} "
            f"composite={best_composite:.1f} "
            f"shape={best_frame.shape}"
        )
        # Evidence photo is pinned to the throw moment if provided;
        # best_frame is still used for plate scanning (car may appear later)
        evidence_frame = force_evidence_frame if force_evidence_frame is not None else best_frame

        # ── Step 3: fast-alpr on full best frame ──────────────────────────
        all_candidates: list[tuple[str, float]] = []
        full_results = _run_alpr(best_frame, save_dir, tag=f"full_{best_offset}")
        all_candidates.extend(full_results)

        # ── Step 3b: EasyOCR fallback ─────────────────────────────────────
        # Triggered when fast-alpr confidence is low (< 0.70) OR no results.
        # EasyOCR handles perspective/angled plates much better than ViT/CCT.
        # Runs on both the raw plate crop and the upscaled version.
        best_full_conf = max((c for _, c in full_results), default=0.0)
        if not full_results or best_full_conf < 0.70:
            logger.info(
                f"[Enhancer] fast-alpr conf={best_full_conf:.2f} < 0.70 "
                "— trying EasyOCR fallback"
            )
            print(f"[Enhancer] EasyOCR fallback (fast-alpr conf={best_full_conf:.2f})")

            crop_paths = sorted(
                Path(save_dir).glob(f"plate_crop_full_{best_offset}_*.jpg")
            )
            up_paths = sorted(
                Path(save_dir).glob(f"plate_crop_upscaled_full_{best_offset}_*.jpg")
            )

            for crop_path in crop_paths:
                crop_img = cv2.imread(str(crop_path))
                if crop_img is not None:
                    easy = _run_easyocr(crop_img, best_full_conf)
                    all_candidates.extend(easy)

            for up_path in up_paths:
                up_img = cv2.imread(str(up_path))
                if up_img is not None:
                    easy = _run_easyocr(up_img, best_full_conf)
                    all_candidates.extend(easy)

            # If crops haven't been saved yet (no ALPR detections),
            # run EasyOCR directly on the best frame's plate region
            if not crop_paths and not up_paths:
                h, w = best_frame.shape[:2]
                plate_region = best_frame[int(h*0.4):, int(w*0.3):]
                easy = _run_easyocr(plate_region, best_full_conf)
                all_candidates.extend(easy)

        # ── Step 4: person-anchored crop fallback ─────────────────────────
        best_conf_so_far = max((c for _, c in all_candidates), default=0.0)
        if (not all_candidates or best_conf_so_far < 0.50) and person_bbox is not None:
            logger.info("[Enhancer] Still no good result — trying person-anchored crops")
            for ci, crop in enumerate(_person_anchored_crops(best_frame, person_bbox)):
                if crop.shape[0] < 30 or crop.shape[1] < 80:
                    continue
                cv2.imwrite(str(Path(save_dir)/f"debug_crop_{ci}.jpg"), crop)
                all_candidates.extend(
                    _run_alpr(crop, save_dir, tag=f"crop{ci}_{best_offset}")
                )

        # ── Step 5: pick best plate across all models & crop variants ────
        plate_text, plate_conf = _best_result(all_candidates)
        result.plate_text = plate_text
        result.plate_conf = plate_conf

        # ── Step 5b: debug annotated frame ───────────────────────────────
        try:
            preds       = _get_alpr().predict(best_frame)
            debug_frame = best_frame.copy()
            for r in preds:
                if r.detection is None:
                    continue
                bb = r.detection.bounding_box
                if hasattr(bb, "x1"):
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                else:
                    x1, y1 = int(bb[0]), int(bb[1])
                    x2, y2 = int(bb[0]+bb[2]), int(bb[1]+bb[3])
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_y = y1 - 6 if y1 > 20 else y1 + 20
                cv2.putText(
                    debug_frame,
                    f"{plate_text or 'NO PLATE'} {plate_conf:.0%}",
                    (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            debug_path = str(Path(save_dir)/f"debug_alpr_corrected_{best_offset}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            result.saved_paths.append(debug_path)
        except Exception as e:
            logger.debug(f"[Enhancer] debug image error: {e}")

        # ── Step 6: annotated evidence frame ─────────────────────────────
        # ── Step 6: annotated evidence frame ─────────────────────────────
        ev = evidence_frame.copy()
        if person_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in person_bbox]
            cv2.rectangle(ev, (px1, py1), (px2, py2), (0, 0, 255), 2)
            label_y = py1 - 8 if py1 > 20 else py1 + 25
            cv2.putText(
                ev, plate_text or "NO PLATE", (px1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                ev, plate_text or "NO PLATE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        ev_path = str(Path(save_dir)/f"{pair_id}_evidence.jpg")
        cv2.imwrite(ev_path, ev)
        result.saved_paths.append(ev_path)

        # ── Step 7: collect plate crop paths ─────────────────────────────
        result.saved_paths.extend([
            str(p) for p in sorted(Path(save_dir).glob("plate_crop_*.jpg"))
        ])

        result.elapsed_ms = int((time.time() - t0) * 1000)
        if plate_text:
            print(f"[Enhancer] 🚗 PLATE: {plate_text} (conf={plate_conf:.2f})\n")
        else:
            print(
                f"[Enhancer] ⚠️  No plate read | "
                f"plate_region_blur={best_blur:.1f} | "
                f"scanned={result.frames_scanned}f\n"
            )
        logger.info(
            f"[Enhancer] {'✅' if plate_text else '⚠️ '} "
            f"pair={pair_id} | plate={plate_text} conf={plate_conf:.2f} | "
            f"plate_region_blur={best_blur:.1f} best_frame=+{best_offset} "
            f"scanned={result.frames_scanned} | {result.elapsed_ms}ms"
        )
        return result