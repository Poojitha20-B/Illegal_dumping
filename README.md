# A Causal, Agentic AI Framework for Illegal Dumping Detection in Video

VidTrace is a modular surveillance framework that detects illegal dumping in public spaces using computer vision, temporal reasoning, and an LLM-driven agentic reasoning layer, and closes the loop with automated enforcement.

The system processes CCTV streams or recorded footage, tracks people/vehicles and objects across frames, determines whether disposal behaviour is legal or illegal, identifies the violator (vehicle plate OCR or face recognition), and automatically generates a challan (penalty notice) with evidence extraction and email delivery — plus a dashboard for reviewing violations and hotspots.

---

## ✨ Features

- Illegal dumping detection from recorded or live video
- Multi-object tracking with identity preservation (ByteTrack + ReID)
- Trash throw and slow-drop detection, bin-aware disposal reasoning
- Temporal memory & behavioural feature extraction
- ROI recovery for lost tracks
- **Layer 5 agentic controller**: a single-LLM-call-per-case belief/attribution agent (Groq · Llama-3.3-70B) that reasons over calibrated physical/kinematic evidence — not hardcoded rules — to confirm or reject candidate violations and resolve spurious person↔object attributions
- Vehicle plate detection & OCR (fast-ALPR + ensemble OCR + EasyOCR fallback, EDSR super-resolution)
- **Pedestrian violator identification** via face recognition (InsightFace `buffalo_l`) when no plate is available
- Automated PDF challan generation with UPI QR codes
- Email notification & escalation workflow (with retry handling)
- Hotspot tracking (repeat-violation locations) backed by SQLite
- Web dashboard (Flask) for browsing violations, evidence, and hotspots
- Batch evaluation & threshold-tuning harness against a labeled dataset
- Modular layer-based architecture (Layers 1–5, independently testable)

---

# 🏗️ System Architecture

```text
Video Stream / CCTV Feed
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 1 — Perception & Detection              │
│ RT-DETR + Trash Detection + Bin Detection     │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 2 — Multi-Object Tracking               │
│ ByteTrack + ReID + ROI Recovery               │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 3 — Memory & Feature Extraction         │
│ Sliding Window + Bin Interaction Features     │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 4 — Dumping Inference                   │
│ Context-Aware Temporal Evaluation             │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Layer 5 — Agentic Belief Controller           │
│ Single LLM-call-per-case (Groq / Llama-3.3)   │
│ Belief state, attribution, intent scoring     │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Enforcement Subsystem                         │
│ Enhancer (Plate OCR) / FaceID → Challan       │
│ → Email Delivery → Hotspot Tracking           │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│ Dashboard (Flask) — review & escalate         │
└──────────────────────────────────────────────┘
```

---

# 🧠 Core Pipeline

## Layer 1 — Perception & Detection
- RT-DETR based object detection
- Slow-drop and fast-throw trash detection
- Dedicated trash-bin detector
- Frame calibration & preprocessing

## Layer 2 — Multi-Object Tracking
- Pure NumPy ByteTrack implementation
- ReID embeddings for identity consistency
- ROI recovery for uncertain tracks
- Stable trash and bin tracking

## Layer 3 — Memory & Feature Extraction
- Sliding-window temporal memory
- Trash↔bin interaction modelling
- Behavioural feature vector extraction

## Layer 4 — Dumping Inference
Classifies events into:
- `legal_disposal`
- `illegal_dumping`
- `pending`

Uses release behaviour, bin proximity, object trajectory, and post-release motion analysis. A logistic-regression scorer trained on five evidence features assists final confidence scoring.

## Layer 5 — Agentic Belief Controller
Final reasoning layer that validates or overrides Layer 4 decisions. Redesigned around a **single LLM call per case** (Groq `llama-3.3-70b-versatile`) instead of an intermediate hand-coded state machine:
- Maintains a running belief state (confidence, phase, reasoning) per person↔object pair
- Receives calibrated kinematic evidence (motion coupling, divergence, rest-onset, velocity spikes) — the LLM reasons over raw signal crossings rather than being told pre-baked conclusions
- Resolves spurious person-track attribution (e.g. rejecting incidental nearby tracks in favour of the actual violator) using relative-size gating and possession/divergence analysis
- Deduplicates multi-ID events belonging to a single physical violation
- Produces a final verdict + reasoning log used to decide whether a challan is issued

---

# 🚨 Enforcement Pipeline

## Enhancer (Layer 1.5 — Plate Detection)
Triggered only when Layer 4 fires `illegal_dumping` on a vehicle:
- Best-frame scan using composite plate scoring (blur × region area)
- fast-ALPR plate detection + super-resolution (EDSR ×4) upscaling
- Ensemble OCR: `global-plates-mobile-vit-v2-model` + `cct-s-v2-global-model`, with an EasyOCR fallback when ensemble confidence < 0.70
- Falls back to person-anchored crops if no plate is found in the full frame

## FaceID Module (Pedestrian Violators)
Triggered when Layer 5 confirms a violation but no vehicle plate was detected:
- Detects and embeds faces with InsightFace (`buffalo_l`)
- Matches against an enrolled-persons database (SQLite)
- On match: creates a challan and emails the violator directly
- On no match: logs to an `unknown_violations` table and saves the face crop for later review

## Penalty Manager
- SQLite-based challan management
- Automated PDF generation with dynamic UPI QR codes
- Escalation handling for overdue challans

## Delivery Agent
- Email notification system (SMTP)
- Escalation reminders with retry mechanism for failed sends

## Hotspot Manager
- Tracks locations with repeat violations in SQLite
- Feeds the dashboard's hotspot view

## Dashboard
- Flask app (`dashboard/app.py`) for browsing violations, evidence images/PDFs, and hotspots
- Runs on `http://localhost:5050`

---

# 🚀 Setup

## Prerequisites

- Python 3.10+
- OpenCV-compatible environment
- CUDA GPU recommended (CPU works but is slower, especially for OCR/ReID/FaceID)
- A [Groq API key](https://console.groq.com) for the Layer 5 agentic controller
- A Gmail account + [app password](https://support.google.com/accounts/answer/185833) for email delivery (or your own SMTP provider)

---

## Installation

```bash
git clone https://github.com/Poojitha20-B/Illegal_dumping.git

cd Illegal_dumping

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```


---

## Environment Variables

Create a `.env` file in the project root (auto-loaded by `python-dotenv`):

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Email delivery (`delivery_agent.py`, `FaceID/notifier.py`) currently reads SMTP credentials from constants at the top of those files (`SMTP_USER`, `SMTP_PASSWORD`, `SENDER_EMAIL`, `SENDER_NAME`) rather than the environment — open each file and set your own Gmail address + app password before running anything that sends email. **Do not commit real credentials** to the repo.

---

# 📦 Model Weights

Place the required weights inside the `weights/` directory (or project root, where noted):

```text
weights/
├── rtdetr-l.pt              # Layer 1 — object detection
└── trash_bin_detector.pt    # Layer 1 — bin detector (included in this repo)
```

Additional models downloaded/cached on first use, with download links:

| Model | Used by | Notes |
|---|---|---|
| [`trash_bin_detector.pt`](https://github.com/Poojitha20-B/Illegal_dumping/blob/main/weights/trash_bin_detector.pt) | Layer 1 | Bundled in this repo at `weights/trash_bin_detector.pt` — no separate download needed |
| [`rtdetr-l.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt) | Layer 1 | Not bundled. Either download directly and place in `weights/`, or let `ultralytics` auto-download it on first run (`from ultralytics import RTDETR; RTDETR("rtdetr-l.pt")`) |
| [`EDSR_x4.pb`](https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb) | Enhancer (plate super-resolution) | `curl -L -o EDSR_x4.pb https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb` — place in project root |
| [`cct-s-v2-global-model`](https://github.com/ankandrew/fast-plate-ocr) | Enhancer (OCR ensemble) | Auto-downloaded (~5 MB) via `fast_plate_ocr` on first run |
| [`buffalo_l`](https://github.com/deepinsight/insightface/tree/master/model_zoo) | FaceID module | Auto-downloaded by `insightface` on first run |

---

# ▶️ Running the Project

### Process a video file
```bash
python run_pipeline.py --source test2.mp4
```

### Use a live camera / stream
```bash
python run_pipeline.py --source 0
```

### Save processed output video
```bash
python run_pipeline.py --source test2.mp4 --save
```

### Run with a custom violation location (written into the challan)
```bash
python run_pipeline.py --source test2.mp4 --save --location "MG Road, Bengaluru"
```

### Skip auto-calibration
```bash
python run_pipeline.py --source test2.mp4 --no-calibrate
```

### Debug mode (verbose logging)
```bash
python run_pipeline.py --source test2.mp4 --debug
```

### Launch the dashboard
```bash
python dashboard/app.py
# Serves at http://localhost:5050
```

---

# ⚡ Penalty Escalation Simulation

Simulate overdue challan escalation directly from the terminal:

```bash
python -c "
from penalty_manager import PenaltyManager
pm = PenaltyManager()
pm.simulate_days_passed('BBMP-VH-KA05KK5546-1DE0619F', 10)
"
```

This applies escalation rules and updates the challan amount based on overdue duration.

---

# 🧪 Evaluation & Threshold Tuning

`tools/` contains a harness for checking the full Layer 1–5 pipeline against a labeled dataset and auto-tuning Layer 4/5's rule thresholds instead of hand-guessing them. See `tools/Readme.md` for the full walkthrough. Quick start:

```bash
# 1. Sanity check on a handful of videos (headless — no window/OCR/email/challan)
python tools/batch_eval.py --videos videos/ --labels labels/ --limit 15 --out quick_report.json

# 2. Full baseline evaluation (accuracy/precision/recall/F1, incl. day/night + static/dynamic breakdown)
python tools/batch_eval.py --videos videos/ --labels labels/ --out baseline_report.json

# 3. Grid-search key thresholds against ground truth
python tools/tune_thresholds.py --videos videos/ --labels labels/ --limit 60
```

Tuning does **not** edit source automatically — it writes the best combination to `tuning_report.json`; copy the winning values into `Layer4/config.py` / `Layer5/config.py` by hand, then re-run the full evaluation to confirm the improvement.

This tunes Layer 4/Layer 5 rule thresholds only — it does not retrain the Layer 1 detectors (`rtdetr-l.pt`, `trash_bin_detector.pt`) or the Layer 2 ReID model, both of which need labels this dataset doesn't provide.

---

# 📊 Dataset

Evaluation uses the **MIVIA IWDD 2026** dataset (University of Salerno). Expected layout, alongside `run_pipeline.py`:

```text
Illegal_dumping/
├── videos/          <- vid0001.mp4, vid0002.mp4, ...
├── labels/           <- vid0001.json, vid0002.json, ...   (bundled in this repo)
└── tools/
```

`labels/` (timestamp-matched ground truth) is included in this repo; the corresponding `videos/` are not — download them separately from Google Drive and place them alongside `labels/`:

- [Dataset / videos — folder 1](https://drive.google.com/drive/u/1/folders/1O8KncUX0MN-p7CAxTfrhnb9bjVrDZZjl)
- [Dataset / videos — folder 2](https://drive.google.com/drive/u/1/folders/1ugWhKRfhEUI3tki7zJMtqgRtQlVIl49e)
- [Annotations / labels](https://drive.google.com/drive/u/1/folders/102g9UgWcJNCaRRoStdw9vznRb3YNRM64)

Access to these folders is restricted — request access if you land on a permissions page. Once downloaded, unzip so the folder names/contents match the `videos/` and `labels/` layout above.

---

# 📂 Project Structure

```text
Illegal_dumping/
│
├── run_pipeline.py          # Entry point — orchestrates Layers 1-5 + enforcement
├── enhancer.py               # Layer 1.5 — plate detection & OCR
├── penalty_manager.py        # Challan DB, PDF generation, escalation
├── delivery_agent.py         # Email delivery + retry/escalation
│
├── Layer1/                   # Perception & detection
├── Layer2/                   # Multi-object tracking
├── Layer3/                   # Memory & feature extraction
├── Layer4/                   # Dumping inference (+ trained scorer weights)
├── Layer5/                   # Agentic belief controller (LLM)
│
├── FaceID/                   # Pedestrian violator identification
├── hotspot/                  # Repeat-violation location tracking
├── dashboard/                # Flask review dashboard
├── tools/                    # Batch evaluation & threshold tuning
│
├── weights/                  # Model weights
├── labels/                   # Ground-truth labels (MIVIA IWDD 2026)
├── evidence/                 # Saved violation evidence (crops, plates, faces)
└── challans/                 # Generated PDF challans
```

---

# 🧪 Tech Stack

| Component | Technology |
|---|---|
| Detection | RT-DETR (Ultralytics) |
| Tracking | ByteTrack (custom NumPy) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| ReID | TorchReID |
| Agentic Reasoning | Groq — Llama-3.3-70B-Versatile |
| Plate OCR | fast-ALPR, fast-plate-ocr (ViT + CCT-S ensemble), EasyOCR, EDSR super-resolution |
| Face Recognition | InsightFace (`buffalo_l`) |
| Database | SQLite |
| PDF Generation | ReportLab |
| Web Dashboard | Flask |
| Scheduling | APScheduler |

---

# 🌍 Applications

- Smart city surveillance
- Municipal sanitation monitoring
- Public space monitoring
- Railway & bus station surveillance
- Campus & gated community monitoring

---

# ⚠️ Note

Detection accuracy and enforcement reliability may vary depending on:
- video quality
- lighting conditions
- camera angle
- object visibility
- environmental occlusions
