# Evaluation & Threshold-Tuning Harness

These scripts let you check the full Layer 1-5 pipeline against your labeled
dataset, and auto-tune Layer 4/5's hand-picked rule thresholds instead of
guessing them.

## 1. Get the dataset onto your machine

Your `videos/`, `labels/`, and `FOO_TEST/` folders are in Google Drive. Easiest
path: open each folder in Drive → right-click → **Download** (Drive zips it
automatically). Unzip so you end up with, next to `run_pipeline.py`:

```
Illegal_dumping/
├── videos/          <- vid0011.mp4, vid0202.mp4, ...
├── labels/           <- vid0011.json, vid0202.json, ...
├── run_pipeline.py
├── Layer1/ ... Layer5/
└── tools/
    ├── batch_eval.py
    ├── tune_thresholds.py
    └── README.md   (this file)
```

## 2. Sanity check on a handful of videos first

```bash
python tools/batch_eval.py --videos videos/ --labels labels/ --limit 15 --out quick_report.json
```

This runs Layers 1-5 headlessly (no window, no OCR, no email/challan) on 15
videos and compares the predicted verdict + timestamp against the label JSON.
Check `quick_report.json` — if `accuracy` is near 0 or every video errors out,
something's off with paths/weights before you commit to a full run.

## 3. Full evaluation (baseline, before any tuning)

```bash
python tools/batch_eval.py --videos videos/ --labels labels/ --out baseline_report.json
```

This is your starting point — accuracy/precision/recall/F1, plus a breakdown
by day/night and static/dynamic dumping, using the *current* hand-picked
thresholds in `Layer4/config.py` and `Layer5/config.py`.

## 4. Tune the thresholds against the labels

```bash
python tools/tune_thresholds.py --videos videos/ --labels labels/ --limit 60
```

This grid-searches a small set of the most impactful constants (bin-proximity
radius, throw-velocity threshold, minimum confidence to act, etc. — see
`SEARCH_SPACE` at the top of the script) and reports the combination that
scores best (F1) against your ground truth.

- `--limit 60` evaluates each grid combination on 60 videos to keep the search
  fast. Once you have a winning combination, re-run `batch_eval.py` on the
  **full** dataset with that combination applied to confirm it holds up.
- Expand `SEARCH_SPACE` in `tune_thresholds.py` to search more constants —
  just know the number of combinations multiplies fast, so add one at a time.

## 5. Apply the winning thresholds

The script does **not** edit your source automatically — it prints the best
combination and writes it to `tuning_report.json`. Copy those values into
`Layer4/config.py` / `Layer5/config.py` by hand, then re-run step 3 on the
full dataset to confirm the improvement.

## What this does and doesn't do

- ✅ Tunes Layer 4 (`DumpingInference`) and Layer 5 (`DumpingAgent`) rule
  thresholds using real labeled outcomes instead of guessed constants.
- ✅ Gives you an honest accuracy/F1 number broken down by edge case
  (day/night, static/dynamic dumping) — exactly what this dataset is suited for.
- ❌ Does **not** retrain `rtdetr-l.pt` or `trash_bin_detector.pt` (Layer 1) —
  that needs bounding-box labels, which this dataset doesn't have.
- ❌ Does **not** retrain the ReID model (Layer 2) — needs track-level identity
  crops, also not in this dataset.