"""
tools/tune_thresholds.py
=========================
Replaces guesswork in Layer4/config.py and Layer5/config.py with values
found by grid-searching against your labeled dataset (videos/ + labels/).

Because Layer 4 & 5 have no learnable weights (they're rule engines), "training"
them means: try combinations of their threshold constants, run the pipeline,
score against ground truth, keep the best combination.

NOTE: This is expensive — each grid point re-runs the ENTIRE video set.
Start small: --limit 40 --quick to sanity check before a full run.

Usage:
    python tools/tune_thresholds.py --videos videos/ --labels labels/ --limit 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Layer4.config as l4cfg
import Layer5.config as l5cfg

from tools.batch_eval import evaluate_dataset, summarize, ModelBundle

# ── Search space ────────────────────────────────────────────────────────────
# Keep this small at first. Each combination = one full pass over the dataset.
# Expand once you've confirmed the harness works end-to-end.
SEARCH_SPACE = {
    # Layer 4 — how close an object must land to a bin to count as legal
    ("Layer4", "BIN_NEAR_PX"):        [90, 120, 150, 180],
    # Layer 4 — speed after release that counts as a "throw"
    ("Layer4", "THROW_VEL_THRESHOLD"): [4.0, 5.0, 6.5],
    # Layer 4 — minimum speed after release to count as ANY motion at all.
    # Lowering this is what should help static (no-throw) dumping get picked up.
    ("Layer4", "MOTION_VEL_THRESHOLD"): [1.0, 1.5, 2.5],
    # Layer 4 — frames of independent post-release motion required before the
    # case is even evaluated. Static dumping may never satisfy a high value here.
    ("Layer4", "MIN_POST_RELEASE"): [1, 2, 4],
    # Layer 5 — minimum confidence before a verdict is acted on
    ("Layer5", "MIN_CONFIDENCE_TO_ACT"): [0.35, 0.45, 0.55],
    # Layer 5 — how close counts as "near the bin" for legal override
    ("Layer5", "BIN_LEGAL_RADIUS_PX"): [180, 210, 250],
}

CONFIG_MODULES = {"Layer4": l4cfg, "Layer5": l5cfg}


def apply_combo(combo: dict) -> dict:
    """Patch config modules in-place, return the previous values so we can restore."""
    previous = {}
    for (layer, key), value in combo.items():
        mod = CONFIG_MODULES[layer]
        previous[(layer, key)] = getattr(mod, key)
        setattr(mod, key, value)
    return previous


def restore(previous: dict) -> None:
    for (layer, key), value in previous.items():
        setattr(CONFIG_MODULES[layer], key, value)


def grid_search(videos_dir: str, labels_dir: str, limit: int | None, verbose: bool) -> dict:
    keys = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[k] for k in keys]
    combos = list(itertools.product(*value_lists))

    print(f"[Tune] {len(combos)} combinations to evaluate.")
    shared_models = ModelBundle()  # loaded ONCE for the whole grid search
    best = None  # (score, combo_dict, summary)
    all_runs = []

    for i, values in enumerate(combos):
        combo = dict(zip(keys, values))
        combo_readable = {f"{l}.{k}": v for (l, k), v in combo.items()}
        print(f"\n[Tune] Combo {i+1}/{len(combos)}: {combo_readable}")

        previous = apply_combo(combo)
        try:
            records = evaluate_dataset(videos_dir, labels_dir, do_calibrate=True,
                                        limit=limit, verbose=verbose, models=shared_models)
            summary = summarize(records)
        finally:
            restore(previous)

        # Score: prioritize F1 (balances false positives vs missed dumping),
        # tie-break on lower mean timestamp error.
        score = summary.get("f1", 0.0)
        print(f"[Tune]   -> accuracy={summary.get('accuracy')} f1={summary.get('f1')} "
              f"precision={summary.get('precision')} recall={summary.get('recall')}")

        all_runs.append({"combo": combo_readable, "summary": summary})

        if best is None or score > best[0]:
            best = (score, combo_readable, summary)

    return {"best_combo": best[1], "best_summary": best[2], "all_runs": all_runs}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--limit", type=int, default=60,
                    help="Videos per combo (keep small for the search, run full set to confirm the winner)")
    p.add_argument("--out", default="tuning_report.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    result = grid_search(args.videos, args.labels, args.limit, args.verbose)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== BEST COMBINATION ===")
    print(json.dumps(result["best_combo"], indent=2))
    print("\n=== BEST SUMMARY ===")
    print(json.dumps(result["best_summary"], indent=2))
    print(f"\nFull tuning report written to {args.out}")
    print("\nTo apply: manually copy these values into Layer4/config.py and Layer5/config.py")