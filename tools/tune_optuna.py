"""
tools/tune_optuna.py
======================
Replaces the brute-force grid search with Bayesian optimization (Optuna).
Same idea as tune_thresholds.py — patch Layer4/Layer5 config constants, run
the pipeline, score against ground truth — but Optuna picks the next
combination intelligently instead of exhausting every grid point, so you can
search a wider, continuous range in far fewer runs.

Objective: F-beta with beta=0.5 (precision weighted 2x recall) by default —
i.e. it optimizes specifically for FEWER FALSE POSITIVES, not just "best
accuracy". Change --beta if you want a different tradeoff:
    beta < 1  -> prioritizes precision (fewer false positives)   [default 0.5]
    beta = 1  -> balanced (F1, same as before)
    beta > 1  -> prioritizes recall (fewer missed dumping events)

Install:
    pip install optuna

Usage:
    python tools/tune_optuna.py --videos videos/ --labels labels/ --n_trials 40 --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import Layer4.config as l4cfg
import Layer5.config as l5cfg

from tools.batch_eval import evaluate_dataset, summarize, ModelBundle

CONFIG_MODULES = {"Layer4": l4cfg, "Layer5": l5cfg}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def apply_params(params: dict) -> dict:
    previous = {}
    for (layer, key), value in params.items():
        mod = CONFIG_MODULES[layer]
        previous[(layer, key)] = getattr(mod, key)
        setattr(mod, key, value)
    return previous


def restore(previous: dict) -> None:
    for (layer, key), value in previous.items():
        setattr(CONFIG_MODULES[layer], key, value)


def make_objective(videos_dir: str, labels_dir: str, limit: int | None, beta: float, verbose: bool,
                    models: ModelBundle):
    def objective(trial: optuna.Trial) -> float:
        params = {
            ("Layer4", "BIN_NEAR_PX"):          trial.suggest_int("BIN_NEAR_PX", 70, 220),
            ("Layer4", "THROW_VEL_THRESHOLD"):   trial.suggest_float("THROW_VEL_THRESHOLD", 3.0, 8.0),
            ("Layer4", "MOTION_VEL_THRESHOLD"):  trial.suggest_float("MOTION_VEL_THRESHOLD", 0.5, 3.0),
            ("Layer4", "MIN_POST_RELEASE"):      trial.suggest_int("MIN_POST_RELEASE", 1, 6),
            ("Layer4", "MIN_HELD_FRAMES"):       trial.suggest_int("MIN_HELD_FRAMES", 5, 25),
            ("Layer5", "MIN_CONFIDENCE_TO_ACT"): trial.suggest_float("MIN_CONFIDENCE_TO_ACT", 0.3, 0.75),
            ("Layer5", "BIN_LEGAL_RADIUS_PX"):   trial.suggest_int("BIN_LEGAL_RADIUS_PX", 140, 300),
        }

        previous = apply_params(params)
        try:
            records = evaluate_dataset(videos_dir, labels_dir, do_calibrate=True,
                                        limit=limit, verbose=verbose, models=models)
            summary = summarize(records)
        finally:
            restore(previous)

        score = fbeta(summary.get("precision", 0.0), summary.get("recall", 0.0), beta)

        # stash extra info on the trial so we can inspect it later without re-running
        trial.set_user_attr("accuracy", summary.get("accuracy"))
        trial.set_user_attr("precision", summary.get("precision"))
        trial.set_user_attr("recall", summary.get("recall"))
        trial.set_user_attr("f1", summary.get("f1"))
        trial.set_user_attr("fp", summary.get("fp"))
        trial.set_user_attr("fn", summary.get("fn"))

        print(f"[trial {trial.number}] fbeta={score:.3f} "
              f"precision={summary.get('precision')} recall={summary.get('recall')} "
              f"fp={summary.get('fp')} fn={summary.get('fn')}")
        return score

    return objective


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--limit", type=int, default=60,
                    help="Videos per trial. Keep small for the search, confirm winner on full set.")
    p.add_argument("--beta", type=float, default=0.5,
                    help="F-beta: <1 favors precision/fewer false positives (default 0.5), 1=balanced, >1 favors recall")
    p.add_argument("--out", default="optuna_report.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    shared_models = ModelBundle()  # loaded ONCE for the entire 40-trial study

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_objective(args.videos, args.labels, args.limit, args.beta, args.verbose, shared_models),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    result = {
        "beta": args.beta,
        "best_score": best.value,
        "best_params": best.params,
        "best_metrics": {
            "accuracy": best.user_attrs.get("accuracy"),
            "precision": best.user_attrs.get("precision"),
            "recall": best.user_attrs.get("recall"),
            "f1": best.user_attrs.get("f1"),
            "fp": best.user_attrs.get("fp"),
            "fn": best.user_attrs.get("fn"),
        },
        "all_trials": [
            {"number": t.number, "params": t.params, "score": t.value, **t.user_attrs}
            for t in study.trials
        ],
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== BEST PARAMS (highest F-beta, beta={}) ===".format(args.beta))
    print(json.dumps(best.params, indent=2))
    print("\n=== BEST METRICS ===")
    print(json.dumps(result["best_metrics"], indent=2))
    print(f"\nFull report written to {args.out}")
    print("\nTo apply: copy these values into Layer4/config.py and Layer5/config.py by hand,")
    print("then re-run tools/batch_eval.py on the FULL dataset to confirm the improvement holds.")