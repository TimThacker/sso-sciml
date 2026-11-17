"""Step 50: active learning on task T1.

    uv run scripts/phase13_active.py            # about 10 minutes
    uv run scripts/phase13_active.py --reuse    # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_t1.py. The training fold is the pool; a
case's label (the solver's growth rate) is revealed when the case is chosen, which
stands for one solver call. Starting from 500 random cases, labels are added in rounds
up to 8,000, by three rules:

    random     at random
    spread     the cases where an ensemble of three disagrees most
    boundary   the cases whose predicted sign is least certain: smallest
               |growth rate| / (spread + floor), the confidence of step 40

The model is the baseline arm R of step 09 (perceptron, 2,000 steps), an ensemble of
three. After each round it is scored on the test fold by its mean absolute error, its
error within 2 1/s of the boundary, and as a screen with the fallback rule of step 40
set on half of the validation fold. Three repeats with different starting sets.

Writes docs/results/phase13/active.json and fig122. Every number in
docs/steps/50-active-learning.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import fallback as fb
from sso_sciml import report
from sso_sciml.data import t1
from sso_sciml.models import compare as cmp
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "phase13"
COUNTS = (500, 1000, 2000, 4000, 8000)
RULES = ("random", "spread", "boundary")
MEMBERS, REPEATS, FACTOR = 3, (0, 1, 2), 1.5
CFG = mt.TrainConfig(steps=2000)
COLOUR = {"random": report.MUTED, "spread": report.SERIES[0], "boundary": report.SERIES[1]}


def fit(d: dict, seed: int, device: str):
    x = mt.inputs("R", d["theta"], d["mech"])
    y = mt.targets("R", d["sigma"], d["mech"])
    h = max(len(y) // 5, 1)
    nets = [mt.train_mlp(x[h:], y[h:], x[:h], y[:h], 100 * seed + m, CFG, device)[0]
            for m in range(MEMBERS)]  # fmt: skip

    def ens(dd):
        xx = mt.inputs("R", dd["theta"], dd["mech"])
        p = np.array([mt.to_sigma("R", mt.predict(n, xx, device), dd["mech"]) for n in nets])
        return p.mean(axis=0), p.std(axis=0)

    return ens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "active.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.reuse:
        pool, val, tests = cmp.sets(ROOT / "data" / t1.VERSION)
        pool = {k: v for k, v in pool.items() if k != "fold"}
        test = {k: v for k, v in tests["in_distribution"].items() if k != "fold"}
        half = val["sigma"].size // 2
        tune = {k: v[half:] for k, v in val.items() if k != "fold"}
        near = np.abs(test["sigma"]) < 2.0
        res: dict = {"counts": list(COUNTS), "pool": int(pool["sigma"].size), "runs": []}
        n_pool = pool["sigma"].size
        for rep in REPEATS:
            start = np.random.default_rng(500 + rep).permutation(n_pool)[: COUNTS[0]]
            for rule in RULES:
                chosen = start.copy()
                rows = []
                for k in range(len(COUNTS)):
                    sub = {key: v[chosen] for key, v in pool.items()}
                    ens = fit(sub, rep * 10 + k, device)
                    p, s = ens(test)
                    pt, st = ens(tune)
                    floor = float(np.median(st))
                    level = FACTOR * fb.threshold(fb.confidence(pt, st, floor), pt, tune["sigma"])
                    pol = fb.outcome(fb.decide(fb.confidence(p, s, floor), p, level),
                                     test["sigma"])  # fmt: skip
                    err = np.abs(p - test["sigma"])
                    rows.append({"labels": int(chosen.size), "mae": float(err.mean()),
                                 "mae_near_boundary": float(err[near].mean()),
                                 "unstable_share_of_labels": float((sub["sigma"] > 0).mean()),
                                 "near_share_of_labels": float((np.abs(sub["sigma"]) < 2).mean()),
                                 "policy": pol})  # fmt: skip
                    print(rep, rule, chosen.size, f"mae {err.mean():.3f}",
                          f"near {err[near].mean():.3f} solver {100 * pol['solver_share']:.1f}%",
                          f"missed {pol['missed_unstable']}", flush=True)  # fmt: skip
                    if k + 1 == len(COUNTS):
                        break
                    add = COUNTS[k + 1] - chosen.size
                    rest = np.setdiff1d(np.arange(n_pool), chosen)
                    if rule == "random":
                        rng = np.random.default_rng(1000 * rep + k)
                        pick = rng.choice(rest, add, replace=False)
                    else:
                        cand = {key: v[rest] for key, v in pool.items()}
                        pc, sc = ens(cand)
                        score = sc if rule == "spread" else -np.abs(pc) / (sc + floor)
                        pick = rest[np.argsort(score)[::-1][:add]]
                    chosen = np.concatenate([chosen, pick])
                res["runs"].append({"repeat": rep, "rule": rule, "rows": rows})
        path.write_text(json.dumps(res))
    res = json.loads(path.read_text())

    def series(rule, key):
        return np.array([[r[key] if not isinstance(key, tuple) else r[key[0]][key[1]]
                          for r in run["rows"]] for run in res["runs"] if run["rule"] == rule])

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    counts = np.array(res["counts"])
    panels = (("mae", "Mean absolute error, test fold (1/s)"),
              ("mae_near_boundary", "Within 2 1/s of the boundary (1/s)"),
              (("policy", "solver_share"), "Share sent to the solver by the rule"))  # fmt: skip
    for ax, (key, label) in zip(axes, panels, strict=True):
        for rule in RULES:
            v = series(rule, key) * (100 if isinstance(key, tuple) else 1)
            report.curves(ax, [(counts, row) for row in v], COLOUR[rule], rule)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Labels (solver calls)")
        ax.set_title(label, loc="left", fontsize=10)
        report.style(ax)
    axes[2].set_title("Sent to the solver by the rule (%)", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "Active learning on T1: three rules for choosing the next solver calls "
                 "(baseline arm, ensembles of three, three repeats)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig122_active.png", dpi=160)
    plt.close(fig)
    for rule in RULES:
        print(rule, "mae", series(rule, "mae").mean(axis=0).round(4).tolist(),
              "near", series(rule, "mae_near_boundary").mean(axis=0).round(4).tolist(),
              "solver %", (100 * series(rule, ("policy", "solver_share"))).mean(axis=0)
              .round(2).tolist(), "missed", series(rule, ("policy", "missed_unstable"))
              .sum(axis=0).tolist(),
              "near share of labels", series(rule, "near_share_of_labels").mean(axis=0)
              .round(3).tolist())  # fmt: skip


if __name__ == "__main__":
    main()
