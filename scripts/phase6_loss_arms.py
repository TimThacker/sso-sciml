"""Phase 6, task T1: the physics as a penalty (arm L) and with features too (F+L).

    uv run scripts/phase6_loss_arms.py            # 105 training runs, about an hour
    uv run scripts/phase6_loss_arms.py --reuse    # statistics and figures from the saved runs

Needs the dataset from scripts/phase3_generate_t1.py and the runs of
scripts/phase6_compare.py (arms N, F and R on the same subsets and seeds). The network
of step 13 predicts growth rate and frequency. The penalty is the squared smallest
singular value of A(theta) - lambda I on 10,000 cases whose state matrix is known and
whose eigenvalues are not.

    N2    parameters in, no penalty: arm N with the two outputs of arm L
    F2    parameters and hand-calculation quantities in, no penalty
    L     parameters in, penalty at a weight of 10 (the best of step 13)
    FL    parameters and hand-calculation quantities in, the same penalty
    L@w   arm L at another weight, at 100 and 1,000 labels
    L, 0  arm L with no labels at all

Four data budgets and five seeds; a seed fixes the same subset as in the other arms, so
differences are paired. Writes data/t1-v2/pool_matrices.npz (not committed),
docs/results/phase6/loss_arms.json and figures fig84 and fig85. Every number on arms L
and F+L in docs/steps/34-arm-comparison.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.models import compare as cmp
from sso_sciml.models import eigen
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase6"
N_POOL = 10000
WEIGHT = 10.0
OTHER_WEIGHTS = (1.0, 100.0)
# arm: (inputs, weight of the penalty)
ARMS = {"N2": ("N", 0.0), "F2": ("F", 0.0), "L": ("N", WEIGHT), "FL": ("F", WEIGHT)}
PAIRS = (("L", "N"), ("L", "N2"), ("FL", "F"), ("FL", "F2"), ("FL", "L"), ("N2", "N"),
         ("L", "R"))  # fmt: skip
COLOUR = {**report.ARM_COLOUR, "L": report.SERIES[4], "FL": report.SERIES[5]}
NAME = {**report.ARM_NAME, "L": "L: physics as a penalty", "FL": "F+L: features and penalty"}


def pool_matrices(theta: np.ndarray) -> np.ndarray:
    path = DATA / "pool_matrices.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["theta"], theta):
            return z["a"]
    a = data.state_matrices(theta, processes=8)
    np.savez_compressed(path, theta=theta, a=a)
    return a


def train_all(device: str) -> dict:
    train, val, tests = cmp.sets(DATA)
    pool = mt.budget_subset(train, N_POOL, 0)
    t0 = time.perf_counter()
    pool["a"] = pool_matrices(pool["theta"])
    res: dict = {"pool": N_POOL, "weight": WEIGHT, "matrices_s": time.perf_counter() - t0,
                 "runs": []}  # fmt: skip
    jobs = [(arm, kind, w, b) for b in cmp.BUDGETS for arm, (kind, w) in ARMS.items()]
    jobs += [(f"L@{w:g}", "N", w, b) for b in (100, 1000) for w in OTHER_WEIGHTS]
    jobs += [("L", "N", 1.0, 0)]
    for arm, kind, weight, budget in jobs:
        for seed in cmp.SEEDS:
            lab = held = None
            if budget:
                sub = mt.budget_subset(train, budget, seed)
                both = {"x": mt.inputs(kind, sub["theta"], sub["mech"]), "sigma": sub["sigma"],
                        "f": sub["f"]}  # fmt: skip
                h = budget // 5
                held = {k: v[:h] for k, v in both.items()}
                lab = {k: v[h:] for k, v in both.items()}
            unl = None
            if weight > 0:
                unl = {"x": mt.inputs(kind, pool["theta"], pool["mech"]), "a": pool["a"]}
            cfg = eigen.EigenConfig(weight=weight, form="singular")
            model, info = eigen.train_eigen(lab, held, unl, seed, cfg, device=device)

            def sigma_of(d, kind=kind, model=model):
                x = mt.inputs(kind, d["theta"], d["mech"])
                return eigen.predict(model, x, device)["sigma"]

            margin = cmp.safe_margin(sigma_of(val), val["sigma"])
            res["runs"].append({
                "family": "mlp", "arm": arm, "budget": budget, "seed": seed,
                "fit_s": info["train_s"], "margin": margin,
                "scores": {k: cmp.scores(v["sigma"], sigma_of(v), margin)
                           for k, v in tests.items()},
            })  # fmt: skip
            s = res["runs"][-1]["scores"]["in_distribution"]
            print(arm, budget, seed, f"mae {s['mae']:.3f} missed {s['missed_unstable']:.3f}",
                  f"{info['train_s']:.0f} s", flush=True)  # fmt: skip
    return res


def statistics(res: dict, others: list[dict]) -> None:
    runs = res["runs"] + [r for r in others if r["family"] == "mlp"]

    def get(arm, budget, test, metric) -> np.ndarray:
        rows = sorted((r for r in runs if (r["arm"], r["budget"]) == (arm, budget)),
                      key=lambda r: r["seed"])  # fmt: skip
        if metric in ("margin", "fit_s"):
            return np.array([r[metric] for r in rows])
        return np.array([r["scores"][test][metric] for r in rows], dtype=float)

    summary, paired = {}, {}
    arms = sorted({(r["arm"], r["budget"]) for r in runs})
    for arm, budget in arms:
        entry = {"fit_s": float(get(arm, budget, "", "fit_s").mean()),
                 "margin": float(get(arm, budget, "", "margin").mean())}  # fmt: skip
        for test in cmp.TESTS:
            entry[test] = {}
            for m in ("mae", "mae_near_boundary", "missed_unstable", "false_alarm", "cleared",
                      "unstable_cleared"):  # fmt: skip
                v = get(arm, budget, test, m)
                entry[test][m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        summary[f"{arm}/{budget}"] = entry
    pairs = [*PAIRS, *((f"L@{w:g}", "N2") for w in OTHER_WEIGHTS)]
    for a, b in pairs:
        for budget in cmp.BUDGETS:
            if (a, budget) not in arms or (b, budget) not in arms:
                continue
            paired[f"{a}-{b}/{budget}"] = {
                test: {
                    "mae_change": cmp.change(get(a, budget, test, "mae"),
                                             get(b, budget, test, "mae")),
                    **{m: cmp.interval(get(a, budget, test, m) - get(b, budget, test, m))
                       for m in ("missed_unstable", "cleared")},
                } for test in cmp.TESTS
            }  # fmt: skip
    res["summary"], res["paired"] = summary, paired


def bars(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([100 * r["mean"] for r in rows])
    err = np.array([[100 * (r["mean"] - r["low"]), 100 * (r["high"] - r["mean"])] for r in rows]).T
    return mean, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    path = OUT / "loss_arms.json"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        t0 = time.perf_counter()
        res = train_all("cuda" if torch.cuda.is_available() else "cpu")
        res["wall_s"] = time.perf_counter() - t0
    statistics(res, json.loads((OUT / "comparison.json").read_text())["runs"])
    path.write_text(json.dumps(res, indent=2))
    summary, paired = res["summary"], res["paired"]
    x = np.arange(len(cmp.BUDGETS))

    # ---------------- figure: the five arms against budget, and the paired changes
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), facecolor=report.SURFACE)
    for arm in ("N", "F", "R", "L", "FL"):
        for ax, metric in zip(axes[:2], ("mae", "missed_unstable"), strict=True):
            rows = [summary[f"{arm}/{n}"]["in_distribution"][metric] for n in cmp.BUDGETS]
            mean, sd = (np.array([r[k] for r in rows]) for k in ("mean", "sd"))
            scale = 100 if metric == "missed_unstable" else 1
            ax.errorbar(x, scale * mean, yerr=scale * sd, color=COLOUR[arm], marker="o", ms=5,
                        lw=2, capsize=3, label=NAME[arm])  # fmt: skip
    axes[0].set_ylabel("Mean absolute error in growth rate (1/s)")
    axes[1].set_ylabel("Unstable cases called stable (%)")
    for ax in axes[:2]:
        ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    shown = (("L", "N", COLOUR["L"]), ("FL", "F", COLOUR["FL"]), ("L", "R", report.MUTED))
    for k, (a, b, colour) in enumerate(shown):
        mean, err = bars([paired[f"{a}-{b}/{n}"]["in_distribution"]["mae_change"]
                          for n in cmp.BUDGETS])  # fmt: skip
        axes[2].errorbar(x + (k - 1) * 0.12, mean, yerr=err, color=colour, marker="o", ms=5,
                         lw=1.5, capsize=3, label=f"{a} against {b}")  # fmt: skip
    axes[2].axhline(0, color=report.INK, lw=0.8, ls="--")
    axes[2].set_ylabel("Change of the mean error, paired (%)")
    axes[2].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    for ax in axes:
        ax.set_xticks(x, [f"{n:,}" for n in cmp.BUDGETS])
        ax.set_xlabel("Labelled cases")
        report.style(ax)
    report.title(fig, "Task T1, perceptron, in distribution: five arms over five seeds "
                 "(bars: one standard deviation; right, 95 % interval)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig84_five_arms.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: every test set, and the weight of the penalty
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    ypos = np.arange(len(cmp.TESTS))[::-1]
    for ax, budget in zip(axes[:2], (1000, 100000), strict=True):
        for k, (a, b, colour) in enumerate(shown[:2]):
            mean, err = bars([paired[f"{a}-{b}/{budget}"][t]["mae_change"] for t in cmp.TESTS])
            ax.errorbar(mean, ypos + (0.5 - k) * 0.24, xerr=err, color=colour, marker="o", ms=5,
                        lw=0, elinewidth=1.5, capsize=3, label=f"{a} against {b}")  # fmt: skip
        ax.axvline(0, color=report.INK, lw=0.8, ls="--")
        ax.set_yticks(ypos, [t.replace("_", " ") for t in cmp.TESTS] if budget == 1000 else [])
        ax.set_xlabel("Change of the mean error, paired (%)")
        ax.set_title(f"{budget:,} labelled cases", loc="left", fontsize=10)
        report.style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    weights = (0.0, OTHER_WEIGHTS[0], WEIGHT, OTHER_WEIGHTS[1])
    names = ("N2", f"L@{OTHER_WEIGHTS[0]:g}", "L", f"L@{OTHER_WEIGHTS[1]:g}")
    for budget, colour in zip((100, 1000), report.SERIES, strict=False):
        rows = [summary[f"{n}/{budget}"]["in_distribution"]["mae"] for n in names]
        axes[2].errorbar(range(len(weights)), [r["mean"] for r in rows],
                         yerr=[r["sd"] for r in rows], color=colour, marker="o", ms=5, lw=2,
                         capsize=3, label=f"{budget:,} labels")  # fmt: skip
    axes[2].set_xticks(range(len(weights)), [f"{w:g}" for w in weights])
    axes[2].set_xlabel("Weight of the penalty")
    axes[2].set_ylabel("Mean absolute error in growth rate (1/s)")
    axes[2].set_yscale("log")
    axes[2].set_title("In distribution, against the weight", loc="left", fontsize=10)
    axes[2].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(axes[2])
    report.title(fig, "The penalty on every test set, and its weight")
    fig.tight_layout()
    fig.savefig(OUT / "fig85_penalty_splits.png", dpi=160)
    plt.close(fig)

    for key, entry in paired.items():
        c = entry["in_distribution"]
        print(f"{key:14s} mae {100 * c['mae_change']['mean']:+6.1f}% "
              f"[{100 * c['mae_change']['low']:+6.1f}, {100 * c['mae_change']['high']:+6.1f}] "
              f"missed {100 * c['missed_unstable']['mean']:+5.2f} pts "
              f"{'*' if c['mae_change']['differs'] else ''}")  # fmt: skip
    for key, entry in summary.items():
        d = entry["in_distribution"]
        print(f"{key:12s} mae {d['mae']['mean']:.3f} ± {d['mae']['sd']:.3f} "
              f"missed {100 * d['missed_unstable']['mean']:.1f} % "
              f"cleared {100 * d['cleared']['mean']:.1f} % ({d['unstable_cleared']['mean']:.1f}) "
              f"fit {entry['fit_s']:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
