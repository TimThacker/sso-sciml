"""Phase 6: where the physics goes, with paired statistics.

    uv run scripts/phase6_compare.py            # 120 training runs, about 10 minutes
    uv run scripts/phase6_compare.py --reuse    # statistics and figures from the saved runs

Needs the dataset from scripts/phase3_generate_t1.py. Arms N, F and R of task T1 (step
09: same perceptron, same trees, same schedule) at four data budgets and five seeds. A
seed fixes the training subset and the initial weights, and is shared by the three arms,
so every difference between arms is taken within a seed. Each difference is given with a
95 % interval over the five seeds (Student t, 4 degrees of freedom).

Each model is also used as a screen: a margin is set on the validation fold so that no
unstable case there is cleared, and the share of test cases cleared at that margin is
counted, with the unstable ones among them. The hand calculation alone gets the same
treatment.

Writes docs/results/phase6/comparison.json and figures fig81 to fig83. Every number in
docs/steps/34-arm-comparison.md comes from this script.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase6"
BUDGETS = (100, 1000, 10000, 100000)
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = {"mlp": "Multilayer perceptron", "gbt": "Gradient-boosted trees"}
TESTS = ("in_distribution", "boundary", *list(data.SPLITS)[1:])
PAIRS = (("F", "N"), ("R", "N"), ("R", "F"))
METRICS = ("mae", "mae_near_boundary", "missed_unstable", "false_alarm", "cleared",
           "unstable_cleared")  # fmt: skip
T_975_4 = 2.7764451051977987  # 97.5th percentile of Student t with 4 degrees of freedom


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solver"][:, 4] > 0
    out = {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "mech": z["mechanism"][ok]}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def safe_margin(pred: np.ndarray, sigma: np.ndarray) -> float:
    """Smallest margin (1/s) at which no unstable case is predicted below minus it."""
    unstable = sigma > 0
    return float(max(0.0, -pred[unstable].min())) if unstable.any() else 0.0


def scores(sigma: np.ndarray, pred: np.ndarray, margin: float) -> dict:
    out = mt.metrics(sigma, pred)
    cleared = pred < -margin
    out["cleared"] = float(cleared.mean())
    out["unstable_cleared"] = int((cleared & (sigma > 0)).sum())
    return out


def train_all(device: str) -> dict:
    full = load("in_distribution")
    train, val = take(full, full["fold"] == 0), take(full, full["fold"] == 1)
    tests = {"in_distribution": take(full, full["fold"] == 2), "boundary": load("boundary")}
    tests.update({s: load(s) for s in TESTS[2:]})
    hand_margin = safe_margin(mt.baseline(val["mech"]), val["sigma"])
    res: dict = {
        "train_pool": int(train["sigma"].size),
        "validation_cases_for_the_margin": int(val["sigma"].size),
        "tests": {k: {"cases": int(v["sigma"].size), "unstable": int((v["sigma"] > 0).sum())}
                  for k, v in tests.items()},
        "hand_calculation": {
            "margin": hand_margin,
            **{k: scores(v["sigma"], mt.baseline(v["mech"]), hand_margin)
               for k, v in tests.items()},
        },
        "runs": [],
    }  # fmt: skip
    for budget in BUDGETS:
        for seed in SEEDS:
            sub = mt.budget_subset(train, budget, seed)
            n_hold = budget // 5
            for arm in mt.ARMS:
                x = mt.inputs(arm, sub["theta"], sub["mech"])
                y = mt.targets(arm, sub["sigma"], sub["mech"])
                for family in FAMILIES:
                    t0 = time.perf_counter()
                    if family == "gbt":
                        model = HistGradientBoostingRegressor(
                            max_iter=600, learning_rate=0.08, random_state=seed
                        ).fit(x, y)
                        run = model.predict
                    else:
                        model, _ = mt.train_mlp(x[n_hold:], y[n_hold:], x[:n_hold], y[:n_hold],
                                                seed, device=device)  # fmt: skip

                        def run(xx, model=model):
                            return mt.predict(model, xx, device)

                    fit_s = time.perf_counter() - t0

                    def sigma_of(d, arm=arm, run=run):
                        return mt.to_sigma(arm, run(mt.inputs(arm, d["theta"], d["mech"])),
                                           d["mech"])  # fmt: skip

                    margin = safe_margin(sigma_of(val), val["sigma"])
                    res["runs"].append({
                        "family": family, "arm": arm, "budget": budget, "seed": seed,
                        "fit_s": fit_s, "margin": margin,
                        "scores": {k: scores(v["sigma"], sigma_of(v), margin)
                                   for k, v in tests.items()},
                    })  # fmt: skip
                    s = res["runs"][-1]["scores"]["in_distribution"]
                    note = f"cleared {s['cleared']:.3f} ({s['unstable_cleared']})"
                    print(family, arm, budget, seed, f"mae {s['mae']:.3f}",
                          f"margin {margin:.2f}", note, flush=True)  # fmt: skip
    return res


def interval(values: np.ndarray) -> dict:
    mean, half = float(values.mean()), float(T_975_4 * values.std(ddof=1) / np.sqrt(values.size))
    return {"mean": mean, "low": mean - half, "high": mean + half,
            "differs": bool(abs(mean) > half)}  # fmt: skip


def statistics(res: dict) -> None:
    def get(family, arm, budget, test, metric) -> np.ndarray:
        rows = [r for r in res["runs"]
                if (r["family"], r["arm"], r["budget"]) == (family, arm, budget)]  # fmt: skip
        rows.sort(key=lambda r: r["seed"])
        if metric == "margin":
            return np.array([r["margin"] for r in rows])
        return np.array([r["scores"][test][metric] for r in rows], dtype=float)

    summary: dict = {}
    paired: dict = {}
    for family in FAMILIES:
        for budget in BUDGETS:
            for arm in mt.ARMS:
                margins = get(family, arm, budget, "", "margin")
                entry = {"margin": {"mean": float(margins.mean()), "max": float(margins.max())}}
                for test in TESTS:
                    entry[test] = {}
                    for m in METRICS:
                        v = get(family, arm, budget, test, m)
                        entry[test][m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                          "max": float(v.max())}  # fmt: skip
                summary[f"{family}/{arm}/{budget}"] = entry
            for a, b in PAIRS:
                entry = {}
                for test in TESTS:
                    va, vb = (get(family, k, budget, test, "mae") for k in (a, b))
                    ratio = interval(np.log(va / vb))
                    entry[test] = {
                        "mae": interval(va - vb),
                        # Relative change of the error, from the paired log ratio.
                        "mae_change": {k: (float(np.expm1(v)) if k != "differs" else v)
                                       for k, v in ratio.items()},
                        **{m: interval(get(family, a, budget, test, m)
                                       - get(family, b, budget, test, m))
                           for m in ("mae_near_boundary", "missed_unstable", "cleared")},
                    }  # fmt: skip
                paired[f"{family}/{a}-{b}/{budget}"] = entry
    res["summary"], res["paired"] = summary, paired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "comparison.json"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()
        res = train_all(device)
        res["wall_s"] = time.perf_counter() - t0
    statistics(res)
    path.write_text(json.dumps(res, indent=2))
    summary, paired = res["summary"], res["paired"]
    colour = {"F-N": report.ARM_COLOUR["F"], "R-N": report.ARM_COLOUR["R"], "R-F": report.MUTED}
    x = np.arange(len(BUDGETS))

    # ---------------- figure: paired change of the error against budget
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=report.SURFACE, sharey=True)
    for ax, (family, name) in zip(axes, FAMILIES.items(), strict=True):
        for k, (a, b) in enumerate(PAIRS):
            rows = [paired[f"{family}/{a}-{b}/{n}"]["in_distribution"]["mae_change"]
                    for n in BUDGETS]  # fmt: skip
            mean = np.array([100 * r["mean"] for r in rows])
            err = np.array([[100 * (r["mean"] - r["low"]), 100 * (r["high"] - r["mean"])]
                            for r in rows]).T  # fmt: skip
            ax.errorbar(x + (k - 1) * 0.12, mean, yerr=err, color=colour[f"{a}-{b}"], marker="o",
                        ms=5, lw=1.5, capsize=3, label=f"{a} against {b}")  # fmt: skip
        ax.axhline(0, color=report.INK, lw=0.8, ls="--")
        ax.set_xticks(x, [f"{n:,}" for n in BUDGETS])
        ax.set_xlabel("Labelled cases")
        ax.set_title(name, loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Change of the mean error (%)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.title(fig, "In distribution: change of error between arms, paired within a seed "
                 "(bars: 95 % interval over five seeds)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig81_paired_budget.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: paired change on every test set, perceptron
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(TESTS))[::-1]
    for ax, budget in zip(axes, (1000, 100000), strict=True):
        for k, (a, b) in enumerate(PAIRS[:2]):
            rows = [paired[f"mlp/{a}-{b}/{budget}"][t]["mae_change"] for t in TESTS]
            mean = np.array([100 * r["mean"] for r in rows])
            err = np.array([[100 * (r["mean"] - r["low"]), 100 * (r["high"] - r["mean"])]
                            for r in rows]).T  # fmt: skip
            ax.errorbar(mean, ypos + (0.5 - k) * 0.24, xerr=err, color=colour[f"{a}-{b}"],
                        marker="o", ms=5, lw=0, elinewidth=1.5, capsize=3,
                        label=f"{a} against {b}")  # fmt: skip
        ax.axvline(0, color=report.INK, lw=0.8, ls="--")
        ax.set_xlabel("Change of the mean error (%)")
        ax.set_title(f"{budget:,} labelled cases", loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_yticks(ypos, [t.replace("_", " ") for t in TESTS])
    axes[1].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.title(fig, "Perceptron on every test set: change of error against arm N, paired "
                 "within a seed")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig82_paired_splits.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: each model as a screen
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=report.SURFACE, sharey=True)
    hand = res["hand_calculation"]
    counts = res["tests"]["in_distribution"]
    stable = 1 - counts["unstable"] / counts["cases"]
    for ax, (family, name) in zip(axes, FAMILIES.items(), strict=True):
        for arm in mt.ARMS:
            rows = [summary[f"{family}/{arm}/{n}"]["in_distribution"] for n in BUDGETS]
            mean = np.array([100 * r["cleared"]["mean"] for r in rows])
            sd = np.array([100 * r["cleared"]["sd"] for r in rows])
            ax.errorbar(x, mean, yerr=sd, color=report.ARM_COLOUR[arm], marker="o", ms=5, lw=2,
                        capsize=3, label=report.ARM_NAME[arm])  # fmt: skip
            for xi, r, m in zip(x, rows, mean, strict=True):
                ax.annotate(f"{r['unstable_cleared']['mean']:.1f}", (xi, m), fontsize=8,
                            color=report.ARM_COLOUR[arm], xytext=(5, -10),
                            textcoords="offset points")  # fmt: skip
        ax.axhline(100 * hand["in_distribution"]["cleared"], color=report.MUTED, lw=1, ls="--",
                   label=f"hand calculation alone ({hand['in_distribution']['unstable_cleared']})")
        ax.axhline(100 * stable, color=report.INK, lw=0.8, ls=":", label="share that is stable")
        ax.set_xticks(x, [f"{n:,}" for n in BUDGETS])
        ax.set_xlabel("Labelled cases")
        ax.set_title(name, loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Test cases cleared at the margin set on the validation fold (%)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower right")
    report.title(fig, "Each model as a screen, in distribution (numbers: unstable cases cleared, "
                 "mean over seeds)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig83_models_as_screens.png", dpi=160)
    plt.close(fig)

    for family in FAMILIES:
        for budget in BUDGETS:
            line = [f"{family} {budget:>6}"]
            for arm in mt.ARMS:
                s = summary[f"{family}/{arm}/{budget}"]
                d = s["in_distribution"]
                line.append(f"{arm} mae {d['mae']['mean']:.3f} margin {s['margin']['mean']:.2f} "
                            f"cleared {d['cleared']['mean']:.3f} "
                            f"({d['unstable_cleared']['mean']:.1f})")  # fmt: skip
            for a, b in PAIRS:
                c = paired[f"{family}/{a}-{b}/{budget}"]["in_distribution"]["mae_change"]
                line.append(f"{a}-{b} {100 * c['mean']:+.0f}% [{100 * c['low']:+.0f}, "
                            f"{100 * c['high']:+.0f}]")  # fmt: skip
            print(" | ".join(line))
    print("hand", {k: v for k, v in hand["in_distribution"].items()}, "margin", hand["margin"])


if __name__ == "__main__":
    main()
