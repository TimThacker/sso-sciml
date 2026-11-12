"""Phase 4: task T1 with the physics nowhere (N), as a feature (F), or as the baseline (R).

Needs the dataset from scripts/phase3_generate_t1.py. Trains gradient-boosted trees on
the CPU and multilayer perceptrons on the GPU at four data budgets and three seeds, and
scores each on the in-distribution test fold, the boundary set and five extrapolation
splits.

Writes docs/results/phase4/t1_arms.json and two figures. Every number in
docs/steps/09-t1-arms.md comes from this script.

Run with:  uv run scripts/phase4_t1_arms.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from sso_sciml.data import t1 as data
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase4"
COLOUR = {"N": "#2a78d6", "F": "#eb6834", "R": "#1baf7a"}
ARM_NAME = {"N": "N: no physics", "F": "F: physics as features", "R": "R: physics as baseline"}
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"
BUDGETS = (100, 1000, 10000, 100000)
SEEDS = (0, 1, 2)
TESTS = ("in_distribution", "boundary", *list(data.SPLITS)[1:])


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(INK)


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solver"][:, 4] > 0
    out = {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "mech": z["mechanism"][ok]}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def mean_sd(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load("in_distribution")
    train = take(full, full["fold"] == 0)
    tests = {"in_distribution": take(full, full["fold"] == 2), "boundary": load("boundary")}
    tests.update({s: load(s) for s in TESTS[2:]})
    res: dict = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "train_pool": int(train["sigma"].size),
        "held_out_share_of_budget": 0.2,
        "tests": {k: int(v["sigma"].size) for k, v in tests.items()},
    }

    # ---------------- hand calculation alone: needs no training data
    hand = {}
    for name, column in (("impedance_root", "sigma_impedance"), ("quick_estimate", "sigma_hand")):
        j = data.MECHANISM_COLUMNS.index(column)
        hand[name] = {
            k: mt.metrics(v["sigma"], np.clip(v["mech"][:, j], -40, 40)) for k, v in tests.items()
        }
    res["hand_calculation"] = hand

    # ---------------- learned models
    runs = []
    for budget in BUDGETS:
        for seed in SEEDS:
            # The budget counts every label a model sees. The perceptron holds out a fifth
            # of it to choose its checkpoint; the trees train on all of it.
            sub = mt.budget_subset(train, budget, seed)
            n_hold = budget // 5
            for arm in mt.ARMS:
                x, y = (
                    mt.inputs(arm, sub["theta"], sub["mech"]),
                    mt.targets(arm, sub["sigma"], sub["mech"]),
                )
                for family in ("gbt", "mlp"):
                    t0 = time.perf_counter()
                    if family == "gbt":
                        model = HistGradientBoostingRegressor(
                            max_iter=600, learning_rate=0.08, random_state=seed
                        ).fit(x, y)
                        fit_s = time.perf_counter() - t0

                        def run_model(xx, model=model):
                            return model.predict(xx)
                    else:
                        model, _ = mt.train_mlp(
                            x[n_hold:], y[n_hold:], x[:n_hold], y[:n_hold], seed, device=device
                        )
                        fit_s = time.perf_counter() - t0

                        def run_model(xx, model=model):
                            return mt.predict(model, xx, device)

                    scores = {}
                    for name, d in tests.items():
                        pred = run_model(mt.inputs(arm, d["theta"], d["mech"]))
                        scores[name] = mt.metrics(d["sigma"], mt.to_sigma(arm, pred, d["mech"]))
                    runs.append({"family": family, "arm": arm, "budget": budget, "seed": seed,
                                 "fit_s": fit_s, "scores": scores})  # fmt: skip
                    print(family, arm, budget, seed, f"{fit_s:.1f}s",
                          f"mae {scores['in_distribution']['mae']:.3f}",
                          f"missed {scores['in_distribution']['missed_unstable']:.3f}")  # fmt: skip
    res["runs"] = runs

    # ---------------- summary over seeds
    summary: dict = {}
    for family in ("gbt", "mlp"):
        for arm in mt.ARMS:
            for budget in BUDGETS:
                sel = [
                    r for r in runs if (r["family"], r["arm"], r["budget"]) == (family, arm, budget)
                ]
                entry = {"fit_s": mean_sd([r["fit_s"] for r in sel])}
                for name in tests:
                    entry[name] = {
                        m: mean_sd([r["scores"][name][m] for r in sel])
                        for m in ("mae", "mae_near_boundary", "missed_unstable", "false_alarm")
                    }
                summary[f"{family}/{arm}/{budget}"] = entry
    res["summary"] = summary
    (OUT / "t1_arms.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: error against data budget
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, facecolor=SURFACE)
    for ax, family, title in ((axes[0], "mlp", "Multilayer perceptron"),
                              (axes[1], "gbt", "Gradient-boosted trees")):  # fmt: skip
        for arm in mt.ARMS:
            m = [summary[f"{family}/{arm}/{b}"]["in_distribution"]["mae"] for b in BUDGETS]
            mean, sd = np.array([v["mean"] for v in m]), np.array([v["sd"] for v in m])
            ax.fill_between(BUDGETS, mean - sd, mean + sd, color=COLOUR[arm], alpha=0.18, lw=0)
            ax.plot(BUDGETS, mean, color=COLOUR[arm], lw=2, marker="o", ms=6, label=ARM_NAME[arm])
        ref = hand["impedance_root"]["in_distribution"]["mae"]
        ax.axhline(ref, color=MUTED, lw=1.0, ls="--")
        ax.text(BUDGETS[0], ref * 1.07, "hand calculation alone", fontsize=9, color=LABEL)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Labelled cases used for training")
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel("Mean absolute error in growth rate (1/s)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower left")
    fig.suptitle("In-distribution test error against data budget (band: one standard deviation "
                 "over three seeds)", x=0.02, ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig18_t1_budget.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error by split at one budget
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, facecolor=SURFACE)
    names = list(tests)
    ypos = np.arange(len(names))[::-1]
    for ax, budget in zip(axes, (1000, 100000), strict=True):
        for k, arm in enumerate(mt.ARMS):
            vals = [summary[f"mlp/{arm}/{budget}"][n]["mae"]["mean"] for n in names]
            ax.barh(ypos + (1 - k) * 0.26, vals, height=0.24, color=COLOUR[arm],
                    label=ARM_NAME[arm])  # fmt: skip
        ax.set_xlabel("Mean absolute error in growth rate (1/s)")
        ax.set_title(f"{budget:,} training cases", loc="left", fontsize=10)
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [n.replace("_", " ") for n in names])
    axes[1].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower right")
    fig.suptitle("Multilayer perceptron: error on each test set", x=0.02, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig19_t1_splits.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"hand": {k: v["in_distribution"] for k, v in hand.items()}}, indent=1))


if __name__ == "__main__":
    main()
