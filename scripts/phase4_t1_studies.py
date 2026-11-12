"""Phase 4: what the hand calculation contributes to task T1, and what its errors cost.

Needs the dataset from scripts/phase3_generate_t1.py. Four studies, all with the
multilayer perceptron of scripts/phase4_t1_arms.py and three seeds:

    ablation     which hand-calculation quantity carries the gain of arm F
    validity     error of each arm against the size of the hand calculation's own error
    wrong data   hand calculation done with one parameter wrong, models trained clean
    noise        hand calculation with a random error in training and in test

Writes docs/results/phase4/t1_studies.json and two figures. Every number in
docs/steps/10-t1-feature-studies.md comes from this script.

Run with:  uv run scripts/phase4_t1_studies.py        # about 15 minutes
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
SEEDS = (0, 1, 2)
GROUPS = {
    "impedance root": ("sigma_impedance", "f_impedance"),
    "circuit root": ("sigma_circuit", "f_circuit"),
    "quick estimate": ("sigma_hand",),
    "resistances": ("r_machine", "r_net"),
    "resonance": ("has_resonance", "f_er", "slip_res"),
}
J_IMP = data.MECHANISM_COLUMNS.index("sigma_impedance")
WRONG = (("rsc_hz", 1.25), ("r_r", 1.25), ("x_line", 1.10))
N_WRONG = 2048


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


def mean_sd(values) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def with_noise(d: dict, sd: float, seed: int) -> dict:
    """The same cases with a random error of standard deviation `sd` (1/s) added to the
    impedance root."""
    if sd == 0.0:
        return d
    mech = d["mech"].copy()
    mech[:, J_IMP] += np.random.default_rng(seed).normal(0.0, sd, len(mech))
    return {**d, "mech": mech}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    full = load("in_distribution")
    train, test = take(full, full["fold"] == 0), take(full, full["fold"] == 2)
    extra = {s: load(s) for s in list(data.SPLITS)[1:]}
    res: dict = {"test_cases": int(test["sigma"].size)}

    def fit(family, arm, features, budget, seed, pool=train):
        """Train one model; return a function from a set of cases to growth rates."""
        sub = mt.budget_subset(pool, budget, seed)
        x = mt.inputs(arm, sub["theta"], sub["mech"], features)
        y = mt.targets(arm, sub["sigma"], sub["mech"])
        if family == "gbt":
            model = HistGradientBoostingRegressor(
                max_iter=600, learning_rate=0.08, random_state=seed
            ).fit(x, y)
            raw = model.predict
        else:
            h = budget // 5
            model, _ = mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed, device=device)

            def raw(xx):
                return mt.predict(model, xx, device)

        def run(d: dict) -> np.ndarray:
            return mt.to_sigma(arm, raw(mt.inputs(arm, d["theta"], d["mech"], features)), d["mech"])

        return run

    # ---------------- ablation: which quantity carries the gain of arm F
    variants: dict[str, tuple[str, tuple[str, ...] | None]] = {"none (arm N)": ("N", None)}
    variants.update({f"only {g}": ("F", cols) for g, cols in GROUPS.items()})
    variants.update(
        {
            f"all but {g}": ("F", tuple(c for c in data.MECHANISM_COLUMNS if c not in cols))
            for g, cols in GROUPS.items()
        }
    )
    variants["all (arm F)"] = ("F", None)
    budget_abl = 1000
    ablation: dict = {}
    for family in ("mlp", "gbt"):
        for name, (arm, cols) in variants.items():
            rows = []
            for seed in SEEDS:
                run = fit(family, arm, cols, budget_abl, seed)
                m = mt.metrics(test["sigma"], run(test))
                m["mae_extrapolation"] = float(
                    np.mean([mt.metrics(d["sigma"], run(d))["mae"] for d in extra.values()])
                )
                rows.append(m)
            ablation[f"{family}/{name}"] = {
                k: mean_sd([r[k] for r in rows])
                for k in ("mae", "mae_near_boundary", "missed_unstable", "mae_extrapolation")
            }
            print("ablation", family, name, f"{ablation[f'{family}/{name}']['mae']['mean']:.3f}")
    res["ablation"] = {"budget": budget_abl, "variants": ablation}

    # ---------------- validity: error against the hand calculation's own error
    hand_err = np.abs(mt.baseline(test["mech"]) - test["sigma"])
    edges = np.quantile(hand_err, [0.0, 0.25, 0.5, 0.75, 1.0])
    quart = np.clip(np.searchsorted(edges, hand_err, side="right") - 1, 0, 3)
    has_res = test["mech"][:, 0] > 0
    groups = {f"quartile {q + 1}": quart == q for q in range(4)}
    groups["resonance found"] = has_res
    groups["no resonance"] = ~has_res
    validity: dict = {
        "quartile_edges": edges.tolist(),
        "hand": {g: float(hand_err[m].mean()) for g, m in groups.items()},
        "share": {g: float(m.mean()) for g, m in groups.items()},
    }
    clean_models: dict = {}
    for budget in (1000, 100000):
        for arm in mt.ARMS:
            errs = []
            for seed in SEEDS:
                run = fit("mlp", arm, None, budget, seed)
                clean_models[(arm, budget, seed)] = run
                errs.append(np.abs(run(test) - test["sigma"]))
            validity[f"{arm}/{budget}"] = {
                g: mean_sd([e[m].mean() for e in errs]) for g, m in groups.items()
            }
            # Does the model's error follow the hand calculation's? Rank correlation.
            rank = [
                float(
                    np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(hand_err)))[0, 1]
                )
                for e in errs
            ]
            validity[f"{arm}/{budget}"]["rank_correlation_with_hand_error"] = mean_sd(rank)
            print("validity", arm, budget, f"{np.mean(rank):.3f}")
    res["validity"] = validity

    # ---------------- wrong data: hand calculation with one parameter off, at test time
    sub = take(test, np.arange(test["sigma"].size) < N_WRONG)
    wrong: dict = {"cases": int(sub["sigma"].size)}
    cases = {"exact": sub}
    for name, factor in WRONG:
        theta = sub["theta"].copy()
        theta[:, data.NAMES.index(name)] *= factor
        cases[f"{name} x {factor}"] = {**sub, "mech": np.array([data.mechanism(r) for r in theta])}
    for label, d in cases.items():
        entry = {"hand": mt.metrics(d["sigma"], mt.baseline(d["mech"]))}
        for budget in (1000, 100000):
            for arm in ("F", "R"):
                rows = [mt.metrics(d["sigma"], clean_models[(arm, budget, s)](d)) for s in SEEDS]
                entry[f"{arm}/{budget}"] = {
                    k: mean_sd([r[k] for r in rows]) for k in ("mae", "missed_unstable")
                }
        wrong[label] = entry
        print("wrong data", label, f"hand {entry['hand']['mae']:.3f}",
              {a: round(entry[f"{a}/100000"]["mae"]["mean"], 3) for a in ("F", "R")})  # fmt: skip
    res["wrong_data"] = wrong

    # ---------------- noise: a hand calculation with a random error, in training and test
    budget_noise = 10000
    noise: dict = {"budget": budget_noise}
    only_root = GROUPS["impedance root"]
    for sd in (0.0, 1.0, 3.0):
        noisy_train = with_noise(train, sd, 7)
        noisy_test = with_noise(test, sd, 8)
        entry = {"hand": mt.metrics(test["sigma"], mt.baseline(noisy_test["mech"]))}
        for arm, cols in (("N", None), ("F", only_root), ("R", None)):
            if arm == "N" and sd > 0:
                entry["N"] = noise["0.0"]["N"]
                continue
            rows = [
                mt.metrics(
                    test["sigma"],
                    fit("mlp", arm, cols, budget_noise, seed, pool=noisy_train)(noisy_test),
                )
                for seed in SEEDS
            ]
            entry[arm] = {k: mean_sd([r[k] for r in rows]) for k in ("mae", "missed_unstable")}
        noise[str(sd)] = entry
        print("noise", sd, {a: round(entry[a]["mae"]["mean"], 3) for a in ("N", "F", "R")})
    res["noise"] = noise
    res["wall_s"] = time.perf_counter() - t_start
    (OUT / "t1_studies.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: ablation
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True, facecolor=SURFACE)
    names = list(variants)
    ypos = np.arange(len(names))[::-1]
    for ax, family, title in ((axes[0], "mlp", "Multilayer perceptron"),
                              (axes[1], "gbt", "Gradient-boosted trees")):  # fmt: skip
        vals = [ablation[f"{family}/{n}"]["mae"] for n in names]
        colours = [COLOUR["N"] if n.startswith("none") else COLOUR["F"] for n in names]
        ax.barh(ypos, [v["mean"] for v in vals], xerr=[v["sd"] for v in vals], height=0.62,
                color=colours, ecolor=LABEL, error_kw={"lw": 1})  # fmt: skip
        ax.set_xlabel("Mean absolute error in growth rate (1/s)")
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, names)
    fig.suptitle(f"Hand-calculation quantities given as inputs, {budget_abl:,} training cases "
                 "(bar: one standard deviation over three seeds)", x=0.02, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig20_t1_ablation.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error against the hand calculation's error
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, facecolor=SURFACE)
    qx = np.arange(4)
    for ax, budget in zip(axes, (1000, 100000), strict=True):
        for arm in mt.ARMS:
            v = [validity[f"{arm}/{budget}"][f"quartile {q + 1}"] for q in range(4)]
            ax.errorbar(qx, [a["mean"] for a in v], yerr=[a["sd"] for a in v], color=COLOUR[arm],
                        lw=2, marker="o", ms=6, label=ARM_NAME[arm])  # fmt: skip
        ax.plot(qx, [validity["hand"][f"quartile {q + 1}"] for q in range(4)], color=MUTED,
                lw=1.2, ls="--", marker="s", ms=4, label="hand calculation alone")  # fmt: skip
        ax.set_yscale("log")
        ax.set_xticks(qx, [f"{edges[q]:.2f} to {edges[q + 1]:.2f}" for q in range(4)])
        ax.set_xlabel("Error of the hand calculation (1/s), by quartile of test cases")
        ax.set_title(f"{budget:,} training cases", loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel("Mean absolute error in growth rate (1/s)")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="center right")
    fig.suptitle("Model error where the hand calculation is good and where it is poor", x=0.02,
                 ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig21_t1_validity.png", dpi=160)
    plt.close(fig)
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
