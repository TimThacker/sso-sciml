"""Phase 4: uncertainty, a Gaussian process baseline, trends and inference time for T1.

Needs the dataset from scripts/phase3_generate_t1.py.

    ensemble     five perceptrons per arm and budget; spread as an error bar
    dropout      one perceptron with dropout left on at prediction
    conformal    intervals calibrated on 1,000 cases of the validation fold
    gp           Gaussian process at 100 and 1,000 cases, with its own posterior
    trends       does the model follow the solver when compensation is swept?
    timing       seconds per case for solver, hand calculation and models

Writes docs/results/phase4/t1_uncertainty.json and two figures. Every number in
docs/steps/11-t1-uncertainty.md comes from this script.

Run with:  uv run scripts/phase4_t1_uncertainty.py       # about 15 minutes
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from sso_sciml.data import t1 as data
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase4"
COLOUR = {"N": "#2a78d6", "F": "#eb6834", "R": "#1baf7a"}
ARM_NAME = {"N": "N: no physics", "F": "F: physics as features", "R": "R: physics as baseline"}
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"
MEMBERS = 5
BUDGETS = (1000, 100000)
GP_BUDGETS = (100, 1000)
SEEDS = (0, 1, 2)
Z90 = 1.6448536269514722
N_CAL = 1000
N_TREND, K_GRID = 256, np.linspace(0.10, 0.80, 8)


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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    full = load("in_distribution")
    train, test = take(full, full["fold"] == 0), take(full, full["fold"] == 2)
    cal = take(take(full, full["fold"] == 1), np.arange((full["fold"] == 1).sum()) < N_CAL)
    tests = {"in_distribution": test, "boundary": load("boundary")}
    tests.update({s: load(s) for s in list(data.SPLITS)[1:]})
    res: dict = {"members": MEMBERS, "calibration_cases": int(cal["sigma"].size)}

    def gaussian(d, mean, sd):
        return mt.interval_scores(d["sigma"], mean - Z90 * sd, mean + Z90 * sd)

    # ---------------- deep ensemble, dropout and conformal intervals
    nets: dict = {}
    keep: dict = {}
    for budget in BUDGETS:
        sub = mt.budget_subset(train, budget, 0)
        h = budget // 5
        for arm in mt.ARMS:
            x = mt.inputs(arm, sub["theta"], sub["mech"])
            y = mt.targets(arm, sub["sigma"], sub["mech"])
            models = [
                mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed, device=device)[0]
                for seed in range(MEMBERS)
            ]
            drop, _ = mt.train_mlp(
                x[h:], y[h:], x[:h], y[:h], 0, mt.TrainConfig(dropout=0.1), device=device
            )

            def ensemble(d, arm=arm, models=models):
                xx = mt.inputs(arm, d["theta"], d["mech"])
                p = np.array(
                    [mt.to_sigma(arm, mt.predict(m, xx, device), d["mech"]) for m in models]
                )
                return p.mean(0), p.std(0), p[0]

            def dropout(d, arm=arm, drop=drop):
                mean, sd = mt.predict_mc(drop, mt.inputs(arm, d["theta"], d["mech"]), device=device)
                return mt.to_sigma(arm, mean, d["mech"]), sd * mt.SIGMA_SCALE

            # Conformal calibration: one width for all cases, or one scaled by the spread.
            c_mean, c_sd, _ = ensemble(cal)
            c_err = np.abs(c_mean - cal["sigma"])
            q_const = mt.conformal_quantile(c_err)
            q_scaled = mt.conformal_quantile(c_err / (c_sd + 1e-6))

            entry: dict = {"q_constant": q_const, "q_scaled": q_scaled}
            for name, d in tests.items():
                mean, sd, single = ensemble(d)
                d_mean, d_sd = dropout(d)
                err = np.abs(mean - d["sigma"])
                entry[name] = {
                    "single": mt.metrics(d["sigma"], single),
                    "ensemble": mt.metrics(d["sigma"], mean),
                    "dropout": mt.metrics(d["sigma"], d_mean),
                    "ensemble_raw": gaussian(d, mean, sd),
                    "dropout_raw": gaussian(d, d_mean, d_sd),
                    "conformal_constant": mt.interval_scores(
                        d["sigma"], mean - q_const, mean + q_const
                    ),
                    "conformal_scaled": mt.interval_scores(
                        d["sigma"], mean - q_scaled * (sd + 1e-6), mean + q_scaled * (sd + 1e-6)
                    ),
                    "spread_error_rank_correlation": spearman(sd, err),
                    "dropout_error_rank_correlation": spearman(d_sd, np.abs(d_mean - d["sigma"])),
                }
                if name == "in_distribution":
                    keep[(arm, budget)] = (sd, err)
            nets[(arm, budget)] = (ensemble, models)
            res[f"mlp/{arm}/{budget}"] = entry
            e = entry["in_distribution"]
            print("ensemble", arm, budget, f"mae {e['ensemble']['mae']:.3f}",
                  f"raw {e['ensemble_raw']['coverage']:.3f}",
                  f"scaled {e['conformal_scaled']['coverage']:.3f}")  # fmt: skip

    # ---------------- Gaussian process
    for budget in GP_BUDGETS:
        for arm in mt.ARMS:
            rows = {name: [] for name in tests}
            fit_s = []
            for seed in SEEDS:
                sub = mt.budget_subset(train, budget, seed)
                x = mt.inputs(arm, sub["theta"], sub["mech"])
                y = mt.targets(arm, sub["sigma"], sub["mech"])
                kernel = ConstantKernel(1.0) * Matern(np.ones(x.shape[1]), nu=2.5) + WhiteKernel(
                    1e-3, (1e-8, 1e1)
                )
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed)
                    gp.fit(x, y)
                fit_s.append(time.perf_counter() - t0)
                for name, d in tests.items():
                    mean, sd = gp.predict(mt.inputs(arm, d["theta"], d["mech"]), return_std=True)
                    mean, sd = mt.to_sigma(arm, mean, d["mech"]), sd * mt.SIGMA_SCALE
                    rows[name].append({**mt.metrics(d["sigma"], mean), **gaussian(d, mean, sd)})
            entry = {"fit_s": mean_sd(fit_s)}
            for name, rr in rows.items():
                # `missed_unstable` here is the interval screen; the point screen is not kept.
                entry[name] = {
                    k: mean_sd([r[k] for r in rr]) for k in ("mae", "coverage", "width", "flagged")
                }
            res[f"gp/{arm}/{budget}"] = entry
            print("gp", arm, budget, f"{np.mean(fit_s):.0f}s",
                  f"mae {entry['in_distribution']['mae']['mean']:.3f}",
                  f"cov {entry['in_distribution']['coverage']['mean']:.3f}")  # fmt: skip

    # ---------------- trends: sweep compensation on test cases and compare slopes
    base = test["theta"][:N_TREND]
    k_col = data.NAMES.index("k")
    sweep = np.repeat(base, K_GRID.size, axis=0)
    sweep[:, k_col] = np.tile(K_GRID, N_TREND)
    out = data.generate(sweep, processes=8)
    ok = (out[:, 4] > 0).reshape(N_TREND, K_GRID.size).all(axis=1)
    d_sweep = {"theta": sweep, "sigma": out[:, 0], "mech": out[:, 5:]}

    def slopes(sigma):
        return np.diff(sigma.reshape(N_TREND, K_GRID.size)[ok], axis=1)

    true = slopes(np.nan_to_num(d_sweep["sigma"]))
    clear = np.abs(true) > 0.2
    trends: dict = {
        "cases": int(ok.sum()),
        "segments": int(clear.sum()),
        "hand": float(
            (np.sign(slopes(mt.baseline(d_sweep["mech"])))[clear] == np.sign(true)[clear]).mean()
        ),
    }
    safe = {**d_sweep, "theta": sweep, "mech": np.nan_to_num(d_sweep["mech"])}
    for (arm, budget), (ensemble, _) in nets.items():
        pred = slopes(ensemble(safe)[0])
        trends[f"{arm}/{budget}"] = float((np.sign(pred)[clear] == np.sign(true)[clear]).mean())
    res["trends"] = trends
    print("trends", {k: round(v, 4) if isinstance(v, float) else v for k, v in trends.items()})

    # ---------------- timing
    timing: dict = {}
    rows = test["theta"][:200]
    t0 = time.perf_counter()
    for r in rows:
        data.solve(r)
    timing["solver_s_per_case"] = (time.perf_counter() - t0) / len(rows)
    t0 = time.perf_counter()
    for r in rows:
        data.mechanism(r)
    timing["hand_s_per_case"] = (time.perf_counter() - t0) / len(rows)
    model = nets[("N", 100000)][1][0]
    x_all = mt.inputs("N", test["theta"], test["mech"])
    for dev in ("cuda", "cpu") if device == "cuda" else ("cpu",):
        model.to(dev)
        mt.predict(model, x_all, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            mt.predict(model, x_all, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        timing[f"mlp_{dev}_batch_s_per_case"] = (time.perf_counter() - t0) / 20 / len(x_all)
        t0 = time.perf_counter()
        for j in range(200):
            mt.predict(model, x_all[j : j + 1], dev)
        timing[f"mlp_{dev}_single_s_per_case"] = (time.perf_counter() - t0) / 200
    model.to(device)
    sub = mt.budget_subset(train, 100000, 0)
    gbt = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, random_state=0).fit(
        mt.inputs("N", sub["theta"], sub["mech"]), mt.targets("N", sub["sigma"], sub["mech"])
    )
    t0 = time.perf_counter()
    for _ in range(5):
        gbt.predict(x_all)
    timing["gbt_batch_s_per_case"] = (time.perf_counter() - t0) / 5 / len(x_all)
    res["timing"] = timing
    print("timing", {k: f"{v:.2e}" for k, v in timing.items()})
    res["wall_s"] = time.perf_counter() - t_start
    (OUT / "t1_uncertainty.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: coverage of each interval on each test set
    names = list(tests)
    methods = (("ensemble_raw", "Ensemble spread", "#898781"),
               ("dropout_raw", "Dropout", "#c3a13a"),
               ("conformal_constant", "Conformal, one width", "#2a78d6"),
               ("conformal_scaled", "Conformal, scaled by spread", "#1baf7a"))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, facecolor=SURFACE)
    ypos = np.arange(len(names))[::-1]
    for ax, arm in zip(axes, ("N", "R"), strict=True):
        for k, (key, label, colour) in enumerate(methods):
            vals = [res[f"mlp/{arm}/100000"][n][key]["coverage"] for n in names]
            ax.barh(ypos + (1.5 - k) * 0.2, vals, height=0.18, color=colour, label=label)
        ax.axvline(0.9, color=INK, lw=1.0, ls="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of cases inside the 90 % interval")
        ax.set_title(f"Arm {ARM_NAME[arm]}, 100,000 training cases", loc="left", fontsize=10)
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [n.replace("_", " ") for n in names])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=LABEL, loc="lower center",
               ncol=4)  # fmt: skip
    fig.suptitle("Coverage of 90 % intervals on each test set (dashed: target)", x=0.02,
                 ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig22_t1_coverage.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: does the spread track the error?
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, facecolor=SURFACE)
    for ax, budget in zip(axes, BUDGETS, strict=True):
        for arm in mt.ARMS:
            sd, err = keep[(arm, budget)]
            order = np.argsort(sd)
            bins = np.array_split(order, 10)
            ax.plot([sd[b].mean() for b in bins], [err[b].mean() for b in bins],
                    color=COLOUR[arm], lw=2, marker="o", ms=5, label=ARM_NAME[arm])  # fmt: skip
        lim = np.array([3e-3, 1.0])
        ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Spread of the five members (1/s), tenths of the test fold")
        ax.set_title(f"{budget:,} training cases", loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel("Mean absolute error of the ensemble (1/s)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper left")
    fig.suptitle("Ensemble spread against actual error (dashed: equal)", x=0.02, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig23_t1_spread.png", dpi=160)
    plt.close(fig)
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
