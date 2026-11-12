"""Phase 5: task T1 with the physics as a penalty during training (arm L).

Needs the dataset from scripts/phase3_generate_t1.py. A network predicts the eigenvalue
of the sub-synchronous mode from the 12 parameters. Besides a data loss on labelled
cases it is penalised on 10,000 cases for which only the state matrix A(theta) is known,
not its eigenvalues. Two forms of the penalty:

    vector      ||A v - lambda v||^2 with the eigenvector v predicted too
    singular    smallest singular value of (A - lambda I), squared; no eigenvector

    labels      0, 100 or 1,000 solver labels
    weight      of the penalty against the data loss

Writes docs/results/phase5/eigen_residual.json and two figures. Every number in
docs/steps/13-eigen-residual.md comes from this script.

Run with:  uv run scripts/phase5_eigen_residual.py        # about 20 minutes
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

from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.models import eigen
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase5"
N_POOL, N_CHECK = 10000, 2000
SEEDS = (0, 1, 2)
VECTOR_WEIGHTS = (0.01, 0.1, 1.0, 10.0)
SINGULAR_WEIGHTS = (1.0, 10.0, 100.0, 1000.0)
RUNS = (  # (labels, weight of the penalty, form)
    (100, 0.0, "none"),
    *((100, w, "vector") for w in VECTOR_WEIGHTS),
    *((100, w, "singular") for w in SINGULAR_WEIGHTS),
    (1000, 0.0, "none"),
    (1000, 1.0, "vector"),
    (1000, 10.0, "singular"),
    (1000, 100.0, "singular"),
    (0, 1.0, "vector"),
    (0, 1.0, "singular"),
)
SHOWN = (
    (100, 0.0, "none"),
    (100, 1.0, "vector"),
    (100, 10.0, "singular"),
    (100, 100.0, "singular"),
    (1000, 0.0, "none"),
    (1000, 10.0, "singular"),
    (0, 1.0, "singular"),
)


def name(labels: int, weight: float, form: str) -> str:
    if labels == 0:
        return f"no labels, {form} penalty"
    if form == "none":
        return f"{labels:,} labels, data only"
    return f"{labels:,} labels, {form} penalty x {weight:g}"


def load() -> dict:
    z = np.load(DATA / "in_distribution.npz")
    ok = z["solver"][:, 4] > 0
    return {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "f": z["solver"][ok, 1],
            "mech": z["mechanism"][ok], "fold": z["fold"][ok]}  # fmt: skip


def take(d: dict, mask) -> dict:
    return {k: v[mask] for k, v in d.items()}


def mean_sd(values) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    full = load()
    train, test = take(full, full["fold"] == 0), take(full, full["fold"] == 2)
    pool = mt.budget_subset(train, N_POOL, 0)
    check = take(test, np.arange(test["sigma"].size) < N_CHECK)

    # ---------------- state matrices: the solver without its eigenvalue step
    t0 = time.perf_counter()
    pool["a"] = data.state_matrices(pool["theta"], processes=8)
    check["a"] = data.state_matrices(check["theta"], processes=8)
    matrices_s = time.perf_counter() - t0
    rows = check["theta"][:200]
    data.solve(rows[0])  # warm up before timing
    t0 = time.perf_counter()
    for r in rows:
        data.solve(r)
    solve_s = (time.perf_counter() - t0) / len(rows)
    t0 = time.perf_counter()
    for r in rows:
        data.state_matrix(r)
    matrix_s = (time.perf_counter() - t0) / len(rows)
    for d in (pool, check, test):
        d["x"] = mt.scale_theta(d["theta"])
    eig = np.linalg.eigvals(check["a"])  # (n, 25)
    upper = np.where(eig.imag >= -1e-9, eig, np.inf)
    target = check["sigma"] + 2j * np.pi * check["f"]
    res: dict = {
        "pool": N_POOL,
        "check_cases": N_CHECK,
        "test_cases": int(test["sigma"].size),
        "cost": {
            "state_matrix_s_per_case": matrix_s,
            "full_solve_s_per_case": solve_s,
            "matrices_wall_s_8_processes": matrices_s,
        },
        "runs": {},
    }
    kept: dict = {}

    for labels, weight, form in RUNS:
        rows = []
        for seed in SEEDS:
            lab = held = None
            if labels:
                pick = np.random.default_rng(seed).permutation(N_POOL)[:labels]
                sub = take(pool, pick)
                h = labels // 5
                held = {k: v[:h] for k, v in sub.items()}
                lab = {k: v[h:] for k, v in sub.items()}
            cfg = eigen.EigenConfig(weight=weight, form=form if form != "none" else "vector")
            model, info = eigen.train_eigen(lab, held, pool if weight > 0 else None, seed, cfg,
                                            device=device)  # fmt: skip
            out = eigen.predict(model, test["x"], device)
            m = mt.metrics(test["sigma"], out["sigma"])
            c = eigen.predict(model, check["x"], device)
            lam_hat = c["sigma"] + 2j * np.pi * np.abs(c["f"])
            dist = np.abs(upper - lam_hat[:, None])
            nearest = upper[np.arange(N_CHECK), dist.argmin(axis=1)]
            rows.append(
                {
                    "mae": m["mae"],
                    "mae_near_boundary": m["mae_near_boundary"],
                    "missed_unstable": m["missed_unstable"],
                    "false_alarm": m["false_alarm"],
                    "f_mae": float(np.abs(np.abs(out["f"]) - test["f"]).mean()),
                    "distance_to_nearest_eigenvalue": float(np.median(dist.min(axis=1))),
                    "nearest_is_sub_synchronous": float((np.abs(nearest - target) < 1e-6).mean()),
                    "median_f_hz": float(np.median(np.abs(c["f"]))),
                    "median_sigma": float(np.median(c["sigma"])),
                    "train_s": info["train_s"],
                    "history": info["history"],
                }
            )
            if seed == SEEDS[0]:
                kept[(labels, weight, form)] = lam_hat
        entry = {k: mean_sd([r[k] for r in rows]) for k in rows[0] if k != "history"}
        entry["histories"] = [r["history"] for r in rows]
        key = name(labels, weight, form)
        res["runs"][key] = {"labels": labels, "weight": weight, "form": form, **entry}
        print(f"{key:40s} mae {entry['mae']['mean']:.3f} "
              f"missed {entry['missed_unstable']['mean']:.3f} "
              f"nearest is sso {entry['nearest_is_sub_synchronous']['mean']:.3f} "
              f"dist {entry['distance_to_nearest_eigenvalue']['mean']:.2f} "
              f"{entry['train_s']['mean']:.0f} s")  # fmt: skip
    res["wall_s"] = time.perf_counter() - t_start
    (OUT / "eigen_residual.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: training curves
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), facecolor=report.SURFACE)
    panels = (("data_loss", "Data loss on labelled cases (scaled)"),
              ("residual", "Penalty on the unlabelled pool (scaled)"),
              ("held_out_mae", "Held-out error in growth rate (1/s)"))  # fmt: skip
    for run, colour in zip(SHOWN, report.SERIES, strict=True):
        hs = res["runs"][name(*run)]["histories"]
        for ax, (key, _) in zip(axes, panels, strict=True):
            if np.isfinite(hs[0][key]).all():
                report.curves(ax, [(np.array(h["step"]), np.array(h[key])) for h in hs], colour,
                              name(*run))  # fmt: skip
    handles = {}
    for ax, (_, label) in zip(axes, panels, strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_title(label, loc="left", fontsize=10)
        report.style(ax)
        hs, ls = ax.get_legend_handles_labels()
        handles.update(dict(zip(ls, hs, strict=True)))
    fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=9,
               labelcolor=report.LABEL, loc="lower center", ncol=4)  # fmt: skip
    report.title(fig, "Training with and without a physics penalty: thin lines are seeds, "
                 "thick lines their mean")  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig32_eigen_training.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: what the penalty buys, and where label-free training lands
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), facecolor=report.SURFACE)
    blue, orange, green = report.SERIES[:3]
    ax = axes[0]
    bars = (("none", 0.0, report.MUTED, "data only"),
            ("vector", 1.0, orange, "vector penalty x 1"),
            ("singular", 10.0, green, "singular penalty x 10"))  # fmt: skip
    for k, labels in enumerate((100, 1000)):
        for j, (form, weight, colour, label) in enumerate(bars):
            e = res["runs"][name(labels, weight, form)]["mae"]
            ax.bar(k + (j - 1) * 0.27, e["mean"], yerr=e["sd"], width=0.25, color=colour,
                   ecolor=report.LABEL, label=label if k == 0 else None)  # fmt: skip
    ax.set_xticks([0, 1], ["100 labels", "1,000 labels"])
    ax.set_ylabel("Mean absolute error in growth rate (1/s)")
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for form, weights, colour in (("vector", VECTOR_WEIGHTS, orange),
                                  ("singular", SINGULAR_WEIGHTS, green)):  # fmt: skip
        e = [res["runs"][name(100, w, form)]["mae"] for w in weights]
        ax.errorbar(weights, [v["mean"] for v in e], yerr=[v["sd"] for v in e], color=colour, lw=2,
                    marker="o", ms=6, label=f"{form} penalty")  # fmt: skip
    ax.axhline(res["runs"][name(100, 0.0, "none")]["mae"]["mean"], color=report.MUTED, lw=1.0,
               ls="--", label="data only")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Weight of the penalty")
    ax.set_title("100 labels: error against the weight", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    ax = axes[2]
    other = eig[eig.imag > 2 * np.pi * 0.2]
    ax.scatter(other.real, other.imag / 2 / np.pi, s=1, color=report.GRID, lw=0, zorder=0,
               label="every mode of the same cases")  # fmt: skip
    ax.scatter(target.real, target.imag / 2 / np.pi, s=4, color=report.MUTED, alpha=0.5, lw=0,
               label="sub-synchronous mode")  # fmt: skip
    for form, colour in (("vector", orange), ("singular", green)):
        lam = kept[(0, 1.0, form)]
        ax.scatter(lam.real, lam.imag / 2 / np.pi, s=5, color=colour, alpha=0.7, lw=0,
                   label=f"network, no labels, {form} penalty")  # fmt: skip
    ax.set_xlim(-60, 40)
    ax.set_ylim(-3, 130)
    ax.set_xlabel("Growth rate (1/s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Where training without labels lands", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=3, loc="upper right")
    report.style(ax)
    report.title(fig, "A physics penalty in training, for the growth rate of the "
                 "sub-synchronous mode")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig33_eigen_results.png", dpi=160)
    plt.close(fig)
    print(json.dumps(res["cost"], indent=1))
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
