"""Phase 3: generate the T1 dataset and its report.

Writes data/t1-v2/*.npz and manifest.json (not committed), data/samples/t1_sample.npz
(committed, used by tests), docs/results/phase3/t1_report.json and two figures. Every
number in docs/steps/08-t1-dataset.md comes from this script.

Run with:  uv run scripts/phase3_generate_t1.py            # full size, about 7 minutes
           uv run scripts/phase3_generate_t1.py --small    # 1/16 size, for a quick check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from sso_sciml.data import t1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t1.VERSION
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "docs" / "results" / "phase3"
C1, C2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"
N_SOLVER = len(t1.SOLVER_COLUMNS)


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


def col(arr: np.ndarray, name: str) -> np.ndarray:
    names = t1.SOLVER_COLUMNS + t1.MECHANISM_COLUMNS
    return arr[:, names.index(name)]


def summarise(theta: np.ndarray, out: np.ndarray) -> dict:
    ok = col(out, "solved") > 0
    res = ok & (col(out, "has_resonance") > 0)
    sso, top = col(out, "sigma_sso"), col(out, "sigma_max")
    other = ok & (top > 0) & (np.abs(top - sso) > 1e-9)
    hand, circ = col(out, "sigma_hand"), col(out, "sigma_circuit")
    imp = col(out, "sigma_impedance")
    return {
        "rows": int(len(theta)),
        "solver_failures": int((~ok).sum()),
        "unstable_sub_synchronous": float((sso[ok] > 0).mean()),
        "unstable_other_mode": float(other.mean()),
        "has_resonance": float(res.mean()),
        "sigma_sso_min": float(sso[ok].min()),
        "sigma_sso_max": float(sso[ok].max()),
        "within_1_of_boundary": float((np.abs(sso[ok]) < 1.0).mean()),
        "hand_mean_abs_error": float(np.abs(hand[res] - sso[res]).mean()),
        "circuit_mean_abs_error": float(np.abs(circ[res] - sso[res]).mean()),
        "hand_sign_agreement": float(((hand[res] > 0) == (sso[res] > 0)).mean()),
        "circuit_sign_agreement": float(((circ[res] > 0) == (sso[res] > 0)).mean()),
        "impedance_mean_abs_error": float(np.abs(imp[ok] - sso[ok]).mean()),
        "impedance_sign_agreement": float(((imp[ok] > 0) == (sso[ok] > 0)).mean()),
        "impedance_mean_error_unstable": float((imp[ok] - sso[ok])[sso[ok] > 0].mean()),
        "no_resonance_unstable": float((sso[ok & ~res] > 0).mean()) if (ok & ~res).any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    scale = 16 if args.small else 1
    n_in, n_ex, n_bd = 131072 // scale, 4096 // scale, 16384 // scale

    DATA.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"version": t1.VERSION, "spec_hash": t1.spec_hash(), "splits": {}}
    t_start = time.perf_counter()
    keep = {}

    for split in t1.SPLITS:
        n = n_in if split == "in_distribution" else n_ex
        theta = t1.sample(n, split)
        t0 = time.perf_counter()
        out = t1.generate(theta, args.processes)
        info = summarise(theta, out)
        info["wall_s"] = time.perf_counter() - t0
        extra = {"fold": t1.folds(n)} if split == "in_distribution" else {}
        np.savez_compressed(DATA / f"{split}.npz", theta=theta, solver=out[:, :N_SOLVER],
                            mechanism=out[:, N_SOLVER:], **extra)  # fmt: skip
        report["splits"][split] = info
        keep[split] = (theta, out)
        print(split, json.dumps(info))

    # boundary-focused cases: a separate Sobol stream, moved along compensation to sigma = 0
    t0 = time.perf_counter()
    theta_b = t1.boundary_cases(t1.sample(n_bd, "in_distribution", stream=1), args.processes)
    out_b = t1.generate(theta_b, args.processes)
    info = summarise(theta_b, out_b)
    info.update({"candidates": n_bd, "wall_s": time.perf_counter() - t0})
    np.savez_compressed(DATA / "boundary.npz", theta=theta_b, solver=out_b[:, :N_SOLVER],
                        mechanism=out_b[:, N_SOLVER:], fold=t1.folds(len(theta_b)))  # fmt: skip
    report["splits"]["boundary"] = info
    print("boundary", json.dumps(info))

    # reproducibility: the first rows again, in this process, compared bit for bit
    theta, out = keep["in_distribution"]
    again = t1.generate(theta[:256], 1)
    report["regenerated_rows_bit_identical"] = bool(np.array_equal(again, out[:256]))
    report["regenerated_rows_max_abs_diff"] = float(np.nanmax(np.abs(again - out[:256])))

    fold = t1.folds(n_in)
    report["folds"] = {name: int((fold == k).sum()) for k, name in
                       enumerate(("train", "validation", "test"))}  # fmt: skip
    report["wall_s_total"] = time.perf_counter() - t_start
    report["cases_per_second"] = (n_in + 5 * n_ex) / sum(
        report["splits"][s]["wall_s"] for s in t1.SPLITS
    )

    if not args.small:
        head = out[:1000]
        np.savez_compressed(SAMPLES / "t1_sample.npz", theta=theta[:1000],
                            solver=head[:, :N_SOLVER], mechanism=head[:, N_SOLVER:])  # fmt: skip
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()  # fmt: skip
        manifest = {**report, "git_commit": commit, "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__,
                    "columns": {"theta": t1.NAMES, "solver": t1.SOLVER_COLUMNS,
                                "mechanism": t1.MECHANISM_COLUMNS}}  # fmt: skip
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (OUT / "t1_report.json").write_text(json.dumps(report, indent=2))

        # ---------------- figures
        ok = (col(out, "solved") > 0) & (col(out, "has_resonance") > 0)
        idx = np.where(ok)[0][:6000]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True, facecolor=SURFACE)
        for ax, name, title in (
            (axes[0], "sigma_hand", "Quick estimate: net resistance over reactance slope"),
            (axes[1], "sigma_circuit", "Root of the equivalent circuit"),
        ):
            ax.scatter(col(out, name)[idx], col(out, "sigma_sso")[idx], s=5, color=C1, alpha=0.35,
                       linewidth=0)  # fmt: skip
            ax.plot([-20, 25], [-20, 25], color=MUTED, lw=0.8)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.axvline(0, color=MUTED, lw=0.6)
            ax.set_xlim(-22, 27)
            ax.set_ylim(-22, 27)
            ax.set_xlabel("Hand calculation (1/s)")
            ax.set_title(title, loc="left", fontsize=10)
            style(ax)
        axes[0].set_ylabel("Full solver growth rate (1/s)")
        fig.suptitle("Hand calculation against the full solver, 6,000 in-distribution cases",
                     x=0.02, ha="left", fontsize=11, color=INK)  # fmt: skip
        fig.tight_layout()
        fig.savefig(OUT / "fig16_hand_vs_solver.png", dpi=160)
        plt.close(fig)

        names = list(report["splits"])
        frac = [100 * report["splits"][s]["unstable_sub_synchronous"] for s in names]
        fig, ax = plt.subplots(figsize=(7.6, 3.8), facecolor=SURFACE)
        ypos = np.arange(len(names))[::-1]
        ax.barh(ypos, frac, height=0.6, color=C1)
        for y, v in zip(ypos, frac, strict=True):
            ax.text(v + 1, y, f"{v:.0f} %", va="center", fontsize=9, color=LABEL)
        ax.set_yticks(ypos, [s.replace("_", " ") for s in names])
        ax.set_xlabel("Cases with an unstable sub-synchronous mode (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Share of unstable cases in each split", loc="left", fontsize=11)
        style(ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(OUT / "fig17_split_balance.png", dpi=160)
        plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=1))


if __name__ == "__main__":
    main()
