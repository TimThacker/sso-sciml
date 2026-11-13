"""Generate the T2 dataset: the response in time of System B2 after a capacitor-voltage
offset.

    uv run scripts/phase3_generate_t2.py            # 38,912 runs of 0.3 s on the GPU
    uv run scripts/phase3_generate_t2.py --small    # one sixteenth, for a quick look

Writes data/t2-v1/<split>.npz and manifest.json (not committed) and
docs/results/phase3/t2_dataset.json with one figure. Every number in
docs/steps/35-t2-dataset.md comes from this script.
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
from sso_sciml.data import t2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t2.VERSION
OUT = ROOT / "docs" / "results" / "phase3"
SIZES = {"in_distribution": 32768, "high_compensation": 2048, "low_speed": 2048,
         "fast_current_loop": 2048}  # fmt: skip


def relative_error(obs: np.ndarray, red: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Error of the reduced model on the four quantities it has, relative to the size of
    the response, over the times before any trip."""
    mask = (np.arange(t2.N_T)[None, :] < valid[:, None])[:, :, None]
    num = (((obs[:, :, :4] - red) ** 2) * mask).sum(axis=(1, 2))
    return np.sqrt(num / ((obs[:, :, :4] ** 2) * mask).sum(axis=(1, 2)))


def quantiles(x: np.ndarray) -> list[float]:
    return [float(q) for q in np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res: dict = {"version": t2.VERSION, "spec_hash": t2.spec_hash(), "offset_pu": t2.OFFSET,
                 "window_s": t2.T_END, "times": t2.N_T, "splits": {}}  # fmt: skip
    kept = {}
    for split, n in SIZES.items():
        n = n // 16 if args.small else n
        t0 = time.perf_counter()
        d = t2.generate(t2.sample(n, split), args.processes, device)
        wall = time.perf_counter() - t0
        if split == "in_distribution":
            d["fold"] = t2.folds(n)
        np.savez_compressed(DATA / f"{split}.npz", **d)
        ok = d["solved"]
        err = relative_error(d["observed"][ok], d["reduced"][ok], d["valid"][ok])
        amp = np.abs(d["observed"][ok]).max(axis=1)
        res["splits"][split] = {
            "cases": int(n), "kept": int(ok.sum()),
            "no_steady_state_or_early_trip": int((~ok).sum()),
            "trips_within_the_window": int((d["valid"][ok] < t2.N_T).sum()),
            "unstable": int((d["sigma"][ok] > 0).sum()),
            "growth_rate_quantiles": quantiles(d["sigma"][ok]),
            "frequency_quantiles_hz": quantiles(d["f"][ok]),
            "largest_deviation_quantiles": {name: quantiles(amp[:, j])
                                            for j, name in enumerate(t2.OBSERVED)},
            "reduced_model_relative_error": quantiles(err),
            "wall_s": wall,
        }  # fmt: skip
        if split == "in_distribution":
            res["splits"][split]["folds"] = [int((d["fold"][ok] == k).sum()) for k in range(3)]
            kept = {"d": d, "err": err, "ok": ok}
        print(split, json.dumps(res["splits"][split]), flush=True)
    if args.small:
        return
    (DATA / "manifest.json").write_text(json.dumps(res, indent=2))
    (OUT / "t2_dataset.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: three cases, and the error of the reduced model
    d, err, ok = kept["d"], kept["err"], kept["ok"]
    idx = np.where(ok)[0]
    order = idx[np.argsort(d["sigma"][idx])]
    picks = [order[len(order) // 10], order[len(order) // 2], order[(9 * len(order)) // 10]]
    t = t2.times()
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4), facecolor=report.SURFACE)
    for ax, i in zip(axes[:3], picks, strict=True):
        stop = int(d["valid"][i])
        ax.plot(t[:stop], d["observed"][i, :stop, 0], color=report.SERIES[0], lw=1.3,
                label="full model")  # fmt: skip
        ax.plot(t[:stop], d["reduced"][i, :stop, 0], color=report.SERIES[1], lw=1.0, ls="--",
                label="reduced model")  # fmt: skip
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Growth rate {d['sigma'][i]:+.1f} 1/s, {d['f'][i]:.0f} Hz", loc="left",
                     fontsize=10)  # fmt: skip
        report.style(ax)
    axes[0].set_ylabel("Line current, D component, deviation (pu)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    axes[3].scatter(d["sigma"][ok], err, s=3, color=report.SERIES[0], lw=0, alpha=0.4)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Growth rate of the full model (1/s)")
    axes[3].set_ylabel("Error of the reduced model, relative")
    report.style(axes[3])
    report.title(fig, "Dataset T2: response to a capacitor-voltage offset of "
                 f"{t2.OFFSET} pu, full model against the reduced model")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig86_t2_dataset.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
