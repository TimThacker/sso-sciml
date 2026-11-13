"""Scan noise and uncertain network data together.

    uv run scripts/noise_and_ranges.py            # a random quarter, 24,659 cases
    uv run scripts/noise_and_ranges.py --small    # every 64th, for a quick look
    uv run scripts/noise_and_ranges.py --reuse    # numbers and figures from the saved pass

A noisy 126-point scan is fitted by least squares (step 20) and the fitted plant is
closed on a network known to within a bound, as given and at the 32 ends of the ranges
(step 23). Needs dataset T1 with data/t1-v2/scan_<set>.npz and dataset C1. Writes
data/neterror-v1/both_<kind>_<set>.npz (not committed),
docs/results/network/noise_and_ranges.json and figure fig62. Every number in
docs/steps/24-noise-and-ranges.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import neterror, report
from sso_sciml.data import c1, t1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neterror-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"t1": "doubly-fed plant", "pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(neterror.BOTH_COLUMNS)}
MARGINS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
BAND = {"t1": "sub", "pll": "all", "follow": "all"}


def label(level: float, bound: float) -> str:
    noise = "clean scan" if level == 0 else f"{100 * level:g} % noise"
    net = "exact network" if bound == 0 else f"network within {100 * bound:g} %"
    return f"{noise}, {net}"


def sets_of(kind: str) -> list[str]:
    if kind == "t1":
        return ["in_distribution", "boundary", *list(t1.SPLITS)[1:]]
    return [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]


def thetas(kind: str, name: str) -> np.ndarray:
    if kind == "t1":
        rows = np.load(ROOT / "data" / t1.VERSION / f"scan_{name}.npz")["rows"]
        return np.load(ROOT / "data" / t1.VERSION / f"{name}.npz")["theta"][rows]
    return np.load(ROOT / "data" / c1.VERSION / f"{kind}_{name}.npz")["theta"]


def load(kind: str) -> dict:
    out = np.concatenate([np.load(DATA / f"both_{kind}_{n}.npz")["out"] for n in sets_of(kind)])
    out = out[np.isfinite(out[:, 0])]
    return {name: out[:, j] for name, j in COL.items()}


def banded(d: dict, prefix: str, tag: str, band: str) -> np.ndarray:
    if band != "all":
        return d[f"{prefix}_{band}{tag}"]
    return np.max([d[f"{prefix}_{b}{tag}"] for b in neterror.BANDS], axis=0)


def outcome(cleared: np.ndarray, unstable: np.ndarray) -> dict:
    return {
        "cleared": float(cleared.mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
    }


def study(kind: str, d: dict) -> dict:
    band = BAND[kind]
    true = banded(d, "true", "", band)
    unstable = true > 0
    res: dict = {"cases": int(true.size), "band": band, "unstable": int(unstable.sum()),
                 "stable_share": float((~unstable).mean()), "variants": {}}  # fmt: skip
    for level, bound in neterror.BOTH:
        tag = f"_n{level:g}_b{bound:g}"
        entry = {}
        for name in ("nominal", "corners"):
            est = banded(d, name, tag, band)
            rows = {f"{m:g}": outcome(est < -m, unstable) for m in MARGINS}
            safe = [m for m in MARGINS if rows[f"{m:g}"]["unstable_cleared"] == 0]
            entry[name] = {"margins": rows, "smallest_safe_margin": safe[0] if safe else None}
        res["variants"][label(level, bound)] = entry
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        for kind in PLANTS:
            for name in sets_of(kind):
                theta = thetas(kind, name)
                theta = theta[part(len(theta), 64 if args.small else 4)]
                out = neterror.both_cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"both_{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"margins": MARGINS, "plants": {}}
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(kind, load(kind))
        print(plant, {k: v for k, v in r.items() if k != "variants"})
        for name, e in r["variants"].items():
            for which in ("nominal", "corners"):
                rows = e[which]["margins"]
                print(f"  {name:40s} {which:8s} " + "  ".join(
                    f"{m}: {o['cleared']:.1%}/{o['unstable_cleared']}" for m, o in rows.items())
                    + f" | safe {e[which]['smallest_safe_margin']}")  # fmt: skip
    if not args.small:
        (OUT / "noise_and_ranges.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: cleared share against margin, at the ends of the ranges
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        r = res["plants"][plant]
        for (level, bound), colour in zip(neterror.BOTH, report.SERIES, strict=False):
            name = label(level, bound)
            rows = r["variants"][name]["corners"]["margins"]
            ax.plot(range(len(MARGINS)), [rows[f"{m:g}"]["cleared"] for m in MARGINS],
                    color=colour, lw=2, marker="o", ms=4, label=name)  # fmt: skip
            for j, m in enumerate(MARGINS):
                n = rows[f"{m:g}"]["unstable_cleared"]
                if n:
                    ax.annotate(f"{n}", (j, rows[f"{m:g}"]["cleared"]), fontsize=8, color=colour,
                                xytext=(0, 5), textcoords="offset points", ha="center")  # fmt: skip
        ax.axhline(r["stable_share"], color=report.INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(MARGINS)), [f"{m:g}" for m in MARGINS])
        ax.set_xlabel("Margin on the largest growth rate over the ends (1/s)")
        which = "sub-synchronous band" if r["band"] == "sub" else "all modes"
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {which}", loc="left", fontsize=10)
        ax.set_ylim(0, 1)
        report.style(ax)
    axes[0].set_ylabel("Share of cases cleared (dashed: share that is stable)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower left")
    report.title(fig, "Fitted noisy scan closed on the ends of the network ranges. Numbers: "
                 "unstable cases cleared")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig62_noise_and_ranges.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
