"""How few scans cover the loadings and operating points of a converter.

    uv run scripts/power_axis.py            # a random quarter of the converter cases, 15,034
    uv run scripts/power_axis.py --small    # one in 64, for a quick look
    uv run scripts/power_axis.py --reuse    # numbers and figures from the saved pass

Needs dataset C1. Writes data/poweraxis-v1/<family>_<set>.npz (not committed),
docs/results/network/power_axis.json and figures fig75 and fig76. Every number in
docs/steps/31-power-axis.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import poweraxis, report
from sso_sciml.data import c1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "poweraxis-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(poweraxis.COLUMNS)}
LABEL = {
    "rated": "1 scan: rated power, reference point",
    "rated, right u": "1 scan: rated power, right point",
    "right P, ref u": "1 scan: right power, reference point",
    "2 x 2": "4 scans, straight lines",
    "3 x 3": "9 scans, parabolas",
    "3 in power": "3 scans in power at the right point",
    "exact": "1 scan at the operating point",
}
EDGES = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0001])


def sets_of(kind: str) -> list[str]:
    return [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]


def load(kind: str) -> dict:
    out = np.concatenate([np.load(DATA / f"{kind}_{n}.npz")["out"] for n in sets_of(kind)])
    out = out[np.isfinite(out[:, COL["true"]])]
    return {name: out[:, j] for name, j in COL.items()}


def study(kind: str, d: dict) -> dict:
    true, size = d["true"], np.abs(d["p"])
    unstable, near = true > 0, np.abs(true) < 5.0
    lo, hi = poweraxis.U_GRID[kind][[0, 2]]
    res: dict = {
        "cases": int(true.size),
        "unstable": int(unstable.sum()),
        "u_quantiles": [float(q) for q in np.quantile(d["u"], [0.01, 0.25, 0.5, 0.75, 0.99])],
        "u_outside_the_scanned_range": float(((d["u"] < lo) | (d["u"] > hi)).mean()),
        "power_below_the_lowest_scan": float((size < poweraxis.POWER[0]).mean()),
        "methods": {},
    }
    for m in poweraxis.METHODS:
        est = d[m]
        ok = np.isfinite(est)
        err = np.abs(est - true)
        says_stable = ok & (est < 0)
        res["methods"][m] = {
            "scans": poweraxis.SCANS[m],
            "no_answer": int((~ok).sum()),
            "near_boundary_error_median": float(np.median(err[ok & near])),
            "near_boundary_error_p95": float(np.quantile(err[ok & near], 0.95)),
            "near_boundary_error_p99": float(np.quantile(err[ok & near], 0.99)),
            "unstable_cleared": int((says_stable & unstable).sum()),
            "stable_flagged": int((~says_stable & ~unstable).sum()),
            "error_p95_by_power": [
                float(np.quantile(err[ok & near & (size >= a) & (size < b)], 0.95))
                if (ok & near & (size >= a) & (size < b)).sum() >= 30
                else None
                for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)
            ],
        }
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
                theta = np.load(ROOT / "data" / c1.VERSION / f"{kind}_{name}.npz")["theta"]
                theta = theta[part(len(theta), 64 if args.small else 4)]
                out = poweraxis.cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"power": poweraxis.POWER.tolist(), "plants": {},
                 "u_grid": {k: v.tolist() for k, v in poweraxis.U_GRID.items()},
                 "power_edges": EDGES.tolist()}  # fmt: skip
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(kind, load(kind))
        print(plant, {k: v for k, v in r.items() if k != "methods"})
        for m, row in r["methods"].items():
            by_power = [None if v is None else round(v, 3) for v in row["error_p95_by_power"]]
            print(f"  {m:16s} {row['scans']} scans: error median "
                  f"{row['near_boundary_error_median']:.1e} "
                  f"p95 {row['near_boundary_error_p95']:.1e} "
                  f"p99 {row['near_boundary_error_p99']:.1e} | "
                  f"{row['unstable_cleared']} unstable cleared, {row['stable_flagged']} stable "
                  f"flagged, {row['no_answer']} no answer | by power {by_power}")  # fmt: skip
    if not args.small:
        (OUT / "power_axis.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: wrong verdicts by method
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(poweraxis.METHODS))[::-1]
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        r = res["plants"][plant]
        missed = [r["methods"][m]["unstable_cleared"] for m in poweraxis.METHODS]
        ax.barh(
            ypos,
            [v + 0.5 for v in missed],
            color=[report.SERIES[1] if v else report.SERIES[2] for v in missed],
            height=0.6,
        )
        for y, v, m in zip(ypos, missed, poweraxis.METHODS, strict=True):
            ax.text(v + 0.5, y, f"  {v} of {r['unstable']} unstable cleared; "
                    f"{r['methods'][m]['stable_flagged']} stable flagged", va="center",
                    fontsize=8, color=report.LABEL)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(0.4, 2e4)
        ax.set_xlabel("Unstable cases cleared (0.5 = none)")
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {r['cases']:,} cases", loc="left",
                     fontsize=10)  # fmt: skip
        report.style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [LABEL[m] for m in poweraxis.METHODS], fontsize=9)
    report.title(fig, "The scan at the operating point of a case, built from scans at other "
                 "loadings and operating points")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig75_power_axis.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error against loading
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE, sharey=True)
    x = np.arange(len(EDGES) - 1)
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        for m, colour in zip(poweraxis.METHODS, report.SERIES, strict=False):
            y = [np.nan if v is None else max(v, 1e-9)
                 for v in res["plants"][plant]["methods"][m]["error_p95_by_power"]]  # fmt: skip
            ax.plot(x, y, color=colour, lw=2, marker="o", ms=4, label=LABEL[m])
        ax.set_yscale("log")
        ticks = [f"{a:g} to {min(b, 1):g}" for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)]
        ax.set_xticks(x, ticks, fontsize=8)
        ax.set_xlabel("Size of the power of the case (pu)")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Error of the growth rate, 95th percentile (1/s)")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=report.LABEL, loc="center left",
                   bbox_to_anchor=(0.02, 0.43))  # fmt: skip
    report.title(fig, "Error of the growth rate against the loading of the case")
    fig.tight_layout()
    fig.savefig(OUT / "fig76_error_by_power.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
