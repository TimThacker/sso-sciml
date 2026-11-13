"""How few scans cover the operating points a converter visits after an outage.

    uv run scripts/operating_point.py            # a random quarter of the converter cases, 15,034
    uv run scripts/operating_point.py --small    # every 64th, for a quick look
    uv run scripts/operating_point.py --reuse    # numbers and figures from the saved pass

Needs dataset C1. Writes data/opscan-v1/<family>_<set>.npz (not committed),
docs/results/network/operating_point.json and figures fig65 and fig66. Every number in
docs/steps/26-operating-point.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import opscan, report
from sso_sciml.data import c1
from sso_sciml.outage import OUTAGES
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "opscan-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(opscan.COLUMNS)}
NAMES = list(OUTAGES)
LABEL = {
    "reuse": "the scan before the outage",
    "two": "2 scans, 0.1 apart, straight line",
    "three": "3 scans, 0.1 apart, parabola",
    "lib3": "3 scans over the whole range",
    "lib5": "5 scans over the whole range",
    "exact": "a new scan after the outage",
}
SCANS = {"reuse": 1, "two": 2, "three": 3, "lib3": 3, "lib5": 5, "exact": 1}


def sets_of(kind: str) -> list[str]:
    return [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]


def load(kind: str) -> dict:
    out = np.concatenate([np.load(DATA / f"{kind}_{n}.npz")["out"] for n in sets_of(kind)])
    out = out[np.isfinite(out[:, 0])]
    return {name: out[:, j] for name, j in COL.items()}


def study(d: dict) -> dict:
    ok = d["intact"] < 0
    res: dict = {
        "cases": int(ok.size),
        "stable_intact": int(ok.sum()),
        "outages": {},
        "u_before_quantiles": [float(q) for q in np.quantile(d["u0"], [0.01, 0.5, 0.99])],
    }
    total = {m: {"cleared": 0, "unstable_cleared": 0, "wrong": 0} for m in opscan.METHODS}
    n_total = n_unstable = 0
    for j, name in enumerate(NAMES):
        both = ok & (d[f"solved_{j}"] > 0)
        true = d[f"true_{j}"][both]
        move = np.abs(d[f"u1_{j}"] - d["u0"])[both]
        unstable = true > 0
        near = np.abs(true) < 5.0
        entry: dict = {
            "cases": int(both.sum()),
            "unstable_after": int(unstable.sum()),
            "u_move_quantiles": [float(q) for q in np.quantile(move, [0.5, 0.95, 0.99, 1.0])],
            "methods": {},
        }
        n_total += int(both.sum())
        n_unstable += int(unstable.sum())
        for m in opscan.METHODS:
            est = d[f"{m}_{j}"][both]
            err = np.abs(est - true)
            row = {
                "near_boundary_error_median": float(np.median(err[near])),
                "near_boundary_error_p95": float(np.quantile(err[near], 0.95)),
                "near_boundary_error_p99": float(np.quantile(err[near], 0.99)),
                "cleared": float((est < 0).mean()),
                "unstable_cleared": int(((est < 0) & unstable).sum()),
                "stable_flagged": int(((est >= 0) & ~unstable).sum()),
            }
            entry["methods"][m] = row
            total[m]["cleared"] += int((est < 0).sum())
            total[m]["unstable_cleared"] += row["unstable_cleared"]
            total[m]["wrong"] += row["unstable_cleared"] + row["stable_flagged"]
        # Error against how far the operating point moved, for the parallel path out.
        if j == 0:
            edges = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0])
            entry["error_p95_by_move"] = {
                m: [float(np.quantile(np.abs(d[f"{m}_{j}"][both] - true)[(move >= lo) & (move < hi)
                                                                           & near], 0.95))
                    if ((move >= lo) & (move < hi) & near).sum() >= 20 else None
                    for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
                for m in opscan.METHODS
            }  # fmt: skip
            entry["move_edges"] = edges.tolist()
        res["outages"][name] = entry
    res["all outages"] = {
        "cases": n_total, "unstable_after": n_unstable,
        "methods": {m: {"cleared": t["cleared"] / n_total,
                        "unstable_cleared": t["unstable_cleared"], "wrong": t["wrong"]}
                    for m, t in total.items()},
    }  # fmt: skip
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
                out = opscan.cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"step": opscan.STEP, "range": opscan.RANGE, "plants": {}}
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(load(kind))
        print(plant, r["cases"], "cases", r["stable_intact"], "stable intact; u before",
              [round(q, 3) for q in r["u_before_quantiles"]])  # fmt: skip
        for name, e in r["outages"].items():
            print(f"  {name}: {e['cases']} cases, {e['unstable_after']} unstable, u moves by "
                  f"{[round(q, 3) for q in e['u_move_quantiles']]}")  # fmt: skip
            for m, row in e["methods"].items():
                print(f"    {m:6s} error median {row['near_boundary_error_median']:.1e} p95 "
                      f"{row['near_boundary_error_p95']:.1e} p99 "
                      f"{row['near_boundary_error_p99']:.1e} | cleared {row['cleared']:.1%}, "
                      f"{row['unstable_cleared']} unstable, {row['stable_flagged']} stable "
                      "flagged")  # fmt: skip
        print("  all outages", json.dumps(r["all outages"]))
    if not args.small:
        (OUT / "operating_point.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: unstable cases cleared, by method
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(opscan.METHODS))[::-1]
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        tot = res["plants"][plant]["all outages"]
        vals = [tot["methods"][m]["unstable_cleared"] for m in opscan.METHODS]
        ax.barh(ypos, [v + 0.5 for v in vals], color=[report.SERIES[1] if v else report.SERIES[2]
                                                      for v in vals], height=0.6)  # fmt: skip
        for y, v, m in zip(ypos, vals, opscan.METHODS, strict=True):
            ax.text(v + 0.5, y, f"  {v} of {tot['unstable_after']} unstable; "
                    f"{tot['methods'][m]['wrong']} wrong verdicts", va="center", fontsize=8,
                    color=report.LABEL)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(0.4, 3e4)
        ax.set_xlabel("Unstable cases cleared over the four outages (0.5 = none)")
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {tot['cases']:,} outage cases",
                     loc="left", fontsize=10)  # fmt: skip
        report.style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [LABEL[m] for m in opscan.METHODS], fontsize=9)
    report.title(fig, "The scan after an outage, built from scans at other operating points")
    fig.tight_layout()
    fig.savefig(OUT / "fig65_operating_point.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error against how far the operating point moved
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE, sharey=True)
    for ax, (kind, plant) in zip(axes, PLANTS.items(), strict=True):
        e = res["plants"][plant]["outages"][NAMES[0]]
        edges = e["move_edges"]
        x = np.arange(len(edges) - 1)
        for m, colour in zip(opscan.METHODS, report.SERIES, strict=False):
            y = [np.nan if v is None else max(v, 1e-9) for v in e["error_p95_by_move"][m]]
            ax.plot(x, y, color=colour, lw=2, marker="o", ms=4, label=LABEL[m])
        ax.set_yscale("log")
        ticks = [f"{lo:g} to {hi:g}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
        ax.set_xticks(x, ticks, fontsize=8)
        what = "reactive current (pu)" if kind == "pll" else "terminal voltage (pu)"
        ax.set_xlabel(f"Change of {what} when the parallel path is lost")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Error of the growth rate, 95th percentile (1/s)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "Error of the growth rate after losing the parallel path, against how far "
                 "the operating point moved")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig66_error_by_move.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
