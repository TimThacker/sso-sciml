"""Screening at the unfavourable end of each network range, against a wider margin.

    uv run scripts/network_corners.py            # every 4th case, about 24,500 cases
    uv run scripts/network_corners.py --small    # every 64th, for a quick look
    uv run scripts/network_corners.py --reuse    # numbers and figures from the saved pass

The screen is given a network wrong by up to a bound in each of five quantities, and it
knows the bound. Needs dataset T1 with data/t1-v2/scan_<set>.npz and dataset C1. Writes
data/neterror-v1/corner_<kind>_<set>.npz (not committed),
docs/results/network/network_corners.json and figures fig60 and fig61. Every number in
docs/steps/23-network-corners.md comes from this script.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neterror-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"t1": "doubly-fed plant", "pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(neterror.CORNER_COLUMNS)}
MARGINS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
P = {name: j for j, name in enumerate(t1.NAMES)}
# The band each plant is screened on: the resonance for the doubly-fed plant, whose slow
# mode goes to the phasor-domain model (step 21); everything for a converter (step 22).
BAND = {"t1": "sub", "pll": "all", "follow": "all"}


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
    out, theta = [], []
    for n in sets_of(kind):
        z = np.load(DATA / f"corner_{kind}_{n}.npz")
        out.append(z["out"])
        theta.append(z["theta"])
    out, theta = np.concatenate(out), np.concatenate(theta)
    ok = np.isfinite(out[:, 0])
    d = {name: out[ok, j] for name, j in COL.items()}
    d["theta"] = theta[ok]
    return d


def banded(d: dict, prefix: str, tag: str, band: str) -> np.ndarray:
    if band != "all":
        return d[f"{prefix}_{band}{tag}"]
    return np.max([d[f"{prefix}_{b}{tag}"] for b in neterror.BANDS], axis=0)


def outcome(cleared: np.ndarray, unstable: np.ndarray) -> dict:
    return {
        "cleared": float(cleared.mean()),
        "stable_cleared": float(cleared[~unstable].mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
    }


def study(kind: str, d: dict) -> dict:
    band = BAND[kind]
    true = banded(d, "true", "", band)
    unstable = true > 0
    res: dict = {"cases": int(true.size), "band": band, "unstable": int(unstable.sum()),
                 "stable_share": float((~unstable).mean()), "bounds": {}}  # fmt: skip
    for bound in neterror.BOUNDS:
        tag = f"_b{bound:g}"
        nominal, corners = banded(d, "nominal", tag, band), banded(d, "corners", tag, band)
        entry: dict = {
            "nominal with a margin": {f"{m:g}": outcome(nominal < -m, unstable) for m in MARGINS},
            "all 32 ends": outcome(corners < 0, unstable),
            # How far the worst end lies above the network as given: the margin it implies.
            "implied_margin_quartiles": [float(q) for q in np.quantile(
                (corners - nominal)[np.isfinite(corners - nominal)], [0.25, 0.5, 0.75, 0.95])],
            "truth_above_the_worst_end": int((true > corners + 1e-9).sum()),
        }  # fmt: skip
        if kind == "t1":
            known = banded(d, "known", tag, band)
            entry["the one unfavourable end"] = outcome(known < 0, unstable)
            entry["unfavourable_end_is_the_worst_end"] = float(
                (np.abs(known - corners) < 1e-6).mean()
            )
            entry["truth_above_the_unfavourable_end"] = int((true > known + 1e-9).sum())
            ratio = d["theta"][:, P["pll_hz"]]
            for label, key in (("step 14 rule as given", "nominal"),
                               ("step 14 rule at the unfavourable end", "known")):  # fmt: skip
                root = d[f"root_{key}{tag}"]
                slow_pll = ratio / np.maximum(d[f"root_f_{key}{tag}"], 1e-6) < 1.0
                entry[label] = outcome(np.where(slow_pll, root < -1.0, root < -5.0), unstable)
        # The margin on the nominal screen that clears as few unstable cases as the ends.
        target = entry["all 32 ends"]["unstable_cleared"]
        margins = entry["nominal with a margin"]
        match = [m for m in MARGINS if margins[f"{m:g}"]["unstable_cleared"] <= target]
        entry["margin_as_safe_as_all_ends"] = match[0] if match else None
        res["bounds"][f"{bound:g}"] = entry
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
                theta = thetas(kind, name)[:: 64 if args.small else 4]
                out = neterror.corner_cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"corner_{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"bounds": neterror.BOUNDS, "margins": MARGINS, "plants": {}}
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(kind, load(kind))
        print(plant, {k: v for k, v in r.items() if k != "bounds"})
        for bound, e in r["bounds"].items():
            print(f"  within {bound}:")
            rows = e["nominal with a margin"].items()
            table = {m: (round(o["cleared"], 3), o["unstable_cleared"]) for m, o in rows}
            print("    nominal + margin", table)
            for key, o in e.items():
                if isinstance(o, dict) and "cleared" in o:
                    print(f"    {key:38s} {o['cleared']:.1%} cleared, {o['unstable_cleared']} "
                          f"unstable, {o['stable_cleared']:.1%} of stable")  # fmt: skip
            print("    ", {k: v for k, v in e.items() if not isinstance(v, dict)})
    if not args.small:
        (OUT / "network_corners.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: cleared share against unstable cases cleared
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=report.SURFACE)
    for ax, (kind, plant) in zip(axes, PLANTS.items(), strict=True):
        r = res["plants"][plant]
        for bound, colour in zip(neterror.BOUNDS, report.SERIES, strict=False):
            e = r["bounds"][f"{bound:g}"]
            rows = [e["nominal with a margin"][f"{m:g}"] for m in MARGINS]
            ax.plot([o["unstable_cleared"] + 0.5 for o in rows], [o["cleared"] for o in rows],
                    color=colour, lw=1.6, marker="o", ms=4,
                    label=f"within {100 * bound:.0f} %: as given, margins 0 to 10 1/s")  # fmt: skip
            o = e["all 32 ends"]
            ax.plot([o["unstable_cleared"] + 0.5], [o["cleared"]], color=colour, marker="*",
                    ms=15, ls="none", label=f"within {100 * bound:.0f} %: all 32 ends")  # fmt: skip
            if kind == "t1":
                o = e["the one unfavourable end"]
                ax.plot([o["unstable_cleared"] + 0.5], [o["cleared"]], color=colour, marker="D",
                        ms=8, ls="none", mfc="none",
                        label=f"within {100 * bound:.0f} %: the one unfavourable end")  # fmt: skip
        ax.axhline(r["stable_share"], color=report.INK, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Unstable cases cleared (0.5 = none)")
        which = "sub-synchronous band" if r["band"] == "sub" else "all modes"
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {which}", loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Share of cases cleared (dashed: share that is stable)")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=report.LABEL, loc="lower right")
    report.title(fig, "Network known to within a bound: a margin on the network as given, "
                 "against the ends of the ranges")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig60_network_corners.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the margin the ends imply
    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=report.SURFACE)
    x = np.arange(len(neterror.BOUNDS))
    for k, (plant, colour) in enumerate(zip(PLANTS.values(), report.SERIES, strict=False)):
        q = np.array([res["plants"][plant]["bounds"][f"{b:g}"]["implied_margin_quartiles"]
                      for b in neterror.BOUNDS])  # fmt: skip
        ax.bar(x + (k - 1) * 0.26, q[:, 1], 0.24, color=colour, label=plant)
        ax.vlines(x + (k - 1) * 0.26, q[:, 0], q[:, 3], color=report.INK, lw=1)
    ax.set_xticks(x, [f"within {100 * b:.0f} %" for b in neterror.BOUNDS])
    ax.set_yscale("log")
    ax.set_ylabel("Worst end above the network as given (1/s)")
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    report.title(fig, "The margin the ends of the ranges imply, case by case "
                 "(bar: median; line: lower quartile to 95th percentile)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig61_implied_margin.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
