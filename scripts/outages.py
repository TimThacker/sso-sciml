"""One outage at a time: what it does to stability, and whether one scan of the plant can
serve every network condition.

    uv run scripts/outages.py            # 98,635 cases
    uv run scripts/outages.py --small    # one sixteenth, for a quick look
    uv run scripts/outages.py --reuse    # numbers and figures from the saved pass

Needs dataset T1 with data/t1-v2/scan_<set>.npz and dataset C1. Writes
data/outage-v1/<kind>_<set>.npz (not committed), docs/results/network/outages.json and
figures fig63 and fig64. Every number in docs/steps/25-outages.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import outage, report
from sso_sciml.data import c1, t1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "outage-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"t1": "doubly-fed plant", "pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(outage.COLUMNS)}
MARGINS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
NAMES = list(outage.OUTAGES)


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
    out = np.concatenate([np.load(DATA / f"{kind}_{n}.npz")["out"] for n in sets_of(kind)])
    out = out[np.isfinite(out[:, 0])]
    return {name: out[:, j] for name, j in COL.items()}


# The band each plant is judged on, as in steps 22 to 24: the resonance for the
# doubly-fed plant, whose slow mode goes to the phasor-domain model; everything for a
# converter.
BAND = {"t1": "sub", "pll": "all", "follow": "all"}


def top(d: dict, prefix: str, band: str = "all") -> np.ndarray:
    if band != "all":
        return d[prefix.format(b=band)]
    return np.max([d[f"{prefix.format(b=b)}"] for b in outage.BANDS], axis=0)


def band_of(d: dict, prefix: str) -> np.ndarray:
    """Which band holds the least-damped mode: 0 slow, 1 sub-synchronous, 2 fast."""
    return np.argmax([d[f"{prefix.format(b=b)}"] for b in outage.BANDS], axis=0)


def study(kind: str, d: dict) -> dict:
    band = BAND[kind]
    intact = top(d, "intact_{b}", band)
    ok = intact < 0
    res: dict = {"cases": int(intact.size), "band": band, "stable_intact": int(ok.sum()),
                 "outages": {}}  # fmt: skip
    worst_bad = np.zeros(intact.size, dtype=bool)
    for j, name in enumerate(NAMES):
        solved = d[f"solved_{j}"] > 0
        true, reuse = top(d, f"true_{{b}}_{j}", band), top(d, f"reuse_{{b}}_{j}", band)
        true = np.where(solved, true, np.nan)
        bad = ~solved | (true > 0)  # insecure: unstable, or no steady state at all
        worst_bad |= bad
        both = ok & solved
        err = np.abs(reuse - true)[both]
        near = both & (np.abs(true) < 5.0)
        where = band_of(d, f"true_{{b}}_{j}")
        entry = {
            "no_steady_state_after": float((~solved)[ok].mean()),
            "unstable_after": float((solved & (true > 0))[ok].mean()),
            "insecure_after": float(bad[ok].mean()),
            "unstable_intact_and_stable_after": float((solved & (true < 0))[~ok].mean())
            if (~ok).any() else None,
            "growth_shift_median": float(np.median((true - intact)[both])),
            "growth_shift_quartiles": [float(q) for q in
                                       np.quantile((true - intact)[both], [0.25, 0.5, 0.75])],
            "band_of_the_unstable_mode": {
                b: float((where[ok & solved & (true > 0)] == k).mean())
                for k, b in enumerate(outage.BANDS)
            } if (ok & solved & (true > 0)).any() else None,
            "one scan": {
                "error_median": float(np.median(err)),
                "error_p95": float(np.quantile(err, 0.95)),
                "near_boundary_error_median": float(np.median(np.abs(reuse - true)[near])),
                "near_boundary_error_p95": float(np.quantile(np.abs(reuse - true)[near], 0.95)),
                # Among cases stable before the outage. A case with no steady state
                # after it counts as insecure whatever the scan says.
                "margins": {
                    f"{m:g}": {
                        "cleared": float((reuse < -m)[ok].mean()),
                        "insecure_cleared": int(((reuse < -m) & bad & ok).sum()),
                        "unstable_cleared": int(((reuse < -m) & solved & (true > 0) & ok).sum()),
                        "no_steady_state_cleared": int(((reuse < -m) & ~solved & ok).sum()),
                    }
                    for m in MARGINS
                },
            },
            # Does a margin before the outage say anything about after it?
            "insecure_after_given_margin_intact": {
                f"{m:g}": float(bad[intact < -m].mean()) if (intact < -m).any() else None
                for m in MARGINS
            },
        }  # fmt: skip
        res["outages"][name] = entry
    res["insecure_after_any_outage"] = float(worst_bad[ok].mean())
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
                theta = theta[part(len(theta), 16 if args.small else 1)]
                out = outage.cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"outages": {k: list(v) for k, v in outage.OUTAGES.items()}, "plants": {}}
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(kind, load(kind))
        print(plant, r["cases"], "cases,", r["stable_intact"], "stable intact, insecure after "
              f"any outage {r['insecure_after_any_outage']:.1%}")  # fmt: skip
        for name, e in r["outages"].items():
            one = e["one scan"]
            print(f"  {name:34s} unstable {e['unstable_after']:.1%} no steady state "
                  f"{e['no_steady_state_after']:.1%} shift {e['growth_shift_median']:+.2f} "
                  f"bands {e['band_of_the_unstable_mode']} cured "
                  f"{e['unstable_intact_and_stable_after']}")  # fmt: skip
            cells = "  ".join(
                f"{m}: {o['cleared']:.1%}/{o['unstable_cleared']}+{o['no_steady_state_cleared']}"
                for m, o in one["margins"].items()
            )
            print(f"      one scan: error near boundary median "
                  f"{one['near_boundary_error_median']:.3f} p95 "
                  f"{one['near_boundary_error_p95']:.2f} | {cells}")  # fmt: skip
            print("      insecure given intact margin",
                  {m: (None if v is None else round(v, 3))
                   for m, v in e["insecure_after_given_margin_intact"].items()})  # fmt: skip
    if not args.small:
        (OUT / "outages.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: what each outage does
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    x = np.arange(len(NAMES))
    for k, (plant, colour) in enumerate(zip(PLANTS.values(), report.SERIES, strict=False)):
        o = res["plants"][plant]["outages"]
        un = np.array([o[n]["unstable_after"] for n in NAMES])
        none = np.array([o[n]["no_steady_state_after"] for n in NAMES])
        axes[0].bar(x + (k - 1) * 0.26, 100 * un, 0.24, color=colour, label=plant)
        axes[0].bar(x + (k - 1) * 0.26, 100 * none, 0.24, bottom=100 * un, color=colour,
                    alpha=0.4)  # fmt: skip
        q = np.array([o[n]["growth_shift_quartiles"] for n in NAMES])
        spread = [q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]]
        axes[1].errorbar(x + (k - 1) * 0.2, q[:, 1], yerr=spread, color=colour, marker="o",
                         ms=5, ls="none", capsize=3, label=plant)  # fmt: skip
    for ax in axes:
        ax.set_xticks(x, [n.replace(" out", "\nout") for n in NAMES], fontsize=9)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cases stable before that are not after (%)")
    axes[0].set_title("Solid: unstable. Pale: no steady state", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    axes[1].axhline(0, color=report.INK, lw=0.8)
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_ylabel("Change of the largest growth rate (1/s), median and quartiles")
    report.title(fig, "One outage at a time, cases stable on the intact network")
    fig.tight_layout()
    fig.savefig(OUT / "fig63_outages.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: one scan for every network condition
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        o = res["plants"][plant]["outages"]
        for name, colour in zip(NAMES, report.SERIES, strict=False):
            rows = o[name]["one scan"]["margins"]
            ax.plot(range(len(MARGINS)), [rows[f"{m:g}"]["cleared"] for m in MARGINS],
                    color=colour, lw=2, marker="o", ms=4, label=name)  # fmt: skip
            for j, m in enumerate(MARGINS):
                n = rows[f"{m:g}"]["insecure_cleared"]
                if n:
                    ax.annotate(f"{n}", (j, rows[f"{m:g}"]["cleared"]), fontsize=8, color=colour,
                                xytext=(0, 5), textcoords="offset points", ha="center")  # fmt: skip
        ax.set_xticks(range(len(MARGINS)), [f"{m:g}" for m in MARGINS])
        ax.set_xlabel("Margin on the growth rate (1/s)")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        ax.set_ylim(0, 1)
        report.style(ax)
    axes[0].set_ylabel("Share of cases cleared for the outage")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower left")
    report.title(fig, "The plant scanned once, before the outage, closed on the network after "
                 "it. Numbers: insecure cases cleared")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig64_one_scan.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
