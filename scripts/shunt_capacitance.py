"""A network with shunt capacitance on the plant bus.

    uv run scripts/shunt_capacitance.py            # 40,960 cases
    uv run scripts/shunt_capacitance.py --small    # one sixteenth, for a quick look
    uv run scripts/shunt_capacitance.py --reuse    # numbers and figures from the saved pass

Each in-range case of dataset C1 is run with the four converter variants of step 28, on
the network without and with a shunt capacitor. Needs dataset C1. Writes
data/shunt-v1/<family>.npz (not committed), docs/results/hf/shunt_capacitance.json and
figures fig73 and fig74. Every number in docs/steps/30-shunt-capacitance.md comes from
this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import hf, report, shunt
from sso_sciml.data import c1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "shunt-v1"
OUT = ROOT / "docs" / "results" / "hf"
PLANTS = {"pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(shunt.COLUMNS)}
P = {name: j for j, name in enumerate(c1.NAMES)}
LOW, HIGH = ("slow", "sub"), ("mid", "high")
NAMES = {"base": "as in step 17", "delay": "with delay", "capacitor": "with filter capacitor",
         "both": "with both"}  # fmt: skip
SCREENS = {
    "ignore": "shunt capacitance left out",
    "given": f"wrong by up to {shunt.BOUND:.0%}, as given",
    "ends": f"wrong by up to {shunt.BOUND:.0%}, two ends",
    "fit": "exact, rational fit to 10 kHz",
}
TURNS = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 5.0])


def load(kind: str) -> dict:
    z = np.load(DATA / f"{kind}.npz")
    ok = np.all([z["out"][:, COL[f"solved_{v}"]] > 0 for v in hf.VARIANTS], axis=0)
    d = {name: z["out"][ok, j] for name, j in COL.items()}
    d["theta"] = z["theta"][ok]
    d["rows"], d["all_solved"] = len(ok), int(ok.sum())
    return d


def band(d: dict, variant: str, names: tuple[str, ...], sh: bool) -> np.ndarray:
    return np.max([d[f"{b}{'_sh' if sh else ''}_{variant}"] for b in names], axis=0)


def binned(x: np.ndarray, flag: np.ndarray, edges: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "cases": int(m.sum()),
                     "share": float(flag[m].mean()) if m.sum() >= 30 else None})  # fmt: skip
    return rows


def quantiles(x: np.ndarray, q=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
    return [float(v) for v in np.quantile(x, q)] if x.size else []


def study(d: dict) -> dict:
    res: dict = {
        "cases": d["rows"],
        "solved_in_every_variant": d["all_solved"],
        "parallel_resonance_hz_quantiles": quantiles(d["f_par"]),
        "variants": {},
        "screens": {},
    }
    for v in hf.VARIANTS:
        plain, sh = d[f"top_{v}"] > 0, d[f"top_sh_{v}"] > 0
        low0, low1 = band(d, v, LOW, False), band(d, v, LOW, True)
        high0, high1 = band(d, v, HIGH, False) > 0, band(d, v, HIGH, True) > 0
        near = np.abs(low0) < 5.0
        f_top = d[f"f_top_sh_{v}"][high1 & sh]
        entry = {
            "unstable": [float(plain.mean()), float(sh.mean())],
            "unstable_above_58_hz": [float(high0.mean()), float(high1.mean())],
            "unstable_below_58_hz": [float((low0 > 0).mean()), float((low1 > 0).mean())],
            "stable_to_unstable": int((~plain & sh).sum()),
            "unstable_to_stable": int((plain & ~sh).sum()),
            "above_58_hz_made": int((~high0 & high1).sum()),
            "above_58_hz_cured": int((high0 & ~high1).sum()),
            "below_58_hz_verdict_changed": int(((low0 > 0) != (low1 > 0)).sum()),
            "below_58_hz_abs_shift_median": float(np.median(np.abs(low1 - low0)[near])),
            "below_58_hz_abs_shift_p95": float(np.quantile(np.abs(low1 - low0)[near], 0.95)),
            "plant_alone_unstable": int((d[f"alone_{v}"] > 0).sum()),
            "mode_hz_quantiles": quantiles(f_top),
            "mode_over_parallel_resonance": quantiles(f_top / d["f_par"][high1 & sh],
                                                      (0.25, 0.5, 0.75)),
            "above_58_hz_share_by": {
                "delay in periods of the parallel resonance": binned(
                    d["f_par"] * d["t_d"], high1, TURNS),
                "delay in periods of the filter resonance": binned(
                    d["f_lcl"] * d["t_d"], high1, TURNS),
                "shunt susceptance (pu)": binned(d["b_sh"], high1, np.geomspace(0.01, 0.3, 6)),
                "short-circuit ratio": binned(d["theta"][:, P["scr"]], high1,
                                              np.geomspace(1.5, 8.0, 6)),
                "damping resistor (pu)": binned(d["r_d"], high1, np.geomspace(0.02, 0.5, 6)),
            },
        }  # fmt: skip
        res["variants"][v] = entry
    for v in shunt.SCREENED:
        unstable = d[f"top_sh_{v}"] > 0
        res["screens"][v] = {"unstable": int(unstable.sum()), "cases": int(unstable.size)}
        for key in SCREENS:
            says = d[f"{key}_{v}"] > 0
            res["screens"][v][key] = {
                "unstable_called_stable": int((unstable & ~says).sum()),
                "stable_called_unstable": int((~unstable & says).sum()),
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
            theta = np.load(ROOT / "data" / c1.VERSION / f"{kind}_in_distribution.npz")["theta"]
            theta = theta[part(len(theta), 16 if args.small else 1)]
            np.savez_compressed(DATA / f"{kind}.npz", theta=theta,
                                out=shunt.cases(kind, theta, args.processes))  # fmt: skip
            print("done", kind, len(theta))

    data = {kind: load(kind) for kind in PLANTS}
    res = {"b_sh": shunt.B_SH, "bound": shunt.BOUND,
           "plants": {PLANTS[k]: study(d) for k, d in data.items()}}  # fmt: skip
    for name, r in res["plants"].items():
        print(name, r["cases"], "rows,", r["solved_in_every_variant"], "solved; parallel "
              "resonance", [round(q) for q in r["parallel_resonance_hz_quantiles"]])  # fmt: skip
        for v, e in r["variants"].items():
            print(f"  {v}")
            for k, x in e.items():
                if k != "above_58_hz_share_by":
                    print("     ", k, json.dumps(x))
            for k, rows in e["above_58_hz_share_by"].items():
                print("      by", k, [None if x["share"] is None else round(x["share"], 3)
                                      for x in rows])  # fmt: skip
        for v, e in r["screens"].items():
            print("  screens,", v, json.dumps(e))
    if not args.small:
        (OUT / "shunt_capacitance.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: each variant without and with the shunt capacitor
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE, sharey=True)
    x = np.arange(len(hf.VARIANTS))
    for ax, name in zip(axes, res["plants"], strict=True):
        v = res["plants"][name]["variants"]
        for k, label in enumerate(("network of R, X and a series capacitor",
                                   "with shunt capacitance")):  # fmt: skip
            low = np.array([100 * v[n]["unstable_below_58_hz"][k] for n in hf.VARIANTS])
            tot = np.array([100 * v[n]["unstable"][k] for n in hf.VARIANTS])
            colour = report.SERIES[k]
            ax.bar(x + (k - 0.5) * 0.38, low, 0.36, color=colour, label=f"{label}: below 58 Hz")
            ax.bar(x + (k - 0.5) * 0.38, tot - low, 0.36, bottom=low, color=colour, alpha=0.4,
                   label=f"{label}: above 58 Hz only")  # fmt: skip
            for xx, t in zip(x, tot, strict=True):
                ax.text(xx + (k - 0.5) * 0.38, t + 0.8, f"{t:.1f}", ha="center", fontsize=8,
                        color=report.LABEL)  # fmt: skip
        ax.set_xticks(x, [NAMES[n] for n in hf.VARIANTS])
        ax.set_title(name[0].upper() + name[1:], loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cases unstable (%)")
    axes[0].set_ylim(0, 68)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="upper left")
    report.title(fig, "The four converter variants of step 28 on a network without and with "
                 "shunt capacitance")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig73_shunt_capacitance.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: what decides it, and what a study misses
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    mid = np.arange(len(TURNS) - 1)
    styles = {"delay": "-", "both": "--"}
    for name, colour in zip(res["plants"], report.SERIES, strict=False):
        for v, ls in styles.items():
            rows = res["plants"][name]["variants"][v]["above_58_hz_share_by"][
                "delay in periods of the parallel resonance"]  # fmt: skip
            y = [np.nan if r["share"] is None else 100 * r["share"] for r in rows]
            axes[0].plot(mid, y, ls, color=colour, lw=2, marker="o", ms=4,
                         label=f"{name}, {NAMES[v]}")  # fmt: skip
    ticks = [f"{lo:g} to {hi:g}" for lo, hi in zip(TURNS[:-1], TURNS[1:], strict=True)]
    ticks[-1] = f"above {TURNS[-2]:g}"
    axes[0].set_xticks(mid, ticks, fontsize=8)
    axes[0].set_xlabel("Delay, in periods of the parallel resonance of the network")
    axes[0].set_ylabel("Cases unstable above 58 Hz, with shunt capacitance (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[0])
    ypos = np.arange(len(SCREENS))[::-1]
    k = 0
    for name in res["plants"]:
        for v in shunt.SCREENED:
            e = res["plants"][name]["screens"][v]
            vals = [e[key]["unstable_called_stable"] for key in SCREENS]
            off = (1.5 - k) * 0.2
            axes[1].barh(ypos + off, [val + 0.5 for val in vals], 0.18,
                         color=report.SERIES[k], label=f"{name}, {NAMES[v]}")  # fmt: skip
            for y, val in zip(ypos, vals, strict=True):
                axes[1].text(val + 0.5, y + off, f"  {val} of {e['unstable']}", va="center",
                             fontsize=7, color=report.LABEL)  # fmt: skip
            k += 1
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.4, 3e4)
    axes[1].set_yticks(ypos, list(SCREENS.values()), fontsize=9)
    axes[1].set_xlabel("Unstable cases called stable (0.5 = none)")
    axes[1].legend(frameon=False, fontsize=7, labelcolor=report.LABEL, loc="lower right")
    report.style(axes[1])
    axes[1].grid(axis="y", visible=False)
    report.title(fig, "Shunt capacitance in the network: where it makes an instability, and "
                 "what a study of the network misses")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig74_shunt_screens.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
