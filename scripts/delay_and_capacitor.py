"""Control delay and a filter capacitor in the generic converter.

    uv run scripts/delay_and_capacitor.py            # 40,960 cases, about 8 minutes
    uv run scripts/delay_and_capacitor.py --small    # one sixteenth, for a quick look
    uv run scripts/delay_and_capacitor.py --reuse    # numbers and figures from the saved pass

Each in-range case of dataset C1 is run as it is, with a delay, with a filter capacitor
and with both. Needs dataset C1. Writes data/hf-v1/<family>.npz (not committed),
docs/results/hf/delay_and_capacitor.json and figures fig69 to fig71. Every number in
docs/steps/28-delay-and-capacitor.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import hf, report
from sso_sciml.data import c1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hf-v1"
OUT = ROOT / "docs" / "results" / "hf"
PLANTS = {"pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(hf.COLUMNS)}
P = {name: j for j, name in enumerate(c1.NAMES)}
LOW, HIGH = ("slow", "sub"), ("mid", "high")


def load(kind: str) -> dict:
    z = np.load(DATA / f"{kind}.npz")
    ok = np.all([z["out"][:, COL[f"solved_{v}"]] > 0 for v in hf.VARIANTS], axis=0)
    d = {name: z["out"][ok, j] for name, j in COL.items()}
    d["theta"] = z["theta"][ok]
    d["rows"], d["all_solved"] = len(ok), int(ok.sum())
    return d


def band(d: dict, variant: str, names: tuple[str, ...]) -> np.ndarray:
    return np.max([d[f"{b}_{variant}"] for b in names], axis=0)


def binned(x: np.ndarray, flag: np.ndarray, edges: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "cases": int(m.sum()),
                     "share": float(flag[m].mean()) if m.sum() >= 30 else None})  # fmt: skip
    return rows


def study(d: dict) -> dict:
    res: dict = {"cases": d["rows"], "solved_in_every_variant": d["all_solved"], "variants": {}}
    for v in hf.VARIANTS:
        top = d[f"top_{v}"]
        res["variants"][v] = {
            "unstable": float((top > 0).mean()),
            "unstable_below_58_hz": float((band(d, v, LOW) > 0).mean()),
            "unstable_above_58_hz": float((band(d, v, HIGH) > 0).mean()),
            "plant_alone_unstable": float((d[f"alone_{v}"] > 0).mean()),
        }
    # Do the low bands care?
    low_base, low_both = band(d, "base", LOW), band(d, "both", LOW)
    near = np.abs(low_base) < 5.0
    shift = (low_both - low_base)[near]
    res["below 58 Hz, both against base"] = {
        "cases_near_boundary": int(near.sum()),
        "shift_quantiles": [float(q) for q in np.quantile(shift, [0.05, 0.25, 0.5, 0.75, 0.95])],
        "abs_shift_median": float(np.median(np.abs(shift))),
        "abs_shift_p95": float(np.quantile(np.abs(shift), 0.95)),
        "abs_shift_p99": float(np.quantile(np.abs(shift), 0.99)),
        "verdict_changed": int(((low_both > 0) != (low_base > 0)).sum()),
        "stable_to_unstable": int(((low_both > 0) & (low_base <= 0)).sum()),
        "unstable_to_stable": int(((low_both <= 0) & (low_base > 0)).sum()),
        "unstable_base": int((low_base > 0).sum()),
    }
    for v in ("delay", "capacitor"):
        s = (band(d, v, LOW) - low_base)[near]
        res[f"below 58 Hz, {v} against base"] = {
            "abs_shift_median": float(np.median(np.abs(s))),
            "abs_shift_p95": float(np.quantile(np.abs(s), 0.95)),
        }
    # The new instability.
    high = band(d, "both", HIGH) > 0
    f_top = d["f_top_both"][high & (d["top_both"] > 0)]
    res["above 58 Hz, both"] = {
        "unstable": int(high.sum()),
        "mode_hz_quartiles": [float(q) for q in np.quantile(f_top, [0.05, 0.25, 0.5, 0.75, 0.95])]
        if f_top.size else [],
        "above_2_khz": int((d["high_both"] > 0).sum()),
        "growth_median": float(np.median(band(d, "both", HIGH)[high])) if high.any() else None,
        "mode_over_lcl_frequency_quartiles": [
            float(q) for q in np.quantile(f_top / d["f_lcl"][high & (d["top_both"] > 0)],
                                          [0.25, 0.5, 0.75])] if f_top.size else [],
    }  # fmt: skip
    turns = d["f_lcl"] * d["t_d"]  # delay in periods of the filter resonance
    scr = d["theta"][:, P["scr"]]
    res["above 58 Hz, unstable share by"] = {
        "delay in periods of the filter resonance": binned(
            turns, high, np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 2.0])),
        "damping resistor (pu)": binned(d["r_d"], high, np.geomspace(0.02, 0.5, 6)),
        "capacitor (pu)": binned(d["b_c"], high, np.geomspace(0.02, 0.1, 5)),
        "short-circuit ratio": binned(scr, high, np.geomspace(1.5, 8.0, 6)),
        "delay times current-loop bandwidth": binned(
            d["t_d"] * d["theta"][:, P["cc_hz"]], high, np.geomspace(0.02, 0.1, 5)),
    }  # fmt: skip
    e_t = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 2.0])
    e_r = np.geomspace(0.02, 0.5, 6)
    grid = np.full((len(e_r) - 1, len(e_t) - 1), np.nan)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            m = ((d["r_d"] >= e_r[i]) & (d["r_d"] < e_r[i + 1])
                 & (turns >= e_t[j]) & (turns < e_t[j + 1]))  # fmt: skip
            if m.sum() >= 30:
                grid[i, j] = high[m].mean()
    cells = [[None if np.isnan(x) else float(x) for x in row] for row in grid]
    res["grid"] = {"turns_edges": e_t.tolist(), "r_d_edges": e_r.tolist(), "unstable": cells}
    # Passivity and the scan-based screen.
    unstable = d["top_both"] > 0
    res["passivity, both"] = {
        "not_passive_somewhere_above_58_hz": float(d["passive_above_58"].mean()),
        "not_passive_up_to_hz_quartiles": [float(q) for q in
                                           np.quantile(d["f_passive"], [0.25, 0.5, 0.75])],
    }  # fmt: skip
    for key, label in (("fit_2k", "scan to 2 kHz"), ("fit_10k", "scan to 10 kHz")):
        est = d[key] > 0
        res.setdefault("rational fit, both", {})[label] = {
            "unstable": int(unstable.sum()),
            "unstable_called_stable": int((unstable & ~est).sum()),
            "stable_called_unstable": int((~unstable & est).sum()),
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
                                out=hf.cases(kind, theta, args.processes))  # fmt: skip
            print("done", kind, len(theta))

    data = {kind: load(kind) for kind in PLANTS}
    res = {"ranges": hf.RANGES, "plants": {PLANTS[k]: study(d) for k, d in data.items()}}
    for name, r in res["plants"].items():
        print(name)
        for k, v in r.items():
            if k != "grid":
                print("  ", k, json.dumps(v))
    if not args.small:
        (OUT / "delay_and_capacitor.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: what each addition does, and where the new modes are
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE)
    x = np.arange(len(hf.VARIANTS))
    for k, (name, colour) in enumerate(zip(res["plants"], report.SERIES, strict=False)):
        v = res["plants"][name]["variants"]
        low = np.array([100 * v[n]["unstable_below_58_hz"] for n in hf.VARIANTS])
        tot = np.array([100 * v[n]["unstable"] for n in hf.VARIANTS])
        axes[0].bar(x + (k - 0.5) * 0.38, low, 0.36, color=colour, label=f"{name}: below 58 Hz")
        axes[0].bar(x + (k - 0.5) * 0.38, tot - low, 0.36, bottom=low, color=colour, alpha=0.4,
                    label=f"{name}: above 58 Hz only")  # fmt: skip
    axes[0].set_xticks(x, ["as in step 17", "with delay", "with capacitor", "with both"])
    axes[0].set_ylabel("Cases unstable (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for (kind, d), colour in zip(data.items(), report.SERIES, strict=False):
        m = d["top_both"] > 0
        axes[1].hist(d["f_top_both"][m], bins=np.geomspace(0.5, 4000, 60), color=colour,
                     alpha=0.6, label=PLANTS[kind])  # fmt: skip
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency of the fastest-growing mode, with both (Hz, synchronous frame)")
    axes[1].set_ylabel("Unstable cases")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    for ax in axes:
        report.style(ax)
    report.title(fig, "Control delay and a filter capacitor in the converter")
    fig.tight_layout()
    fig.savefig(OUT / "fig69_delay_and_capacitor.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: where the high-frequency instability is
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE)
    for ax, name in zip(axes, res["plants"], strict=True):
        g = res["plants"][name]["grid"]
        grid = np.array([[np.nan if v is None else v for v in row] for row in g["unstable"]])
        im = ax.imshow(100 * grid, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(g["turns_edges"])) - 0.5, [f"{v:g}" for v in g["turns_edges"]])
        ax.set_yticks(np.arange(len(g["r_d_edges"])) - 0.5, [f"{v:.2f}" for v in g["r_d_edges"]])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center",
                            fontsize=8, color=report.INK)  # fmt: skip
        ax.set_xlabel("Delay, in periods of the filter resonance")
        ax.set_ylabel("Damping resistor (pu)")
        ax.set_title(f"{name[0].upper() + name[1:]}: unstable above 58 Hz (%)", loc="left",
                     fontsize=10)  # fmt: skip
        fig.colorbar(im, ax=ax, fraction=0.046)
    report.title(fig, "The instability the delay and the capacitor make together")
    fig.tight_layout()
    fig.savefig(OUT / "fig70_high_frequency.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the low bands, with both against as in step 17
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), facecolor=report.SURFACE)
    lim = (-10.0, 10.0)
    for ax, (kind, d) in zip(axes, data.items(), strict=True):
        report.parity(ax, np.clip(band(d, "base", LOW), *lim), np.clip(band(d, "both", LOW), *lim),
                      lim)  # fmt: skip
        ax.set_xlabel("Largest growth rate below 58 Hz, as in step 17 (1/s)")
        ax.set_title(PLANTS[kind][0].upper() + PLANTS[kind][1:], loc="left", fontsize=10)
    axes[0].set_ylabel("The same with delay and capacitor (1/s)")
    report.title(fig, "Modes below 58 Hz with and without delay and capacitor")
    fig.tight_layout()
    fig.savefig(OUT / "fig71_low_bands.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
