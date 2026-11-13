"""A doubly-fed plant beside a converter, at one terminal and a transformer apart.

    uv run scripts/machine_beside_converter.py            # 32,768 cases, about 20 minutes
    uv run scripts/machine_beside_converter.py --small    # one sixteenth, for a quick look
    uv run scripts/machine_beside_converter.py --reuse    # numbers and figures from the saved pass

A doubly-fed plant (machine and controls of dataset T1) and a generic converter (dataset
C1, with a phase-locked loop as source or load, or without one as a load) share the
network of the converter's case. Scored against the modes of the two together:

    alone        each unit by itself on the network is stable
    scans added  each unit scanned by itself on the network, the two set side by side
    one bus      the two taken to be at one terminal when a transformer each lies between
    reference    the screen of step 24 on scans of each unit at the joint steady state

Writes data/beside-v1/<kind>.npz (not committed),
docs/results/multi/machine_beside_converter.json and figures fig77 and fig78. Every number
in docs/steps/32-machine-beside-converter.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import beside, report
from sso_sciml.data import c1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "beside-v1"
OUT = ROOT / "docs" / "results" / "multi"
COL = {name: j for j, name in enumerate(beside.COLUMNS)}
N = 16384
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def load(kind: str) -> dict:
    out = np.load(DATA / f"{kind}.npz")["out"]
    return {name: out[:, j] for name, j in COL.items()}


def osc(d: dict, tag: str, prefix: str = "") -> np.ndarray:
    """Largest growth rate above 2 Hz in the synchronous frame."""
    return np.maximum(d[f"{prefix}sub_{tag}"], d[f"{prefix}fast_{tag}"])


def screen(says_stable: np.ndarray, unstable: np.ndarray) -> dict:
    return {
        "cleared": float(says_stable.mean()),
        "unstable_cleared": int((says_stable & unstable).sum()),
        "unstable_cleared_share": float(says_stable[unstable].mean()) if unstable.any() else None,
        "stable_flagged": int((~says_stable & ~unstable).sum()),
        "stable_flagged_share": float((~says_stable)[~unstable].mean()),
    }


def quantiles(x: np.ndarray) -> list[float]:
    return [float(q) for q in np.quantile(x, QUANTILES)] if x.size else []


def study(kind: str, d: dict) -> dict:
    n = int(d["share"].size)
    solved = (d["solved_bus"] > 0) & (d["solved_apart"] > 0)
    cc_hz = np.array([job[2][c1.NAMES.index("cc_hz")] for job in beside.sample(n, kind)])[solved]
    d = {k: v[solved] for k, v in d.items()}
    bus, apart = osc(d, "bus"), osc(d, "apart")
    unstable = bus > 0
    m_ok, c_ok = d["solved_machine"] > 0, d["solved_converter"] > 0
    with np.errstate(invalid="ignore"):
        m_stable = m_ok & (osc(d, "machine") < 0)
        c_stable = c_ok & (osc(d, "converter") < 0)
    res: dict = {
        "cases": n,
        "solved": int(solved.sum()),
        "unstable_together": int(unstable.sum()),
        "unstable_share": float(unstable.mean()),
        "unstable_share_apart": float((apart > 0).mean()),
        "unstable_below_58": float((d["sub_bus"] > 0).mean()),
        "unstable_above_58": float((d["fast_bus"] > 0).mean()),
        "slow_mode_unstable": float((d["slow_bus"] > 0).mean()),
        "slow_mode_unstable_machine_alone": float((d["slow_machine"][m_ok] > 0).mean()),
        "machine_alone_unstable": float((~m_stable)[m_ok].mean()),
        "converter_alone_unstable": float((~c_stable)[c_ok].mean()),
        "machine_alone_no_steady_state": int((~m_ok).sum()),
        "converter_alone_no_steady_state": int((~c_ok).sum()),
        "unstable_poles_on_an_ideal_voltage": int((d["poles_bus"] > 0).sum()),
        "frequency_of_the_unstable_mode_hz": quantiles(d["f_top_bus"][unstable & (d["top_bus"] > 0)
                                                                       & (d["slow_bus"] < 0)]),
    }  # fmt: skip
    table = {}
    for a, sa in (("machine stable alone", m_stable), ("machine not", ~m_stable)):
        for b, sb in (("converter stable alone", c_stable), ("converter not", ~c_stable)):
            m = sa & sb
            table[f"{a}, {b}"] = {"cases": int(m.sum()), "share": float(m.mean()),
                                  "unstable_together": int(unstable[m].sum()),
                                  "unstable_together_share": float(unstable[m].mean())
                                  if m.any() else None}  # fmt: skip
    res["alone against together"] = table

    # What the converter does to the machine's mode below 58 Hz.
    shift = d["sub_bus"] - d["sub_machine"]
    near = m_ok & (np.abs(d["sub_machine"]) < 5.0)
    groups = {"all": np.ones_like(unstable)}
    if (d["p_converter"] > 0).any():
        groups["converter a source"] = d["p_converter"] > 0
        groups["converter a load"] = d["p_converter"] < 0
    res["shift of the growth rate below 58 Hz by the converter"] = {}
    for name, g in groups.items():
        k = g & near
        res["shift of the growth rate below 58 Hz by the converter"][name] = {
            "cases": int(g.sum()),
            "unstable_together": float(unstable[g].mean()),
            "machine_alone_unstable": float((~m_stable)[g & m_ok].mean()),
            "quantiles": quantiles(shift[k]),
            "raised": float((shift[k] > 0).mean()),
            "machine unstable alone, stable beside the converter":
                float((~unstable)[g & m_ok & ~m_stable].mean()),
            "machine stable alone, unstable beside the converter":
                float(unstable[g & m_stable].mean()),
        }  # fmt: skip
    edges = np.array([0.2, 0.4, 0.6, 0.8])
    res["shift by share of the machine"] = [
        {"lo": float(lo), "hi": float(hi),
         "median": float(np.median(shift[near & (d["share"] >= lo) & (d["share"] < hi)])),
         "unstable_together": float(unstable[(d["share"] >= lo) & (d["share"] < hi)].mean()),
         "machine_alone_unstable": float((~m_stable)[m_ok & (d["share"] >= lo)
                                                     & (d["share"] < hi)].mean())}
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]  # fmt: skip
    cuts = np.quantile(cc_hz, [0.0, 1 / 3, 2 / 3, 1.0])
    res["shift by current-loop bandwidth of the converter"] = [
        {"lo_hz": float(lo), "hi_hz": float(hi),
         "median": float(np.median(shift[near & (cc_hz >= lo) & (cc_hz <= hi)]))}
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)
    ]  # fmt: skip

    # Shortcuts and screens.
    with np.errstate(invalid="ignore"):
        added = np.isfinite(d["top_added"]) & (osc(d, "added") < 0)
    res["screens"] = {
        "each stable alone": screen(m_stable & c_stable, unstable),
        "scans taken alone, set side by side": screen(added, unstable),
        "reference, network as given": screen(osc(d, "bus", "given_") < 0, unstable),
        "reference, 32 ends": screen(osc(d, "bus", "ends_") < 0, unstable),
    }
    un_apart = apart > 0
    res["screens, a transformer apart"] = {
        "taken to be at one terminal": screen(bus < 0, un_apart),
        "reference, network as given": screen(osc(d, "apart", "given_") < 0, un_apart),
        "reference, 32 ends": screen(osc(d, "apart", "ends_") < 0, un_apart),
    }
    have = np.isfinite(d["top_added"])
    err = np.abs(osc(d, "added") - bus)[have & (np.abs(bus) < 5.0)]
    res["scans_added_error"] = {"median": float(np.median(err)),
                                "p95": float(np.quantile(err, 0.95))}  # fmt: skip
    res["reference_slow_band"] = {
        "true slow mode unstable": int((d["slow_bus"] > 0).sum()),
        "of those, the fit at the ends says stable": int(((d["slow_bus"] > 0)
                                                           & (d["ends_slow_bus"] < 0)).sum()),
        "stable, the fit at the ends says unstable": int(((d["slow_bus"] < 0)
                                                           & (d["ends_slow_bus"] > 0)).sum()),
    }  # fmt: skip

    # One terminal against a transformer apart.
    diff = apart - bus
    res["one terminal against apart"] = {
        "stable at one terminal, unstable apart": int((~unstable & un_apart).sum()),
        "unstable at one terminal, stable apart": int((unstable & ~un_apart).sum()),
        "shift_quantiles_near_the_boundary": quantiles(diff[np.abs(bus) < 5.0]),
        "abs_shift_median": float(np.median(np.abs(diff[np.abs(bus) < 5.0]))),
        "abs_shift_p95": float(np.quantile(np.abs(diff[np.abs(bus) < 5.0]), 0.95)),
        "machine_voltage_one_terminal": quantiles(d["v_machine_bus"]),
        "machine_voltage_apart": quantiles(d["v_machine_apart"]),
    }
    res["_plot"] = {
        "shift": {name: np.histogram(np.clip(shift[g & near], -6, 6), bins=48,
                                     range=(-6, 6))[0].tolist() for name, g in groups.items()},
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
        for kind in beside.KINDS:
            jobs = beside.sample(N // (16 if args.small else 1), kind)
            np.savez_compressed(DATA / f"{kind}.npz", out=beside.cases(jobs, args.processes))
            print("done", kind, len(jobs), flush=True)

    names = {"pll": "converter with PLL", "follow": "converter without PLL"}
    res = {names[kind]: study(kind, load(kind)) for kind in beside.KINDS}
    for name, r in res.items():
        print(name)
        for k, v in r.items():
            if k != "_plot":
                print("  ", k, json.dumps(v))
    plot = {name: r.pop("_plot") for name, r in res.items()}
    if not args.small:
        (OUT / "machine_beside_converter.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: alone and together, and what the converter does
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    bars = [("machine alone", lambda r: r["machine_alone_unstable"]),
            ("converter alone", lambda r: r["converter_alone_unstable"]),
            ("together,\none terminal", lambda r: r["unstable_share"]),
            ("together,\na transformer apart", lambda r: r["unstable_share_apart"])]  # fmt: skip
    x = np.arange(len(bars))
    for k, (name, colour) in enumerate(zip(res, report.SERIES, strict=False)):
        vals = [100 * f(res[name]) for _, f in bars]
        axes[0].bar(x + (k - 0.5) * 0.38, vals, 0.36, color=colour, label=name)
        for xi, v in zip(x, vals, strict=True):
            axes[0].text(xi + (k - 0.5) * 0.38, v + 0.3, f"{v:.1f}", ha="center", fontsize=8,
                         color=report.LABEL)  # fmt: skip
    axes[0].set_xticks(x, [b[0] for b in bars])
    axes[0].set_ylabel("Unstable above 2 Hz in the synchronous frame (%)")
    axes[0].set_title("Share of cases unstable", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    mid = np.linspace(-6, 6, 49)[:-1] + 0.125
    k = 0
    for name in res:
        for group, counts in plot[name]["shift"].items():
            if group == "all" and len(plot[name]["shift"]) > 1:
                continue
            label = f"{name}: {group}" if group != "all" else f"{name} (a load)"
            counts = np.array(counts, dtype=float)
            axes[1].plot(mid, 100 * counts / counts.sum(), color=report.SERIES[k], lw=2,
                         label=label)  # fmt: skip
            k += 1
    axes[1].axvline(0, color=report.INK, lw=0.8, ls="--")
    axes[1].set_xlabel("Growth rate below 58 Hz: together minus the machine alone (1/s)")
    axes[1].set_ylabel("Share of cases within 5 1/s of the boundary (%)")
    axes[1].set_title("What the converter does to the machine's mode", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for ax in axes:
        report.style(ax)
    axes[0].grid(axis="x", visible=False)
    report.title(fig, "A doubly-fed plant beside a converter")
    fig.tight_layout()
    fig.savefig(OUT / "fig77_machine_beside_converter.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the screens
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    rows = [("screens", "each stable alone", "each stable\nalone"),
            ("screens", "scans taken alone, set side by side", "scans taken\nalone"),
            ("screens, a transformer apart", "taken to be at one terminal",
             "one terminal\nassumed"),
            ("screens", "reference, network as given", "reference,\nas given"),
            ("screens", "reference, 32 ends", "reference,\n32 ends"),
            ("screens, a transformer apart", "reference, 32 ends",
             "reference, 32 ends,\napart")]  # fmt: skip
    x = np.arange(len(rows))
    for ax, key, what in ((axes[0], "unstable_cleared", "Unstable cases called stable"),
                          (axes[1], "stable_flagged", "Stable cases called unstable")):  # fmt: skip
        for k, (name, colour) in enumerate(zip(res, report.SERIES, strict=False)):
            vals = [100 * (res[name][g][s][f"{key}_share"] or 0) for g, s, _ in rows]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=colour, label=name)
            for xi, (g, s, _), v in zip(x, rows, vals, strict=True):
                ax.text(xi + (k - 0.5) * 0.38, v + 0.3, f"{res[name][g][s][key]}", ha="center",
                        fontsize=8, color=report.LABEL)  # fmt: skip
        ax.set_xticks(x, [r[2] for r in rows], fontsize=8)
        ax.set_ylabel(f"{what} (%)")
        ax.set_title(f"{what} (labels: cases)", loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.title(fig, "Screens for a doubly-fed plant beside a converter")
    fig.tight_layout()
    fig.savefig(OUT / "fig78_beside_screens.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
