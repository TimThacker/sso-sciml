"""A source and a load on one bus.

    uv run scripts/two_converters.py            # 65,536 cases, about 5 minutes
    uv run scripts/two_converters.py --small    # one sixteenth, for a quick look
    uv run scripts/two_converters.py --reuse    # numbers and figures from the saved pass

A source with a phase-locked loop and a load (with a phase-locked loop, or following the
voltage waveform) share a bus and a network. Three shortcuts are scored against the modes
of the two together:

    alone        each unit by itself on the network is stable
    scans added  each unit scanned by itself on the network, the two scans added
    net power    one converter carrying the net power of the bus

Writes data/multi-v1/<load>.npz (not committed), docs/results/multi/two_converters.json
and figures fig67 and fig68. Every number in docs/steps/27-two-converters.md comes from
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

from sso_sciml import multi, report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "multi-v1"
OUT = ROOT / "docs" / "results" / "multi"
COL = {name: j for j, name in enumerate(multi.COLUMNS)}
NAME = {"pll": "load with PLL", "follow": "load without PLL"}
N = 32768


def load(kind: str) -> dict:
    out = np.load(DATA / f"{kind}.npz")["out"]
    return {name: out[:, j] for name, j in COL.items()}


def shortcut(says_stable: np.ndarray, unstable: np.ndarray) -> dict:
    return {
        "cleared": float(says_stable.mean()),
        "unstable_cleared": int((says_stable & unstable).sum()),
        "unstable_cleared_share": float(says_stable[unstable].mean()),
        "stable_flagged": int((~says_stable & ~unstable).sum()),
    }


def binned(x: np.ndarray, unstable: np.ndarray, edges: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "cases": int(m.sum()),
                     "unstable": float(unstable[m].mean()) if m.sum() >= 30 else None})  # fmt: skip
    return rows


def study(d: dict) -> dict:
    solved = d["solved"] > 0
    n = int(solved.size)
    d = {k: v[solved] for k, v in d.items()}
    together = d["together"]
    unstable = together > 0
    s_ok, l_ok = d["solved_source"] > 0, d["solved_load"] > 0
    s_stable, l_stable = s_ok & (d["source_alone"] < 0), l_ok & (d["load_alone"] < 0)
    res: dict = {
        "cases": n,
        "no_steady_state_together": int(n - solved.sum()),
        "solved": int(solved.sum()),
        "unstable_together": int(unstable.sum()),
        "unstable_share": float(unstable.mean()),
        "source_alone_unstable_or_none": float((~s_stable).mean()),
        "load_alone_unstable_or_none": float((~l_stable).mean()),
        "load_alone_no_steady_state": float((~l_ok).mean()),
    }
    # Alone against together.
    table = {}
    for a, sa in (("source stable alone", s_stable), ("source not", ~s_stable)):
        for b, lb in (("load stable alone", l_stable), ("load not", ~l_stable)):
            m = sa & lb
            table[f"{a}, {b}"] = {"cases": int(m.sum()), "share": float(m.mean()),
                                  "unstable_together": float(unstable[m].mean()) if m.any()
                                  else None}  # fmt: skip
    res["alone against together"] = table
    both = s_stable & l_stable
    worse = together - np.maximum(d["source_alone"], d["load_alone"])
    ok = both & np.isfinite(worse)
    res["together_minus_the_worse_alone_quantiles"] = [
        float(q) for q in np.quantile(worse[ok], [0.05, 0.25, 0.5, 0.75, 0.95])
    ]
    res["the_other_unit_cures"] = {
        "source unstable alone, stable with the load": float((~unstable)[s_ok & ~s_stable].mean())
        if (s_ok & ~s_stable).any() else None,
        "load unstable alone, stable with the source": float((~unstable)[l_ok & ~l_stable].mean())
        if (l_ok & ~l_stable).any() else None,
        "load with no steady state alone, stable with the source": float((~unstable)[~l_ok].mean())
        if (~l_ok).any() else None,
    }  # fmt: skip
    # Shortcuts.
    added = np.isfinite(d["scans_added"]) & (d["scans_added"] < 0)
    net = (d["solved_net"] > 0) & (d["net_equivalent"] < 0)
    res["shortcuts"] = {
        "each stable alone": shortcut(both, unstable),
        "scans taken alone, added": shortcut(added, unstable),
        "one converter with the net power": shortcut(net, unstable),
    }
    have = np.isfinite(d["scans_added"])
    err = np.abs(d["scans_added"] - together)[have]
    near = np.abs(together[have]) < 5.0
    res["scans_added_error"] = {
        "median": float(np.median(err[near])),
        "p95": float(np.quantile(err[near], 0.95)),
        "p99": float(np.quantile(err[near], 0.99)),
    }
    res["reactive_current_shift_of_the_source"] = [
        float(q) for q in np.quantile(
            np.abs(d["iq_together"] - d["iq_source_alone"])[s_ok], [0.5, 0.95, 0.99])
    ]  # fmt: skip
    # Net against gross power.
    gross = d["share"] * d["p_source"] + (1 - d["share"]) * np.abs(d["p_load"])
    edges = np.linspace(0.0, 1.0, 6)
    res["unstable by net power"] = binned(np.abs(d["p_net"]), unstable, edges)
    res["unstable by gross power"] = binned(gross, unstable, edges)
    small = np.abs(d["p_net"]) < 0.1
    res["net power below 0.1 pu"] = {
        "cases": int(small.sum()),
        "unstable": float(unstable[small].mean()),
        "gross_power_median": float(np.median(gross[small])),
    }
    grid = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            size = np.abs(d["p_net"])
            m = ((gross >= edges[i]) & (gross < edges[i + 1])
                 & (size >= edges[j]) & (size < edges[j + 1]))  # fmt: skip
            if m.sum() >= 30:
                grid[i, j] = unstable[m].mean()
    res["unstable_grid_gross_by_net"] = [[None if np.isnan(v) else float(v) for v in row]
                                         for row in grid]  # fmt: skip
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
        for kind in multi.LOADS:
            jobs = multi.sample(N // (16 if args.small else 1), kind)
            np.savez_compressed(DATA / f"{kind}.npz", out=multi.cases(jobs, args.processes))
            print("done", kind, len(jobs))

    res = {NAME[kind]: study(load(kind)) for kind in multi.LOADS}
    for name, r in res.items():
        print(name)
        for k, v in r.items():
            if k not in ("unstable_grid_gross_by_net",):
                print("  ", k, json.dumps(v))
    if not args.small:
        (OUT / "two_converters.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: alone against together, and the shortcuts
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    keys = ["source stable alone, load stable alone", "source stable alone, load not",
            "source not, load stable alone", "source not, load not"]  # fmt: skip
    x = np.arange(len(keys))
    for k, (name, colour) in enumerate(zip(res, report.SERIES, strict=False)):
        t = res[name]["alone against together"]
        vals = [100 * (t[key]["unstable_together"] or 0) for key in keys]
        axes[0].bar(x + (k - 0.5) * 0.38, vals, 0.36, color=colour, label=name)
        for xi, key, v in zip(x, keys, vals, strict=True):
            axes[0].text(xi + (k - 0.5) * 0.38, v + 1, f"{100 * t[key]['share']:.0f} %",
                         ha="center", fontsize=8, color=report.LABEL)  # fmt: skip
    axes[0].set_xticks(x, ["both stable\nalone", "load not", "source not", "neither"])
    axes[0].set_ylabel("Unstable together (%)")
    axes[0].set_title("By what each unit does alone (labels: share of cases)", loc="left",
                      fontsize=10)  # fmt: skip
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    names = list(next(iter(res.values()))["shortcuts"])
    x = np.arange(len(names))
    for k, (name, colour) in enumerate(zip(res, report.SERIES, strict=False)):
        s = res[name]["shortcuts"]
        vals = [100 * s[n]["unstable_cleared_share"] for n in names]
        axes[1].bar(x + (k - 0.5) * 0.38, vals, 0.36, color=colour, label=name)
        for xi, n, v in zip(x, names, vals, strict=True):
            axes[1].text(xi + (k - 0.5) * 0.38, v + 1, f"{s[n]['unstable_cleared']}",
                         ha="center", fontsize=8, color=report.LABEL)  # fmt: skip
    axes[1].set_xticks(x, ["each stable\nalone", "scans taken alone,\nadded",
                           "one converter with\nthe net power"])  # fmt: skip
    axes[1].set_ylabel("Unstable cases the shortcut calls stable (%)")
    axes[1].set_title("Three shortcuts (labels: cases)", loc="left", fontsize=10)
    for ax in axes:
        report.style(ax)
        ax.grid(axis="x", visible=False)
    report.title(fig, "A source and a load on one bus")
    fig.tight_layout()
    fig.savefig(OUT / "fig67_two_converters.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: net against gross power
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), facecolor=report.SURFACE)
    for name, colour in zip(res, report.SERIES, strict=False):
        for key, style in (("unstable by net power", "-"), ("unstable by gross power", ":")):
            rows = [b for b in res[name][key] if b["unstable"] is not None]
            which = "net" if style == "-" else "gross"
            axes[0].plot([0.5 * (b["lo"] + b["hi"]) for b in rows],
                         [100 * b["unstable"] for b in rows], color=colour, ls=style, lw=2,
                         marker="o", ms=4, label=f"{name}: against {which} power")  # fmt: skip
    axes[0].set_xlabel("Power on the common rating (pu)")
    axes[0].set_ylabel("Unstable together (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[0])
    for ax, name in zip(axes[1:], res, strict=True):
        grid = np.array([[np.nan if v is None else v for v in row]
                         for row in res[name]["unstable_grid_gross_by_net"]])  # fmt: skip
        im = ax.imshow(100 * grid, origin="lower", aspect="auto", cmap="Oranges", vmin=0)
        ticks = [f"{v:.1f}" for v in np.linspace(0, 1, 6)]
        ax.set_xticks(np.arange(6) - 0.5, ticks)
        ax.set_yticks(np.arange(6) - 0.5, ticks)
        for i in range(5):
            for j in range(5):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center",
                            fontsize=8, color=report.INK)  # fmt: skip
        ax.set_xlabel("Net power out of the bus, size (pu)")
        ax.set_ylabel("Gross power: source plus load (pu)")
        ax.set_title(f"{name[0].upper() + name[1:]}: unstable (%)", loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    report.title(fig, "Unstable share against the net power of the bus and the gross power of "
                 "its two converters")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig68_net_and_gross.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
