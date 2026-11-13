"""The slow mode of the doubly-fed plant: what it is, what drives it, and how small a
model reproduces it.

    uv run scripts/slow_mode.py            # 38,511 cases, about 10 minutes
    uv run scripts/slow_mode.py --small    # one sixteenth, for a quick look
    uv run scripts/slow_mode.py --reuse    # figures and numbers from the saved pass

Needs dataset T1 and data/t1-v2/scan_<set>.npz from scripts/phase3_scan_t1.py, whose
cases it uses. Writes data/t1-v2/slow_<set>.npz (not committed),
docs/results/slow/slow_mode.json and figures fig54 to fig56. Every number in
docs/steps/21-slow-mode.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import report, slowmode
from sso_sciml.data import t1
from sso_sciml.sampling import part as rows_of

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t1.VERSION
OUT = ROOT / "docs" / "results" / "slow"
COL = {name: j for j, name in enumerate(slowmode.COLUMNS)}
P = {name: j for j, name in enumerate(t1.NAMES)}
SETS = ("in_distribution", "boundary", *list(t1.SPLITS)[1:])
PHASOR_MARGIN = 0.05  # 1/s, on the slow mode of the phasor-domain model
MODELS = {
    "phasor": "phasor-domain, 17 states",
    "loops": "control loops only, 9 states",
    "held": "the same, rotor speed held, 8 states",
    "alone": "plant alone on an ideal voltage",
}
GROUPS = {
    "rotor current controller integrators": (11, 12),
    "outer-loop integrators": (15, 16),
    "rotor speed": (8,),
    "power filters": (13, 14),
    "rotor flux": (2, 3),
    "phase-locked loop": (9, 10),
    "stator, network and capacitor": (0, 1, 4, 5, 23, 24),
    "grid-side converter and DC link": (6, 7, 17, 18, 19, 20, 21, 22),
}


def load(name: str) -> dict:
    z, s = np.load(DATA / f"{name}.npz"), np.load(DATA / f"slow_{name}.npz")
    rows, ok = s["rows"], np.isfinite(s["out"][:, 0])
    d = {col: s["out"][ok, j] for col, j in COL.items()}
    d["theta"] = z["theta"][rows][ok]
    mech = z["mechanism"][rows][ok]
    m = {col: j for j, col in enumerate(t1.MECHANISM_COLUMNS)}
    root = mech[:, m["sigma_impedance"]]
    ratio = d["theta"][:, P["pll_hz"]] / np.maximum(mech[:, m["f_impedance"]], 1e-6)
    d["step14"] = np.where(ratio < 1.0, root < -1.0, root < -5.0)  # the rule of step 14
    return d


def pool(sets: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in sets]) for k in sets[0]}


def agreement(est: np.ndarray, true: np.ndarray) -> dict:
    err = np.abs(est - true)
    near = np.abs(true) < 1.0
    return {
        "error_median": float(np.median(err)),
        "error_p99": float(np.quantile(err, 0.99)),
        "error_max": float(err.max()),
        "near_boundary_error_p99": float(np.quantile(err[near], 0.99)),
        "unstable": int((true > 0).sum()),
        "unstable_called_stable": int(((true > 0) & (est <= 0)).sum()),
        "stable_called_unstable": int(((true <= 0) & (est > 0)).sum()),
    }


def study(d: dict) -> dict:
    slow, sub = d["true_slow"], d["true_sub"]
    un = slow > 0
    res: dict = {
        "cases": int(slow.size),
        "slow_unstable": int(un.sum()),
        "sub_synchronous_unstable": int((sub > 0).sum()),
        "fast_unstable": int((d["true_fast"] > 0).sum()),
        "both_unstable": int((un & (sub > 0)).sum()),
        "slow_growth_quartiles": [float(q) for q in np.quantile(slow, [0.25, 0.5, 0.75])],
        "slow_hz_quartiles_when_unstable": [
            float(q) for q in np.quantile(d["f_slow"][un], [0.25, 0.5, 0.75])
        ] if un.any() else [],
        "models": {name: agreement(d[f"{name}_slow"], slow) for name in MODELS},
        "phasor_on_the_sub_synchronous_band": {
            "unstable": int((sub > 0).sum()),
            "unstable_called_stable": int(((sub > 0) & ~(d["phasor_sub"] > 0)).sum()),
        },
        "participation_when_unstable": {
            label: float(np.mean(sum(d[f"part_{k}"][un] for k in ks))) if un.any() else None
            for label, ks in GROUPS.items()
        },
        "cured_by": {
            "outer loops at half speed": float((d["half_outer_slow"][un] < 0).mean())
            if un.any() else None,
            "rotor current loop twice as fast": float((d["double_rsc_slow"][un] < 0).mean())
            if un.any() else None,
        },
    }  # fmt: skip
    # Two models with no scan: the hand formula of step 14 for the resonance, the
    # phasor-domain model for everything below 2 Hz.
    unstable = (slow > 0) | (sub > 0) | (d["true_fast"] > 0)
    for label, cleared in (("step 14 rule alone", d["step14"]),
                           ("phasor-domain model alone", d["phasor_slow"] < 0),
                           ("both", d["step14"] & (d["phasor_slow"] < 0)),
                           ("both, phasor margin 0.05 1/s",
                            d["step14"] & (d["phasor_slow"] < -PHASOR_MARGIN))):  # fmt: skip
        res.setdefault("no_scan_screen", {})[label] = {
            "cleared": float(cleared.mean()),
            "unstable_cleared": int((cleared & unstable).sum()),
            "unstable": int(unstable.sum()),
        }
    return res


def by_bins(d: dict, name: str, edges: np.ndarray) -> list[dict]:
    x, un = d["theta"][:, P[name]], d["true_slow"] > 0
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        out.append({"lo": float(lo), "hi": float(hi), "cases": int(m.sum()),
                    "unstable": float(un[m].mean()) if m.any() else None})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        for name in SETS:
            rows = np.load(DATA / f"scan_{name}.npz")["rows"]
            rows = rows[rows_of(len(rows), 16 if args.small else 1)]
            theta = np.load(DATA / f"{name}.npz")["theta"][rows]
            np.savez_compressed(DATA / f"slow_{name}.npz", rows=rows,
                                out=slowmode.cases(theta, args.processes))  # fmt: skip
            print("done", name, len(rows))

    sets = {name: load(name) for name in SETS}
    inside = sets["in_distribution"]
    outside = pool([sets[s] for s in SETS[2:]])
    everything = pool(list(sets.values()))
    res: dict = {
        "in range": study(inside),
        "outside": study(outside),
        "by set": {name: {"cases": int(d["true_slow"].size),
                          "slow_unstable": int((d["true_slow"] > 0).sum())}
                   for name, d in sets.items()},
    }  # fmt: skip
    un = everything["true_slow"] > 0
    th = everything["theta"]
    res["all cases"] = {
        "cases": int(un.size),
        "slow_unstable": int(un.sum()),
        "largest_rsc_hz_of_an_unstable_case": float(th[un, P["rsc_hz"]].max()),
        "smallest_rotor_speed_of_an_unstable_case": float(th[un, P["w_r"]].min()),
        "unstable_share_with_rsc_above_10_hz": float(un[th[:, P["rsc_hz"]] > 10].mean()),
        "models": {name: agreement(everything[f"{name}_slow"], everything["true_slow"])
                   for name in MODELS},
    }  # fmt: skip
    edges = {
        "rsc_hz": np.geomspace(2.0, 30.0, 9),
        "w_r": np.linspace(0.8, 1.2, 9),
        "outer_ratio": np.geomspace(0.03, 0.15, 7),
        "r_r": np.linspace(0.004, 0.008, 7),
        "x_t": np.linspace(0.05, 0.25, 7),
        "k": np.linspace(0.1, 0.8, 8),
    }
    res["in range"]["unstable_share_by"] = {n: by_bins(inside, n, e) for n, e in edges.items()}
    # Rotor speed against current-loop bandwidth: where the slow mode is unstable.
    grid = np.full((len(edges["w_r"]) - 1, len(edges["rsc_hz"]) - 1), np.nan)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            m = ((inside["theta"][:, P["w_r"]] >= edges["w_r"][i])
                 & (inside["theta"][:, P["w_r"]] < edges["w_r"][i + 1])
                 & (inside["theta"][:, P["rsc_hz"]] >= edges["rsc_hz"][j])
                 & (inside["theta"][:, P["rsc_hz"]] < edges["rsc_hz"][j + 1]))  # fmt: skip
            grid[i, j] = (inside["true_slow"][m] > 0).mean() if m.any() else np.nan
    res["in range"]["unstable_share_grid_speed_by_rsc"] = grid.tolist()
    if not args.small:
        (OUT / "slow_mode.json").write_text(json.dumps(res, indent=2))
    for key in ("in range", "outside", "all cases"):
        r = res[key]
        skip = ("models", "unstable_share_by", "unstable_share_grid_speed_by_rsc")
        print(key, {k: v for k, v in r.items() if k not in skip})
        for name, a in r["models"].items():
            print(f"  {name:7s}", {k: (round(v, 5) if isinstance(v, float) else v)
                                   for k, v in a.items()})  # fmt: skip
    for n, rows in res["in range"]["unstable_share_by"].items():
        print(n, [None if b["unstable"] is None else round(b["unstable"], 3) for b in rows])

    # ---------------- figure: what the mode is made of and where it is unstable
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=report.SURFACE)
    part = res["in range"]["participation_when_unstable"]
    keys = sorted(part, key=part.get)
    axes[0].barh(range(len(keys)), [part[k] for k in keys], color=report.SERIES[0])
    axes[0].set_yticks(range(len(keys)), keys, fontsize=9)
    axes[0].set_xlabel("Share of the mode (participation), mean over unstable cases")
    axes[0].set_title("What the slow mode is made of", loc="left", fontsize=10)
    im = axes[1].imshow(100 * grid, origin="lower", aspect="auto", cmap="Oranges", vmin=0)
    axes[1].set_xticks(np.arange(len(edges["rsc_hz"])) - 0.5,
                       [f"{v:.1f}" for v in edges["rsc_hz"]])  # fmt: skip
    axes[1].set_yticks(np.arange(len(edges["w_r"])) - 0.5, [f"{v:.2f}" for v in edges["w_r"]])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            axes[1].text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                         color=report.INK)  # fmt: skip
    axes[1].set_xlabel("Rotor current loop bandwidth (Hz)")
    axes[1].set_ylabel("Rotor speed (pu)")
    axes[1].set_title("Cases with the slow mode unstable (%)", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for name, colour in zip(("outer_ratio", "r_r", "x_t", "k"), report.SERIES, strict=False):
        rows = res["in range"]["unstable_share_by"][name]
        lo, hi = rows[0]["lo"], rows[-1]["hi"]
        axes[2].plot([(0.5 * (b["lo"] + b["hi"]) - lo) / (hi - lo) for b in rows],
                     [100 * b["unstable"] for b in rows], color=colour, lw=2, marker="o", ms=4,
                     label={"outer_ratio": "outer loop speed over current loop",
                            "r_r": "rotor resistance", "x_t": "plant-side reactance",
                            "k": "series compensation"}[name])  # fmt: skip
    axes[2].set_xlabel("Position in the sampled range (0 = low, 1 = high)")
    axes[2].set_ylabel("Cases with the slow mode unstable (%)")
    axes[2].set_title("The other parameters", loc="left", fontsize=10)
    axes[2].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    for ax in (axes[0], axes[2]):
        report.style(ax)
    axes[0].grid(axis="y", visible=False)
    report.title(fig, "The slow mode of the doubly-fed plant, test fold of dataset T1")
    fig.tight_layout()
    fig.savefig(OUT / "fig54_slow_mode.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: reduced models against the full model
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8), facecolor=report.SURFACE)
    both = pool([inside, outside])
    lim = (-1.5, 1.5)
    for ax, (name, label) in zip(axes, MODELS.items(), strict=True):
        report.parity(
            ax, np.clip(both["true_slow"], *lim), np.clip(both[f"{name}_slow"], *lim), lim
        )
        a = agreement(both[f"{name}_slow"], both["true_slow"])
        ax.set_title(f"{label}\n{a['unstable_called_stable']} of {a['unstable']} unstable called "
                     "stable", loc="left", fontsize=9)  # fmt: skip
        ax.set_xlabel("Growth rate of the slow mode, full model (1/s)")
    axes[0].set_ylabel("Growth rate from the reduced model (1/s)")
    report.title(fig, "The slow mode from reduced models, in and outside the training range "
                 "(values beyond 1.5 1/s drawn at the edge)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig55_slow_models.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: which model sees which band
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=report.SURFACE)
    r = res["in range"]
    ph = r["phasor_on_the_sub_synchronous_band"]
    rows = [
        (
            "Phasor-domain model, slow mode",
            1 - r["models"]["phasor"]["unstable_called_stable"] / max(r["slow_unstable"], 1),
        ),
        (
            "Phasor-domain model, sub-synchronous mode",
            1 - ph["unstable_called_stable"] / max(ph["unstable"], 1),
        ),
        ("Step 14 rule, sub-synchronous mode", 1.0 - 0.0),
        ("Step 14 rule, slow mode", 0.0),
    ]
    scr = r["no_scan_screen"]
    alone = scr["step 14 rule alone"]
    rows[2] = ("Step 14 rule, any unstable case",
               1 - alone["unstable_cleared"] / alone["unstable"])  # fmt: skip
    both = scr["both, phasor margin 0.05 1/s"]
    rows[3] = ("Both together, any unstable case",
               1 - both["unstable_cleared"] / both["unstable"])  # fmt: skip
    ypos = np.arange(len(rows))[::-1]
    colours = [report.SERIES[0], report.SERIES[0], report.SERIES[1], report.SERIES[2]]
    ax.barh(ypos, [v for _, v in rows], color=colours, height=0.6)
    for y, (_, v) in zip(ypos, rows, strict=True):
        ax.text(v + 0.01, y, f"{100 * v:.2f} %", va="center", fontsize=9, color=report.LABEL)
    ax.set_yticks(ypos, [n for n, _ in rows], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Share of unstable cases caught")
    report.style(ax)
    ax.grid(axis="y", visible=False)
    report.title(fig, "Two models with no scan and no time-domain run, test fold")
    fig.tight_layout()
    fig.savefig(OUT / "fig56_two_models.png", dpi=160)
    plt.close(fig)
    print("no-scan screen", json.dumps(res["in range"]["no_scan_screen"]),
          json.dumps(res["outside"]["no_scan_screen"]))  # fmt: skip


if __name__ == "__main__":
    main()
