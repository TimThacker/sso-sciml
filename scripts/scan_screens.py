"""One screen for both plant types: tests on the eigenvalues of the loop matrix Y Z, from
2 x 2 scans of the plant admittance and the network impedance.

    uv run scripts/phase3_scan_t1.py         # doubly-fed plant, about 6 minutes
    uv run scripts/phase3_generate_c1.py     # generic converter, about 10 minutes
    uv run scripts/scan_screens.py

Nothing is fitted here, so every solved case is used: the test fold and the sets outside
the training range for the doubly-fed plant, every set for the converter. A case is
unstable if any mode of the full model grows.

Writes docs/results/scan/scan_screens.json and figures fig45 to fig47. Every number in
docs/steps/18-one-screen-two-plants.md comes from this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import admittance as adm
from sso_sciml import report
from sso_sciml.data import c1, t1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "scan"
PLANTS = ("doubly-fed plant", "converter with PLL", "converter without PLL")
POINTS = {1: 4001, **{k: 4000 // k + 1 for k in adm.STEPS}}


def load_t1(name: str) -> dict:
    z = np.load(ROOT / "data" / t1.VERSION / f"{name}.npz")
    s = np.load(ROOT / "data" / t1.VERSION / f"scan_{name}.npz")
    rows, ok = s["rows"], s["out"][:, 0] > 0
    d = {col: s["out"][ok, j] for j, col in enumerate(t1.SCAN_COLUMNS)}
    d["theta"] = z["theta"][rows][ok]
    d["sigma_sso"] = z["solver"][rows, 0][ok]
    mech = z["mechanism"][rows][ok]
    m = {col: j for j, col in enumerate(t1.MECHANISM_COLUMNS)}
    # The step 14 rule: impedance root with a margin that depends on the PLL speed.
    root = mech[:, m["sigma_impedance"]]
    ratio = d["theta"][:, t1.NAMES.index("pll_hz")] / np.maximum(mech[:, m["f_impedance"]], 1e-6)
    d["step14"] = np.where(ratio < 1.0, root < -1.0, root < -5.0)
    return d


def load_c1(family: str, split: str) -> dict:
    z = np.load(ROOT / "data" / c1.VERSION / f"{family}_{split}.npz")
    ok = z["out"][:, 0] > 0
    d = {col: z["out"][ok, j] for j, col in enumerate(c1.COLUMNS)}
    d["theta"] = z["theta"][ok]
    return d


def pool(sets: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in sets]) for k in sets[0]}


def outcome(cleared: np.ndarray, d: dict) -> dict:
    unstable = d["sigma"] > 0
    missed = d["sigma"][cleared & unstable]
    return {
        # Growth rate (1/s) of the unstable cases that were cleared: median and largest.
        "missed_growth_median": float(np.median(missed)) if missed.size else None,
        "missed_growth_max": float(missed.max()) if missed.size else None,
        "cleared": float(cleared.mean()),
        "stable_cleared": float(cleared[~unstable].mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
        "unstable": int(unstable.sum()),
        "cases": int(unstable.size),
    }


def screens(d: dict) -> dict[str, np.ndarray]:
    """True where a screen clears the case. All need the plant alone to be stable."""
    alone = d["poles"] == 0
    out = {
        "loop gain (norm) below 1 where not passive": alone & (d["norm"] < 1),
        "loop gain (eigenvalue) below 1 where not passive": alone & (d["radius"] < 1),
    }
    for g in adm.GUARDS:
        out[f"gain margin, guard {g:g} deg"] = alone & (d[f"guard_{g:g}"] < 1)
    for g in (10, 30):
        out[f"positive sequence only, gain margin, guard {g} deg"] = alone & (
            d[f"pos_guard_{g}"] < 1
        )
    out["positive sequence only, Nyquist count"] = d["count_pos"] == 0
    out["Nyquist count"] = d["count"] == 0
    return out


def coarse(d: dict) -> dict:
    """The same tests on scans with fewer points, and with noise."""
    alone = d["poles"] == 0
    by_points = {}
    for k in (1, *adm.STEPS):
        tag = "" if k == 1 else f"_s{k}"
        by_points[POINTS[k]] = {
            "Nyquist count": outcome(d[f"count{tag}"] == 0, d),
            **{f"guard {g} deg": outcome(alone & (d[f"guard_{g}{tag}"] < 1), d)
               for g in (10, 30, 60)},
        }  # fmt: skip
    by_noise = {}
    for level in adm.NOISE:
        tag = f"_n{round(100 * level)}"
        by_noise[level] = {
            "Nyquist count": outcome(d[f"count{tag}"] == 0, d),
            **{f"guard {g} deg": outcome(alone & (d[f"guard_{g}{tag}"] < 1), d) for g in (30, 60)},
        }
    return {"points": by_points, "noise_on_251_points": by_noise}


def loci(theta: np.ndarray) -> np.ndarray:
    """Eigenvalues of Y Z along the scan for one doubly-fed case, shape (points, 2)."""
    y, z, _ = t1.scan_pair(theta)
    return np.linalg.eigvals(y @ z)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t1_sets = {
        name: load_t1(name) for name in ("in_distribution", "boundary", *list(t1.SPLITS)[1:])
    }
    c1_sets = {
        family: {s: load_c1(family, s) for s in c1.SPLITS
                 if not (family == "follow" and s == "fast_pll")}
        for family in c1.FAMILIES
    }  # fmt: skip
    groups = {
        PLANTS[0]: {"in range": t1_sets["in_distribution"],
                    "outside": pool([t1_sets[s] for s in list(t1.SPLITS)[1:]]),
                    "boundary": t1_sets["boundary"]},
        PLANTS[1]: {"in range": c1_sets["pll"]["in_distribution"],
                    "outside": pool([v for k, v in c1_sets["pll"].items()
                                     if k != "in_distribution"])},
        PLANTS[2]: {"in range": c1_sets["follow"]["in_distribution"],
                    "outside": pool([v for k, v in c1_sets["follow"].items()
                                     if k != "in_distribution"])},
    }  # fmt: skip

    grid = adm.scan_grid()
    res: dict = {
        "plants": {},
        "points": POINTS,
        # Gap between neighbouring scan points as a share of their frequency.
        "relative_spacing": {POINTS[k]: float(grid[-1] / grid[-1 - k] - 1) for k in POINTS},
    }
    for plant, sets in groups.items():
        entry: dict = {"sets": {}, "screens": {}, "coarse": {}}
        for name, d in sets.items():
            unstable = d["sigma"] > 0
            entry["sets"][name] = {
                "cases": int(unstable.size),
                "unstable": int(unstable.sum()),
                "plant_alone_unstable": int((d["poles"] > 0).sum()),
                "count_agrees_with_eigenvalues": int(((d["count"] > 0) == unstable).sum()),
                "not_passive_up_to_hz_median": float(np.median(d["f_passive"])),
                "most_negative_conductance_median": float(np.median(d["g_min"])),
                "norm_median": float(np.median(d["norm"])),
            }
            if "sigma_sso" in d:
                other = unstable & (d["sigma_sso"] <= 0)
                entry["sets"][name].update({
                    "unstable_in_the_sub_synchronous_mode": int((d["sigma_sso"] > 0).sum()),
                    "unstable_in_another_mode_only": int(other.sum()),
                    "other_mode_hz_quartiles": [float(q) for q in
                                                np.quantile(d["f"][other], [0.25, 0.5, 0.75])]
                    if other.any() else [],
                    "other_mode_growth_median": float(np.median(d["sigma"][other]))
                    if other.any() else None,
                })  # fmt: skip
            for label, cleared in screens(d).items():
                entry["screens"].setdefault(label, {})[name] = outcome(cleared, d)
            if "step14" in d:
                o = outcome(d["step14"], d)
                o["sub_synchronous_unstable_cleared"] = int(
                    (d["step14"] & (d["sigma_sso"] > 0)).sum()
                )
                entry["screens"].setdefault("step 14 rule: impedance root with margin", {})[
                    name
                ] = o
            entry["coarse"][name] = coarse(d)
        res["plants"][plant] = entry
        print(plant, json.dumps(entry["sets"]))
        for label, r in entry["screens"].items():
            print(f"  {label:52s} " + "  ".join(
                f"{n}: {o['cleared']:6.1%} {o['unstable_cleared']}/{o['unstable']}"
                for n, o in r.items()))  # fmt: skip
        for n in ("in range", "outside"):
            for pts, r in entry["coarse"][n]["points"].items():
                print(
                    f"  {n} {pts:5d} points "
                    + "  ".join(
                        f"{k}: {o['cleared']:6.1%} {o['unstable_cleared']}" for k, o in r.items()
                    )
                )
            for lvl, r in entry["coarse"][n]["noise_on_251_points"].items():
                print(
                    f"  {n} noise {lvl} "
                    + "  ".join(
                        f"{k}: {o['cleared']:6.1%} {o['unstable_cleared']}" for k, o in r.items()
                    )
                )
    (OUT / "scan_screens.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: the loop eigenvalues of three doubly-fed cases
    d = t1_sets["in_distribution"]
    stable, unstable = d["sigma"] < -0.1, d["sigma"] > 0.5
    picks = {
        "Stable, cleared with a 30 degree guard": np.where(stable & (d["guard_30"] < 1))[0][0],
        "Stable, not cleared: the count must decide":
            np.where(stable & (d["guard_30"] > 2) & (d["guard_5"] < 1))[0][0],
        "Unstable": np.where(unstable & (d["sigma_sso"] > 0.5))[0][0],
    }  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), facecolor=report.SURFACE)
    turn = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    for ax, (label, row) in zip(axes, picks.items(), strict=True):
        lam = loci(d["theta"][row])
        reach = 4.0
        for g, shade in ((30.0, 0.10), (10.0, 0.16)):
            edge = np.tan(np.radians(g)) * reach
            ax.fill(
                [0, -reach, -reach], [0, edge, -edge], color=report.SERIES[1], alpha=shade, lw=0
            )
        ax.plot(turn.real, turn.imag, color=report.MUTED, lw=0.8, ls="--")
        ax.plot(lam.real, lam.imag, ".", ms=2.5, color=report.SERIES[0])
        ax.plot([-1], [0], "x", color=report.INK, ms=8, mew=1.6)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_xlabel("Real part of the eigenvalues of Y Z")
        ax.set_title(f"{label}\ngrowth {d['sigma'][row]:+.1f} 1/s, loop gain {d['norm'][row]:.0f}",
                     loc="left", fontsize=10)  # fmt: skip
        report.style(ax)
    axes[0].set_ylabel("Imaginary part")
    report.title(fig, "Doubly-fed plant: loop eigenvalues along the scan. Cross: -1. Dashed: "
                 "size 1. Shaded: within 10 and 30 degrees of the negative real axis")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig45_loop_eigenvalues.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: what each test clears on each plant
    labels = [
        "loop gain (norm) below 1 where not passive",
        "loop gain (eigenvalue) below 1 where not passive",
        "gain margin, guard 60 deg",
        "gain margin, guard 30 deg",
        "gain margin, guard 10 deg",
        "positive sequence only, gain margin, guard 30 deg",
        "positive sequence only, Nyquist count",
        "Nyquist count",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(labels))[::-1]
    for ax, plant in zip(axes, PLANTS, strict=True):
        scr = res["plants"][plant]["screens"]
        for y, label in zip(ypos, labels, strict=True):
            a, b = scr[label]["in range"], scr[label]["outside"]
            missed = a["unstable_cleared"] + b["unstable_cleared"]
            ax.barh(y, a["cleared"], color=report.SERIES[1] if missed else report.SERIES[2],
                    height=0.6)  # fmt: skip
            ax.text(a["cleared"] + 0.01, y, f"{a['cleared']:.0%}, {a['unstable_cleared']} + "
                    f"{b['unstable_cleared']} unstable", va="center", fontsize=8,
                    color=report.LABEL)  # fmt: skip
        stable = 1 - res["plants"][plant]["sets"]["in range"]["unstable"] / res["plants"][plant][
            "sets"]["in range"]["cases"]  # fmt: skip
        ax.axvline(stable, color=report.INK, lw=0.8, ls="--")
        ax.set_xlim(0, 1.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Share of cases cleared (dashed: share that is stable)")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, labels, fontsize=9)
    report.title(fig, "Tests on 2 x 2 scans, nothing fitted. Green: no unstable case cleared. "
                 "Label: unstable cleared in + outside the training range")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig46_scan_screens.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: guard angle, number of points, noise
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=report.SURFACE)
    for plant, colour in zip(PLANTS, report.SERIES, strict=False):
        scr = res["plants"][plant]["screens"]
        share = [scr[f"gain margin, guard {g:g} deg"]["in range"]["cleared"] for g in adm.GUARDS]
        axes[0].plot(adm.GUARDS, share, color=colour, lw=2, marker="o", ms=4, label=plant)
        pts = res["plants"][plant]["coarse"]["in range"]["points"]
        out = res["plants"][plant]["coarse"]["outside"]["points"]
        n = list(pts)
        for ax, key in ((axes[1], "Nyquist count"), (axes[2], "guard 30 deg")):
            ax.plot(n, [pts[k][key]["cleared"] for k in n], color=colour, lw=2, marker="o", ms=4,
                    label=plant)  # fmt: skip
            for k in n:
                missed = pts[k][key]["unstable_cleared"] + out[k][key]["unstable_cleared"]
                if missed:
                    ax.annotate(f"{missed}", (k, pts[k][key]["cleared"]), fontsize=8,
                                color=colour, xytext=(0, 6), textcoords="offset points",
                                ha="center")  # fmt: skip
    axes[0].set_xlabel("Guard angle either side of the negative real axis (degrees)")
    axes[0].set_ylabel("Share of cases cleared")
    axes[0].set_title("A wider guard clears less", loc="left", fontsize=10)
    for ax, name in ((axes[1], "Nyquist count"), (axes[2], "Gain margin, guard 30 degrees")):
        ax.set_xscale("log")
        ax.set_xlabel("Points in the scan, -2,000 to 2,000 Hz")
        ax.set_title(f"{name}: numbers are unstable cases cleared", loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylim(0, 1)
        report.style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.title(fig, "The guard angle, and how many points the scan needs")
    fig.tight_layout()
    fig.savefig(OUT / "fig47_guard_and_points.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
