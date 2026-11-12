"""Growth rate from a rational fit to a thin scan, as a screen with a margin in 1/s.

    uv run scripts/phase3_scan_t1.py         # doubly-fed plant scans
    uv run scripts/phase3_generate_c1.py     # converter dataset
    uv run scripts/phase3_fit_scans.py       # the fits, about 30 minutes
    uv run scripts/fit_screens.py

A case is cleared if the fitted growth rate is below minus the margin. It is scored as in
step 18, on every solved case, against the largest growth rate of any mode of the full
model. The tests of step 18 on the same thinned scans are shown beside it.

Writes docs/results/scan/fit_screens.json and figures fig48 to fig50. Every number in
docs/steps/19-rational-fit.md comes from this script.
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
POINTS = {k: 4000 // k + 1 for k in adm.STEPS}
MARGINS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
FIT = {name: j for j, name in enumerate(adm.FIT_COLUMNS)}


def load_t1(name: str) -> dict:
    folder = ROOT / "data" / t1.VERSION
    s, f = np.load(folder / f"scan_{name}.npz"), np.load(folder / f"fit_{name}.npz")
    assert np.array_equal(s["rows"], f["rows"])
    ok = s["out"][:, 0] > 0
    d = {col: s["out"][ok, j] for j, col in enumerate(t1.SCAN_COLUMNS)}
    d.update({col: f["out"][ok, j] for col, j in FIT.items()})
    return d


def load_c1(family: str, split: str) -> dict:
    folder = ROOT / "data" / c1.VERSION
    z, f = np.load(folder / f"{family}_{split}.npz"), np.load(folder / f"fit_{family}_{split}.npz")
    ok = z["out"][:, 0] > 0
    d = {col: z["out"][ok, j] for j, col in enumerate(c1.COLUMNS)}
    d.update({col: f["out"][ok, j] for col, j in FIT.items()})
    return d


def pool(sets: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in sets]) for k in sets[0]}


def outcome(cleared: np.ndarray, sigma: np.ndarray) -> dict:
    unstable = sigma > 0
    missed = sigma[cleared & unstable]
    return {
        "cleared": float(cleared.mean()),
        "stable_cleared": float(cleared[~unstable].mean()),
        "unstable_cleared": int(missed.size),
        "missed_growth_max": float(missed.max()) if missed.size else None,
    }


def accuracy(est: np.ndarray, sigma: np.ndarray) -> dict:
    """Error of the fitted growth rate (1/s), over all cases and over those within
    3 1/s of the stability boundary, where it decides the verdict."""
    err = np.abs(est - sigma)
    near = np.abs(sigma) < 3.0
    return {
        "median": float(np.median(err)),
        "p99": float(np.quantile(err, 0.99)),
        "share_within_0.01": float((err < 0.01).mean()),
        "near_boundary_cases": int(near.sum()),
        "near_boundary_p99": float(np.quantile(err[near], 0.99)),
        "near_boundary_max": float(err[near].max()),
        "near_boundary_share_within_0.01": float((err[near] < 0.01).mean()),
        "wrong_sign": int(((est > 0) != (sigma > 0)).sum()),
    }


def study(d: dict) -> dict:
    sigma, alone = d["sigma"], d["poles"] == 0
    res: dict = {"cases": int(sigma.size), "unstable": int((sigma > 0).sum()), "points": {},
                 "noise_on_251_points": {}}  # fmt: skip
    for k, n in POINTS.items():
        est = d[f"fit_s{k}"]
        res["points"][n] = {
            "terms_median": float(np.median(d[f"fit_terms_s{k}"])),
            "accuracy": accuracy(est, sigma),
            "fit": {f"{m:g}": outcome(est < -m, sigma) for m in MARGINS},
            "Nyquist count": outcome(d[f"count_s{k}"] == 0, sigma),
            "guard 30 deg": outcome(alone & (d[f"guard_30_s{k}"] < 1), sigma),
        }
    for level in adm.NOISE:
        tag = f"_n{round(100 * level)}"
        est = d[f"fit{tag}"]
        res["noise_on_251_points"][level] = {
            "terms_median": float(np.median(d[f"fit_terms{tag}"])),
            "accuracy": accuracy(est, sigma),
            "fit": {f"{m:g}": outcome(est < -m, sigma) for m in MARGINS},
            "Nyquist count": outcome(d[f"count{tag}"] == 0, sigma),
            "guard 30 deg": outcome(alone & (d[f"guard_30{tag}"] < 1), sigma),
        }
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c1_sets = {
        family: {s: load_c1(family, s) for s in c1.SPLITS
                 if not (family == "follow" and s == "fast_pll")}
        for family in c1.FAMILIES
    }  # fmt: skip
    groups = {
        PLANTS[0]: {"in range": load_t1("in_distribution"),
                    "outside": pool([load_t1(s) for s in list(t1.SPLITS)[1:]]),
                    "boundary": load_t1("boundary")},
        PLANTS[1]: {"in range": c1_sets["pll"]["in_distribution"],
                    "outside": pool([v for k, v in c1_sets["pll"].items()
                                     if k != "in_distribution"])},
        PLANTS[2]: {"in range": c1_sets["follow"]["in_distribution"],
                    "outside": pool([v for k, v in c1_sets["follow"].items()
                                     if k != "in_distribution"])},
    }  # fmt: skip
    res: dict = {"points": POINTS, "margins": MARGINS, "plants": {}}
    for plant, sets in groups.items():
        res["plants"][plant] = {name: study(d) for name, d in sets.items()}
        for name, r in res["plants"][plant].items():
            print(plant, name, r["cases"], "cases", r["unstable"], "unstable")
            for n, row in r["points"].items():
                a = row["accuracy"]
                print(f"  {n:5d} points, {row['terms_median']:.0f} terms: error median "
                      f"{a['median']:.1e} p99 {a['p99']:.1e}; near boundary p99 "
                      f"{a['near_boundary_p99']:.1e} max {a['near_boundary_max']:.1e}; wrong sign "
                      f"{a['wrong_sign']} | fit clears " + "  ".join(
                          f"{m}: {o['cleared']:.1%}/{o['unstable_cleared']}"
                          for m, o in row["fit"].items() if m in ("0", "0.1", "1"))
                      + f" | count {row['Nyquist count']['unstable_cleared']} guard "
                      f"{row['guard 30 deg']['unstable_cleared']}")  # fmt: skip
            for level, row in r["noise_on_251_points"].items():
                a = row["accuracy"]
                print(f"  noise {level}: {row['terms_median']:.0f} terms, error median "
                      f"{a['median']:.1e}; wrong sign {a['wrong_sign']} | fit clears " + "  ".join(
                          f"{m}: {o['cleared']:.1%}/{o['unstable_cleared']}"
                          for m, o in row["fit"].items() if m in ("0", "1", "5"))
                      + f" | count {row['Nyquist count']['cleared']:.1%}/"
                      f"{row['Nyquist count']['unstable_cleared']} guard "
                      f"{row['guard 30 deg']['cleared']:.1%}/"
                      f"{row['guard 30 deg']['unstable_cleared']}")  # fmt: skip
    (OUT / "fit_screens.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: fitted against true growth rate, 101-point scan
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), facecolor=report.SURFACE)
    for ax, plant in zip(axes, PLANTS, strict=True):
        d = pool([groups[plant]["in range"], groups[plant]["outside"]])
        est = np.clip(d["fit_s40"], -30, 30)
        report.parity(ax, np.clip(d["sigma"], -30, 30), est, (-30.0, 30.0))
        ax.set_xlabel("Largest growth rate of the full model (1/s)")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
    axes[0].set_ylabel("Growth rate from the fit (1/s)")
    report.title(fig, "Growth rate from a rational fit to a 101-point scan, in and outside the "
                 "training range (values beyond 30 1/s drawn at the edge)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig48_fit_parity.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: unstable cases cleared against the number of points
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), facecolor=report.SURFACE, sharey=True)
    n = list(POINTS.values())
    for ax, plant in zip(axes, PLANTS, strict=True):
        r = res["plants"][plant]
        for key, label, colour in (
            ("Nyquist count", "Nyquist count at the points", report.SERIES[1]),
            ("guard 30 deg", "gain margin, guard 30°", report.SERIES[0]),
            ("fit", "rational fit, roots", report.SERIES[2]),
        ):

            def missed(row, key=key):
                o = row["fit"]["0"] if key == "fit" else row[key]
                return o["unstable_cleared"]

            total = [missed(r["in range"]["points"][k]) + missed(r["outside"]["points"][k])
                     for k in n]  # fmt: skip
            ax.plot(n, np.array(total) + 0.5, color=colour, lw=2, marker="o", ms=4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Points in the scan, -2,000 to 2,000 Hz")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Unstable cases cleared, in and outside the range (0.5 = none)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.title(fig, "Unstable cases wrongly cleared as the scan is thinned, no margin")
    fig.tight_layout()
    fig.savefig(OUT / "fig49_fit_points.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: what a margin costs, and what noise does
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=report.SURFACE)
    for plant, colour in zip(PLANTS, report.SERIES, strict=False):
        r = res["plants"][plant]["in range"]
        clean = r["points"][251]["fit"]
        axes[0].plot(MARGINS[1:], [clean[f"{m:g}"]["cleared"] for m in MARGINS[1:]], color=colour,
                     lw=2, marker="o", ms=4, label=plant)  # fmt: skip
        axes[0].axhline(clean["0"]["cleared"], color=colour, lw=0.8, ls="--")
        x = [0.0, *adm.NOISE]
        rows = [r["points"][251], *(r["noise_on_251_points"][e] for e in adm.NOISE)]
        axes[1].plot(x, [row["fit"]["0"]["cleared"] for row in rows], color=colour, lw=2,
                     marker="o", ms=4, label=f"{plant}: fit")  # fmt: skip
        axes[1].plot(x, [row["guard 30 deg"]["cleared"] for row in rows], color=colour, lw=1.4,
                     ls=":", marker="s", ms=4, label=f"{plant}: guard 30°")  # fmt: skip
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Margin on the fitted growth rate (1/s)")
    axes[0].set_ylabel("Share of cases cleared")
    axes[0].set_title("Clean 251-point scan: what a margin costs (dashed: no margin)", loc="left",
                      fontsize=10)  # fmt: skip
    axes[1].set_xlabel("Noise on each entry of the admittance (standard deviation)")
    axes[1].set_title("251-point scan with noise: share cleared, no margin", loc="left",
                      fontsize=10)  # fmt: skip
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
        report.style(ax)
    report.title(fig, "Rational fit as a screen, cases in the training range")
    fig.tight_layout()
    fig.savefig(OUT / "fig50_fit_margin_noise.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
