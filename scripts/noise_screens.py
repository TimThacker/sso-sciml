"""What noise on the scan does to a fitted model, mode by mode.

    uv run scripts/phase3_noise_fits.py     # the fits, about 25 minutes
    uv run scripts/noise_screens.py

For each plant type, noise level and frequency band: the error of the fitted growth
rate, and the outcome of clearing a case when the fitted growth rate in that band is
below minus a margin. A band is unstable if a mode of the full model in it grows.

Writes docs/results/scan/noise_screens.json and figures fig51 to fig53. Every number in
docs/steps/20-noise-and-modes.md comes from this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import report, scanfit
from sso_sciml.data import c1, t1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "noise-v1"
OUT = ROOT / "docs" / "results" / "scan"
PLANTS = {"t1": "doubly-fed plant", "pll": "converter with PLL", "follow": "converter without PLL"}
MARGINS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
COL = {name: j for j, name in enumerate(scanfit.COLUMNS)}
LABEL = {
    (16, 0.0): "126 points, clean",
    (16, 0.001): "126 points, 0.1 % noise",
    (16, 0.01): "126 points, 1 % noise",
    (16, 0.05): "126 points, 5 % noise",
    (4, 0.05): "501 points, 5 % noise",
}


def load(kind: str) -> np.ndarray:
    """Every solved case of one plant type, in and outside the training range."""
    if kind == "t1":
        names = ("in_distribution", "boundary", *list(t1.SPLITS)[1:])
    else:
        names = [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]
    out = np.concatenate([np.load(DATA / f"{kind}_{n}.npz")["out"] for n in names])
    return out[np.isfinite(out[:, 0])]


def outcome(est: np.ndarray, true: np.ndarray, margin: float) -> dict:
    cleared, unstable = est < -margin, true > 0
    return {
        "cleared": float(cleared.mean()),
        "stable_cleared": float(cleared[~unstable].mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
    }


def band_study(est: np.ndarray, true: np.ndarray) -> dict:
    """Error where both the full model and the fit have a mode in the band, and the
    screen on this band alone."""
    both = np.isfinite(est) & np.isfinite(true)
    err = np.abs(est - true)[both]
    near = both & (np.abs(true) < 5.0)
    res = {
        "cases_with_a_mode": int(np.isfinite(true).sum()),
        "unstable": int((true > 0).sum()),
        "fit_has_no_mode": int((np.isfinite(true) & ~np.isfinite(est)).sum()),
        "error_median": float(np.median(err)) if err.size else None,
        "error_p95": float(np.quantile(err, 0.95)) if err.size else None,
        "near_boundary_cases": int(near.sum()),
        "near_boundary_error_median": float(np.median(np.abs(est - true)[near]))
        if near.any() else None,
        "near_boundary_error_p95": float(np.quantile(np.abs(est - true)[near], 0.95))
        if near.any() else None,
        "near_boundary_error_p99": float(np.quantile(np.abs(est - true)[near], 0.99))
        if near.any() else None,
        "margins": {f"{m:g}": outcome(est, true, m) for m in MARGINS},
    }  # fmt: skip
    safe = [m for m in MARGINS if res["margins"][f"{m:g}"]["unstable_cleared"] == 0]
    res["smallest_margin_with_no_unstable_cleared"] = safe[0] if safe else None
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"margins": MARGINS, "bands": scanfit.BANDS, "plants": {}}
    data = {kind: load(kind) for kind in PLANTS}
    for kind, plant in PLANTS.items():
        d = data[kind]
        true = {b: d[:, COL[f"true_{b}"]] for b in scanfit.BANDS}
        true["all"] = np.max([true[b] for b in scanfit.BANDS], axis=0)
        entry: dict = {"cases": int(len(d)), "fits": {}, "interpolating_fit": {}}
        for step, level in scanfit.VARIANTS:
            tag = f"_s{step}_n{level:g}"
            est = {b: d[:, COL[f"{b}{tag}"]] for b in scanfit.BANDS}
            est["all"] = np.max([est[b] for b in scanfit.BANDS], axis=0)
            pairs = d[:, COL[f"pairs{tag}"]]
            entry["fits"][LABEL[step, level]] = {
                "residual_median": float(np.median(d[:, COL[f"residual{tag}"]])),
                "states_median": float(np.median(d[:, COL[f"states{tag}"]])),
                "pole_pairs_share": {int(p): float((pairs == p).mean()) for p in np.unique(pairs)},
                "bands": {b: band_study(est[b], true[b]) for b in (*scanfit.BANDS, "all")},
            }
        for level in scanfit.AAA_NOISE:
            est = d[:, COL[f"aaa_n{level:g}"]]
            entry["interpolating_fit"][f"{level:g}"] = {
                "wrong_verdicts": float(((est > 0) != (true["all"] > 0)).mean()),
                **outcome(est, true["all"], 0.0),
            }
        res["plants"][plant] = entry
        print(plant, entry["cases"], "cases")
        for label, fit in entry["fits"].items():
            print(
                f"  {label}: residual {fit['residual_median']:.1e}, "
                f"{fit['states_median']:.0f} states"
            )
            for b, r in fit["bands"].items():
                if r["error_median"] is None:
                    continue
                m = r["margins"]
                print(f"    {b:5s} unstable {r['unstable']:5d} no mode {r['fit_has_no_mode']:5d} "
                      f"near-boundary error median {r['near_boundary_error_median'] or 0:.1e} "
                      f"p95 {r['near_boundary_error_p95'] or 0:.1e} | " + "  ".join(
                          f"{k}: {m[k]['cleared']:.1%}/{m[k]['unstable_cleared']}"
                          for k in ("0", "0.5", "1", "2", "5"))
                      + f" | safe {r['smallest_margin_with_no_unstable_cleared']}")  # fmt: skip
        print("  interpolating fit:", {k: (round(v["wrong_verdicts"], 4), v["unstable_cleared"])
                                       for k, v in entry["interpolating_fit"].items()})  # fmt: skip
    (OUT / "noise_screens.json").write_text(json.dumps(res, indent=2))

    labels = list(LABEL.values())
    noise_x = [1e-4, 1e-3, 1e-2, 5e-2]  # the clean scan is drawn at 0.01 %

    # ---------------- figure: error of the fitted growth rate, by band and noise
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        fits = res["plants"][plant]["fits"]
        for band, colour in zip(scanfit.BANDS, report.SERIES, strict=False):
            rows = [fits[lab]["bands"][band] for lab in labels[:4]]
            if rows[0]["near_boundary_cases"] < 30:
                continue
            for key, style, width in (("near_boundary_error_median", "-", 2.0),
                                      ("near_boundary_error_p95", ":", 1.4)):  # fmt: skip
                ax.plot(noise_x, [max(r[key], 1e-12) for r in rows], color=colour, ls=style,
                        lw=width, marker="o", ms=4,
                        label=f"{band}: {'median' if style == '-' else '95th pct'}")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(noise_x, ["clean", "0.1 %", "1 %", "5 %"])
        ax.set_xlabel("Noise on each entry of the admittance")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
        report.style(ax)
    axes[0].set_ylabel("Error of the fitted growth rate (1/s), modes within 5 1/s of zero")
    report.title(fig, "Least-squares fit to a 126-point scan: error of the growth rate, by band "
                 "(slow: below 2 Hz, sub: 2 to 58 Hz, fast: above)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig51_noise_error.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: margin against cleared share and misses
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, (plant, band) in zip(axes, zip(PLANTS.values(), ("sub", "all", "all"), strict=True),
                                 strict=True):  # fmt: skip
        fits = res["plants"][plant]["fits"]
        for lab, colour in zip(labels, report.SERIES, strict=False):
            m = fits[lab]["bands"][band]["margins"]
            ax.plot(range(len(MARGINS)), [m[f"{k:g}"]["cleared"] for k in MARGINS], color=colour,
                    lw=2, marker="o", ms=4, label=lab)  # fmt: skip
            for j, k in enumerate(MARGINS):
                n = m[f"{k:g}"]["unstable_cleared"]
                if n:
                    ax.annotate(f"{n}", (j, m[f"{k:g}"]["cleared"]), fontsize=8, color=colour,
                                xytext=(0, 5), textcoords="offset points", ha="center")  # fmt: skip
        ax.set_xticks(range(len(MARGINS)), [f"{k:g}" for k in MARGINS])
        ax.set_xlabel("Margin on the fitted growth rate (1/s)")
        which = "sub-synchronous band" if band == "sub" else "all modes"
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {which}", loc="left", fontsize=10)
        ax.set_ylim(0, 1)
        report.style(ax)
    axes[0].set_ylabel("Share of cases cleared")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower left")
    report.title(fig, "Clearing on the fitted growth rate. Numbers: unstable cases cleared")
    fig.tight_layout()
    fig.savefig(OUT / "fig52_noise_margin.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the two fits against noise
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=report.SURFACE)
    for plant, colour in zip(PLANTS.values(), report.SERIES, strict=False):
        e = res["plants"][plant]
        ax.plot(list(scanfit.AAA_NOISE), [e["interpolating_fit"][f"{k:g}"]["wrong_verdicts"]
                                          for k in scanfit.AAA_NOISE],
                color=colour, lw=2, marker="o", ms=4, label=f"{plant}: interpolating")  # fmt: skip
        wrong = []
        for lab in labels[1:4]:
            b = e["fits"][lab]["bands"]["all"]["margins"]["0"]
            n_un = e["fits"][lab]["bands"]["all"]["unstable"]
            stable = e["cases"] - n_un
            wrong.append((b["unstable_cleared"] + (1 - b["stable_cleared"]) * stable) / e["cases"])
        ax.plot(noise_x[1:], wrong, color=colour, lw=1.6, ls=":", marker="s", ms=4,
                label=f"{plant}: least squares")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise on each entry of the admittance (standard deviation)")
    ax.set_ylabel("Share of cases with the wrong verdict")
    ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(ax)
    report.title(fig, "Verdict on all modes, no margin: a fit through the points against a "
                 "least-squares fit")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig53_two_fits.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
