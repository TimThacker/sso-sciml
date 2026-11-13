"""What error in the network data does to each screen.

    uv run scripts/network_error.py            # about 98,000 cases, about 25 minutes
    uv run scripts/network_error.py --small    # one sixteenth, for a quick look
    uv run scripts/network_error.py --reuse    # numbers and figures from the saved pass

Needs dataset T1 with data/t1-v2/scan_<set>.npz (scripts/phase3_scan_t1.py) and dataset
C1. Writes data/neterror-v1/<kind>_<set>.npz (not committed),
docs/results/network/network_error.json and figures fig57 to fig59. Every number in
docs/steps/22-network-error.md comes from this script.
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
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neterror-v1"
OUT = ROOT / "docs" / "results" / "network"
PLANTS = {"t1": "doubly-fed plant", "pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(neterror.COLUMNS)}
MARGINS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
PHASOR_MARGINS = (0.0, 0.05, 0.1, 0.2)
P = {name: j for j, name in enumerate(t1.NAMES)}


def sets_of(kind: str) -> list[str]:
    if kind == "t1":
        return ["in_distribution", "boundary", *list(t1.SPLITS)[1:]]
    return [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]


def thetas(kind: str, name: str) -> np.ndarray:
    if kind == "t1":
        rows = np.load(ROOT / "data" / t1.VERSION / f"scan_{name}.npz")["rows"]
        return np.load(ROOT / "data" / t1.VERSION / f"{name}.npz")["theta"][rows]
    return np.load(ROOT / "data" / c1.VERSION / f"{kind}_{name}.npz")["theta"]


def load(kind: str, names: list[str]) -> dict:
    out, theta = [], []
    for n in names:
        z = np.load(DATA / f"{kind}_{n}.npz")
        out.append(z["out"])
        theta.append(z["theta"])
    out, theta = np.concatenate(out), np.concatenate(theta)
    ok = np.isfinite(out[:, 0])
    d = {name: out[ok, j] for name, j in COL.items()}
    d["theta"] = theta[ok]
    return d


def outcome(cleared: np.ndarray, unstable: np.ndarray) -> dict:
    return {
        "cleared": float(cleared.mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
        "stable_flagged": float((~cleared)[~unstable].mean()),
    }


def _q(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if values.size else None


def band_study(est: np.ndarray, true: np.ndarray) -> dict:
    both = np.isfinite(est) & np.isfinite(true)
    err = np.abs(est - true)[both]
    close = np.abs(est - true)[both & (np.abs(true) < 5.0)]
    res = {
        "unstable": int((true > 0).sum()),
        "error_median": _q(err, 0.5),
        "error_p95": _q(err, 0.95),
        "near_boundary_error_median": _q(close, 0.5),
        "near_boundary_error_p95": _q(close, 0.95),
        "near_boundary_error_p99": _q(close, 0.99),
        "margins": {f"{m:g}": outcome(est < -m, true > 0) for m in MARGINS},
    }
    safe = [m for m in MARGINS if res["margins"][f"{m:g}"]["unstable_cleared"] == 0]
    res["smallest_margin_with_no_unstable_cleared"] = safe[0] if safe else None
    return res


def study(kind: str, d: dict) -> dict:
    true = {b: d[f"true_{b}"] for b in neterror.BANDS}
    true["all"] = np.max([true[b] for b in neterror.BANDS], axis=0)
    res: dict = {"cases": int(true["all"].size), "levels": {}, "one at a time": {}}
    for level in neterror.LEVELS:
        tag = f"_e{level:g}"
        est = {b: d[f"exact_{b}{tag}"] for b in neterror.BANDS}
        est["all"] = np.max([est[b] for b in neterror.BANDS], axis=0)
        entry: dict = {"exact": {b: band_study(est[b], true[b]) for b in (*neterror.BANDS, "all")}}
        if kind == "t1":
            root = d[f"root{tag}"]
            ratio = d["theta"][:, P["pll_hz"]] / np.maximum(d[f"root_f{tag}"], 1e-6)
            sub_unstable, any_unstable = true["sub"] > 0, true["all"] > 0
            # The step 14 rule and the same rule with each margin widened by a constant.
            entry["step 14 rule"] = {
                f"{extra:g}": outcome(
                    np.where(ratio < 1.0, root < -1.0 - extra, root < -5.0 - extra), sub_unstable
                )
                for extra in (0.0, 1.0, 2.0, 5.0)
            }
            ph = d[f"phasor_slow{tag}"]
            solved = np.isfinite(ph)
            entry["phasor"] = {
                "no_steady_state_on_the_wrong_network": int((~solved).sum()),
                "error_median": float(np.median(np.abs(ph - true["slow"])[solved])),
                "error_p99": float(np.quantile(np.abs(ph - true["slow"])[solved], 0.99)),
                "margins": {f"{m:g}": outcome(solved & (ph < -m), true["slow"] > 0)
                            for m in PHASOR_MARGINS},
            }  # fmt: skip
            pair = np.where(ratio < 1.0, root < -1.0, root < -5.0) & solved & (ph < -0.05)
            entry["step 14 rule and phasor model"] = outcome(pair, any_unstable)
        res["levels"][f"{level:g}"] = entry
    for j, name in enumerate(neterror.FACTORS):
        row = {}
        for band in ("sub", "slow"):
            t = true[band]
            near = np.isfinite(t) & (np.abs(t) < 5.0)
            up, down = d[f"{band}_{j}_up"] - t, d[f"{band}_{j}_down"] - t
            ok = near & np.isfinite(up) & np.isfinite(down)
            row[band] = {
                "shift_up_median": float(np.median(up[ok])),
                "shift_down_median": float(np.median(down[ok])),
                "abs_shift_median": float(np.median(0.5 * (np.abs(up) + np.abs(down))[ok])),
                "abs_shift_p95": float(np.quantile(0.5 * (np.abs(up) + np.abs(down))[ok], 0.95)),
                "verdict_changed": float(((d[f"{band}_{j}_up"] > 0) != (t > 0))[ok].mean()),
            }
        res["one at a time"][name] = row
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
                np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta,
                                    out=neterror.cases(kind, theta, args.processes))  # fmt: skip
                print("done", kind, name, len(theta))

    res: dict = {"levels": neterror.LEVELS, "factors": neterror.FACTORS,
                 "single_factor": neterror.SINGLE, "plants": {}}  # fmt: skip
    for kind, plant in PLANTS.items():
        res["plants"][plant] = r = study(kind, load(kind, sets_of(kind)))
        print(plant, r["cases"], "cases")
        for level, e in r["levels"].items():
            for band in ("sub", "slow", "all"):
                b = e["exact"][band]
                m = b["margins"]
                print(f"  error {level} exact {band:4s} unstable {b['unstable']:5d} near-boundary "
                      f"error median {b['near_boundary_error_median'] or 0:.2f} p95 "
                      f"{b['near_boundary_error_p95'] or 0:.2f} | " + "  ".join(
                          f"{k}: {m[k]['cleared']:.1%}/{m[k]['unstable_cleared']}"
                          for k in ("0", "0.5", "1", "2", "5", "10"))
                      + f" | safe {b['smallest_margin_with_no_unstable_cleared']}")  # fmt: skip
            if kind == "t1":
                print("    step 14 rule", {k: (round(v["cleared"], 3), v["unstable_cleared"])
                                           for k, v in e["step 14 rule"].items()})  # fmt: skip
                print("    phasor", e["phasor"]["no_steady_state_on_the_wrong_network"],
                      round(e["phasor"]["error_median"], 4), round(e["phasor"]["error_p99"], 3),
                      {k: (round(v["cleared"], 3), v["unstable_cleared"])
                       for k, v in e["phasor"]["margins"].items()})  # fmt: skip
                print("    pair", e["step 14 rule and phasor model"])
        for name, row in r["one at a time"].items():
            print(f"  +-10 % {name:22s} " + "  ".join(
                f"{band}: median |shift| {row[band]['abs_shift_median']:.3f} p95 "
                f"{row[band]['abs_shift_p95']:.2f} up {row[band]['shift_up_median']:+.3f} "
                f"verdict changed {row[band]['verdict_changed']:.2%}" for band in row))  # fmt: skip
    if not args.small:
        (OUT / "network_error.json").write_text(json.dumps(res, indent=2))

    levels = [f"{e:g}" for e in neterror.LEVELS]
    x = [100 * e for e in neterror.LEVELS]

    # ---------------- figure: which network quantity matters
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(neterror.FACTORS))[::-1]
    names = {"sub": "2 to 58 Hz", "slow": "below 2 Hz"}
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        one = res["plants"][plant]["one at a time"]
        for k, (band, colour) in enumerate(zip(("sub", "slow"), report.SERIES, strict=False)):
            ax.barh(ypos + (0.5 - k) * 0.36, [one[f][band]["abs_shift_median"]
                                              for f in neterror.FACTORS], height=0.34,
                    color=colour, label=names[band])  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("Median shift of the growth rate for a 10 % error (1/s)")
        ax.set_title(plant[0].upper() + plant[1:], loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, neterror.FACTORS, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        labelcolor=report.LABEL,
        loc="upper right",
        ncols=2,
    )
    report.title(fig, "One network quantity wrong by 10 %: shift of the least-damped mode in "
                 "each band, cases within 5 1/s of the boundary")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig57_network_sensitivity.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: growth-rate error and safe margin against data error
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=report.SURFACE)
    for (plant, band), colour in zip(zip(PLANTS.values(), ("sub", "all", "all"), strict=True),
                                     report.SERIES, strict=False):  # fmt: skip
        rows = [res["plants"][plant]["levels"][k]["exact"][band] for k in levels]
        label = f"{plant}, {'sub-synchronous band' if band == 'sub' else 'all modes'}"
        axes[0].plot(x, [r["near_boundary_error_median"] for r in rows], color=colour, lw=2,
                     marker="o", ms=4, label=label)  # fmt: skip
        axes[0].plot(x, [r["near_boundary_error_p95"] for r in rows], color=colour, lw=1.4,
                     ls=":", marker="s", ms=4)  # fmt: skip
        for m, style in ((0.0, ":"), (2.0, "-")):
            axes[1].plot(x, [r["margins"][f"{m:g}"]["unstable_cleared"] / max(r["unstable"], 1)
                             for r in rows], color=colour, lw=2 if m else 1.4, ls=style,
                         marker="o", ms=4, label=f"{label}, margin {m:g} 1/s")  # fmt: skip
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error of the growth rate (1/s)")
    axes[0].set_title("Solid: median. Dotted: 95th percentile", loc="left", fontsize=10)
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_ylabel("Share of unstable cases cleared")
    for ax in axes:
        ax.set_xlabel("Error in each network quantity (standard deviation, %)")
        ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
        report.style(ax)
    report.title(fig, "The exact plant model on a network known with error")
    fig.tight_layout()
    fig.savefig(OUT / "fig58_network_error.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the screens of the doubly-fed plant under network error
    t = res["plants"][PLANTS["t1"]]["levels"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=report.SURFACE)
    series = (
        (
            "exact model, sub-synchronous band, margin 2 1/s",
            [t[k]["exact"]["sub"]["margins"]["2"] for k in levels],
            report.SERIES[0],
        ),
        (
            "step 14 rule as it stands",
            [t[k]["step 14 rule"]["0"] for k in levels],
            report.SERIES[1],
        ),
        (
            "step 14 rule, margins widened by 2 1/s",
            [t[k]["step 14 rule"]["2"] for k in levels],
            report.SERIES[2],
        ),
        (
            "phasor-domain model, slow mode, margin 0.05 1/s",
            [t[k]["phasor"]["margins"]["0.05"] for k in levels],
            report.SERIES[3],
        ),
    )
    for label, rows, colour in series:
        axes[0].plot(x, [r["cleared"] for r in rows], color=colour, lw=2, marker="o", ms=4,
                     label=label)  # fmt: skip
        axes[1].plot(x, [r["unstable_cleared"] for r in rows], color=colour, lw=2, marker="o",
                     ms=4, label=label)  # fmt: skip
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of cases cleared")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("Unstable cases cleared")
    for ax in axes:
        ax.set_xlabel("Error in each network quantity (standard deviation, %)")
        report.style(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower left")
    report.title(fig, "Doubly-fed plant: each screen given a network known with error")
    fig.tight_layout()
    fig.savefig(OUT / "fig59_network_screens.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
