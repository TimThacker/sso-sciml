"""Screens for a generic converter (System C), ordered by the data they need.

    uv run scripts/phase3_generate_c1.py     # dataset, a few minutes
    uv run scripts/converter_screens.py

Each screen clears cases. It is scored on the share of cases it clears and on the share
of unstable cases among those it clears, on the test fold and on the sets outside the
training range. A threshold is either fixed by theory or set on the training and
validation folds as the largest that clears no unstable case there.

Writes docs/results/converter/screens.json and figures fig42 to fig44.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from sso_sciml import admittance as adm
from sso_sciml import report
from sso_sciml.data import c1
from sso_sciml.physics.converter import Converter, ConverterControls
from sso_sciml.screening import screen, zero_miss_threshold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / c1.VERSION
OUT = ROOT / "docs" / "results" / "converter"
GROUPS = ("PLL, source", "PLL, load", "no PLL, load")


def load(family: str, split: str) -> dict:
    z = np.load(DATA / f"{family}_{split}.npz")
    ok = z["out"][:, 0] > 0
    d = {name: z["out"][ok, j] for j, name in enumerate(c1.COLUMNS)}
    d["theta"] = z["theta"][ok]
    d.update({name: z["theta"][ok, j] for j, name in enumerate(c1.NAMES)})
    d["fold"] = z["fold"][ok] if "fold" in z else np.full(ok.sum(), 2, dtype=np.int8)
    d["rows"], d["solved"] = len(ok), int(ok.sum())
    return d


def take(d: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def pool(sets: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in sets]) for k, v in sets[0].items()
            if isinstance(v, np.ndarray)}  # fmt: skip


def scores(d: dict) -> dict[str, np.ndarray]:
    """Screens with a threshold to be set. Larger means more at risk."""
    return {
        "short-circuit ratio": 1.0 / d["scr"],
        "power over short-circuit ratio": np.abs(d["p"]) / d["scr"],
        "negative conductance over short-circuit ratio": np.maximum(-d["g_min"], 0) / d["scr"],
        "loop gain where not passive, margin set on data": d["small_gain"],
        "positive-sequence scan, reactance crossing, margin set on data":
            np.maximum(d["scan_sigma"], -1e3),
    }  # fmt: skip


def fixed(d: dict) -> dict[str, np.ndarray]:
    """Screens whose threshold is not fitted: True where the case is cleared."""
    return {
        "loop gain where not passive, below 1": d["small_gain"] < 1.0,
        "positive-sequence scan, reactance crossing": d["scan_sigma"] < 0,
        "positive-sequence scan, Nyquist count": d["count_pos"] == 0,
        "2 x 2 scan at zero reactive current, Nyquist count": d["count_rated"] == 0,
        "2 x 2 scan at the operating point, Nyquist count": d["count"] == 0,
    }


def outcome(cleared: np.ndarray, sigma: np.ndarray) -> dict:
    unstable = sigma > 0
    return {
        "cleared": float(cleared.mean()),
        "unstable_cleared": int((cleared & unstable).sum()),
        "unstable": int(unstable.sum()),
        "cases": int(len(sigma)),
    }


def ladder(d: dict, outside: dict) -> dict:
    fit, test = take(d, d["fold"] < 2), take(d, d["fold"] == 2)
    res: dict = {}
    for name, cleared in fixed(test).items():
        res[name] = {"test": outcome(cleared, test["sigma"]),
                     "outside": outcome(fixed(outside)[name], outside["sigma"])}  # fmt: skip
    s_fit, s_test, s_out = scores(fit), scores(test), scores(outside)
    for name in s_fit:
        t = zero_miss_threshold(s_fit[name], fit["sigma"] > 0)
        res[name] = {"threshold": t,
                     "test": outcome(s_test[name] < t, test["sigma"]),
                     "outside": outcome(s_out[name] < t, outside["sigma"])}  # fmt: skip
    # Trees on the 12 parameters, trained on the solver's answer.
    train, val = take(d, d["fold"] == 0), take(d, d["fold"] == 1)
    gbt = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(
        train["theta"], np.clip(train["sigma"], -30, 30)
    )
    t = zero_miss_threshold(gbt.predict(val["theta"]), val["sigma"] > 0)
    res["trees on the 12 parameters"] = {
        "threshold": t,
        "test": outcome(gbt.predict(test["theta"]) < t, test["sigma"]),
        "outside": outcome(gbt.predict(outside["theta"]) < t, outside["sigma"]),
    }
    imp = permutation_importance(gbt, test["theta"], np.clip(test["sigma"], -30, 30),
                                 n_repeats=5, random_state=0)  # fmt: skip
    res["_importance"] = dict(zip(c1.NAMES, map(float, imp.importances_mean), strict=True))
    res["_exact_check"] = screen(-(test["count"] == 0).astype(float), test["sigma"], -0.5)
    return res


def attribution(d: dict) -> dict:
    """Of the unstable cases, the share made stable by one change to the controls."""
    u = d["sigma"] > 0
    out = {"unstable": int(u.sum()), "cases": int(len(u)), "share_unstable": float(u.mean())}
    for key, name in (("sigma_slow_pll", "PLL ten times slower"),
                      ("sigma_slow_dc", "DC voltage loop ten times slower"),
                      ("sigma_no_ff", "no voltage feed-forward")):  # fmt: skip
        out[name] = float((d[key][u] < 0).mean()) if u.any() else float("nan")
    f = d["f"][u]
    out["mode_hz_quartiles"] = (
        [float(q) for q in np.quantile(f, [0.25, 0.5, 0.75])] if u.any() else []
    )
    out["passive_everywhere"] = float((d["f_passive"] == 0).mean())
    out["not_passive_up_to_hz_median"] = float(np.median(d["f_passive"]))
    return out


def conductance_curves() -> dict:
    """Passivity index against frequency for one converter in three roles."""
    f = np.geomspace(0.2, 500, 400)
    f = np.concatenate([-f[::-1], f])
    s = 2j * np.pi * f
    out = {"f": f}
    for name, p_src, follow in zip(GROUPS, (0.8, -0.8, -0.8), (0.0, 0.0, 1.0), strict=True):
        model = Converter(ConverterControls(follow=follow, t_ff=0.0002 if follow else 0.002))
        i = model.current_at(1.0 + 0j, p_src)
        scan = adm.ConverterScan(model, 1.0 + 0j, i, model.params(p_src))
        out[name] = adm.passivity_index(scan.admittance(s))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"groups": {}, "ladder": {}, "sets": {}}
    data = {}
    for family in c1.FAMILIES:
        splits = [s for s in c1.SPLITS if not (family == "follow" and s == "fast_pll")]
        sets = {s: load(family, s) for s in splits}
        for s, d in sets.items():
            res["sets"][f"{family}_{s}"] = {
                "rows": d["rows"], "solved": d["solved"],
                "unstable": float((d["sigma"] > 0).mean()),
                "count_agrees_with_modes": float(((d["count"] > 0) == (d["sigma"] > 0)).mean()),
                "unstable_cleared_by_loop_gain_below_1":
                    int(((d["small_gain"] < 1) & (d["sigma"] > 0)).sum()),
            }  # fmt: skip
        inside = sets["in_distribution"]
        outside = pool([sets[s] for s in splits if s != "in_distribution"])
        data[family] = inside
        res["ladder"][family] = ladder(inside, outside)
        print(family)
        for name, r in res["ladder"][family].items():
            if not name.startswith("_"):
                a, b = r["test"], r["outside"]
                print(f"  {name:66s} clears {a['cleared']:6.1%}  unstable among them "
                      f"{a['unstable_cleared']:4d}/{a['unstable']:<4d} outside "
                      f"{b['cleared']:6.1%} {b['unstable_cleared']}/{b['unstable']}")  # fmt: skip

    pll = data["pll"]
    groups = {
        GROUPS[0]: take(pll, pll["p"] > 0),
        GROUPS[1]: take(pll, pll["p"] < 0),
        GROUPS[2]: data["follow"],
    }
    for name, d in groups.items():
        res["groups"][name] = attribution(d)
        print(name, json.dumps(res["groups"][name]))
    (OUT / "screens.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: negative conductance of one converter in three roles
    curves = conductance_curves()
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor=report.SURFACE)
    f = curves["f"]
    for name, colour in zip(GROUPS, report.SERIES, strict=False):
        # The index is the same at +f and -f, so one side is drawn.
        ax.plot(f[f > 0], curves[name][f > 0], color=colour, lw=1.8, label=name)
    ax.axhline(0, color=report.INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Frequency in the synchronous frame (Hz): 60 Hz plus or minus this in the phases")
    ax.set_ylabel("Smallest conductance (pu)")
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    report.title(fig, "Where one converter can feed an oscillation: 0.8 pu, PLL 10 Hz, "
                 "DC voltage loop 10 Hz, current loop 200 Hz")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig42_conductance.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: which loop, and where the unstable cases are
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor=report.SURFACE)
    changes = (
        "PLL ten times slower",
        "DC voltage loop ten times slower",
        "no voltage feed-forward",
    )
    width = 0.26
    for k, (name, colour) in enumerate(zip(GROUPS, report.SERIES, strict=False)):
        vals = [res["groups"][name][c] for c in changes]
        axes[0].bar(np.arange(3) + (k - 1) * width, vals, width, color=colour, label=name)
    axes[0].set_xticks(np.arange(3), ["PLL 10x slower", "DC loop 10x slower", "no feed-forward"])
    axes[0].set_ylabel("Share of unstable cases made stable")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    edges = np.linspace(-1, 1, 11)
    mid = 0.5 * (edges[:-1] + edges[1:])
    share = [float((pll["sigma"][(pll["p"] >= lo) & (pll["p"] < hi)] > 0).mean())
             for lo, hi in zip(edges[:-1], edges[1:], strict=True)]  # fmt: skip
    axes[1].bar(mid, share, 0.18, color=report.SERIES[0], label="PLL")
    fl = data["follow"]
    share_f = [float((fl["sigma"][(fl["p"] >= lo) & (fl["p"] < hi)] > 0).mean())
               if ((fl["p"] >= lo) & (fl["p"] < hi)).any() else np.nan
               for lo, hi in zip(edges[:-1], edges[1:], strict=True)]  # fmt: skip
    axes[1].plot(mid, share_f, color=report.SERIES[2], lw=2, marker="o", ms=4, label="no PLL")
    axes[1].set_xlabel("Power out of the converter (pu). Negative: a load")
    axes[1].set_ylabel("Share of cases unstable")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    e_scr = np.geomspace(1.5, 8.0, 9)
    for (name, d), colour in zip(groups.items(), report.SERIES, strict=False):
        y = [float((d["sigma"][(d["scr"] >= lo) & (d["scr"] < hi)] > 0).mean())
             for lo, hi in zip(e_scr[:-1], e_scr[1:], strict=True)]  # fmt: skip
        axes[2].plot(np.sqrt(e_scr[:-1] * e_scr[1:]), y, color=colour, lw=2, marker="o", ms=4,
                     label=name)  # fmt: skip
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Short-circuit ratio")
    axes[2].set_ylabel("Share of cases unstable")
    axes[2].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    for ax in axes:
        report.style(ax)
    report.title(fig, "Which control loop the instability needs, and where it is found")
    fig.tight_layout()
    fig.savefig(OUT / "fig43_which_loop.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the ladder
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.6), facecolor=report.SURFACE)
    for ax, family, label in zip(axes, c1.FAMILIES, ("With a PLL", "No PLL"), strict=True):
        rows = [(n, r) for n, r in res["ladder"][family].items() if not n.startswith("_")]
        rows.sort(key=lambda nr: nr[1]["test"]["cleared"])
        for j, (_, r) in enumerate(rows):
            t = r["test"]
            clean = t["unstable_cleared"] == 0 and r["outside"]["unstable_cleared"] == 0
            ax.barh(j, t["cleared"], color=report.SERIES[2] if clean else report.SERIES[1])
            ax.text(t["cleared"] + 0.01, j, f"{t['cleared']:.0%}, {t['unstable_cleared']} + "
                    f"{r['outside']['unstable_cleared']} unstable", va="center", fontsize=8,
                    color=report.LABEL)  # fmt: skip
        ax.set_yticks(range(len(rows)), [n for n, _ in rows], fontsize=8)
        ax.set_xlim(0, 1.6)
        ax.set_xlabel("Share of test cases cleared")
        ax.set_title(label, loc="left", fontsize=10)
        report.style(ax)
    report.title(fig, "Screens for a generic converter. Green: no unstable case cleared, in or "
                 "outside the training range. Label: unstable cleared in + outside")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig44_converter_ladder.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
