"""How little information decides stability? Deterministic screens on the T1 dataset.

Needs the dataset from scripts/phase3_generate_t1.py. No training on the GPU.

Screens, in order of the data they need:

    topology     network reactances only: is there a series resonance, and where?
    slip         + rotor speed: slip at that resonance
    resistance   + machine resistances and current-loop gain: sign of the loop resistance
    quick        the same data: growth rate from resistance over reactance slope
    impedance    the same data: least-damped root of the loop impedance
    phasor       full plant and controls, network and stator transients dropped
    trees        gradient-boosted trees given the solver's answer on 20,000 cases and a
                 growing subset of the 12 parameters

Each is scored two ways: by its natural rule, and by the share of cases it clears when
its threshold is set so that no unstable training case is cleared.

Writes docs/results/screening/ladder.json and four figures. Every number in
docs/steps/14-screening-ladder.md comes from this script.

Run with:  uv run scripts/screening_ladder.py        # about 3 minutes, CPU only
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

from sso_sciml import report, screening
from sso_sciml.data import t1 as data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "screening"
M = {name: j for j, name in enumerate(data.MECHANISM_COLUMNS)}
P = {name: j for j, name in enumerate(data.NAMES)}
LOW = -1e9  # score of a case a screen has nothing to say about: cleared at any threshold
N_PHASOR, N_TREE = 3000, 20000
TOPOLOGY = ("k", "x_line", "xr_line", "x_t", "par_ratio", "n_online")
TIERS = {
    "topology": TOPOLOGY,
    "+ rotor speed": (*TOPOLOGY, "w_r"),
    "+ current-loop bandwidth": (*TOPOLOGY, "w_r", "rsc_hz"),
    "+ rotor resistance": (*TOPOLOGY, "w_r", "rsc_hz", "r_r"),
    "+ PLL, outer loop, feed-forward": data.NAMES,
}
NEEDS = {
    "topology": "network reactances, units online",
    "slip": "+ rotor speed",
    "resistance": "+ machine resistances, current-loop gains",
    "quick": "same as resistance",
    "impedance": "same as resistance",
}


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solver"][:, 4] > 0
    out = {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "mech": z["mechanism"][ok]}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, mask) -> dict:
    return {k: v[mask] for k, v in d.items()}


def scores(d: dict) -> dict[str, np.ndarray]:
    """Score of every deterministic screen that needs no state matrix."""
    mech, found = d["mech"], d["mech"][:, M["has_resonance"]] > 0
    return {
        "topology": screening.network_resonance(d["theta"]),
        "slip": screening.slip_score(d["theta"]),
        "resistance": np.where(found, -mech[:, M["r_net"]], LOW),
        "quick": np.where(found, mech[:, M["sigma_hand"]], LOW),
        "impedance": mech[:, M["sigma_impedance"]],
    }


NATURAL = {"topology": 1e-9, "slip": 1e-9, "resistance": 0.0, "quick": 0.0, "impedance": 0.0}


def auc(score: np.ndarray, sigma: np.ndarray) -> float:
    return float(roc_auc_score(sigma > 0, score))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    full = load("in_distribution")
    train, val, test = (take(full, full["fold"] == k) for k in range(3))
    sets = {"in_distribution": test, "boundary": load("boundary")}
    sets.update({s: load(s) for s in list(data.SPLITS)[1:]})
    res: dict = {
        "cases": {k: int(v["sigma"].size) for k, v in sets.items()},
        "unstable_share": {k: float((v["sigma"] > 0).mean()) for k, v in sets.items()},
        "needs": NEEDS,
    }

    # ---------------- deterministic screens
    s_train = scores(train)
    s_sets = {name: scores(d) for name, d in sets.items()}
    ladder: dict = {}
    for name in NATURAL:
        tau = screening.zero_miss_threshold(s_train[name], train["sigma"] > 0)
        entry = {"zero_miss_threshold": tau, "auc": {}, "natural": {}, "zero_miss": {}}
        for split, d in sets.items():
            s = s_sets[split][name]
            entry["auc"][split] = auc(s, d["sigma"])
            entry["natural"][split] = screening.screen(s, d["sigma"], NATURAL[name])
            entry["zero_miss"][split] = screening.screen(s, d["sigma"], tau)
        ladder[name] = entry
        e = entry
        print(f"{name:11s} auc {e['auc']['in_distribution']:.3f} "
              f"natural missed {e['natural']['in_distribution']['missed_unstable']:.4f} "
              f"cleared {e['natural']['in_distribution']['cleared']:.3f} | zero-miss tau "
              f"{tau:.3f} cleared {e['zero_miss']['in_distribution']['cleared']:.3f}")  # fmt: skip
    res["ladder"] = ladder

    # Two steps in sequence: clear a case if the network scan finds no resonance, or if
    # the impedance root is below its zero-miss threshold.
    tau_imp = ladder["impedance"]["zero_miss_threshold"]
    two_step: dict = {}
    for split, d in sets.items():
        s = s_sets[split]
        cleared = (s["topology"] == 0) | (s["impedance"] < tau_imp)
        unstable = d["sigma"] > 0
        two_step[split] = {
            "cleared": float(cleared.mean()),
            "cleared_by_network_scan": float((s["topology"] == 0).mean()),
            "missed_unstable": float(cleared[unstable].mean()),
            "unstable_cases": int(unstable.sum()),
            "unstable_cleared": int(cleared[unstable].sum()),
            "stable_cleared": float(cleared[~unstable].mean()),
        }
    res["two_step"] = two_step
    print("two step", two_step["in_distribution"])

    # ---------------- where the impedance root errs: the phase-locked loop
    # The impedance model has no PLL. Compare the PLL's natural frequency with the
    # frequency of the mode the impedance model finds (both in the synchronous frame).
    def pll_ratio(d):
        return d["theta"][:, P["pll_hz"]] / np.maximum(d["mech"][:, M["f_impedance"]], 1e-6)

    def root_of(d):
        return d["mech"][:, M["sigma_impedance"]]

    r_train, un_train = pll_ratio(train), train["sigma"] > 0
    gap_train = train["sigma"] - root_of(train)
    edges = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 100.0)
    by_ratio = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (r_train >= lo) & (r_train < hi)
        by_ratio.append({
            "ratio_lo": lo, "ratio_hi": hi, "share": float(m.mean()),
            "unstable": int((m & un_train).sum()),
            "unstable_with_negative_root": int((m & un_train & (root_of(train) < 0)).sum()),
            "lowest_root_of_an_unstable_case": float(root_of(train)[m & un_train].min()),
            "mean_error": float(-gap_train[m].mean()),
            "underestimate_p999": float(np.quantile(gap_train[m], 0.999)),
        })  # fmt: skip
    ff_train = train["theta"][:, P["ff"]]
    by_ff = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (ff_train >= lo) & (ff_train < hi)
        by_ff.append({
            "ff_lo": lo, "ff_hi": min(hi, 1.0), "mean_error": float(-gap_train[m].mean()),
            "unstable_with_negative_root": int((m & un_train & (root_of(train) < 0)).sum()),
        })  # fmt: skip
    slow = r_train < 1.0
    tau_slow = screening.zero_miss_threshold(root_of(train)[slow], un_train[slow])
    tau_fast = screening.zero_miss_threshold(root_of(train)[~slow], un_train[~slow])

    def apply(t_slow: float, t_fast: float) -> dict:
        out = {}
        for split, d in sets.items():
            s, is_slow, unstable = root_of(d), pll_ratio(d) < 1.0, d["sigma"] > 0
            cleared = np.where(is_slow, s < t_slow, s < t_fast)
            out[split] = {
                "slow_pll_share": float(is_slow.mean()),
                "cleared": float(cleared.mean()),
                "missed_unstable": float(cleared[unstable].mean()),
                "unstable_cleared": int((cleared & unstable).sum()),
                "unstable_cases": int(unstable.sum()),
                "stable_cleared": float(cleared[~unstable].mean()),
            }
        return out

    rule = apply(tau_slow, tau_fast)
    res["pll"] = {
        "by_ratio": by_ratio,
        "by_feed_forward": by_ff,
        "threshold_slow_pll": tau_slow,
        "threshold_fast_pll": tau_fast,
        "rule": rule,
        # Round margins, wider than the fitted ones.
        "rounded_margins": {"slow_pll": -1.0, "fast_pll": -5.0},
        "rounded_rule": apply(-1.0, -5.0),
    }
    print("rounded", {k: (round(v["cleared"], 3), v["unstable_cleared"])
                      for k, v in res["pll"]["rounded_rule"].items()})  # fmt: skip
    print("pll rule", round(tau_slow, 3), round(tau_fast, 3), rule["in_distribution"])
    for row in by_ratio:
        print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})

    # Worst case of the impedance root: how far below the truth can it be?
    gap = train["sigma"] - train["mech"][:, M["sigma_impedance"]]
    res["impedance_gap"] = {
        "max_underestimate": float(gap.max()),
        "p999_underestimate": float(np.quantile(gap, 0.999)),
        "mean": float(gap.mean()),
        "max_underestimate_unstable": float(gap[train["sigma"] > 0].max()),
    }

    # ---------------- what the hand calculation leaves out
    def root(d):
        return d["mech"][:, M["sigma_impedance"]]

    unmodelled = ("pll_hz", "outer_ratio", "ff")
    pick = np.random.default_rng(0).permutation(train["sigma"].size)[:N_TREE]
    left_out: dict = {}
    for label, cols in (("the three controls it leaves out", unmodelled),
                        ("the nine parameters it uses",
                         tuple(n for n in data.NAMES if n not in unmodelled)),
                        ("all 12 parameters", data.NAMES)):  # fmt: skip
        j = [P[c] for c in cols]
        gbt = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(
            train["theta"][pick][:, j], (train["sigma"] - root(train))[pick]
        )
        left_out[label] = float(
            r2_score(test["sigma"] - root(test), gbt.predict(test["theta"][:, j]))
        )
    res["residual_of_impedance_root_explained"] = left_out
    print("residual explained", left_out)

    # The impedance root with that learned correction (all 12 parameters), as a screen.
    tau = screening.zero_miss_threshold(root(val) + gbt.predict(val["theta"]), val["sigma"] > 0)
    corrected = {"zero_miss_threshold": tau, "auc": {}, "natural": {}, "zero_miss": {}}
    for split, d in sets.items():
        s = root(d) + gbt.predict(d["theta"])
        corrected["auc"][split] = auc(s, d["sigma"])
        corrected["natural"][split] = screening.screen(s, d["sigma"], 0.0)
        corrected["zero_miss"][split] = screening.screen(s, d["sigma"], tau)
    res["impedance_with_learned_correction"] = corrected
    print("corrected", tau, corrected["zero_miss"]["in_distribution"],
          corrected["natural"]["in_distribution"])  # fmt: skip

    # ---------------- trees on growing subsets of the parameters
    tiers: dict = {}
    tier_scores: dict = {}
    for label, cols in TIERS.items():
        j = [P[c] for c in cols]
        gbt = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(
            train["theta"][pick][:, j], train["sigma"][pick]
        )
        tau = screening.zero_miss_threshold(gbt.predict(val["theta"][:, j]), val["sigma"] > 0)
        entry = {"parameters": len(cols), "zero_miss_threshold": tau, "auc": {}, "natural": {},
                 "zero_miss": {}}  # fmt: skip
        for split, d in sets.items():
            s = gbt.predict(d["theta"][:, j])
            entry["auc"][split] = auc(s, d["sigma"])
            entry["natural"][split] = screening.screen(s, d["sigma"], 0.0)
            entry["zero_miss"][split] = screening.screen(s, d["sigma"], tau)
            if split == "in_distribution":
                entry["mae"] = float(np.abs(s - d["sigma"]).mean())
                tier_scores[label] = s
        tiers[label] = entry
        print(f"trees {label:32s} auc {entry['auc']['in_distribution']:.3f} mae {entry['mae']:.2f} "
              f"zero-miss cleared {entry['zero_miss']['in_distribution']['cleared']:.3f} "
              f"missed {entry['zero_miss']['in_distribution']['missed_unstable']:.4f}")  # fmt: skip
        if cols is data.NAMES:
            imp = permutation_importance(gbt, test["theta"][:5000], test["sigma"][:5000],
                                         n_repeats=3, random_state=0,
                                         scoring="neg_mean_absolute_error")  # fmt: skip
            res["importance"] = dict(zip(data.NAMES, imp.importances_mean.tolist(), strict=True))
    res["trees"] = tiers

    # ---------------- phasor-domain model: transients of network and stator dropped
    sub = take(test, np.arange(test["sigma"].size) < N_PHASOR)
    a = data.state_matrices(sub["theta"], processes=8)
    qs = np.array([screening.phasor_growth(m) for m in a])
    sub_scores = scores(sub)
    res["phasor"] = {
        "cases": N_PHASOR,
        "unstable": int((sub["sigma"] > 0).sum()),
        "auc": auc(qs, sub["sigma"]),
        "natural": screening.screen(qs, sub["sigma"], 0.0),
        "phasor_unstable_share": float((qs > 0).mean()),
        "same_cases": {
            name: {"auc": auc(sub_scores[name], sub["sigma"]),
                   "natural": screening.screen(sub_scores[name], sub["sigma"], NATURAL[name])}
            for name in NATURAL
        },
    }  # fmt: skip
    print("phasor", {k: res["phasor"][k] for k in ("auc", "natural", "phasor_unstable_share")})

    # ---------------- what the topology says
    col = {n: test["theta"][:, j] for n, j in P.items()}
    unstable = test["sigma"] > 0
    k_edges = np.linspace(0.10, 0.80, 8)
    p_edges = np.geomspace(0.5, 8.0, 7)
    grid = np.full((len(p_edges) - 1, len(k_edges) - 1), np.nan)
    for i in range(len(p_edges) - 1):
        for j in range(len(k_edges) - 1):
            m = ((col["par_ratio"] >= p_edges[i]) & (col["par_ratio"] < p_edges[i + 1])
                 & (col["k"] >= k_edges[j]) & (col["k"] < k_edges[j + 1]))  # fmt: skip
            grid[i, j] = unstable[m].mean()
    w_res = s_sets["in_distribution"]["topology"]
    none = w_res == 0
    f_edges = np.linspace(0.0, 60.0, 13)
    by_f = []
    for lo, hi in zip(f_edges[:-1], f_edges[1:], strict=True):
        m = ~none & (w_res * 60 >= lo) & (w_res * 60 < hi)
        by_f.append({"f_lo": lo, "f_hi": hi, "cases": int(m.sum()),
                     "unstable": float(unstable[m].mean()) if m.sum() >= 30 else None})  # fmt: skip
    res["topology"] = {
        "k_edges": k_edges.tolist(),
        "par_ratio_edges": p_edges.tolist(),
        "unstable_share_grid": grid.tolist(),
        "no_resonance_share": float(none.mean()),
        "no_resonance_share_without_network_resistance": float(
            (screening.network_resonance(test["theta"], lossless=True) == 0).mean()
        ),
        "unstable_without_resonance": float(unstable[none].mean()),
        "unstable_with_resonance": float(unstable[~none].mean()),
        "by_resonance_frequency": by_f,
    }
    res["wall_s"] = time.perf_counter() - t_start
    (OUT / "ladder.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: the ladder
    order = list(NATURAL)
    ind = "in_distribution"
    n_unstable = int((test["sigma"] > 0).sum())

    def row(label, outcome, colour):
        return (label, outcome["cleared"], round(outcome["missed_unstable"] * n_unstable), colour)

    blue, orange, green, grey = *report.SERIES[:3], report.MUTED
    ph = res["phasor"]["natural"]
    rows = [
        ("Phasor-domain model, sign", ph["cleared"],
         "every unstable case", grey),
        row("Network scan: no resonance", ladder["topology"]["natural"][ind], blue),
        row("Impedance root, sign", ladder["impedance"]["natural"][ind], orange),
        row("Impedance root, one margin", ladder["impedance"]["zero_miss"][ind], green),
        row("Impedance root, margin by PLL speed", res["pll"]["rule"][ind], green),
        row("The same with round margins", res["pll"]["rounded_rule"][ind], green),
        row("Trees on all 12 parameters, margin", tiers[list(TIERS)[-1]]["zero_miss"][ind], grey),
        row("Impedance root with learned correction, margin", corrected["zero_miss"][ind], grey),
    ]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), facecolor=report.SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.6]})  # fmt: skip
    ax = axes[0]
    names = ["network scan", "+ rotor speed (slip)", "+ resistances, gain (sign of R)",
             "same data (quick estimate)", "same data (impedance root)"]  # fmt: skip
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, [ladder[n]["auc"][ind] for n in order], color=blue, height=0.6)
    ax.barh(-1, res["phasor"]["auc"], color=grey, height=0.6)
    ax.axvline(0.5, color=report.INK, lw=0.8, ls="--")
    ax.set_yticks([*ypos, -1], [*names, "phasor-domain model"], fontsize=9)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Ranking of unstable above stable cases (0.5 = chance, 1 = perfect)")
    ax.set_title("Screens in order of the data they need", loc="left", fontsize=10)
    ax = axes[1]
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, [r[1] for r in rows], color=[r[3] for r in rows], height=0.6)
    for y, r in zip(ypos, rows, strict=True):
        note = r[2] if isinstance(r[2], str) else f"{r[2]} of {n_unstable} unstable cases"
        ax.text(r[1] + 0.01, y, f"{100 * r[1]:.0f} % cleared, with {note}", va="center",
                fontsize=8, color=report.LABEL)  # fmt: skip
    ax.axvline(1 - res["unstable_share"][ind], color=report.INK, lw=0.8, ls="--")
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of cases cleared without a detailed study (dashed: share that is stable)")
    ax.set_title("What each rule clears, and what it wrongly clears", loc="left", fontsize=10)
    for ax in axes:
        report.style(ax)
        ax.grid(axis="y", visible=False)
    report.title(fig, "Deterministic screens on the in-distribution test fold")
    fig.tight_layout()
    fig.savefig(OUT / "fig34_screening_ladder.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: ROC curves and robustness
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), facecolor=report.SURFACE)
    ax = axes[0]
    for name, colour in zip(order, report.SERIES, strict=False):
        fpr, tpr, _ = roc_curve(unstable, s_sets["in_distribution"][name])
        ax.plot(fpr, tpr, color=colour, lw=2, label=name)
    fpr, tpr, _ = roc_curve(sub["sigma"] > 0, qs)
    ax.plot(fpr, tpr, color=report.INK, lw=1.5, ls=":", label="phasor-domain model")
    ax.plot([0, 1], [0, 1], color=report.MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Stable cases flagged")
    ax.set_ylabel("Unstable cases flagged")
    ax.set_title("Every threshold of every screen", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL, loc="lower right")
    report.style(ax)
    ax = axes[1]
    names = list(sets)
    ypos = np.arange(len(names))[::-1]
    shown = (("impedance root, sign", ladder["impedance"]["natural"], orange),
             ("one margin fitted in distribution", ladder["impedance"]["zero_miss"], blue),
             ("margin by PLL speed, fitted", res["pll"]["rule"], green),
             ("margin by PLL speed, round", res["pll"]["rounded_rule"], grey))  # fmt: skip
    for k, (label, outcome, colour) in enumerate(shown):
        ax.barh(ypos + (1.5 - k) * 0.2, [100 * outcome[s]["missed_unstable"] for s in names],
                height=0.18, color=colour, label=label)  # fmt: skip
    ax.set_yticks(ypos, [n.replace("_", " ") for n in names])
    ax.set_xlabel("Unstable cases cleared (%)")
    ax.set_title("Does a threshold set in distribution hold outside it?", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    ax.grid(axis="y", visible=False)
    report.title(fig, "Screens compared at every threshold, and tested outside the training range")
    fig.tight_layout()
    fig.savefig(OUT / "fig35_screening_roc.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: topology
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor=report.SURFACE)
    ax = axes[0]
    im = ax.imshow(100 * grid, origin="lower", aspect="auto", cmap="Oranges", vmin=0)
    ax.set_xticks(np.arange(len(k_edges)) - 0.5, [f"{100 * v:.0f}" for v in k_edges])
    ax.set_yticks(np.arange(len(p_edges)) - 0.5, [f"{v:.1f}" for v in p_edges])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color=report.INK)  # fmt: skip
    ax.set_xlabel("Series compensation of the line (%)")
    ax.set_ylabel("Parallel path impedance / line impedance")
    ax.set_title("Unstable cases (%)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    mid = [0.5 * (b["f_lo"] + b["f_hi"]) for b in by_f if b["unstable"] is not None]
    ax.bar(mid, [100 * b["unstable"] for b in by_f if b["unstable"] is not None], width=4.2,
           color=report.SERIES[1])  # fmt: skip
    ax.set_xlabel("Network resonance frequency from a frequency scan (Hz)")
    ax.set_ylabel("Unstable cases (%)")
    ax.set_title(f"No resonance: {100 * none.mean():.0f} % of cases, "
                 f"{100 * unstable[none].mean():.2f} % of them unstable",
                 loc="left", fontsize=10)  # fmt: skip
    report.style(ax)
    ax = axes[2]
    imp = res["importance"]
    keys = sorted(imp, key=imp.get)
    ax.barh(np.arange(len(keys)), [imp[k] for k in keys],
            color=[report.SERIES[k not in TOPOLOGY] for k in keys])  # fmt: skip
    ax.set_yticks(np.arange(len(keys)), keys)
    ax.set_xlabel("Rise in error when the parameter is shuffled (1/s)")
    ax.set_title("What the growth rate depends on (blue: network)", loc="left", fontsize=10)
    report.style(ax)
    ax.grid(axis="y", visible=False)
    report.title(fig, "What the network alone says, and which parameters carry the answer")
    fig.tight_layout()
    fig.savefig(OUT / "fig36_screening_topology.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: trees by information tier
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=report.SURFACE)
    tl = list(TIERS)
    x = np.arange(len(tl))
    axes[0].bar(x, [tiers[t]["mae"] for t in tl], color=report.SERIES[0], width=0.6)
    axes[0].set_title("Error in growth rate (1/s)", loc="left", fontsize=10)
    axes[1].bar(x, [tiers[t]["zero_miss"]["in_distribution"]["cleared"] for t in tl], width=0.6,
                color=report.SERIES[2])  # fmt: skip
    axes[1].set_title("Share of cases cleared with no unstable case among them", loc="left",
                      fontsize=10)  # fmt: skip
    short = ["network only", "+ rotor speed", "+ current-loop\nbandwidth", "+ rotor\nresistance",
             "+ PLL, outer loop,\nfeed-forward"]  # fmt: skip
    for ax in axes:
        ax.set_xticks(x, short, fontsize=8)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    report.title(fig, "Trees given the solver's answer and a growing set of parameters "
                 f"({N_TREE:,} training cases)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig37_screening_information.png", dpi=160)
    plt.close(fig)
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
