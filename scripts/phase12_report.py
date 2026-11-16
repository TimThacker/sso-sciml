"""Phase 12: the numbers of the final report, gathered from the saved results.

    uv run scripts/phase12_report.py

No training and no solver. Reads the result files of Phases 4 and 6 to 11 and writes
docs/results/phase12/summary.json and fig111_all_models.png. Every number in
docs/REPORT.md is either in that file or quoted from the step it names.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import report

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
OUT = RES / "phase12"
T1_BUDGET = 100000


def read(*parts: str) -> dict:
    return json.loads(RES.joinpath(*parts).read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unc, cmp6, rep6 = read("phase4", "t1_uncertainty.json"), read("phase6", "comparison.json"), \
        read("phase6", "report.json")  # fmt: skip
    t2m, meas, fall = read("phase7", "t2_models.json"), read("phase8", "measure.json"), \
        read("phase9", "fallback.json")  # fmt: skip
    design, reg = read("phase10", "design.json"), read("phase11", "registry.json")
    tests1 = list(fall["tests"])
    out: dict = {}

    # ---------------- task T1: growth rate
    t1: dict = {"tests": tests1, "models": {}}
    t1["models"]["hand calculation"] = {
        "mae": {t: fall["hand_calculation"][t]["mae"] for t in tests1}, "cases": 0}
    for arm in "NFR":
        t1["models"][f"trees, arm {arm}"] = {
            "mae": {t: cmp6["summary"][f"gbt/{arm}/{T1_BUDGET}"][t]["mae"]["mean"] for t in tests1},
            "cases": T1_BUDGET}  # fmt: skip
        t1["models"][f"perceptron, arm {arm}"] = {
            "mae": {t: cmp6["summary"][f"mlp/{arm}/{T1_BUDGET}"][t]["mae"]["mean"] for t in tests1},
            "cases": T1_BUDGET}  # fmt: skip
        t1["models"][f"ensemble of five, arm {arm}"] = {
            "mae": fall["models"][f"{arm}/{T1_BUDGET}"]["mae"], "cases": T1_BUDGET}
    j = rep6["t1"]["budgets"].index(T1_BUDGET)
    for arm, name in (("L", "perceptron, penalty (L)"), ("FL", "perceptron, features and penalty")):
        t1["models"][name] = {"mae": {"in_distribution": rep6["t1"]["inside"][arm][j]},
                              "cases": T1_BUDGET}  # fmt: skip
    for arm in "NFR":
        t1["models"][f"Gaussian process, arm {arm}"] = {
            "mae": {"in_distribution": unc[f"gp/{arm}/1000"]["in_distribution"]["mae"]["mean"]},
            "cases": 1000}  # fmt: skip
    t1["timing_s_per_case"] = unc["timing"]
    out["t1"] = t1

    # ---------------- task T2: response in time
    big = max(t2m["budgets"])
    tests2 = list(t2m["tests"])
    out["t2"] = {
        "tests": tests2, "budget": big,
        "reduced_model_alone": {t: t2m["reduced_model_alone"][t]["median"] for t in tests2},
        "models": {k.split("@")[0]: {"median": {t: v[t]["median"]["mean"] for t in tests2},
                                     "fit_s": v["fit_s"],
                                     "trusted_to_s": {t: v[t]["trusted_to_s"] for t in tests2}}
                   for k, v in t2m["summary"].items() if k.endswith(f"@{big}")},
    }  # fmt: skip

    # ---------------- task T3: from a measurement
    rows = meas["classical"]["rows"]
    out["t3"] = {
        "growth_rate_error": {
            snr: {"matrix pencil": rows[snr]["pencil"]["sigma"],
                  "prony": rows[snr]["prony"]["sigma"],
                  "phasors 120": rows[snr]["pmu120"]["sigma"],
                  "phasors 60": rows[snr]["pmu60"]["sigma"],
                  "phasors 30": rows[snr]["pmu30"]["sigma"],
                  "network": meas["network"]["rows"][snr]["sigma"]}
            for snr in ("40", "20", "0")},
        "bound": meas["parameters"]["bound"], "sensitivity": meas["parameters"]["sensitivity"],
    }  # fmt: skip

    # ---------------- the screen with its fallback
    screen = {}
    for key, det in (("R", "ensemble spread"), ("R", "margin only"), ("N", "ensemble spread"),
                     ("N", "margin only")):  # fmt: skip
        e = fall["models"][f"{key}/{T1_BUDGET}"]["detectors"][det]["sets"]
        screen[f"arm {key}, {det}"] = {t: e[t]["policy"] for t in tests1}
    screen["hand calculation"] = {
        t: {k: fall["hand_calculation"][t][k]
            for k in ("solver_share", "missed_unstable", "false_unstable", "unstable", "cases")}
        for t in tests1}  # fmt: skip
    out["screen"] = screen

    # ---------------- design search and service
    out["design"] = {
        "summary": design["summary"],
        "baseline": design["baseline"],
        "per_segment": [{"best": r["best"], "design": r["design"]} for r in design["per_segment"]],
        "time_domain": {k: {"score": v["score"],
                            "decays": sum(r["final_over_initial"] < 1.0 for r in v["scenarios"]),
                            "scenarios": len(v["scenarios"])}
                        for k, v in design["time_domain"].items()},
    }  # fmt: skip
    out["service"] = {"card": {k: v for k, v in reg["card"].items() if k != "inputs"},
                      **{k: v for k, v in reg.items() if k not in ("card", "rows")}}  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))

    # ---------------- figure: every model on its task
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), facecolor=report.SURFACE)
    names = list(t1["models"])
    vals = [t1["models"][n]["mae"]["in_distribution"] for n in names]
    order = np.argsort(vals)[::-1]
    colour = [report.MUTED if "hand" in names[k] else report.ARM_COLOUR.get(names[k][-1],
              report.SERIES[3]) for k in order]  # fmt: skip
    axes[0].barh(np.arange(len(names)), [vals[k] for k in order], color=colour)
    axes[0].set_yticks(np.arange(len(names)),
                       [f"{names[k]} ({t1['models'][names[k]]['cases']:,})"
                        if t1["models"][names[k]]["cases"] else names[k] for k in order],
                       fontsize=8)  # fmt: skip
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean absolute error of the growth rate, in distribution (1/s)")
    axes[0].set_title("T1: growth rate (training cases in brackets)", loc="left", fontsize=10)
    m2 = out["t2"]["models"]
    names2 = sorted(m2, key=lambda k: -m2[k]["median"]["in_distribution"])
    y = np.arange(len(names2))
    axes[1].barh(y + 0.2, [m2[k]["median"]["in_distribution"] for k in names2], 0.38,
                 color=report.SERIES[0], label="in distribution")  # fmt: skip
    axes[1].barh(y - 0.2, [np.exp(np.mean([np.log(m2[k]["median"][t]) for t in tests2[1:]]))
                           for k in names2], 0.38, color=report.SERIES[1],
                 label="outside, geometric mean of three sets")  # fmt: skip
    axes[1].axvline(out["t2"]["reduced_model_alone"]["in_distribution"], color=report.INK, lw=0.8,
                    ls="--", label="reduced model alone")  # fmt: skip
    axes[1].set_yticks(y, names2, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Median relative error of the response")
    axes[1].set_title(f"T2: response in time, {big:,} trajectories", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7, labelcolor=report.LABEL, loc="lower center",
                   bbox_to_anchor=(0.5, 1.06), ncol=3)  # fmt: skip
    names3 = list(screen)
    share = [100 * screen[n]["in_distribution"]["solver_share"] for n in names3]
    axes[2].barh(np.arange(len(names3)), share, color=report.SERIES[2])
    for k, n in enumerate(names3):
        p = screen[n]["in_distribution"]
        axes[2].text(share[k] * 1.08, k, f"{share[k]:.1f} %, {p['missed_unstable']} missed",
                     va="center", fontsize=8, color=report.LABEL)  # fmt: skip
    axes[2].set_yticks(np.arange(len(names3)), names3, fontsize=8)
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.3, 400)
    axes[2].set_xlabel("Cases sent to the solver, in distribution (%)")
    axes[2].set_title("The screen with its fallback rule", loc="left", fontsize=10)
    for ax in axes:
        report.style(ax)
        ax.grid(axis="y", visible=False)
    report.title(fig, "Every model on its task")
    fig.tight_layout()
    fig.savefig(OUT / "fig111_all_models.png", dpi=160)
    plt.close(fig)

    for name, m in t1["models"].items():
        print(f"{name:34s}", {t: round(v, 3) for t, v in m["mae"].items()})
    print(json.dumps(out["t3"]["growth_rate_error"]))
    for name, s in screen.items():
        print(f"{name:26s}", {t: (round(100 * p["solver_share"], 1), p["missed_unstable"])
                              for t, p in s.items()})  # fmt: skip
    print({k: v for k, v in out["service"].items() if k != "card"})


if __name__ == "__main__":
    main()
