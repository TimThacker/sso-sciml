"""Phase 6 report: one table and one figure per task for choosing an arm.

    uv run scripts/phase6_report.py

No training. Reads the saved results of scripts/phase6_compare.py, phase6_loss_arms.py
and phase6_t2_arms.py and, for each task, arm and budget, gives

    the error in distribution, mean over five seeds,
    the error outside the distribution against arm N: the geometric mean, over the
        extrapolation sets, of the ratio of the two mean errors,
    the training time against arm N.

Writes docs/results/phase6/report.json and fig91 (task T1) and fig92 (task T2). Every
number in docs/steps/37-phase6-report.md that is not in steps 34 and 36 comes from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import report

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase6"
NAME = {"N": "N: no physics", "F": "F: as inputs", "R": "R: as baseline", "L": "L: as penalty",
        "FL": "F+L: inputs and penalty", "A": "A: in the model"}  # fmt: skip
T1_OUTSIDE = ("high_compensation", "low_speed", "fast_current_loop", "weak_parallel_path",
              "other_machine")  # fmt: skip
T2_OUTSIDE = ("high_compensation", "low_speed", "fast_current_loop")


def task(summary: dict, arms: tuple, budgets: tuple, outside: tuple, metric: str) -> dict:
    def err(arm, budget, test) -> float:
        return summary[f"{arm}/{budget}"][test][metric]["mean"]

    out = {"arms": list(arms), "budgets": list(budgets), "outside_sets": list(outside),
           "inside": {}, "inside_against_n": {}, "outside_against_n": {}, "cost_against_n": {},
           "outside_by_set": {}}  # fmt: skip
    for arm in arms:
        out["inside"][arm] = [err(arm, b, "in_distribution") for b in budgets]
        out["inside_against_n"][arm] = [
            err(arm, b, "in_distribution") / err("N", b, "in_distribution") for b in budgets
        ]
        ratios = np.array([[err(arm, b, t) / err("N", b, t) for t in outside] for b in budgets])
        out["outside_against_n"][arm] = np.exp(np.log(ratios).mean(axis=1)).tolist()
        out["outside_by_set"][arm] = {t: [err(arm, b, t) for b in budgets] for t in outside}
        cost = [summary[f"{arm}/{b}"]["fit_s"] / summary[f"N/{b}"]["fit_s"] for b in budgets]
        out["cost_against_n"][arm] = float(np.mean(cost))
    for key in ("inside", "outside_against_n"):
        table = np.array([out[key][a] for a in arms])
        out[f"best_{key}"] = [arms[int(k)] for k in table.argmin(axis=0)]
    return out


def draw(res: dict, name: str, unit: str, path: Path) -> None:
    arms, budgets = res["arms"], res["budgets"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.62 * len(arms) + 2.2),
                             facecolor=report.SURFACE)  # fmt: skip
    panels = (("inside_against_n", "inside", "In distribution: " + unit),
              ("outside_against_n", None, "Outside the distribution: error against arm N"))
    for ax, (key, shown, label) in zip(axes, panels, strict=True):
        ratio = np.array([res[key][a] for a in arms])
        ax.imshow(np.log2(ratio), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        text = np.array([res[shown][a] for a in arms]) if shown else ratio
        best = text.argmin(axis=0)
        for i in range(len(arms)):
            for j in range(len(budgets)):
                value = f"{text[i, j]:.3g}" if shown else f"×{text[i, j]:.2f}"
                dark = abs(np.log2(ratio[i, j])) > 2.4
                ax.text(j, i, value, ha="center", va="center", fontsize=10,
                        color=report.SURFACE if dark else report.INK,
                        fontweight="bold" if best[j] == i else "normal")  # fmt: skip
        ax.set_xticks(range(len(budgets)), [f"{b:,}" for b in budgets])
        ax.set_yticks(range(len(arms)),
                      [f"{NAME[a]} ({res['cost_against_n'][a]:.0f}× time)" for a in arms])
        ax.set_xlabel("Labelled cases")
        ax.set_title(label, loc="left", fontsize=10, color=report.LABEL)
        ax.tick_params(colors=report.LABEL, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[1].set_yticklabels([])
    report.title(fig, f"{name}: which arm for which budget (bold: best in the column; blue: "
                 "better than arm N, red: worse)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t1 = json.loads((OUT / "loss_arms.json").read_text())["summary"]
    t2 = json.loads((OUT / "t2_arms.json").read_text())["summary"]
    res = {
        "t1": task(t1, ("N", "F", "R", "L", "FL"), (100, 1000, 10000, 100000), T1_OUTSIDE, "mae"),
        "t2": task(t2, ("N", "F", "R", "L", "FL", "A"), (100, 1000, 10000, 25000), T2_OUTSIDE,
                   "median"),
    }  # fmt: skip
    (OUT / "report.json").write_text(json.dumps(res))
    draw(res["t1"], "Task T1, growth rate", "mean absolute error, 1/s",
         OUT / "fig91_t1_choice.png")  # fmt: skip
    draw(res["t2"], "Task T2, response in time", "median relative error",
         OUT / "fig92_t2_choice.png")  # fmt: skip
    for key, r in res.items():
        print(key, "best inside", r["best_inside"], "best outside", r["best_outside_against_n"])
        for arm in r["arms"]:
            print(f"  {arm:3s} cost ×{r['cost_against_n'][arm]:.1f}  inside",
                  " ".join(f"{v:.3f}" for v in r["inside"][arm]), " outside ×",
                  " ".join(f"{v:.2f}" for v in r["outside_against_n"][arm]))  # fmt: skip


if __name__ == "__main__":
    main()
