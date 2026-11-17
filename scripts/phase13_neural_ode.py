"""Step 46: a neural ODE with a nonlinear correction (arm AN) against arm A, on task T2.

    uv run scripts/phase13_neural_ode.py            # 20 training runs, about 3.5 hours
    uv run scripts/phase13_neural_ode.py --reuse    # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_t2.py. Arm A is the linear system of
step 36; arm AN adds a learned term g(z, theta) to its dynamics and starts from arm A
(`models.t2`). Both are trained for 4,000 steps at a quarter of the learning rate, as
in steps 36 and 38, at 1,000 and 25,000 trajectories, with five seeds shared by the two
arms, and scored on every test set. In distribution the cases are also split into those
whose sub-synchronous mode grows and those where it decays, since a nonlinear term can
only matter where the response gets large. The perceptron with physics inputs of step
38 is quoted for comparison. Runs are logged one by one, so the script resumes.

Writes docs/results/phase13/neural_ode.json and fig116. Every number in
docs/steps/46-neural-ode.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.data import t2
from sso_sciml.models import compare as cmp
from sso_sciml.models import t1 as mt1
from sso_sciml.models import t2 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t2.VERSION
OUT = ROOT / "docs" / "results" / "phase13"
BUDGETS = (1000, 25000)
ARMS = ("A", "AN")
LABEL = {"A": "A: linear system", "AN": "AN: with a nonlinear term"}
KEYS = ("theta", "observed", "reduced", "valid", "states", "a_red", "sigma")


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solved"]
    out = {k: z[k][ok] for k in KEYS}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, m) -> dict:
    return {k: v[m] for k, v in d.items() if k != "fold"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "neural_ode.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load("in_distribution")
    test = take(full, full["fold"] == 2)
    tests = {"in_distribution": test,
             "growing": take(test, test["sigma"] > 0), "decaying": take(test, test["sigma"] <= 0)}
    tests.update({s: take(load(s), slice(None)) for s in list(t2.SPLITS)[1:]})  # fmt: skip
    if not args.reuse:
        train = take(full, full["fold"] == 0)
        log = OUT / "neural_ode_runs.jsonl"
        done = {}
        if log.exists():
            for line in log.read_text().splitlines():
                r = json.loads(line)
                done[(r["arm"], r["budget"], r["seed"])] = r
        for budget in BUDGETS:
            for seed in cmp.SEEDS:
                sub = mt1.budget_subset(train, budget, seed)
                h = budget // 5
                held, lab = ({k: v[:h] for k, v in sub.items()},
                             {k: v[h:] for k, v in sub.items()})  # fmt: skip
                for arm in ARMS:
                    if (arm, budget, seed) in done:
                        continue
                    predict, info = mt.train(arm, lab, held, seed, mt.Config(steps=4000),
                                             device=device)  # fmt: skip
                    r = {"arm": arm, "budget": budget, "seed": seed, "train_s": info["train_s"],
                         "skipped_steps": info["skipped_steps"],
                         "scores": {k: mt.metrics(predict(v), v) for k, v in tests.items()},
                         "history": {k: info["history"][k]
                                     for k in ("step", "held_out")}}  # fmt: skip
                    with log.open("a") as fh:
                        fh.write(json.dumps(r) + "\n")
                    done[(arm, budget, seed)] = r
                    print(arm, budget, seed, {k: round(v["median"], 4)
                                              for k, v in r["scores"].items()},
                          f"{info['train_s']:.0f} s skipped {info['skipped_steps']}",
                          flush=True)  # fmt: skip
        runs = list(done.values())
        res: dict = {"tests": {k: int(len(v["theta"])) for k, v in tests.items()}, "runs": runs}
        summary, paired = {}, {}
        for budget in BUDGETS:
            for arm in ARMS:
                rs = sorted((r for r in runs if (r["arm"], r["budget"]) == (arm, budget)),
                            key=lambda r: r["seed"])  # fmt: skip
                summary[f"{arm}@{budget}"] = {
                    "train_s": float(np.mean([r["train_s"] for r in rs])),
                    **{k: {"mean": float(np.mean([r["scores"][k]["median"] for r in rs])),
                           "sd": float(np.std([r["scores"][k]["median"] for r in rs], ddof=1))}
                       for k in tests}}  # fmt: skip
            a, b = (sorted((r for r in runs if (r["arm"], r["budget"]) == (x, budget)),
                           key=lambda r: r["seed"]) for x in ARMS)  # fmt: skip
            paired[f"AN-A@{budget}"] = {
                k: cmp.change(np.array([r["scores"][k]["median"] for r in b]),
                              np.array([r["scores"][k]["median"] for r in a]))
                for k in tests}  # fmt: skip
        res["summary"], res["paired"] = summary, paired
        prev = json.loads((ROOT / "docs/results/phase7/t2_models.json").read_text())["summary"]
        res["step38_mlp_F"] = {str(b): {k: prev[f"mlp/F@{b}"][k]["median"]["mean"]
                                        for k in t2.SPLITS} for b in BUDGETS}  # fmt: skip
        path.write_text(json.dumps(res))
    res = json.loads(path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE, sharey=True)
    sets = list(res["tests"])
    xs = np.arange(len(sets))
    for ax, budget in zip(axes, BUDGETS, strict=True):
        for k, arm in enumerate(ARMS):
            s = res["summary"][f"{arm}@{budget}"]
            ax.bar(xs + (k - 0.5) * 0.3, [s[t]["mean"] for t in sets], 0.28,
                   yerr=[s[t]["sd"] for t in sets], color=report.SERIES[5],
                   alpha=1.0 if k else 0.5,
                   label=LABEL[arm])  # fmt: skip
        mlp = res["step38_mlp_F"][str(budget)]
        ax.plot([xs[sets.index(t)] for t in mlp], list(mlp.values()), marker="_", ms=30, lw=0,
                color=report.INK, label="perceptron with physics inputs, step 38")  # fmt: skip
        ax.set_xticks(xs, [t.replace("_", " ") for t in sets], rotation=15, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{budget:,} trajectories", loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Median relative error of the response")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=0.4)
    report.title(fig, "A neural ODE with a nonlinear correction against the linear system "
                 "(five seeds, 4,000 steps)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig116_neural_ode.png", dpi=160)
    plt.close(fig)

    for k, v in res["summary"].items():
        print(k, f"{v['train_s']:.0f} s", {t: (round(x["mean"], 4), round(x["sd"], 4))
                                            for t, x in v.items() if t != "train_s"})  # fmt: skip
    for k, v in res["paired"].items():
        print(k, {t: f"{100 * x['mean']:+.0f}[{100 * x['low']:+.0f},{100 * x['high']:+.0f}]"
                  f"{'*' if x['differs'] else ''}" for t, x in v.items()})  # fmt: skip
    print("step 38 mlp/F", res["step38_mlp_F"])
    print("skipped", {f"{r['arm']}@{r['budget']}/{r['seed']}": r["skipped_steps"]
                      for r in res["runs"] if r["skipped_steps"]})  # fmt: skip


if __name__ == "__main__":
    main()
