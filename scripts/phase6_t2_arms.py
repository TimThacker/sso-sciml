"""Phase 6, task T2: the response in time with the physics used in six ways.

    uv run scripts/phase6_t2_arms.py            # 195 training runs
    uv run scripts/phase6_t2_arms.py --small    # two budgets and two seeds, for a quick look
    uv run scripts/phase6_t2_arms.py --reuse    # statistics and figures from the saved runs

Needs the dataset from scripts/phase3_generate_t2.py. Arms N, F, R, L, F+L and A of
`sso_sciml.models.t2`, and L and F+L with the residual switched off (L0, FL0), at four
data budgets and five seeds. A seed fixes the labelled cases and the initial weights and
is shared by the arms, so differences are paired. Arm L is also run at two other weights
of the residual at the two small budgets, and arms N, F and R with four times the
training steps at the largest budget.

Writes docs/results/phase6/t2_arms.json and figures fig87 to fig90. Every number in
docs/steps/36-t2-arms.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
import time
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
OUT = ROOT / "docs" / "results" / "phase6"
BUDGETS = (100, 1000, 10000, 25000)
N_POOL = 2048
OTHER_WEIGHTS = (0.1, 10.0)
LONG, LONGER, LONG_ARMS = "@long", 4, ("N", "F", "R")  # four times the training steps
KEYS = ("theta", "observed", "reduced", "valid", "states", "a_red")
PAIRS = (("F", "N"), ("R", "N"), ("R", "F"), ("L", "N"), ("L", "L0"), ("FL", "F"),
         ("FL", "FL0"), ("A", "N"), ("A", "R"), ("A", "F"), ("L0", "N"))  # fmt: skip
SHOWN = ("N", "F", "R", "L", "FL", "A")
COLOUR = {**report.ARM_COLOUR, "L": report.SERIES[4], "FL": report.SERIES[5],
          "A": report.SERIES[3]}  # fmt: skip
NAME = {**report.ARM_NAME, "L": "L: physics as a penalty", "FL": "F+L: features and penalty",
        "A": "A: physics in the architecture"}  # fmt: skip


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solved"]
    out = {k: z[k][ok] for k in KEYS}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in d.items() if k != "fold"}


def train_all(device: str, budgets: tuple, seeds: tuple) -> dict:
    full = load("in_distribution")
    train = take(full, full["fold"] == 0)
    tests = {"in_distribution": take(full, full["fold"] == 2)}
    tests.update({s: take(load(s), slice(None)) for s in list(t2.SPLITS)[1:]})
    pool = mt.unlabelled(mt1.budget_subset({"theta": train["theta"]}, N_POOL, 0)["theta"], 8)
    res: dict = {
        "train_pool": int(train["theta"].shape[0]), "unlabelled_pool": int(pool["theta"].shape[0]),
        "tests": {k: int(v["theta"].shape[0]) for k, v in tests.items()},
        "reduced_model_alone": {k: mt.metrics(mt.reduced_alone(v), v) for k, v in tests.items()},
        "runs": [], "kept": {},
    }  # fmt: skip
    steps = mt.Config().steps
    jobs = [(arm, arm, 1.0, b, steps) for b in budgets for arm in mt.ARMS]
    jobs += [(f"L@{w:g}", "L", w, b, steps) for b in budgets[:2] for w in OTHER_WEIGHTS]
    # Is the largest budget limited by the data or by the length of training?
    jobs += [(f"{arm}{LONG}", arm, 1.0, budgets[-1], LONGER * steps) for arm in LONG_ARMS]
    # The test case on which the reduced model has its median error is kept for a figure.
    inside = tests["in_distribution"]
    order = np.argsort(mt.case_error(mt.reduced_alone(inside), inside))
    show = int(order[len(order) // 2])
    one = {k: v[show : show + 1] for k, v in inside.items()}
    res["kept"]["truth"] = mt.targets(one)[0, :, 0].tolist()
    res["kept"]["reduced"] = mt.reduced_alone(one)[0, :, 0].tolist()
    res["kept"]["valid"] = int(one["valid"][0])
    # Finished runs are kept in a file beside the dataset, so a stopped pass can go on.
    log = DATA / f"arm_runs_{len(budgets)}_{len(seeds)}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[(r["arm"], r["budget"], r["seed"])] = r
    for name, arm, weight, budget, n_steps in jobs:
        budget = min(budget, train["theta"].shape[0])
        for seed in seeds:
            if (name, budget, seed) in done:
                run = done[(name, budget, seed)]
                if "kept" in run:
                    res["kept"][name] = run["kept"]
                res["runs"].append({k: v for k, v in run.items() if k != "kept"})
                continue
            sub = mt1.budget_subset(train, budget, seed)
            h = budget // 5
            held, lab = ({k: v[:h] for k, v in sub.items()}, {k: v[h:] for k, v in sub.items()})
            predict, info = mt.train(arm, lab, held, seed, mt.Config(weight=weight, steps=n_steps),
                                     pool=pool, device=device)  # fmt: skip
            run = {"arm": name, "budget": budget, "seed": seed, "fit_s": info["train_s"],
                   "skipped_steps": info["skipped_steps"],
                   "scores": {k: mt.metrics(predict(v), v) for k, v in tests.items()}}  # fmt: skip
            saved = dict(run)
            if seed == seeds[0]:
                run["history"] = saved["history"] = info["history"]
                if budget == budgets[1]:
                    res["kept"][name] = saved["kept"] = predict(one)[0, :, 0].tolist()
            with log.open("a") as fh:
                fh.write(json.dumps(saved) + "\n")
            res["runs"].append(run)
            s = run["scores"]["in_distribution"]
            print(name, budget, seed, f"median {s['median']:.3f} p90 {s['p90']:.3f}",
                  f"{info['train_s']:.0f} s", flush=True)  # fmt: skip
    return res


def statistics(res: dict) -> None:
    runs = res["runs"]
    budgets = sorted({r["budget"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    def get(arm, budget, test, metric) -> np.ndarray:
        rows = sorted((r for r in runs if (r["arm"], r["budget"]) == (arm, budget)),
                      key=lambda r: r["seed"])  # fmt: skip
        if metric == "fit_s":
            return np.array([r["fit_s"] for r in rows])
        return np.array([r["scores"][test][metric] for r in rows], dtype=float)

    have = sorted({(r["arm"], r["budget"]) for r in runs})
    summary = {}
    for arm, budget in have:
        entry = {"fit_s": float(get(arm, budget, "", "fit_s").mean())}
        for test in res["tests"]:
            entry[test] = {m: {"mean": float(get(arm, budget, test, m).mean()),
                               "sd": float(get(arm, budget, test, m).std(ddof=1))}
                           for m in ("median", "mean", "p90", "within_10_percent")}  # fmt: skip
        summary[f"{arm}/{budget}"] = entry
    paired = {}
    if len(seeds) == 5:
        pairs = [*PAIRS, *((f"L@{w:g}", "L0") for w in OTHER_WEIGHTS),
                 *((f"{a}{LONG}", a) for a in LONG_ARMS),
                 (f"F{LONG}", f"N{LONG}"), (f"R{LONG}", f"N{LONG}")]  # fmt: skip
        for a, b in pairs:
            for budget in budgets:
                if (a, budget) in have and (b, budget) in have:
                    paired[f"{a}-{b}/{budget}"] = {
                        test: cmp.change(get(a, budget, test, "median"),
                                         get(b, budget, test, "median"))
                        for test in res["tests"]
                    }  # fmt: skip
    res["summary"], res["paired"], res["budgets"] = summary, paired, budgets
    res["fit_total_s"] = float(sum(r["fit_s"] for r in runs))
    res["skipped_steps"] = {f"{r['arm']}/{r['budget']}/{r['seed']}": r["skipped_steps"]
                            for r in runs if r.get("skipped_steps")}  # fmt: skip


def bars(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([100 * r["mean"] for r in rows])
    err = np.array([[100 * (r["mean"] - r["low"]), 100 * (r["high"] - r["mean"])] for r in rows]).T
    return mean, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("t2_arms_small.json" if args.small else "t2_arms.json")
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        t0 = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = train_all(device, BUDGETS[:2] if args.small else BUDGETS,
                        (0, 1) if args.small else cmp.SEEDS)  # fmt: skip
        res["wall_s"] = time.perf_counter() - t0
    statistics(res)
    path.write_text(json.dumps(res))
    summary, paired, budgets = res["summary"], res["paired"], res["budgets"]
    for key, e in summary.items():
        d = e["in_distribution"]
        near = 100 * d["within_10_percent"]["mean"]
        print(f"{key:12s} median {d['median']['mean']:.3f} ± {d['median']['sd']:.3f} "
              f"p90 {d['p90']['mean']:.3f} within 10 % {near:.0f} % fit {e['fit_s']:.0f} s")
    for key, e in paired.items():
        c = e["in_distribution"]
        print(f"{key:14s} {100 * c['mean']:+6.1f}% [{100 * c['low']:+6.1f}, "
              f"{100 * c['high']:+6.1f}] {'*' if c['differs'] else ''}")  # fmt: skip
    print("reduced model alone", res["reduced_model_alone"])
    if args.small:
        return
    x = np.arange(len(budgets))
    ticks = [f"{n:,}" for n in budgets]

    # ---------------- figure: error against budget, and the paired changes
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), facecolor=report.SURFACE)
    for arm in SHOWN:
        rows = [summary[f"{arm}/{n}"]["in_distribution"]["median"] for n in budgets]
        mean, sd = [r["mean"] for r in rows], [r["sd"] for r in rows]
        axes[0].errorbar(x, mean, yerr=sd, color=COLOUR[arm], marker="o", ms=5, lw=2, capsize=3,
                         label=NAME[arm])  # fmt: skip
    for arm in LONG_ARMS:
        row = summary[f"{arm}{LONG}/{budgets[-1]}"]["in_distribution"]["median"]
        note = f"hollow: {LONGER} times the steps" if arm == "N" else None
        axes[0].errorbar(x[-1] + 0.12, row["mean"], yerr=row["sd"], color=COLOUR[arm], marker="o",
                         ms=6, mfc=report.SURFACE, capsize=3, label=note)  # fmt: skip
    ref = res["reduced_model_alone"]["in_distribution"]["median"]
    axes[0].axhline(ref, color=report.MUTED, lw=1, ls="--", label="reduced model alone")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative error of the response, median over test cases")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", ncol=4, frameon=False,
               fontsize=8, labelcolor=report.LABEL)  # fmt: skip
    for ax, group in ((axes[1], (("F", "N"), ("R", "N"), ("A", "N"))),
                      (axes[2], (("L", "L0"), ("FL", "FL0"), ("L", "N")))):  # fmt: skip
        for k, (a, b) in enumerate(group):
            mean, err = bars([paired[f"{a}-{b}/{n}"]["in_distribution"] for n in budgets])
            ax.errorbar(x + (k - 1) * 0.12, mean, yerr=err, color=COLOUR[a] if k < 2 or a == "A"
                        else report.MUTED, marker="o", ms=5, lw=1.5, capsize=3,
                        label=f"{a} against {b}")  # fmt: skip
        ax.axhline(0, color=report.INK, lw=0.8, ls="--")
        ax.set_ylabel("Change of the median error, paired (%)")
        ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    for ax in axes:
        ax.set_xticks(x, ticks)
        ax.set_xlabel("Labelled trajectories")
        report.style(ax)
    report.title(fig, "Task T2, in distribution: six arms over five seeds (bars: one standard "
                 "deviation; changes, 95 % interval)")  # fmt: skip
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "fig87_t2_budget.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: every test set
    tests = list(res["tests"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), facecolor=report.SURFACE, sharey=True)
    xs = np.arange(len(tests))
    for ax, budget in zip(axes, (budgets[1], budgets[-1]), strict=True):
        for k, arm in enumerate(SHOWN):
            rows = [summary[f"{arm}/{budget}"][t]["median"] for t in tests]
            ax.bar(xs + (k - 2.5) * 0.13, [r["mean"] for r in rows], 0.12,
                   yerr=[r["sd"] for r in rows], color=COLOUR[arm], label=NAME[arm])  # fmt: skip
        ax.plot(xs, [res["reduced_model_alone"][t]["median"] for t in tests], color=report.INK,
                marker="_", ms=22, lw=0, label="reduced model alone")  # fmt: skip
        ax.set_xticks(xs, [t.replace("_", " ") for t in tests])
        ax.set_yscale("log")
        ax.set_title(f"{budget:,} labelled trajectories", loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Relative error of the response, median")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower left", ncol=4, frameon=False,
               fontsize=8, labelcolor=report.LABEL)  # fmt: skip
    report.title(fig, "Task T2 on every test set")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "fig88_t2_splits.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: training curves at the second budget, first seed
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), facecolor=report.SURFACE)
    for arm in SHOWN:
        h = next(r["history"] for r in res["runs"]
                 if (r["arm"], r["budget"]) == (arm, budgets[1]) and "history" in r)  # fmt: skip
        axes[0].plot(h["step"], h["data_loss"], color=COLOUR[arm], lw=1.6, label=NAME[arm])
        axes[1].plot(h["step"], h["held_out"], color=COLOUR[arm], lw=1.6)
        if np.isfinite(h["residual"]).all():
            axes[2].plot(h["step"], h["residual"], color=COLOUR[arm], lw=1.6, label=NAME[arm])
    for ax, label in zip(axes, ("Data loss (scaled)", "Held-out relative error, median",
                                "Residual of the equations (scaled)"), strict=True):  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_title(label, loc="left", fontsize=10)
        report.style(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, f"Training at {budgets[1]:,} labelled trajectories, first seed")
    fig.tight_layout()
    fig.savefig(OUT / "fig89_t2_training.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: one test case, every arm
    kept, t = res["kept"], t2.times()
    stop = kept["valid"]
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 7.0), facecolor=report.SURFACE, sharex=True,
                             sharey=True)  # fmt: skip
    for ax, arm in zip(axes.ravel(), SHOWN, strict=True):
        ax.plot(t[:stop], np.array(kept["truth"])[:stop], color=report.INK, lw=1.4,
                label="full model")  # fmt: skip
        ax.plot(t[:stop], np.array(kept["reduced"])[:stop], color=report.MUTED, lw=0.9, ls=":",
                label="reduced model")  # fmt: skip
        ax.plot(t[:stop], np.array(kept[arm])[:stop], color=COLOUR[arm], lw=1.2, ls="--",
                label="prediction")  # fmt: skip
        ax.set_title(NAME[arm], loc="left", fontsize=10)
        report.style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Line current, D, deviation (scaled)")
    report.title(fig, f"One test case, models trained on {budgets[1]:,} trajectories, first seed")
    fig.tight_layout()
    fig.savefig(OUT / "fig90_t2_case.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
