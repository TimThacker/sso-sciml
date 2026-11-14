"""Phase 7: operators and sequence models on task T2.

    uv run scripts/phase7_t2_models.py            # 110 training runs
    uv run scripts/phase7_t2_models.py --small    # one budget and one seed, short
    uv run scripts/phase7_t2_models.py --reuse    # statistics and figures from the saved runs

Needs the dataset from scripts/phase3_generate_t2.py. Eleven models at two budgets and
five seeds:

    mlp          the perceptron of step 36, trained four times as long
    deeponet, fno, tcn, transformer     of `sso_sciml.models.t2_seq`
        each without the physics (/N) and with the reduced response as inputs (/F)
    A            the linear system of step 36

Scored on every test set by the relative error of the response, by that error within
six stretches of the window, and by the growth rate read off the predicted response.
Training time, peak GPU memory and time of a prediction are recorded.

Writes docs/results/phase7/t2_models.json and figures fig93 to fig96. Every number in
docs/steps/38-t2-operators.md comes from this script.
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
from sso_sciml.models import t2_seq as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t2.VERSION
OUT = ROOT / "docs" / "results" / "phase7"
BUDGETS = (1000, 25000)
MLP_STEPS = 16000
KEYS = ("theta", "observed", "reduced", "valid", "states", "a_red")
EDGES = np.array([0, 34, 67, 100, 134, 167, 201])  # six stretches of about 50 ms
TRUST = 0.1  # a stretch with a median error above this is not trusted
FAMILIES = ("mlp", *ms.ARCHS)
MODELS = (*(f"{f}/{a}" for f in FAMILIES for a in ("N", "F")), "A")
COLOUR = dict(zip((*FAMILIES, "A"), report.SERIES, strict=False))
PAIRS = (*((f"{f}/F", f"{f}/N") for f in FAMILIES),
         *((f"{f}/F", "mlp/F") for f in ms.ARCHS), ("A", "mlp/F"),
         *((f"{f}/N", "mlp/N") for f in ms.ARCHS))  # fmt: skip


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solved"]
    out = {k: z[k][ok] for k in KEYS}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, m) -> dict:
    return {k: v[m] for k, v in d.items() if k != "fold"}


def fit(model: str, lab: dict, held: dict, seed: int, device: str, short: bool):
    if model == "A":
        return mt.train("A", lab, held, seed, mt.Config(steps=400 if short else 4000),
                        device=device)  # fmt: skip
    family, arm = model.split("/")
    if family == "mlp":
        return mt.train(arm, lab, held, seed, mt.Config(steps=800 if short else MLP_STEPS),
                        device=device)  # fmt: skip
    cfg = ms.Config(steps=400, eval_every=100) if short else ms.Config()
    return ms.train(family, arm == "F", lab, held, seed, cfg, device=device)


def score(predict, d: dict) -> dict:
    pred = predict(d)
    out = mt.metrics(pred, d)
    out["horizon"] = np.nanmedian(ms.horizon_error(pred, d, EDGES), axis=0).tolist()
    out.update(ms.growth_scores(pred, d))
    return out


def train_all(device: str, budgets: tuple, seeds: tuple, short: bool) -> dict:
    full = load("in_distribution")
    train = take(full, full["fold"] == 0)
    tests = {"in_distribution": take(full, full["fold"] == 2)}
    tests.update({s: take(load(s), slice(None)) for s in list(t2.SPLITS)[1:]})
    inside = tests["in_distribution"]

    def fixed(predict):
        return {k: score(predict, v) for k, v in tests.items()}

    res: dict = {
        "train_pool": int(train["theta"].shape[0]),
        "tests": {k: int(v["theta"].shape[0]) for k, v in tests.items()},
        "edges_s": (EDGES * t2.T_END / (t2.N_T - 1)).tolist(),
        "reduced_model_alone": fixed(mt.reduced_alone),
        "runs": [], "kept": {},
    }  # fmt: skip
    order = np.argsort(mt.case_error(mt.reduced_alone(inside), inside))
    show = int(order[len(order) // 2])
    one = {k: v[show : show + 1] for k, v in inside.items()}
    res["kept"]["truth"] = mt.targets(one)[0, :, 0].tolist()
    res["kept"]["reduced"] = mt.reduced_alone(one)[0, :, 0].tolist()
    res["kept"]["valid"] = int(one["valid"][0])
    log = DATA / f"phase7_runs_{len(budgets)}_{len(seeds)}{'_short' if short else ''}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[(r["model"], r["budget"], r["seed"])] = r
    for budget in budgets:
        budget = min(budget, train["theta"].shape[0])
        for model in MODELS:
            for seed in seeds:
                if (model, budget, seed) in done:
                    run = done[(model, budget, seed)]
                    if "kept" in run:
                        res["kept"][model] = run["kept"]
                    res["runs"].append({k: v for k, v in run.items() if k != "kept"})
                    continue
                sub = mt1.budget_subset(train, budget, seed)
                h = budget // 5
                held = {k: v[:h] for k, v in sub.items()}
                lab = {k: v[h:] for k, v in sub.items()}
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                predict, info = fit(model, lab, held, seed, device, short)
                fit_s = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
                predict(one)
                t0 = time.perf_counter()
                predict(inside)
                per_1000 = 1000 * (time.perf_counter() - t0) / inside["theta"].shape[0]
                run = {"model": model, "budget": budget, "seed": seed, "fit_s": fit_s,
                       "peak_memory_mb": peak, "predict_s_per_1000": per_1000,
                       "parameters": info.get("parameters"), "scores": fixed(predict),
                       "skipped_steps": info["skipped_steps"]}  # fmt: skip
                saved = dict(run)
                if seed == seeds[0]:
                    hist = {k: info["history"][k] for k in ("step", "held_out")}
                    run["history"] = saved["history"] = hist
                    if budget == min(budgets[-1], train["theta"].shape[0]):
                        res["kept"][model] = saved["kept"] = predict(one)[0, :, 0].tolist()
                with log.open("a") as fh:
                    fh.write(json.dumps(saved) + "\n")
                res["runs"].append(run)
                s = run["scores"]["in_distribution"]
                print(model, budget, seed, f"median {s['median']:.4f} last stretch "
                      f"{s['horizon'][-1]:.4f} growth {s['growth_error_median']:.2f} 1/s",
                      f"{fit_s:.0f} s {peak:.0f} MB", flush=True)  # fmt: skip
    return res


SCALARS = ("median", "p90", "within_10_percent", "growth_error_median", "growth_error_p90",
           "false_divergence", "missed_growth")  # fmt: skip


def statistics(res: dict) -> None:
    runs = res["runs"]
    budgets = sorted({r["budget"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    def rows(model, budget) -> list:
        return sorted((r for r in runs if (r["model"], r["budget"]) == (model, budget)),
                      key=lambda r: r["seed"])  # fmt: skip

    summary = {}
    for budget in budgets:
        for model in MODELS:
            rs = rows(model, budget)
            if not rs:
                continue
            entry = {k: float(np.mean([r[k] for r in rs]))
                     for k in ("fit_s", "peak_memory_mb", "predict_s_per_1000")}  # fmt: skip
            entry["parameters"] = rs[0]["parameters"]
            for test in res["tests"]:
                e = {}
                for m in SCALARS:
                    v = np.array([r["scores"][test][m] for r in rs], dtype=float)
                    e[m] = {"mean": float(v.mean()),
                            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}  # fmt: skip
                hz = np.array([r["scores"][test]["horizon"] for r in rs]).mean(axis=0)
                e["horizon"] = hz.tolist()
                over = np.nonzero(hz > TRUST)[0]
                e["trusted_to_s"] = float(res["edges_s"][int(over[0])] if over.size
                                          else res["edges_s"][-1])  # fmt: skip
                entry[test] = e
            summary[f"{model}@{budget}"] = entry
    paired = {}
    if len(seeds) == 5:
        for a, b in PAIRS:
            for budget in budgets:
                ra, rb = rows(a, budget), rows(b, budget)
                paired[f"{a}-{b}@{budget}"] = {
                    test: cmp.change(np.array([r["scores"][test]["median"] for r in ra]),
                                     np.array([r["scores"][test]["median"] for r in rb]))
                    for test in res["tests"]
                }  # fmt: skip
    res["summary"], res["paired"], res["budgets"] = summary, paired, budgets
    res["fit_total_s"] = float(sum(r["fit_s"] for r in runs))
    res["skipped_steps"] = {f"{r['model']}@{r['budget']}/{r['seed']}": r["skipped_steps"]
                            for r in runs if r.get("skipped_steps")}  # fmt: skip


def style_of(model: str) -> dict:
    family = model.split("/")[0]
    return {"color": COLOUR[family], "ls": "-" if model.endswith("F") or model == "A" else "--"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("t2_models_small.json" if args.small else "t2_models.json")
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = train_all(device, BUDGETS[:1] if args.small else BUDGETS,
                        (0,) if args.small else cmp.SEEDS, args.small)  # fmt: skip
    statistics(res)
    path.write_text(json.dumps(res))
    summary, budgets, tests = res["summary"], res["budgets"], list(res["tests"])
    for key, e in summary.items():
        d = e["in_distribution"]
        print(f"{key:20s} median {d['median']['mean']:.4f} ± {d['median']['sd']:.4f} "
              f"trusted to {d['trusted_to_s']:.2f} s growth {d['growth_error_median']['mean']:.2f} "
              f"1/s false div {100 * d['false_divergence']['mean']:.1f} % "
              f"fit {e['fit_s']:.0f} s {e['peak_memory_mb']:.0f} MB")  # fmt: skip
    for key, e in res["paired"].items():
        print(f"{key:28s}", " ".join(
            f"{100 * e[t]['mean']:+.0f}[{100 * e[t]['low']:+.0f},{100 * e[t]['high']:+.0f}]"
            f"{'*' if e[t]['differs'] else ' '}" for t in tests))  # fmt: skip
    if args.small:
        return
    big = budgets[-1]
    names = [t.replace("_", " ") for t in tests]

    # ---------------- figure: every model on every test set
    fig, axes = plt.subplots(1, len(budgets), figsize=(15.0, 5.2), facecolor=report.SURFACE,
                             sharey=True)  # fmt: skip
    xs = np.arange(len(tests))
    for ax, budget in zip(axes, budgets, strict=True):
        for k, model in enumerate(MODELS):
            e = summary[f"{model}@{budget}"]
            st = style_of(model)
            ax.bar(xs + (k - 5) * 0.075, [e[t]["median"]["mean"] for t in tests], 0.07,
                   yerr=[e[t]["median"]["sd"] for t in tests], color=st["color"],
                   alpha=1.0 if st["ls"] == "-" else 0.45, label=model)  # fmt: skip
        ax.plot(xs, [res["reduced_model_alone"][t]["median"] for t in tests], color=report.INK,
                marker="_", ms=40, lw=0, label="reduced model alone")  # fmt: skip
        ax.set_xticks(xs, names)
        ax.set_yscale("log")
        ax.set_title(f"{budget:,} labelled trajectories", loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Relative error of the response, median")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", ncol=6, frameon=False,
               fontsize=8, labelcolor=report.LABEL, markerscale=0.3)  # fmt: skip
    report.title(fig, "Task T2: eleven models on every test set (pale: no physics; full: "
                 "reduced response as inputs)")  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig93_t2_models.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error against time within the window
    mid = 0.5 * (np.array(res["edges_s"][:-1]) + np.array(res["edges_s"][1:]))
    shown = ("in_distribution", "high_compensation", "fast_current_loop")
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, test in zip(axes, shown, strict=True):
        for model in MODELS:
            ax.plot(mid, summary[f"{model}@{big}"][test]["horizon"], marker="o", ms=3, lw=1.5,
                    label=model, **style_of(model))  # fmt: skip
        ax.plot(mid, res["reduced_model_alone"][test]["horizon"], color=report.INK, lw=1, ls=":",
                label="reduced model alone")  # fmt: skip
        ax.axhline(TRUST, color=report.MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Time after the disturbance (s)")
        ax.set_title(test.replace("_", " "), loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Relative error within the stretch, median")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", ncol=6, frameon=False,
               fontsize=8, labelcolor=report.LABEL)  # fmt: skip
    report.title(fig, f"Error against time, {big:,} trajectories (dashed: no physics; grey "
                 "line: 10 %)")  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig94_t2_horizon.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: training, cost and growth rate
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    for model in MODELS:
        st = style_of(model)
        h = next(r["history"] for r in res["runs"]
                 if (r["model"], r["budget"]) == (model, big) and "history" in r)  # fmt: skip
        axes[0].plot(np.array(h["step"]) / h["step"][-1], h["held_out"], lw=1.5, label=model, **st)
        e = summary[f"{model}@{big}"]
        mark = "o" if st["ls"] == "-" else "s"
        fill = st["color"] if st["ls"] == "-" else report.SURFACE
        axes[1].plot(e["fit_s"], e["in_distribution"]["median"]["mean"], marker=mark, ms=8,
                     color=st["color"], mfc=fill, lw=0)  # fmt: skip
        axes[1].annotate(model, (e["fit_s"], e["in_distribution"]["median"]["mean"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7,
                         color=report.LABEL)  # fmt: skip
        axes[2].plot(e["in_distribution"]["growth_error_median"]["mean"],
                     100 * e["in_distribution"]["false_divergence"]["mean"], marker=mark, ms=8,
                     color=st["color"], mfc=fill, lw=0)  # fmt: skip
        axes[2].annotate(model, (e["in_distribution"]["growth_error_median"]["mean"],
                                 100 * e["in_distribution"]["false_divergence"]["mean"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7,
                         color=report.LABEL)  # fmt: skip
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Share of the training done")
    axes[0].set_title("Held-out relative error, median, first seed", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, labelcolor=report.LABEL)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training time (s)")
    axes[1].set_title("Test error against training time", loc="left", fontsize=10)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Error of the growth rate read off the prediction, median (1/s)")
    axes[2].set_title("Decaying cases predicted to grow (%)", loc="left", fontsize=10)
    for ax in axes:
        report.style(ax)
    report.title(fig, f"Training, cost and growth rate at {big:,} trajectories, in distribution")
    fig.tight_layout()
    fig.savefig(OUT / "fig95_t2_cost.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: one test case
    kept, t = res["kept"], t2.times()
    stop = kept["valid"]
    fig, axes = plt.subplots(3, 4, figsize=(18.0, 9.0), facecolor=report.SURFACE, sharex=True,
                             sharey=True)  # fmt: skip
    for ax, model in zip(axes.ravel(), MODELS, strict=False):
        ax.plot(t[:stop], np.array(kept["truth"])[:stop], color=report.INK, lw=1.4,
                label="full model")  # fmt: skip
        ax.plot(t[:stop], np.array(kept[model])[:stop], color=COLOUR[model.split("/")[0]], lw=1.2,
                ls="--", label="prediction")  # fmt: skip
        ax.set_title(model, loc="left", fontsize=10)
        report.style(ax)
    axes.ravel()[-1].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Line current, D (scaled)")
    report.title(fig, f"One test case, models trained on {big:,} trajectories, first seed")
    fig.tight_layout()
    fig.savefig(OUT / "fig96_t2_case.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
