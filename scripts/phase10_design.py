"""Phase 10: design search for damping.

    uv run scripts/phase10_design.py            # about 20 minutes
    uv run scripts/phase10_design.py --reuse    # figures from the saved numbers

The design problem is in `sso_sciml.design`: rotor current loop bandwidth (floor 10 Hz),
gain and centre frequency of a damping controller, and the number of series-capacitor
segments in service. A design is scored by the largest growth rate of any oscillatory
mode over 12 scenarios, so one design evaluation is 12 solver calls. Two problems:

    free    all four quantities are chosen, as the plan states it
    full    all five segments stay in service and the controller has to cope

and on each, five seeds of

    random        scrambled Sobol points
    cma           CMA-ES, population 8
    bayes         Gaussian process with expected improvement
    surrogate     a perceptron trained on solver calls spread over designs and
                  scenarios; the search runs on the perceptron and the solver confirms
                  its 16 best designs. Training data: 2,000 and 20,000 solver calls,
                  used for both problems.

Then, with the segments held at each number, CMA-ES finds the best controller, and the
best designs are run in the nonlinear time-domain model on every scenario.

Writes docs/results/phase10/design.json and figures fig105 to fig108. Every number in
docs/steps/41-design-search.md comes from this script.
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
from scipy.stats import qmc

from sso_sciml import design as ds
from sso_sciml import report, search
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "phase10"
EVALS = 240
SEEDS = (0, 1, 2, 3, 4)
PROCESSES = 8
N_SCEN = len(ds.SCENARIOS)
TRAIN_CALLS = (2000, 20000)
CONFIRM = 16
CANDIDATES = 200_000
WITHIN = (1.0, 0.25)  # targets: this far from the best score any method found, 1/s
CAP = 20.0  # scores above this are entered at it in the Gaussian process, 1/s
PROBLEMS = {"free": None, "full": ds.MAX_SEGMENTS}
PROBLEM_NAME = {"free": "Segments chosen freely", "full": "All five segments in service"}
METHODS = {"random": "Random search", "cma": "CMA-ES", "bayes": "Bayesian optimisation"}
ALL = (*METHODS, *[f"surrogate/{n}" for n in TRAIN_CALLS])
LABEL = {**METHODS, **{f"surrogate/{n}": f"Surrogate, {n:,} training calls" for n in TRAIN_CALLS}}
COLOUR = dict(zip(ALL, report.SERIES, strict=False))
GRID = 25


def designs_of(z: np.ndarray, segments: int | None) -> np.ndarray:
    """Points of the unit cube to designs. With the segments held, the cube has three
    coordinates."""
    z = np.atleast_2d(z)
    if segments is None:
        return ds.decode(z)
    d = ds.decode(np.column_stack([z, np.zeros(len(z))]))
    d[:, 3] = segments
    return d


class Objective:
    def __init__(self, segments: int | None = None):
        self.segments = segments
        self.dim = 4 if segments is None else 3

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return ds.worst_case(designs_of(z, self.segments), PROCESSES)[0]


def history(z: np.ndarray, v: np.ndarray, wall: float, segments: int | None) -> dict:
    j = int(np.argmin(v))
    return {"values": v.tolist(), "best": float(v[j]),
            "design": designs_of(z, segments)[j].tolist(), "wall_s": wall}  # fmt: skip


# ---------------------------------------------------------------- surrogate
def surrogate_inputs(designs: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
    """Scaled inputs: bandwidth (log), gain, centre frequency, compensation, speed,
    units online, parallel path (log)."""
    lo, hi = np.log(ds.BOUNDS["rsc_hz"][0]), np.log(ds.BOUNDS["rsc_hz"][1])
    cols = [
        2 * (np.log(designs[:, 0]) - lo) / (hi - lo) - 1,
        2 * designs[:, 1] / ds.BOUNDS["damp_gain"][1] - 1,
        2 * (designs[:, 2] - 5.0) / 45.0 - 1,
        2 * (designs[:, 3] - 1.0) / (ds.MAX_SEGMENTS - 1) - 1,
        (scenarios[:, 0] - 1.0) / 0.2,
        2 * (scenarios[:, 1] - 0.4) / 0.6 - 1,
        2 * np.log(scenarios[:, 2]) / np.log(8.0) - 1,
    ]
    return np.column_stack(cols)


def sample_training(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Designs and scenarios spread over the whole space. The compensation is continuous
    here (a fractional number of segments), and the scenarios fill their ranges."""
    u = qmc.Sobol(d=7, scramble=True, seed=100 + seed).random(n)
    d = ds.decode(u[:, :4])
    d[:, 3] = 1.0 + u[:, 3] * (ds.MAX_SEGMENTS - 1)
    sc = np.column_stack([0.8 + 0.4 * u[:, 4], 0.4 + 0.6 * u[:, 5], np.exp(u[:, 6] * np.log(8.0))])
    return d, sc


def solve_pairs(d: np.ndarray, sc: np.ndarray) -> np.ndarray:
    cases = np.array([np.r_[ds.theta_of(ds.SEGMENT * a[3], a[0], s), a[1], a[2]]
                      for a, s in zip(d, sc, strict=True)])  # fmt: skip
    out = ds.solve_many(cases, PROCESSES)
    return np.where(out[:, 5] > 0, out[:, 2], ds.FAILED)


def surrogate_search(n_train: int, seed: int, device: str, test: tuple) -> dict:
    t0 = time.perf_counter()
    d, sc = sample_training(n_train, seed)
    y = np.clip(solve_pairs(d, sc), -40.0, 40.0) / mt.SIGMA_SCALE
    data_s = time.perf_counter() - t0
    x = surrogate_inputs(d, sc)
    h = n_train // 5
    t1 = time.perf_counter()
    model, info = mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed,
                               mt.TrainConfig(steps=6000), device=device)  # fmt: skip
    fit_s = time.perf_counter() - t1
    test_pred = mt.predict(model, surrogate_inputs(*test[:2]), device) * mt.SIGMA_SCALE
    out = {
        "train_calls": n_train, "confirm_calls": CONFIRM * N_SCEN, "data_s": data_s,
        "fit_s": fit_s, "test_mae": float(np.abs(test_pred - test[2]).mean()),
        "history": [{"step": r[0], "held_out": r[2] * mt.SIGMA_SCALE} for r in info["history"]],
        "parity": {"true": test[2][:1500].tolist(), "pred": test_pred[:1500].tolist()},
        "problems": {},
    }  # fmt: skip
    # Search on the perceptron: many designs, each on the 12 scenarios.
    rng = np.random.default_rng(seed)
    for name, segments in PROBLEMS.items():
        t2 = time.perf_counter()
        cand = designs_of(rng.random((CANDIDATES, 4 if segments is None else 3)), segments)
        xx = surrogate_inputs(np.repeat(cand, N_SCEN, axis=0),
                              np.tile(ds.SCENARIOS, (CANDIDATES, 1)))  # fmt: skip
        pred = mt.predict(model, xx, device) * mt.SIGMA_SCALE
        pred = pred.reshape(CANDIDATES, N_SCEN).max(axis=1)
        top = np.argsort(pred)[:CONFIRM]
        true = ds.worst_case(cand[top], PROCESSES)[0]
        j = int(np.argmin(true))
        out["problems"][name] = {
            "predicted": pred[top].tolist(), "confirmed": true.tolist(), "best": float(true[j]),
            "design": cand[top][j].tolist(), "predicted_at_best": float(pred[top][j]),
            "search_s": time.perf_counter() - t2,
        }  # fmt: skip
    out["wall_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------- the study
def study(device: str) -> dict:
    res: dict = {"evals": EVALS, "scenarios": ds.SCENARIOS.tolist(), "segment": ds.SEGMENT,
                 "floor_hz": ds.FLOOR_HZ, "bounds": ds.BOUNDS, "runs": {},
                 "surrogate": {}}  # fmt: skip
    t0 = time.perf_counter()
    ds.worst_case(ds.decode(np.full((8, 4), 0.5)), PROCESSES)  # starts the workers
    one = time.perf_counter()
    ds.worst_case(ds.decode(np.random.default_rng(0).random((64, 4))), PROCESSES)
    res["solver_s_per_call"] = (time.perf_counter() - one) / (64 * N_SCEN)

    # Baseline: no damping controller, bandwidth at the floor and at 30 Hz.
    base = np.array([[hz, 0.0, 30.0, n] for n in range(1, ds.MAX_SEGMENTS + 1)
                     for hz in (ds.FLOOR_HZ, 30.0)])  # fmt: skip
    score, out = ds.worst_case(base, PROCESSES)
    res["baseline"] = [{"design": b.tolist(), "score": float(s),
                        "worst_scenario": ds.SCENARIOS[int(np.argmax(o[:, 2]))].tolist(),
                        "f_at_worst": float(o[int(np.argmax(o[:, 2])), 3])}
                       for b, s, o in zip(base, score, out, strict=True)]  # fmt: skip
    print("baseline", [(b["design"][0], b["design"][3], round(b["score"], 2))
                       for b in res["baseline"]], flush=True)  # fmt: skip

    run = {"random": search.random_search, "cma": search.cma_es,
           "bayes": lambda f, dim, n, seed: search.bayes(f, dim, n, seed, cap=CAP)}  # fmt: skip
    for problem, segments in PROBLEMS.items():
        res["runs"][problem] = {}
        for name, method in run.items():
            res["runs"][problem][name] = []
            for seed in SEEDS:
                f = Objective(segments)
                t1 = time.perf_counter()
                z, v = method(f, f.dim, EVALS, seed)
                r = history(z, v, time.perf_counter() - t1, segments)
                res["runs"][problem][name].append(r)
                print(problem, name, seed, f"best {r['best']:.2f}",
                      np.round(r["design"], 2).tolist(), f"{r['wall_s']:.0f} s", flush=True)

    td, tsc = sample_training(4000, 99)
    test = (td, tsc, np.clip(solve_pairs(td, tsc), -40.0, 40.0))
    for n_train in TRAIN_CALLS:
        res["surrogate"][str(n_train)] = []
        for seed in SEEDS:
            r = surrogate_search(n_train, seed, device, test)
            res["surrogate"][str(n_train)].append(r)
            print("surrogate", n_train, seed, f"mae {r['test_mae']:.2f}",
                  {k: (round(v["best"], 2), round(v["predicted_at_best"], 2),
                       np.round(v["design"], 2).tolist()) for k, v in r["problems"].items()},
                  f"{r['wall_s']:.0f} s", flush=True)  # fmt: skip

    # The best controller at each number of segments.
    res["per_segment"] = []
    for n in range(1, ds.MAX_SEGMENTS + 1):
        f = Objective(n)
        t1 = time.perf_counter()
        z, v = search.cma_es(f, f.dim, EVALS, 0)
        r = history(z, v, time.perf_counter() - t1, n)
        _, o = ds.worst_case(np.array(r["design"]), PROCESSES)
        r["sigma_osc"], r["f_osc"], r["sigma_max"] = (o[0][:, k].tolist() for k in (2, 3, 4))
        res["per_segment"].append(r)
        print("segments", n, f"best {r['best']:.2f}", np.round(r["design"], 2).tolist(),
              "mode at", np.round(r["f_osc"], 1).tolist(), flush=True)  # fmt: skip

    # The surface around the best design with all segments in service.
    best = np.array(res["per_segment"][-1]["design"])
    gains = np.linspace(*ds.BOUNDS["damp_gain"], GRID)
    freqs = np.linspace(*ds.BOUNDS["damp_hz"], GRID)
    grid = np.array([[best[0], g, f, best[3]] for g in gains for f in freqs])
    surface = ds.worst_case(grid, PROCESSES)[0].reshape(GRID, GRID)
    res["surface"] = {"gain": gains.tolist(), "hz": freqs.tolist(), "rsc_hz": float(best[0]),
                      "segments": int(best[3]), "score": surface.tolist()}  # fmt: skip

    # Nonlinear time-domain runs: the per-segment designs, the best design of each
    # method on the hard problem, and the undamped plant, on every scenario.
    checks = {f"{n} segments": r["design"] for n, r in enumerate(res["per_segment"], start=1)}
    checks["no damping, 3 segments"] = [ds.FLOOR_HZ, 0.0, 30.0, 3.0]
    checks.update({f"{m}, full (seed 0)": res["runs"]["full"][m][0]["design"] for m in METHODS})
    last = res["surrogate"][str(TRAIN_CALLS[-1])][0]["problems"]["full"]["design"]
    checks["surrogate, full (seed 0)"] = last
    res["time_domain"] = {}
    for name, d in checks.items():
        d = np.array(d)
        score, o = ds.worst_case(d, PROCESSES)
        rows, keep = [], None
        worst = int(np.argmax(o[0][:, 2]))
        for k, r in enumerate(ds.time_responses(d)):
            rows.append({"rate": r["rate"], "final_over_initial": r["final_over_initial"],
                         "finite": r["finite"], "sigma_osc": float(o[0][k, 2]),
                         "sigma_max": float(o[0][k, 4])})  # fmt: skip
            if k == worst:
                keep = {"t": r["t"][::2].tolist(), "deviation": r["deviation"][::2].tolist()}
        res["time_domain"][name] = {"design": d.tolist(), "score": float(score[0]),
                                    "worst_scenario": worst, "scenarios": rows,
                                    "trace": keep}  # fmt: skip
        print("time domain", name, f"score {score[0]:.2f}",
              "decays in", sum(r["final_over_initial"] < 1.0 for r in rows), "of", N_SCEN,
              flush=True)  # fmt: skip
    res["wall_s"] = time.perf_counter() - t0
    ds.close_pool()
    return res


def summarise(res: dict) -> None:
    """For each problem: the best score any method found, and for each method the
    solver calls each seed needed to come within each margin of it."""
    res["summary"] = {}
    for problem in PROBLEMS:
        runs = res["runs"][problem]
        sur = {f"surrogate/{n}": [r["problems"][problem] for r in rows]
               for n, rows in res["surrogate"].items()}  # fmt: skip
        best = min([r["best"] for rows in runs.values() for r in rows]
                   + [r["best"] for rows in sur.values() for r in rows])  # fmt: skip
        entry: dict = {"best": best, "methods": {}}
        for m, rows in runs.items():
            finals = [r["best"] for r in rows]
            calls = {f"{w:g}": [search.calls_to_reach(np.array(r["values"]), best + w)
                                for r in rows] for w in WITHIN}  # fmt: skip
            calls = {k: [None if c is None else c * N_SCEN for c in v] for k, v in calls.items()}
            entry["methods"][m] = {"final_mean": float(np.mean(finals)),
                                   "final_sd": float(np.std(finals, ddof=1)),
                                   "final_worst": float(max(finals)), "calls": calls,
                                   "wall_s": float(np.mean([r["wall_s"] for r in rows])),
                                   "budget_calls": EVALS * N_SCEN}  # fmt: skip
        for m, rows in sur.items():
            n = int(m.split("/")[1])
            finals = [r["best"] for r in rows]
            total = n + CONFIRM * N_SCEN
            full_rows = res["surrogate"][str(n)]
            entry["methods"][m] = {
                "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals, ddof=1)),
                "final_worst": float(max(finals)),
                "calls": {f"{w:g}": [total if r["best"] <= best + w else None for r in rows]
                          for w in WITHIN},
                "wall_s": float(np.mean([r["data_s"] + r["fit_s"] for r in full_rows])
                                + np.mean([r["search_s"] for r in rows])),
                "budget_calls": total,
                "promised_mean": float(np.mean([r["predicted_at_best"] for r in rows])),
                "test_mae": float(np.mean([r["test_mae"] for r in full_rows])),
            }  # fmt: skip
        res["summary"][problem] = entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "design.json"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        res = study("cuda" if torch.cuda.is_available() else "cpu")
    summarise(res)
    path.write_text(json.dumps(res))

    # ---------------- figure: best score against solver calls
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE)
    calls = np.arange(1, EVALS + 1) * N_SCEN
    for ax, problem in zip(axes, PROBLEMS, strict=True):
        for name, label in METHODS.items():
            runs = [(calls, search.best_so_far(np.array(r["values"])))
                    for r in res["runs"][problem][name]]  # fmt: skip
            report.curves(ax, runs, COLOUR[name], label)
        for n_train, rows in res["surrogate"].items():
            x = int(n_train) + CONFIRM * N_SCEN
            y = [r["problems"][problem]["best"] for r in rows]
            ax.plot([x] * len(y), y, marker="D", ms=5, lw=0, color=COLOUR[f"surrogate/{n_train}"],
                    label=LABEL[f"surrogate/{n_train}"])  # fmt: skip
        best = res["summary"][problem]["best"]
        ax.axhline(best + WITHIN[0], color=report.MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(best - 0.5, best + 8.0)
        ax.set_xlabel("Solver calls")
        ax.set_title(PROBLEM_NAME[problem], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Best worst-case growth rate so far (1/s)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, f"Design search, five seeds per method (dotted: {WITHIN[0]:g} 1/s from the "
                 "best found)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig105_search.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the surface and the price of a segment
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE)
    sf = res["surface"]
    im = axes[0].pcolormesh(sf["hz"], sf["gain"], np.clip(np.array(sf["score"]), -8, 8),
                            cmap="RdBu_r", vmin=-8, vmax=8, shading="nearest")  # fmt: skip
    axes[0].contour(sf["hz"], sf["gain"], np.array(sf["score"]), levels=[0.0],
                    colors=report.INK, linewidths=1.0)  # fmt: skip
    best = res["per_segment"][-1]["design"]
    axes[0].plot(best[2], best[1], marker="*", ms=14, color=report.INK, mfc="white")
    fig.colorbar(im, ax=axes[0], label="Worst-case growth rate (1/s)")
    axes[0].set_xlabel("Centre frequency of the damping controller (Hz)")
    axes[0].set_ylabel("Share of the loop gain removed")
    axes[0].set_title(f"{sf['segments']} segments, bandwidth {sf['rsc_hz']:.1f} Hz "
                      "(line: zero; star: best found)", loc="left", fontsize=10)  # fmt: skip
    axes[0].tick_params(colors=report.MUTED, labelsize=9)
    seg = np.arange(1, len(res["per_segment"]) + 1)
    axes[1].plot(seg, [r["best"] for r in res["per_segment"]], marker="o", lw=2,
                 color=report.SERIES[0], label="best controller found")  # fmt: skip
    for hz, ls in ((ds.FLOOR_HZ, "--"), (30.0, ":")):
        axes[1].plot(seg, [b["score"] for b in res["baseline"] if b["design"][0] == hz],
                     marker="s", lw=1.5, ls=ls, color=report.MUTED,
                     label=f"no damping controller, {hz:g} Hz")  # fmt: skip
    axes[1].axhline(0, color=report.INK, lw=0.8)
    axes[1].set_xticks(seg, [f"{n}\n({100 * n * res['segment']:.0f} %)" for n in seg])
    axes[1].set_xlabel("Segments in service (compensation)")
    axes[1].set_ylabel("Worst-case growth rate (1/s)")
    axes[1].set_title("What each segment costs in damping", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[1])
    report.title(fig, "The design surface and the price of compensation")
    fig.tight_layout()
    fig.savefig(OUT / "fig106_surface.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: time-domain confirmation
    shown = [k for k in res["time_domain"] if "segments" in k]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 6.6), facecolor=report.SURFACE, sharex=True)
    for ax, name in zip(axes.ravel(), shown, strict=False):
        td = res["time_domain"][name]
        tr = td["trace"]
        ax.plot(tr["t"], tr["deviation"], color=report.SERIES[0], lw=0.8)
        d = td["design"]
        ax.set_title(f"{name}: {d[0]:.0f} Hz, gain {d[1]:.2f} at {d[2]:.0f} Hz; "
                     f"{td['score']:+.2f} 1/s", loc="left", fontsize=9)  # fmt: skip
        report.style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Capacitor voltage deviation, D (pu)")
    report.title(fig, "Nonlinear model after a 0.05 pu capacitor-voltage offset, worst scenario "
                 "of each design (last number: growth rate from the eigenvalues)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig107_time_domain.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the surrogate
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), facecolor=report.SURFACE)
    for n_train, rows in res["surrogate"].items():
        c = COLOUR[f"surrogate/{n_train}"]
        runs = [(np.array([h["step"] for h in r["history"]]),
                 np.array([h["held_out"] for h in r["history"]])) for r in rows]  # fmt: skip
        report.curves(axes[0], runs, c, f"{int(n_train):,} training calls")
        for problem, mark in (("free", "o"), ("full", "s")):
            axes[2].plot([r["problems"][problem]["predicted_at_best"] for r in rows],
                         [r["problems"][problem]["best"] for r in rows], marker=mark, ms=6, lw=0,
                         color=c, label=f"{int(n_train):,} calls, {problem}")  # fmt: skip
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Held-out mean absolute error (1/s)")
    axes[0].set_title("Training, five seeds", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[0])
    par = res["surrogate"][str(TRAIN_CALLS[-1])][0]["parity"]
    report.parity(axes[1], np.array(par["true"]), np.array(par["pred"]), (-30.0, 40.0))
    axes[1].set_xlabel("Growth rate from the solver (1/s)")
    axes[1].set_ylabel("Perceptron (1/s)")
    axes[1].set_title(f"One case, {TRAIN_CALLS[-1]:,} training calls", loc="left", fontsize=10)
    lim = [-10, 4]
    axes[2].plot(lim, lim, color=report.INK, lw=0.8, ls="--")
    axes[2].set_xlabel("Worst-case growth rate the perceptron promised (1/s)")
    axes[2].set_ylabel("What the solver found (1/s)")
    axes[2].set_title("The chosen design", loc="left", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[2])
    report.title(fig, "The surrogate used for the search")
    fig.tight_layout()
    fig.savefig(OUT / "fig108_surrogate.png", dpi=160)
    plt.close(fig)

    # ---------------- printed summary
    print("solver s per call", res["solver_s_per_call"], "wall", round(res.get("wall_s", 0)))
    for b in res["baseline"]:
        print("baseline", b["design"], round(b["score"], 2), b["worst_scenario"],
              round(b["f_at_worst"], 1))  # fmt: skip
    for problem, entry in res["summary"].items():
        print(problem, "best found", round(entry["best"], 3))
        for m, e in entry["methods"].items():
            print(f"  {m:16s} final {e['final_mean']:.2f} ± {e['final_sd']:.2f} worst "
                  f"{e['final_worst']:.2f} wall {e['wall_s']:.0f} s budget {e['budget_calls']}",
                  e["calls"], {k: round(v, 2) for k, v in e.items()
                               if k in ("promised_mean", "test_mae")})  # fmt: skip
        for m in METHODS:
            print("   ", m, [np.round(r["design"], 2).tolist() for r in res["runs"][problem][m]])
    for n, r in enumerate(res["per_segment"], start=1):
        print("segments", n, round(r["best"], 2), np.round(r["design"], 2).tolist(),
              "f", np.round(r["f_osc"], 1).tolist())  # fmt: skip
    for name, td in res["time_domain"].items():
        rows = td["scenarios"]
        w = rows[td["worst_scenario"]]
        print("time", name, round(td["score"], 2), "decays",
              sum(r["final_over_initial"] < 1.0 for r in rows), "rate at worst",
              round(w["rate"], 2), "sigma_max there", round(w["sigma_max"], 2),
              "largest |rate - sigma_osc|",
              round(float(np.nanmax([abs(r["rate"] - r["sigma_osc"]) for r in rows])), 2))


if __name__ == "__main__":
    main()
