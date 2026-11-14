"""Phase 9: out-of-distribution study and the fallback rule, task T1.

    uv run scripts/phase9_fallback.py            # 30 training runs, about 10 minutes
    uv run scripts/phase9_fallback.py --reuse    # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_t1.py. For arms N, F and R at 1,000 and
100,000 cases, five perceptrons are trained from five initial weights on one training
subset. Their mean is the model. Seven detectors are computed for every case:

    margin only           nothing; the rule uses the size of the prediction alone
    ensemble spread       standard deviation of the five members
    learned width         a tree model of the absolute error, fitted on validation cases
    input distance        mean distance to the 5 nearest training cases, scaled inputs
    latent distance       the same in the last hidden layer of the first member
    hand disagreement     |prediction - hand calculation|
    range excess          how far the inputs are outside the training range

The fallback rule: confidence = |prediction| / (detector + floor), where the floor is the
median of the detector on validation. Below a threshold the solver is called; above it
the sign of the prediction is the answer. The threshold is the largest confidence of a
validation case with the wrong sign, times FACTOR. FACTOR was fixed before the test sets
were scored.

The validation fold is cut in three: cases for fitting the learned width, cases for its
conformal quantile, and cases for the threshold.

Writes docs/results/phase9/fallback.json and figures fig100 to fig104. Every number in
docs/steps/40-fallback.md comes from this script.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from sso_sciml import fallback as fb
from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.models import compare as cmp
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase9"
BUDGETS = (1000, 100000)
MEMBERS = 5
FACTOR = 1.5  # safety factor on the validation threshold
N_FIT, N_CAL = 3000, 3000  # validation cases for the learned width and its quantile
DETECTORS = ("margin only", "ensemble spread", "learned width", "input distance",
             "latent distance", "hand disagreement", "range excess")  # fmt: skip
SHARES = np.linspace(0.0, 0.5, 101)
DIRECTION = {s: next(iter(r)) for s, r in list(data.SPLITS.items())[1:]}
TARGET_SHARE = 0.10
COLOUR = dict(zip(DETECTORS, (report.MUTED, *report.SERIES), strict=False))


def latent(model: mt.MLP, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
        return model.net[:-1](xt).cpu().numpy()


def study(device: str) -> dict:
    train, val, tests = cmp.sets(DATA)
    n_val = val["sigma"].size
    idx = np.arange(n_val)
    part = {"fit": cmp.take(val, idx < N_FIT),
            "cal": cmp.take(val, (idx >= N_FIT) & (idx < N_FIT + N_CAL)),
            "tune": cmp.take(val, idx >= N_FIT + N_CAL)}  # fmt: skip
    sets = {**part, **tests}
    res: dict = {
        "factor": FACTOR, "members": MEMBERS,
        "validation": {k: int(v["sigma"].size) for k, v in part.items()},
        "tests": {k: {"cases": int(v["sigma"].size), "unstable": int((v["sigma"] > 0).sum())}
                  for k, v in tests.items()},
        "models": {}, "distance": {}, "history": {}, "parity": {},
    }  # fmt: skip

    # The hand calculation alone, under the same rule.
    hand = {k: mt.baseline(v["mech"]) for k, v in sets.items()}
    level = FACTOR * fb.threshold(np.abs(hand["tune"]), hand["tune"], part["tune"]["sigma"])
    res["hand_calculation"] = {
        "level": level,
        **{k: {"mae": float(np.abs(hand[k] - v["sigma"]).mean()),
               **fb.outcome(fb.decide(np.abs(hand[k]), hand[k], level), v["sigma"])}
           for k, v in tests.items()},
    }  # fmt: skip

    for budget in BUDGETS:
        sub = mt.budget_subset(train, budget, 0)
        h = budget // 5
        u_train = mt.scale_theta(sub["theta"][h:])
        for arm in mt.ARMS:
            t0 = time.perf_counter()
            x = mt.inputs(arm, sub["theta"], sub["mech"])
            y = mt.targets(arm, sub["sigma"], sub["mech"])
            fits = [mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed, device=device)
                    for seed in range(MEMBERS)]  # fmt: skip
            models = [m for m, _ in fits]
            fit_s = time.perf_counter() - t0
            z_train = latent(models[0], x[h:], device)
            z_scale = z_train.std(axis=0).mean() + 1e-9

            pred, det = {}, {}
            for name, d in sets.items():
                xx = mt.inputs(arm, d["theta"], d["mech"])
                p = np.array([mt.to_sigma(arm, mt.predict(m, xx, device), d["mech"])
                              for m in models])  # fmt: skip
                u = mt.scale_theta(d["theta"])
                pred[name] = p.mean(axis=0)
                det[name] = {
                    "margin only": np.zeros(u.shape[0]),
                    "ensemble spread": p.std(axis=0),
                    "input distance": fb.knn_distance(u_train, u, device=device),
                    "latent distance": fb.knn_distance(
                        z_train / z_scale, latent(models[0], xx, device) / z_scale, device=device),
                    "hand disagreement": np.abs(pred[name] - mt.baseline(d["mech"])),
                    "range excess": fb.range_excess(u),
                }  # fmt: skip

            # Learned width: trees on the inputs of arm F and the spread, to the error.
            def width_inputs(name, det=det):
                d = sets[name]
                return np.column_stack([mt.inputs("F", d["theta"], d["mech"]),
                                        np.log(det[name]["ensemble spread"] + 1e-6)])  # fmt: skip

            err = {k: np.abs(pred[k] - v["sigma"]) for k, v in sets.items()}
            trees = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
            trees.fit(width_inputs("fit"), np.log(err["fit"] + 1e-4))
            for name in sets:
                det[name]["learned width"] = np.exp(trees.predict(width_inputs(name)))
            q90 = mt.conformal_quantile(err["cal"] / det["cal"]["learned width"])

            entry: dict = {"fit_s": fit_s, "conformal_quantile": q90, "detectors": {},
                           "mae": {k: float(err[k].mean()) for k in tests},
                           "wrong_sign": {k: int(((pred[k] > 0) != (v["sigma"] > 0)).sum())
                                          for k, v in tests.items()},
                           "coverage_90": {
                               k: float((err[k] <= q90 * det[k]["learned width"]).mean())
                               for k in tests}}  # fmt: skip
            for name in DETECTORS:
                floor = 1.0 if name == "margin only" else float(np.median(det["tune"][name]))
                if name == "range excess":
                    floor = 1.0
                conf = {k: fb.confidence(pred[k], det[k][name], floor) for k in sets}
                sig = part["tune"]["sigma"]
                level = FACTOR * fb.threshold(conf["tune"], pred["tune"], sig)
                level_u = FACTOR * fb.threshold(conf["tune"], pred["tune"], sig,
                                                unstable_only=True)  # fmt: skip
                alarm = float(np.quantile(det["tune"][name], 0.95))
                e: dict = {"floor": floor, "level": level, "level_unstable_only": level_u,
                           "alarm_level": alarm, "sets": {}}  # fmt: skip
                e["validation"] = fb.outcome(fb.decide(conf["tune"], pred["tune"], level), sig)
                for k, d in tests.items():
                    s = d["sigma"]
                    wrong = (pred[k] > 0) != (s > 0)
                    e["sets"][k] = {
                        "policy": fb.outcome(fb.decide(conf[k], pred[k], level), s),
                        "policy_raw": fb.outcome(fb.decide(conf[k], pred[k], level / FACTOR), s),
                        "policy_unstable_only": fb.outcome(fb.decide(conf[k], pred[k], level_u), s),
                        "share_for_no_miss": fb.share_for_no_miss(conf[k], pred[k], s),
                        "rank_correlation_with_error": fb.spearman(det[k][name], err[k]),
                        "auroc_wrong_sign": fb.auroc(-conf[k], wrong),
                        "alarm_rate": float((det[k][name] > alarm).mean()),
                    }  # fmt: skip
                e["tradeoff"] = fb.tradeoff(conf["in_distribution"], pred["in_distribution"],
                                            tests["in_distribution"]["sigma"], SHARES)  # fmt: skip
                at = int(np.argmin(np.abs(SHARES - TARGET_SHARE)))
                e["missed_at_target_share"] = e["tradeoff"]["missed_unstable"][at]
                entry["detectors"][name] = e
            res["models"][f"{arm}/{budget}"] = entry

            # Error against distance outside the range, one direction per test set.
            for split, par in DIRECTION.items():
                d = tests[split]
                j = data.NAMES.index(par)
                out = np.abs(mt.scale_theta(d["theta"])[:, j]) - 1.0
                edges = np.quantile(out, [0, 0.25, 0.5, 0.75, 1.0])
                which = np.clip(np.searchsorted(edges, out, side="right") - 1, 0, 3)
                rows = res["distance"].setdefault(split, {"parameter": par, "models": {}})
                rows["mid"] = [float(out[which == b].mean()) for b in range(4)]
                rows["models"][f"{arm}/{budget}"] = {
                    "mae": [float(err[split][which == b].mean()) for b in range(4)],
                    "spread_alarm": [float((det[split]["ensemble spread"][which == b]
                                            > entry["detectors"]["ensemble spread"]["alarm_level"])
                                           .mean()) for b in range(4)],
                }  # fmt: skip
                rows["hand"] = [float(np.abs(hand[split] - d["sigma"])[which == b].mean())
                                for b in range(4)]  # fmt: skip
            res["history"][f"{arm}/{budget}"] = [
                {"step": [r[0] for r in info["history"]],
                 "held_out": [r[2] * mt.SIGMA_SCALE for r in info["history"]]}
                for _, info in fits
            ]  # fmt: skip
            if budget == BUDGETS[-1]:
                keep = np.random.default_rng(0).permutation(pred["in_distribution"].size)[:4000]
                res["parity"][arm] = {
                    "true": tests["in_distribution"]["sigma"][keep].tolist(),
                    "pred": pred["in_distribution"][keep].tolist(),
                }
            p = entry["detectors"]
            print(arm, budget, f"mae {entry['mae']['in_distribution']:.3f}", f"{fit_s:.0f} s",
                  {k: (round(v["sets"]["in_distribution"]["policy"]["solver_share"], 3),
                       v["sets"]["in_distribution"]["policy"]["missed_unstable"],
                       v["sets"]["in_distribution"]["policy"]["false_unstable"])
                   for k, v in p.items()}, flush=True)  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fallback.json"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        t0 = time.perf_counter()
        res = study("cuda" if torch.cuda.is_available() else "cpu")
        res["wall_s"] = time.perf_counter() - t0
        path.write_text(json.dumps(res))
    models, tests = res["models"], list(res["tests"])
    big = BUDGETS[-1]
    names = [t.replace("_", " ") for t in tests]

    # ---------------- figure: error against distance outside the range
    fig, axes = plt.subplots(1, len(DIRECTION), figsize=(18.0, 4.2), facecolor=report.SURFACE,
                             sharey=True)  # fmt: skip
    for ax, (split, rows) in zip(axes, res["distance"].items(), strict=True):
        for arm in mt.ARMS:
            ax.plot(rows["mid"], rows["models"][f"{arm}/{big}"]["mae"], marker="o", ms=4, lw=2,
                    color=report.ARM_COLOUR[arm], label=report.ARM_NAME[arm])  # fmt: skip
            ax.axhline(models[f"{arm}/{big}"]["mae"]["in_distribution"],
                       color=report.ARM_COLOUR[arm], lw=0.8, ls=":")  # fmt: skip
        ax.plot(rows["mid"], rows["hand"], marker="s", ms=4, lw=1.5, color=report.MUTED, ls="--",
                label="hand calculation")  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel(f"Distance of {rows['parameter']} outside its range (ranges)")
        ax.set_title(split.replace("_", " "), loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Mean absolute error (1/s)")
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="center right")
    report.title(fig, f"Error against distance outside the training range, {big:,} cases "
                 "(dotted: error in distribution; a distance of 1 is half the range)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig100_error_against_distance.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: trade-off in distribution
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, arm in zip(axes, mt.ARMS, strict=True):
        for name in DETECTORS[:-1]:
            t = models[f"{arm}/{big}"]["detectors"][name]["tradeoff"]
            ax.plot(100 * np.array(t["solver_share"]), np.array(t["missed_unstable"]) + 0.5,
                    color=COLOUR[name], lw=1.8, label=name)  # fmt: skip
        ax.axvline(100 * TARGET_SHARE, color=report.INK, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlim(0, 20)
        ax.set_xlabel("Cases sent to the solver, least confident first (%)")
        ax.set_title(f"Arm {report.ARM_NAME[arm]}", loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Unstable cases called stable, plus 0.5")
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    n_u = res["tests"]["in_distribution"]["unstable"]
    report.title(fig, f"In distribution, {big:,} cases: unstable cases missed against solver "
                 f"share ({n_u:,} unstable of {res['tests']['in_distribution']['cases']:,}; "
                 "0.5 means none)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig101_tradeoff.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the rule on every test set, arm R
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.4), facecolor=report.SURFACE, sharex=True)
    xs = np.arange(len(tests))
    det_r = models[f"R/{big}"]["detectors"]
    for k, name in enumerate(DETECTORS):
        rows = [det_r[name]["sets"][t]["policy"] for t in tests]
        off = (k - 3) * 0.12
        axes[0].bar(xs + off, [100 * r["solver_share"] for r in rows], 0.11, color=COLOUR[name],
                    label=name)  # fmt: skip
        axes[1].bar(xs + off, [100 * r["missed_unstable_rate"] + 0.01 for r in rows], 0.11,
                    color=COLOUR[name])  # fmt: skip
    axes[0].set_ylabel("Cases sent to the solver (%)")
    axes[1].set_ylabel("Unstable cases called stable (%)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(xs, names)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, ncol=4)
    for ax in axes:
        report.style(ax)
        ax.grid(axis="x", visible=False)
    report.title(fig, f"The rule set on validation, applied to every test set (arm R, {big:,} "
                 "cases)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig102_rule_on_every_set.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: how well each detector ranks and warns, arm R
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), facecolor=report.SURFACE)
    for ax, key, label in zip(axes, ("rank_correlation_with_error", "alarm_rate"),
                              ("Rank correlation of the detector with the error",
                               "Share of cases above the 95th percentile of validation"),
                              strict=True):  # fmt: skip
        grid = np.array([[det_r[n]["sets"][t][key] for t in tests] for n in DETECTORS[1:]])
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else report.INK)  # fmt: skip
        ax.set_xticks(xs, names, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(DETECTORS) - 1), DETECTORS[1:], fontsize=8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.tick_params(colors=report.MUTED)
        fig.colorbar(im, ax=ax, fraction=0.03)
    report.title(fig, f"Detectors on every test set (arm R, {big:,} cases)")
    fig.tight_layout()
    fig.savefig(OUT / "fig103_detectors.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: training and performance of the ensembles
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), facecolor=report.SURFACE)
    for arm in mt.ARMS:
        runs = [(np.array(h["step"]), np.array(h["held_out"]))
                for h in res["history"][f"{arm}/{big}"]]  # fmt: skip
        report.curves(axes[0], runs, report.ARM_COLOUR[arm], report.ARM_NAME[arm])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Held-out mean absolute error (1/s)")
    axes[0].set_title("Five members per arm", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[0])
    par = res["parity"]["R"]
    report.parity(axes[1], np.array(par["true"]), np.array(par["pred"]), (-25.0, 15.0))
    axes[1].set_xlabel("Growth rate from the solver (1/s)")
    axes[1].set_ylabel("Mean of the five members, arm R (1/s)")
    axes[1].set_title("Predicted against true, test fold", loc="left", fontsize=10)
    w = 0.25
    for k, arm in enumerate(mt.ARMS):
        axes[2].bar(xs + (k - 1) * w, [models[f"{arm}/{big}"]["mae"][t] for t in tests], w,
                    color=report.ARM_COLOUR[arm], label=report.ARM_NAME[arm])  # fmt: skip
    axes[2].plot(xs, [res["hand_calculation"][t]["mae"] for t in tests], color=report.INK,
                 marker="_", ms=30, lw=0, label="hand calculation")  # fmt: skip
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.04, 60.0)
    axes[2].set_xticks(xs, names, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean absolute error (1/s)")
    axes[2].set_title("Ensemble on every test set", loc="left", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=0.4)
    report.style(axes[2])
    report.title(fig, f"Ensembles of five perceptrons at {big:,} cases")
    fig.tight_layout()
    fig.savefig(OUT / "fig104_ensembles.png", dpi=160)
    plt.close(fig)

    # ---------------- printed summary
    hc = res["hand_calculation"]
    print("hand", {t: (round(hc[t]["solver_share"], 3), hc[t]["missed_unstable"]) for t in tests})
    for key, m in models.items():
        print(key, "mae", {t: round(m["mae"][t], 3) for t in tests})
        print(key, "wrong sign", m["wrong_sign"], "coverage",
              {t: round(v, 3) for t, v in m["coverage_90"].items()})  # fmt: skip
        for name, e in m["detectors"].items():
            row = []
            for t in tests:
                s = e["sets"][t]
                p = s["policy"]
                row.append(f"{100 * p['solver_share']:.1f}%/{p['missed_unstable']}/"
                           f"{p['false_unstable']} r{s['rank_correlation_with_error']:+.2f} "
                           f"a{s['alarm_rate']:.2f}")  # fmt: skip
            print(f"  {name:18s} level {e['level']:.3g} miss@10% {e['missed_at_target_share']}",
                  " | ".join(row))  # fmt: skip
    for split, rows in res["distance"].items():
        print(split, rows["parameter"], np.round(rows["mid"], 2).tolist(), "hand",
              np.round(rows["hand"], 2).tolist(),
              {k: np.round(v["mae"], 2).tolist() for k, v in rows["models"].items()
               if k.endswith(str(big))})  # fmt: skip


if __name__ == "__main__":
    main()
