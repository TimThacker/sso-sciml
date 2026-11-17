"""Step 48: physics-residual detectors on T1, and the out-of-distribution study on T2.

    uv run scripts/phase13_ood_residual.py            # about 40 minutes
    uv run scripts/phase13_ood_residual.py --reuse    # figures from the saved numbers

Needs the datasets of steps 08 and 35 and the registered model of step 42
(`uv run scripts/phase11_build.py`).

T1. The registered screen predicts a growth rate and a frequency, so a predicted
eigenvalue lam = sigma + j 2 pi f. Two residuals of physics at that eigenvalue:

    impedance residual   |Z / Z'| at the predicted frequency, for the loop impedance of
                         the hand calculation (network, machine, current loop): the
                         Newton distance, in 1/s, from the prediction to a zero of that
                         impedance. Cheap: one polynomial evaluation.
    state-matrix residual   the smallest singular value of A(theta) - lam I, for the
                         state matrix of the full model, 1/s. Zero when lam is an
                         eigenvalue. Needs A, which is most of the cost of a solve.

Each is used as the detector of the fallback rule of step 40 (floor and level set on the
second half of the validation fold, factor 1.5) and compared with the ensemble spread.

T2. Five perceptrons with the reduced model's response as inputs (arm F, 16,000 steps,
25,000 trajectories) and five without (arm N). Detectors of a bad prediction (error of
the response above 10 %): the ensemble spread, the distance to the nearest training
cases in the parameters, the disagreement with the reduced model, and how far the
parameters are outside their range. A case is sent to the solver when its detector is
above the level that flags every validation case with an error above 25 %, divided by
1.5. Error against distance outside the range is given for each outside set.

Writes docs/results/phase13/ood_residual.json and fig119, fig120. Every number in
docs/steps/48-ood-and-residual.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import fallback as fb
from sso_sciml import registry, report
from sso_sciml.data import t1, t2
from sso_sciml.models import compare as cmp
from sso_sciml.models import t1 as mt1
from sso_sciml.models import t2 as mt2
from sso_sciml.physics import circuit
from sso_sciml.physics.dfig import gains_from_bandwidth
from sso_sciml.physics.induction import IGData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "phase13"
FACTOR = 1.5
T2_BUDGET, T2_STEPS, MEMBERS = 25000, 16000, 5
BAD, VERY_BAD = 0.10, 0.25


# ---------------------------------------------------------------- T1
def impedance_residual(theta: np.ndarray, lam: complex) -> float:
    """|Z / Z'| * wb at s = j + lam / wb for the hand calculation's loop impedance."""
    th = dict(zip(t1.NAMES, map(float, theta), strict=True))
    data = replace(IGData(), r_r=th["r_r"])
    net = t1._network(th, data)
    kp, ki = gains_from_bandwidth(data, th["rsc_hz"])
    z = circuit.parallel(circuit.branch(net["r_e"], net["x_e"], net["x_c"]),
                         circuit.branch(net["r_p"], net["x_p"])) \
        + circuit.branch(net["r_t"], net["x_t"]) \
        + circuit.dfig_impedance(data.r_s, data.x_ls, data.x_m, data.r_r, data.x_lr, th["w_r"],
                                 kp, ki)  # fmt: skip
    s = 1j + lam / data.wb
    h = 1e-6
    dz = (z(s + h) - z(s - h)) / (2 * h)
    return float(abs(z(s) / dz) * data.wb)


def matrix_residual(a: np.ndarray, lam: complex) -> float:
    if not np.isfinite(a).all():
        return float("nan")
    return float(np.linalg.svd(a - lam * np.eye(a.shape[0]), compute_uv=False)[-1])


def t1_part() -> dict:
    bundle = registry.load(version="1.0.0")
    train, val, tests = cmp.sets(ROOT / "data" / t1.VERSION)
    half = val["sigma"].size // 2
    tune = {k: v[half:] for k, v in val.items()}
    sets = {"tune": tune, **tests}
    out: dict = {"timing": {}, "sets": {}}
    det_all, pred_all = {}, {}
    for name, d in sets.items():
        p = bundle.predict(d["theta"], d["mech"])
        lam = p["sigma"] + 2j * np.pi * p["f"]
        t0 = time.perf_counter()
        # The impedance model's root of the mode sits at negative frequency (lambda =
        # wb (s - j)); the state matrix is real, so either sign gives its residual.
        imp = np.array([impedance_residual(th, np.conj(lm))
                        for th, lm in zip(d["theta"], lam, strict=True)])  # fmt: skip
        t_imp = (time.perf_counter() - t0) / len(lam)
        t0 = time.perf_counter()
        mats = t1.state_matrices(d["theta"], processes=8)
        t_mat = (time.perf_counter() - t0) / len(lam)
        res = np.array([matrix_residual(a, lm) for a, lm in zip(mats, lam, strict=True)])
        det_all[name] = {"ensemble spread": p["spread"], "impedance residual": imp,
                         "state-matrix residual": np.nan_to_num(res, nan=1e3)}  # fmt: skip
        pred_all[name] = p["sigma"]
        if name == "in_distribution":
            out["timing"] = {"impedance_s_per_case": t_imp,
                             "state_matrix_s_per_case_8_processes": t_mat}  # fmt: skip
        print("t1", name, len(lam), f"imp {t_imp * 1e3:.2f} ms mat {t_mat * 1e3:.2f} ms",
              flush=True)  # fmt: skip
    for det in det_all["tune"]:
        floor = float(np.median(det_all["tune"][det]))
        conf = {k: fb.confidence(pred_all[k], det_all[k][det], floor) for k in sets}
        level = FACTOR * fb.threshold(conf["tune"], pred_all["tune"], tune["sigma"])
        out["sets"][det] = {"floor": floor, "level": level}
        for k, d in tests.items():
            err = np.abs(pred_all[k] - d["sigma"])
            wrong = (pred_all[k] > 0) != (d["sigma"] > 0)
            out["sets"][det][k] = {
                "policy": fb.outcome(fb.decide(conf[k], pred_all[k], level), d["sigma"]),
                "rank_correlation_with_error": fb.spearman(det_all[k][det], err),
                "auroc_wrong_sign": fb.auroc(-conf[k], wrong),
                "share_for_no_miss": fb.share_for_no_miss(conf[k], pred_all[k], d["sigma"]),
            }  # fmt: skip
        s = out["sets"][det]
        print("t1", det, {k: (round(100 * s[k]["policy"]["solver_share"], 2),
                              s[k]["policy"]["missed_unstable"],
                              round(s[k]["rank_correlation_with_error"], 2))
                          for k in tests}, flush=True)  # fmt: skip
    return out


# ---------------------------------------------------------------- T2
KEYS = ("theta", "observed", "reduced", "valid", "states", "a_red", "sigma")


def load2(split: str) -> dict:
    z = np.load(ROOT / "data" / t2.VERSION / f"{split}.npz")
    ok = z["solved"]
    out = {k: z[k][ok] for k in KEYS}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, m) -> dict:
    return {k: v[m] for k, v in d.items() if k != "fold"}


def t2_part(device: str) -> dict:
    full = load2("in_distribution")
    train, val = take(full, full["fold"] == 0), take(full, full["fold"] == 1)
    tests = {"in_distribution": take(full, full["fold"] == 2)}
    tests.update({s: take(load2(s), slice(None)) for s in list(t2.SPLITS)[1:]})
    sets = {"validation": val, **tests}
    sub = mt1.budget_subset(train, T2_BUDGET, 0)
    h = T2_BUDGET // 5
    held, lab = {k: v[:h] for k, v in sub.items()}, {k: v[h:] for k, v in sub.items()}
    u_train = t2.scale_theta(lab["theta"])
    out: dict = {"arms": {}, "distance": {}}
    for arm in ("F", "N"):
        preds = {k: [] for k in sets}
        t0 = time.perf_counter()
        for seed in range(MEMBERS):
            predict, _ = mt2.train(arm, lab, held, seed, mt2.Config(steps=T2_STEPS), device=device)
            for k, d in sets.items():
                preds[k].append(predict(d))
            print("t2", arm, seed, flush=True)
        fit_s = time.perf_counter() - t0
        e: dict = {"fit_s": fit_s, "sets": {}}
        det, err = {}, {}
        for k, d in sets.items():
            p = np.array(preds[k])
            mean = p.mean(axis=0)
            err[k] = mt2.case_error(mean, d)
            m = mt2.mask(d)[:, :, None]
            size = np.sqrt(((mean**2) * m).sum(axis=(1, 2)) / m.sum(axis=(1, 2)))
            spread = np.sqrt(((p.std(axis=0) ** 2) * m).sum(axis=(1, 2)) / m.sum(axis=(1, 2)))
            red = mt2.reduced_alone(d)
            dis = np.sqrt((((mean[:, :, :4] - red[:, :, :4]) ** 2) * m).sum(axis=(1, 2))
                          / ((mean[:, :, :4] ** 2) * m).sum(axis=(1, 2)))  # fmt: skip
            u = t2.scale_theta(d["theta"])
            det[k] = {"ensemble spread": spread / size,
                      "input distance": fb.knn_distance(u_train, u, device=device),
                      "reduced disagreement": dis, "range excess": fb.range_excess(u)}  # fmt: skip
        for name in det["validation"]:
            v_det, v_err = det["validation"][name], err["validation"]
            level = float(v_det[v_err > VERY_BAD].min() / FACTOR) if (v_err > VERY_BAD).any() \
                else float(np.quantile(v_det, 0.99))  # fmt: skip
            e["sets"][name] = {"level": level}
            for k in tests:
                flag = det[k][name] > level
                e["sets"][name][k] = {
                    "median_error": float(np.median(err[k])),
                    "bad_share": float((err[k] > BAD).mean()),
                    "flagged_share": float(flag.mean()),
                    "very_bad_missed": int(((err[k] > VERY_BAD) & ~flag).sum()),
                    "very_bad": int((err[k] > VERY_BAD).sum()),
                    "bad_missed": int(((err[k] > BAD) & ~flag).sum()),
                    "bad": int((err[k] > BAD).sum()),
                    "rank_correlation_with_error": fb.spearman(det[k][name], err[k]),
                    "auroc_bad": fb.auroc(det[k][name], err[k] > BAD),
                }  # fmt: skip
        out["arms"][arm] = e
        s = e["sets"]
        for name in s:
            print("t2", arm, name, {k: (round(100 * s[name][k]["flagged_share"], 1),
                                        s[name][k]["very_bad_missed"], s[name][k]["very_bad"],
                                        round(s[name][k]["auroc_bad"], 2))
                                    for k in tests}, flush=True)  # fmt: skip
        for split in list(t2.SPLITS)[1:]:
            d = tests[split]
            par = next(iter(t2.SPLITS[split]))
            j = t2.NAMES.index(par)
            dist = np.abs(t2.scale_theta(d["theta"])[:, j]) - 1.0
            edges = np.quantile(dist, [0, 0.25, 0.5, 0.75, 1.0])
            which = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, 3)
            row = out["distance"].setdefault(split, {"parameter": par, "arms": {}})
            row["mid"] = [float(dist[which == b].mean()) for b in range(4)]
            row["arms"][arm] = [float(np.median(err[split][which == b])) for b in range(4)]
            row["reduced"] = [float(np.median(mt2.case_error(mt2.reduced_alone(d), d)[which == b]))
                              for b in range(4)]  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ood_residual.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.reuse:
        res = {"t1": t1_part()}
        path.write_text(json.dumps(res))
        res["t2"] = t2_part(device)
        path.write_text(json.dumps(res))
    res = json.loads(path.read_text())

    # ---------------- figure: T1 residual detectors
    r1 = res["t1"]["sets"]
    tests = [k for k in next(iter(r1.values())) if k not in ("floor", "level")]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE)
    xs = np.arange(len(tests))
    for k, det in enumerate(r1):
        pol = [r1[det][t]["policy"] for t in tests]
        axes[0].bar(xs + (k - 1) * 0.25, [100 * p["solver_share"] for p in pol], 0.23,
                    color=report.SERIES[k], label=det)  # fmt: skip
        axes[1].bar(xs + (k - 1) * 0.25, [p["missed_unstable"] + 0.1 for p in pol], 0.23,
                    color=report.SERIES[k])  # fmt: skip
    axes[0].set_ylabel("Cases sent to the solver (%)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Unstable cases called stable (+0.1)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(xs, [t.replace("_", " ") for t in tests], rotation=20, ha="right", fontsize=8)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "T1: the fallback rule with a physics residual as the detector (registered "
                 "model, no range check)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig119_t1_residual.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: T2 error against distance, and detectors
    r2 = res["t2"]
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.4), facecolor=report.SURFACE)
    for ax, (split, row) in zip(axes[:3], r2["distance"].items(), strict=True):
        for arm, c in (("F", report.ARM_COLOUR["F"]), ("N", report.ARM_COLOUR["N"])):
            ax.plot(row["mid"], row["arms"][arm], marker="o", lw=2, color=c,
                    label=report.ARM_NAME[arm])  # fmt: skip
            ax.axhline(r2["arms"][arm]["sets"]["ensemble spread"]["in_distribution"]
                       ["median_error"], color=c, lw=0.8, ls=":")  # fmt: skip
        ax.plot(row["mid"], row["reduced"], marker="s", lw=1.5, ls="--", color=report.MUTED,
                label="reduced model alone")  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel(f"Distance of {row['parameter']} outside its range (half-ranges)")
        ax.set_title(split.replace("_", " "), loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Median relative error of the response")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    sF = r2["arms"]["F"]["sets"]
    t2tests = [k for k in next(iter(sF.values())) if k != "level"]
    grid = np.array([[sF[d][t]["auroc_bad"] for t in t2tests] for d in sF])
    im = axes[3].imshow(grid, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        axes[3].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v > 0.8 else report.INK)  # fmt: skip
    axes[3].set_xticks(range(len(t2tests)), [t.replace("_", " ") for t in t2tests], rotation=20,
                       ha="right", fontsize=8)  # fmt: skip
    axes[3].set_yticks(range(len(sF)), list(sF), fontsize=8)
    axes[3].set_title("Arm F: how well each detector finds errors above 10 % (AUROC)",
                      loc="left", fontsize=9)  # fmt: skip
    fig.colorbar(im, ax=axes[3], fraction=0.04)
    report.title(fig, "T2 outside its range: five perceptrons per arm, 25,000 trajectories "
                 "(dotted: in distribution)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig120_t2_ood.png", dpi=160)
    plt.close(fig)

    print("timing", res["t1"]["timing"])
    for det, s in r1.items():
        print("T1", det, "level", round(s["level"], 3), {t: (
              round(100 * s[t]["policy"]["solver_share"], 2),
              s[t]["policy"]["missed_unstable"], s[t]["policy"]["false_unstable"],
              round(s[t]["rank_correlation_with_error"], 2), round(s[t]["auroc_wrong_sign"], 4),
              round(s[t]["share_for_no_miss"], 4)) for t in tests})  # fmt: skip
    for arm, e in r2["arms"].items():
        print("T2", arm, f"fit {e['fit_s']:.0f} s")
        for det, s in e["sets"].items():
            print("  ", det, "level", round(s["level"], 4), {t: (round(s[t]["median_error"], 4),
                  round(100 * s[t]["bad_share"], 1), round(100 * s[t]["flagged_share"], 1),
                  s[t]["very_bad_missed"], s[t]["very_bad"], s[t]["bad_missed"], s[t]["bad"],
                  round(s[t]["auroc_bad"], 3)) for t in t2tests})  # fmt: skip
    for split, row in r2["distance"].items():
        print(split, row["parameter"], np.round(row["mid"], 3).tolist(),
              {a: np.round(v, 3).tolist() for a, v in row["arms"].items()},
              "reduced", np.round(row["reduced"], 3).tolist())  # fmt: skip


if __name__ == "__main__":
    main()
