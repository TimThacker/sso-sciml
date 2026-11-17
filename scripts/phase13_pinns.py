"""Step 47: a parametric PINN and an inverse PINN on task T2.

    uv run scripts/phase13_pinns.py               # both parts, about 2.5 hours
    uv run scripts/phase13_pinns.py --reuse       # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_t2.py.

Parametric PINN. A network x_hat(t; theta) of all 23 states over the parameter range,
trained on the residual of the full equations at 8,192 cases with no labels: arm L of
step 36 with its data loss switched off (`Config.data = 0`). Against it, at the same
16,000 steps and on the same 1,000 labelled trajectories: arm L0 (the same network on
data alone) and arm L (data and residual), and the perceptron with physics inputs of
step 38. Three seeds. The labels of the held-out fifth pick the best step for every
arm, the PINN included, and are its only contact with solver data.

Inverse PINN. For 6 test cases, the line current over 0.3 s with white noise at 40 dB,
and `inverse_pinn.fit` started (a) from the truth moved by the convolutional network's
median error at 40 dB (step 39), with random signs, and (b) for 2 of them from the
middle of the range. The Gauss-Newton fit through the solver of step 39 (`identify`)
gets the same records, as phase current, and the same starts.

Writes docs/results/phase13/pinns.json and fig117, fig118. Every number in
docs/steps/47-pinns.md comes from this script.
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

from sso_sciml import identify as idn
from sso_sciml import inverse_pinn as ip
from sso_sciml import measure as me
from sso_sciml import report
from sso_sciml.data import t2
from sso_sciml.models import t1 as mt1
from sso_sciml.models import t2 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t2.VERSION
OUT = ROOT / "docs" / "results" / "phase13"
KEYS = ("theta", "observed", "reduced", "valid", "states", "a_red", "sigma")
BUDGET, POOL, STEPS, SEEDS = 1000, 8192, 16000, (0, 1, 2)
ARMS = {"P": "PINN: residual only", "L0": "L0: data only", "L": "L: data and residual"}
N_INV, N_CENTRE, SNR_DB, INV_STEPS = 6, 2, 40.0, 8000
PARAMS = ("compensation", "rotor speed", "current loop", "PLL")


def load(split: str) -> dict:
    z = np.load(DATA / f"{split}.npz")
    ok = z["solved"]
    out = {k: z[k][ok] for k in KEYS}
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, m) -> dict:
    return {k: v[m] for k, v in d.items() if k != "fold"}


def parametric(device: str) -> dict:
    full = load("in_distribution")
    train = take(full, full["fold"] == 0)
    tests = {"in_distribution": take(full, full["fold"] == 2)}
    tests.update({s: take(load(s), slice(None)) for s in list(t2.SPLITS)[1:]})
    rest = np.random.default_rng(7).permutation(train["theta"].shape[0])[:POOL]
    pool = mt.unlabelled(train["theta"][rest], processes=8)
    out: dict = {"runs": [], "pool": int(pool["theta"].shape[0])}
    for seed in SEEDS:
        sub = mt1.budget_subset(train, BUDGET, seed)
        h = BUDGET // 5
        held, lab = {k: v[:h] for k, v in sub.items()}, {k: v[h:] for k, v in sub.items()}
        for arm in ARMS:
            cfg = mt.Config(steps=STEPS, data=0.0 if arm == "P" else 1.0,
                            weight=0.0 if arm == "L0" else 1.0)  # fmt: skip
            predict, info = mt.train("L" if arm == "P" else arm, lab, held, seed, cfg,
                                     pool=pool if arm != "L0" else None, device=device)  # fmt: skip
            r = {"arm": arm, "seed": seed, "train_s": info["train_s"],
                 "scores": {k: mt.metrics(predict(v), v) for k, v in tests.items()},
                 "history": {k: info["history"][k]
                             for k in ("step", "held_out", "residual")}}  # fmt: skip
            out["runs"].append(r)
            print("parametric", arm, seed, {k: round(v["median"], 4)
                                            for k, v in r["scores"].items()},
                  f"{info['train_s']:.0f} s", flush=True)  # fmt: skip
    out["summary"] = {arm: {k: float(np.mean([r["scores"][k]["median"] for r in out["runs"]
                                              if r["arm"] == arm])) for k in tests}
                      for arm in ARMS}  # fmt: skip
    out["reduced_model_alone"] = {k: mt.metrics(mt.reduced_alone(v), v)["median"]
                                  for k, v in tests.items()}  # fmt: skip
    return out


def inverse(device: str) -> dict:
    full = load("in_distribution")
    test = take(full, full["fold"] == 2)
    ok = np.nonzero(test["valid"] >= t2.N_T)[0][:N_INV]
    size = np.maximum(np.median(np.abs(take(full, full["fold"] == 0)["states"][:2000])
                                .max(axis=1), axis=0), 1e-4)  # fmt: skip
    net_err = np.array(json.loads((ROOT / "docs/results/phase8/measure.json").read_text())
                       ["network"]["rows"]["40"]["u"])  # fmt: skip
    rng = np.random.default_rng(47)
    zero = me.phase_current(np.zeros((1, t2.N_T, 5)))
    out: dict = {"cases": [], "start_error": net_err.tolist(), "snr_db": SNR_DB}
    for n, j in enumerate(ok):
        theta = test["theta"][j]
        u_true = t2.scale_theta(theta[None])[0]
        obs = test["observed"][j : j + 1].astype(float)
        noisy = obs.copy()
        noisy[0, :, :2] = me.add_noise(obs[0, :, :2].T, obs[0, :, :2].T, SNR_DB, rng).T
        base = t2.setup(theta)["p"]
        starts = {"network": u_true + net_err * rng.choice([-1.0, 1.0], 4)}
        if n < N_CENTRE:
            starts["centre"] = np.zeros(4)
        row = {"theta": theta.tolist(), "u_true": u_true.tolist(), "fits": {}}
        for name, u0 in starts.items():
            s0 = t2.setup(idn.unscale(u0[None])[0])
            fit = ip.fit(noisy[0, :, :2], u0, s0["x_eq"], s0["p"]["p_m"], base, size,
                         ip.Config(steps=INV_STEPS), device=device)  # fmt: skip
            y = me.phase_current(noisy) - zero
            gn, hist = idn.gauss_newton(y, u0[None], device=device)
            row["fits"][name] = {
                "start": u0.tolist(), "pinn": fit["u"].tolist(), "pinn_s": fit["train_s"],
                "pinn_history": fit["history"], "gauss_newton": gn[0].tolist(),
                "gauss_newton_misfit": hist[:, 0].tolist(),
            }  # fmt: skip
            print("inverse", n, name, "true", np.round(u_true, 3).tolist(), "pinn",
                  np.round(fit["u"], 3).tolist(), "gn", np.round(gn[0], 3).tolist(),
                  f"{fit['train_s']:.0f} s", flush=True)  # fmt: skip
        out["cases"].append(row)
    for name in ("network", "centre"):
        rows = [c for c in out["cases"] if name in c["fits"]]
        err = {m: np.median([np.abs(np.array(c["fits"][name][m]) - c["u_true"])
                             for c in rows], axis=0).tolist()
               for m in ("start", "pinn", "gauss_newton")}  # fmt: skip
        out[f"median_error_{name}"] = err
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--part", choices=("both", "parametric", "inverse"), default="both")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pinns.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = json.loads(path.read_text()) if path.exists() else {}
    if not args.reuse:
        if args.part in ("both", "parametric"):
            res["parametric"] = parametric(device)
            path.write_text(json.dumps(res))
        if args.part in ("both", "inverse"):
            res["inverse"] = inverse(device)
            path.write_text(json.dumps(res))
        prev = json.loads((ROOT / "docs/results/phase7/t2_models.json").read_text())["summary"]
        res["step38_mlp_F_1000"] = {k: prev["mlp/F@1000"][k]["median"]["mean"]
                                    for k in t2.SPLITS}  # fmt: skip
        path.write_text(json.dumps(res))

    par = res["parametric"]
    sets = list(par["reduced_model_alone"])
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE)
    xs = np.arange(len(sets))
    for k, arm in enumerate(ARMS):
        axes[0].bar(xs + (k - 1) * 0.22, [par["summary"][arm][t] for t in sets], 0.2,
                    color=report.SERIES[k], label=ARMS[arm])  # fmt: skip
    axes[0].plot(xs, [res["step38_mlp_F_1000"][t] for t in sets], marker="_", ms=30, lw=0,
                 color=report.INK, label="perceptron with physics inputs, step 38")  # fmt: skip
    axes[0].plot(xs, [par["reduced_model_alone"][t] for t in sets], marker="x", ms=8, lw=0,
                 color=report.MUTED, label="reduced model alone")  # fmt: skip
    axes[0].set_xticks(xs, [t.replace("_", " ") for t in sets], fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median relative error of the response")
    axes[0].set_title(f"{BUDGET:,} labelled trajectories, {STEPS:,} steps, three seeds",
                      loc="left", fontsize=10)  # fmt: skip
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=0.4)
    for arm in ARMS:
        runs = [(np.array(r["history"]["step"]), np.array(r["history"]["held_out"]))
                for r in par["runs"] if r["arm"] == arm]  # fmt: skip
        report.curves(axes[1], runs, report.SERIES[list(ARMS).index(arm)], ARMS[arm])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Held-out median relative error")
    axes[1].set_title("Training", loc="left", fontsize=10)
    for ax in axes:
        report.style(ax)
    report.title(fig, "A parametric PINN on task T2: the equations alone against data")
    fig.tight_layout()
    fig.savefig(OUT / "fig117_parametric_pinn.png", dpi=160)
    plt.close(fig)

    inv = res["inverse"]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.4), facecolor=report.SURFACE, sharey=True)
    for j, ax in enumerate(axes):
        for name, mark in (("network", "o"), ("centre", "s")):
            for c in inv["cases"]:
                if name not in c["fits"]:
                    continue
                f = c["fits"][name]
                for m, col in (("start", report.MUTED), ("pinn", report.SERIES[1]),
                               ("gauss_newton", report.SERIES[0])):  # fmt: skip
                    ax.plot({"start": 0, "pinn": 1, "gauss_newton": 2}[m],
                            abs(f[m][j] - c["u_true"][j]) + 1e-5, marker=mark, ms=6, lw=0,
                            color=col, alpha=0.8)  # fmt: skip
        ax.set_xticks([0, 1, 2], ["start", "inverse PINN", "Gauss-Newton"], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(PARAMS[j], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Error of the scaled parameter (range = 2)")
    report.title(fig, f"Parameters from a noisy record at {SNR_DB:g} dB (circles: start near the "
                 "answer; squares: start at the middle of the range)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig118_inverse_pinn.png", dpi=160)
    plt.close(fig)

    print("parametric", {a: {t: round(v, 4) for t, v in s.items()}
                         for a, s in par["summary"].items()})  # fmt: skip
    print("train s", {a: round(np.mean([r["train_s"] for r in par["runs"] if r["arm"] == a]))
                      for a in ARMS})  # fmt: skip
    print("reduced alone", par["reduced_model_alone"], "mlp/F", res["step38_mlp_F_1000"])
    for name in ("network", "centre"):
        print(name, {m: np.round(v, 4).tolist() for m, v in inv[f"median_error_{name}"].items()})
    print("pinn s", round(np.mean([f["pinn_s"] for c in inv["cases"] for f in c["fits"].values()])))


if __name__ == "__main__":
    main()
