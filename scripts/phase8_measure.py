"""Phase 8, task T3: the mode and the plant parameters from a noisy waveform.

    uv run scripts/phase8_measure.py            # about 20 minutes
    uv run scripts/phase8_measure.py --reuse    # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_t2.py. The cases are those of its
in-distribution folds that run through the whole 0.3 s window. From each, the phase
current a fault recorder would see (32 samples per cycle) and the phasors a measurement
unit would report (120, 60 and 30 frames per second), with white noise at a stated
signal-to-noise ratio against the deviation from the fundamental.

    matrix pencil and Prony on the recorder trace, matrix pencil on the phasors:
        growth rate and frequency of the sub-synchronous mode
    a convolutional network on the recorder trace: the mode and the four parameters
    the Cramer-Rao bound of each parameter from derivatives of the full model
    a Gauss-Newton fit of the parameters through the full model

Writes docs/results/phase8/measure.json and figures fig97 to fig99. Every number in
docs/steps/39-measurement-inverse.md comes from this script.
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

from sso_sciml import identify as idn
from sso_sciml import measure as me
from sso_sciml import report
from sso_sciml.data import t2
from sso_sciml.models import t3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t2.VERSION
OUT = ROOT / "docs" / "results" / "phase8"
SNRS = (60.0, 40.0, 30.0, 20.0, 10.0, 0.0)
FRAMES = (120, 60, 30)
N_CLASSICAL, N_BOUND, N_FIT = 1024, 256, 64
FIT_SNRS = (40.0, 20.0)
SEEDS = (0, 1, 2)
PARAMS = ("series compensation", "rotor speed", "current loop bandwidth",
          "phase-locked loop frequency")  # fmt: skip


def med(a: np.ndarray) -> float:
    return float(np.nanmedian(np.abs(a)))


def p90(a: np.ndarray) -> float:
    return float(np.nanquantile(np.abs(a), 0.9))


def load() -> dict:
    z = np.load(DATA / "in_distribution.npz")
    ok = z["solved"] & (z["valid"] >= t2.N_T)
    zero = me.phase_current(np.zeros((1, t2.N_T, 5)))
    out = {}
    for name, fold in (("train", 0), ("held", 1), ("test", 2)):
        m = ok & (z["fold"] == fold)
        out[name] = {"clean": me.phase_current(z["observed"][m]) - zero, "sigma": z["sigma"][m],
                     "f": z["f"][m], "u": t2.scale_theta(z["theta"][m])}  # fmt: skip
    out["zero"] = zero
    out["dropped"] = int((z["solved"] & (z["valid"] < t2.N_T)).sum())
    return out


def classical(test: dict, zero: np.ndarray, rng) -> dict:
    clean, sigma, f = (test[k][:N_CLASSICAL] for k in ("clean", "sigma", "f"))
    res: dict = {"cases": int(clean.shape[0]), "rows": {}, "scatter": {}}
    for snr in (np.inf, *SNRS):
        y = me.add_noise(clean, clean, snr, rng) + zero
        row = {}
        for name, method, order in (("pencil", me.matrix_pencil, 12), ("prony", me.prony, 40)):
            t0 = time.perf_counter()
            est = me.from_recorder(y, order, method)
            row[name] = {"sigma": med(est[:, 0] - sigma), "sigma_p90": p90(est[:, 0] - sigma),
                         "f": med(est[:, 1] - f), "none": int((~np.isfinite(est[:, 0])).sum()),
                         "wrong_sign": float(np.mean(np.sign(est[:, 0]) != np.sign(sigma))),
                         "s_per_case": (time.perf_counter() - t0) / len(y)}  # fmt: skip
        for fps in FRAMES:
            ph, dt = me.phasors(y, fps)
            est = me.from_phasors(ph, dt)
            row[f"pmu{fps}"] = {
                "sigma": med(est[:, 0] - sigma), "sigma_p90": p90(est[:, 0] - sigma),
                "f": med(est[:, 1] - f), "f_folded": med(est[:, 1] - me.folded(f, fps)),
                "wrong_sign": float(np.mean(np.sign(est[:, 0]) != np.sign(sigma))),
                "frames": int(ph.shape[1]),
            }  # fmt: skip
            if not np.isfinite(snr):
                res["scatter"][str(fps)] = {"true": f[:400].tolist(), "seen": est[:400, 1].tolist()}
        res["rows"]["clean" if not np.isfinite(snr) else f"{snr:g}"] = row
        print("classical", snr, {k: round(v["sigma"], 3) for k, v in row.items()}, flush=True)
    return res


def network(data: dict, device: str) -> tuple[dict, object]:
    tr, he, te = data["train"], data["held"], data["test"]
    res: dict = {"train_cases": int(tr["clean"].shape[0]), "test_cases": int(te["clean"].shape[0]),
                 "runs": []}  # fmt: skip
    first = None
    for seed in SEEDS:
        predict, info = t3.train(tr["clean"], t3.targets(tr["sigma"], tr["f"], tr["u"]),
                                 (he["clean"], t3.targets(he["sigma"], he["f"], he["u"])), seed,
                                 device=device)  # fmt: skip
        first = first or predict
        rng = np.random.default_rng(100 + seed)
        run = {"seed": seed, "fit_s": info["train_s"], "parameters": info["parameters"],
               "history": info["history"], "rows": {}}  # fmt: skip
        for snr in SNRS:
            out = predict(me.add_noise(te["clean"], te["clean"], snr, rng))
            run["rows"][f"{snr:g}"] = {
                "sigma": med(out["sigma"] - te["sigma"]),
                "sigma_p90": p90(out["sigma"] - te["sigma"]), "f": med(out["f"] - te["f"]),
                "wrong_sign": float(np.mean(np.sign(out["sigma"]) != np.sign(te["sigma"]))),
                "u": [med(out["u"][:, j] - te["u"][:, j]) for j in range(4)],
            }  # fmt: skip
        res["runs"].append(run)
        print("network", seed, {k: round(v["sigma"], 3) for k, v in run["rows"].items()},
              f"{info['train_s']:.0f} s", flush=True)  # fmt: skip
    res["rows"] = {
        key: {m: float(np.mean([r["rows"][key][m] for r in res["runs"]]))
              for m in ("sigma", "sigma_p90", "f", "wrong_sign")}
        | {"u": np.mean([r["rows"][key]["u"] for r in res["runs"]], axis=0).tolist()}
        for key in res["runs"][0]["rows"]
    }  # fmt: skip
    return res, first


def parameters(test: dict, predict, device: str, rng) -> dict:
    clean, u = test["clean"][:N_BOUND], test["u"][:N_BOUND]
    t0 = time.perf_counter()
    jac, ok = idn.jacobian(u, device=device)
    res: dict = {"cases": int(ok.sum()), "jacobian_s": time.perf_counter() - t0, "bound": {},
                 "fit": {}}  # fmt: skip
    jac, clean_ok = jac[ok], clean[ok]
    size = np.sqrt((clean_ok**2).mean(axis=1))
    for snr in SNRS:
        b = idn.bound(jac, size * 10 ** (-snr / 20))
        res["bound"][f"{snr:g}"] = np.median(b, axis=0).tolist()
    # How unequal the directions are: the eigenvalues of J^T J of each case.
    eig = np.linalg.eigvalsh(np.einsum("nti,ntj->nij", jac, jac))
    res["eigen_ratio_median"] = float(np.median(eig[:, 0] / eig[:, -1]))
    vec = np.linalg.eigh(np.einsum("nti,ntj->nij", jac, jac))[1][:, :, 0]
    res["weakest_direction"] = np.median(np.abs(vec), axis=0).tolist()
    res["sensitivity"] = np.median(np.sqrt((jac**2).mean(axis=1)) / size[:, None], axis=0).tolist()
    for snr in FIT_SNRS:
        y = me.add_noise(clean[:N_FIT], clean[:N_FIT], snr, rng)
        truth = u[:N_FIT]
        start = predict(y)["u"].astype(float)
        row = {"network": [med(start[:, j] - truth[:, j]) for j in range(4)]}
        for name, u0 in (("from_network", start), ("from_centre", np.zeros_like(truth))):
            t0 = time.perf_counter()
            fit, history = idn.gauss_newton(y, u0, device=device)
            noise = np.sqrt((clean[:N_FIT] ** 2).mean(axis=1)) * 10 ** (-snr / 20)
            reached = history[-1] < 1.5 * noise
            row[name] = {
                "u": [med(fit[:, j] - truth[:, j]) for j in range(4)],
                "u_reached": [med((fit[:, j] - truth[:, j])[reached]) for j in range(4)],
                "reached_noise_floor": float(reached.mean()),
                "s_per_case": (time.perf_counter() - t0) / N_FIT,
            }  # fmt: skip
        res["fit"][f"{snr:g}"] = row
        print("fit", snr, {k: np.round(v["u"] if isinstance(v, dict) else v, 4).tolist()
                           for k, v in row.items()}, flush=True)  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "measure.json"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(20261114)
        data = load()
        t0 = time.perf_counter()
        res = {"signal_rms_pu_median": float(np.median(
                   np.sqrt((data["test"]["clean"] ** 2).mean(axis=1)))),
               "dropped_for_a_trip": data["dropped"], "samples": int(data["zero"].shape[1]),
               "classical": classical(data["test"], data["zero"], rng)}  # fmt: skip
        res["network"], predict = network(data, device)
        res["parameters"] = parameters(data["test"], predict, device, rng)
        res["wall_s"] = time.perf_counter() - t0
        path.write_text(json.dumps(res))
    cl, net, par = res["classical"]["rows"], res["network"]["rows"], res["parameters"]
    keys = [f"{s:g}" for s in SNRS]

    # ---------------- figure: the mode against noise
    lines = (("pencil", "Matrix pencil, recorder", report.SERIES[0], "-"),
             ("prony", "Prony, recorder", report.SERIES[1], "-"),
             ("pmu120", "Matrix pencil, phasors at 120 /s", report.SERIES[2], "--"),
             ("pmu60", "phasors at 60 /s", report.SERIES[3], "--"),
             ("pmu30", "phasors at 30 /s", report.SERIES[4], "--"))  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    labels = ("Error of the growth rate, median (1/s)", "Error of the frequency, median (Hz)",
              "Sign of the growth rate wrong (% of cases)")  # fmt: skip
    for ax, metric, label in zip(axes, ("sigma", "f", "wrong_sign"), labels, strict=True):
        gain = 100.0 if metric == "wrong_sign" else 1.0
        for key, name, colour, ls in lines:
            m = "f_folded" if metric == "f" and key.startswith("pmu") else metric
            ax.plot(SNRS, [gain * cl[k][key][m] for k in keys], color=colour, ls=ls, marker="o",
                    ms=4, lw=1.6, label=name)  # fmt: skip
        ax.plot(SNRS, [gain * net[k][metric] for k in keys], color=report.INK, marker="s", ms=4,
                lw=1.8, label="Convolutional network, recorder")  # fmt: skip
        if metric != "wrong_sign":
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Signal-to-noise ratio (dB)")
        ax.set_title(label, loc="left", fontsize=10)
        report.style(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "The sub-synchronous mode from 0.3 s of phase current (phasor frequency "
                 "scored against the folded value)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig97_mode_against_noise.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: what a phasor unit reports
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor=report.SURFACE, sharey=True)
    grid = np.linspace(22.0, 50.0, 200)
    for ax, fps in zip(axes, FRAMES, strict=True):
        sc = res["classical"]["scatter"][str(fps)]
        ax.plot(grid, grid, color=report.MUTED, lw=0.8, ls=":", label="true frequency")
        ax.plot(grid, me.folded(grid, fps), color=report.INK, lw=1.0, label="folded by the rate")
        ax.plot(sc["true"], sc["seen"], ls="", marker="o", ms=3, color=report.SERIES[0],
                alpha=0.6, label="read from the phasors")  # fmt: skip
        ax.set_title(f"{fps} frames per second", loc="left", fontsize=10)
        ax.set_xlabel("Frequency of the mode, synchronous frame (Hz)")
        report.style(ax)
    axes[0].set_ylabel("Frequency seen (Hz)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "What a phasor measurement unit reports of the mode, no noise")
    fig.tight_layout()
    fig.savefig(OUT / "fig98_phasor_folding.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: the parameters
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.6), facecolor=report.SURFACE, sharey=True)
    for j, ax in enumerate(axes):
        ax.plot(SNRS, [par["bound"][k][j] for k in keys], color=report.INK, lw=1.4, ls=":",
                label="Cramér-Rao bound")  # fmt: skip
        ax.plot(SNRS, [net[k]["u"][j] for k in keys], color=report.SERIES[0], marker="s", ms=4,
                lw=1.6, label="Convolutional network")  # fmt: skip
        fs = [float(k) for k in par["fit"]]
        ax.plot(fs, [par["fit"][k]["from_network"]["u"][j] for k in par["fit"]], ls="",
                marker="o", ms=7, color=report.SERIES[1], label="Fit through the model, from the "
                "network's answer")  # fmt: skip
        ax.plot(fs, [par["fit"][k]["from_centre"]["u"][j] for k in par["fit"]], ls="",
                marker="x", ms=7, color=report.SERIES[3], label="Fit, from the middle of the "
                "range")  # fmt: skip
        ax.axhline(0.5, color=report.MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Signal-to-noise ratio (dB)")
        ax.set_title(PARAMS[j], loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("Error, median (range of the parameter = 2)")
    axes[3].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="lower right")
    report.title(fig, "The four parameters from the waveform (grey line: a guess at the middle "
                 "of the range)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig99_parameters.png", dpi=160)
    plt.close(fig)

    print("signal rms", res["signal_rms_pu_median"], "dropped", res["dropped_for_a_trip"])
    for k in ("clean", *keys):
        print(k, {m: (round(v["sigma"], 3), round(v["f"], 3)) for m, v in cl[k].items()})
    for k in keys:
        print("net", k, round(net[k]["sigma"], 3), round(net[k]["f"], 3), np.round(net[k]["u"], 3),
              "bound", np.round(par["bound"][k], 4))  # fmt: skip
    print("eigen ratio", par["eigen_ratio_median"], "weakest",
          np.round(par["weakest_direction"], 2), "sensitivity", np.round(par["sensitivity"], 3))
    print(json.dumps(par["fit"], indent=0)[:3000])


if __name__ == "__main__":
    main()
