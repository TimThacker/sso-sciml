"""What does a time-domain run add to the eigenvalues?

Needs the dataset from scripts/phase3_generate_t1.py. For 512 cases of the T1 test fold
the series capacitor is given an initial voltage offset of four sizes and the full
25-state model is integrated with RK4. The same response is computed with no time
stepping from the modes of the state matrix,

    dx(t) = V exp(Lambda t) V^-1 dx(0),

and the two are compared: how large must the oscillation be before they part?

Writes docs/results/phase3/linear_vs_time_domain.json and two figures. Every number in
docs/steps/15-linear-vs-time-domain.md comes from this script.

Run with:  uv run scripts/phase3_linear_vs_time_domain.py        # about 5 minutes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.physics.dfig_meshed import STATE_NAMES as DFIG_STATES
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian
from sso_sciml.solver.modal import linear_response

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase3"
N_CASES = 512
EPS = (0.01, 0.05, 0.2, 0.5)  # pu, initial offset of the D-axis capacitor voltage
T_END, DT, SAVE = 0.5, 2.5e-4, 4
VC = [4, 5]
ELECTRICAL = [0, 1, 2, 3, 4, 5, 6, 7, 23, 24]  # fluxes, capacitor voltage, loop flux
OTHER = [k for k in range(25) if k not in ELECTRICAL]
TOL = 0.10  # relative difference at which the two responses are said to have parted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    z = np.load(DATA / "in_distribution.npz")
    pick = np.where((z["solver"][:, 4] > 0) & (z["fold"] == 2))[0][:N_CASES]
    theta, sigma = z["theta"][pick], z["solver"][pick, 0]

    # ---------------- operating points and state matrices
    t0 = time.perf_counter()
    ops = [data.operating_point(row) for row in theta]
    a = np.array([jacobian(m.rhs, x, p).numpy() for m, x, p in ops])
    setup_s = (time.perf_counter() - t0) / N_CASES
    model = ops[0][0]
    x_eq = torch.stack([o[1] for o in ops])
    p = {k: torch.stack([o[2][k] for o in ops]) for k in ops[0][2]}

    # ---------------- RK4, all cases and sizes in one batch
    x0 = x_eq.repeat(len(EPS), 1)
    for j, eps in enumerate(EPS):
        x0[j * N_CASES : (j + 1) * N_CASES, VC[0]] += eps
    pb = {k: v.repeat(len(EPS)).to(device) for k, v in p.items()}
    t0 = time.perf_counter()
    t, traj = rk4(model.rhs, x0.to(device), pb, T_END, DT, save_every=SAVE)
    rk4_s = time.perf_counter() - t0
    t = t.numpy()
    dev = traj.cpu().numpy() - x_eq.repeat(len(EPS), 1).numpy()[None]  # (n_t, batch, 25)

    # One case alone on the CPU, as a study would run it.
    t0 = time.perf_counter()
    rk4(model.rhs, x0[0], {k: v[0] for k, v in p.items()}, T_END, DT, save_every=SAVE)
    rk4_single_s = time.perf_counter() - t0

    # ---------------- the same from the modes
    t0 = time.perf_counter()
    unit = np.zeros(data.N_STATE)
    unit[VC[0]] = 1.0
    lin_unit = np.array([linear_response(a[i], unit, t) for i in range(N_CASES)])  # (n, n_t, 25)
    modal_s = (time.perf_counter() - t0) / N_CASES
    scale = np.abs(lin_unit).max(axis=1)  # size of each state's response per unit offset

    def so_far(diff, ref):
        """Difference accumulated up to each time, relative to the response up to then."""
        return np.sqrt(np.cumsum((diff**2).sum(axis=1))) / np.sqrt(np.cumsum((ref**2).sum(axis=1)))

    rows = []
    by_state = np.zeros((len(EPS), data.N_STATE))
    k100 = int(np.searchsorted(t, 0.1))
    for j, eps in enumerate(EPS):
        for i in range(N_CASES):
            full = dev[:, j * N_CASES + i, :]
            lin = eps * lin_unit[i]
            s = eps * scale[i]
            diff = np.where(np.isfinite(full), full - lin, np.inf)
            # The oscillation itself: fluxes and capacitor voltage, in per unit.
            rel = so_far(diff[:, ELECTRICAL], lin[:, ELECTRICAL])
            # Speed, PLL, controller and DC-link states, each in units of its own response.
            rel_slow = so_far(diff[:, OTHER] / s[OTHER], lin[:, OTHER] / s[OTHER])
            parted = np.where(rel > TOL)[0]
            amp = np.hypot(lin[:, VC[0]], lin[:, VC[1]])  # capacitor voltage oscillation, pu
            if np.isfinite(diff[: k100 + 1]).all():
                by_state[j] += ((diff[: k100 + 1] / s) ** 2).mean(axis=0) / N_CASES
            rows.append({
                "case": i, "eps": eps, "sigma": float(sigma[i]),
                "error_100ms": float(rel[k100]) if np.isfinite(rel[k100]) else None,
                "error_500ms": float(rel[-1]) if np.isfinite(rel[-1]) else None,
                "other_states_error_100ms": (
                    float(rel_slow[k100]) if np.isfinite(rel_slow[k100]) else None
                ),
                "parted_at_s": float(t[parted[0]]) if parted.size else None,
                "amplitude_when_parted": float(amp[: parted[0] + 1].max()) if parted.size else None,
                "largest_amplitude": float(amp.max()),
            })  # fmt: skip

    def summarise(sel):
        e1 = [r["error_100ms"] for r in sel if r["error_100ms"] is not None]
        e5 = [r["error_500ms"] for r in sel if r["error_500ms"] is not None]
        amp = [r["amplitude_when_parted"] for r in sel if r["parted_at_s"] is not None]
        eo = [r["other_states_error_100ms"] for r in sel
              if r["other_states_error_100ms"] is not None]  # fmt: skip
        return {
            "trajectories": len(sel),
            "median_error_100ms": float(np.median(e1)),
            "p95_error_100ms": float(np.quantile(e1, 0.95)),
            "median_other_states_error_100ms": float(np.median(eo)),
            "median_error_500ms": float(np.median(e5)) if e5 else None,
            "share_never_parted": float(np.mean([r["parted_at_s"] is None for r in sel])),
            "median_amplitude_when_parted": float(np.median(amp)) if amp else None,
            "p10_amplitude_when_parted": float(np.quantile(amp, 0.1)) if amp else None,
            "median_largest_amplitude": float(np.median([r["largest_amplitude"] for r in sel])),
        }

    summary = {}
    for eps in EPS:
        for kind, sign in (("stable", -1), ("unstable", 1)):
            summary[f"{eps}/{kind}"] = summarise(
                [r for r in rows if r["eps"] == eps and (r["sigma"] > 0) == (sign > 0)]
            )
    parted = [r for r in rows if r["parted_at_s"] is not None]
    names = DFIG_STATES
    worst = np.argsort(-by_state[EPS.index(0.2)])[:6]
    res = {
        "cases": N_CASES,
        "unstable_cases": int((sigma > 0).sum()),
        "offsets_pu": list(EPS),
        "window_s": T_END,
        "time_step_s": DT,
        "tolerance": TOL,
        "summary": summary,
        "amplitude_when_parted_all": {
            "trajectories": len(parted),
            "p10": float(np.quantile([r["amplitude_when_parted"] for r in parted], 0.1)),
            "median": float(np.median([r["amplitude_when_parted"] for r in parted])),
            "p90": float(np.quantile([r["amplitude_when_parted"] for r in parted], 0.9)),
        },
        # States that differ most at 0.2 pu, each in units of its own linear response:
        # root-mean-square difference over the first 0.1 s, averaged over cases.
        "states_that_differ_most_at_0.2_pu": {
            names[k]: float(np.sqrt(by_state[EPS.index(0.2), k])) for k in worst
        },
        "cost": {
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "steady_state_and_state_matrix_s_per_case": setup_s,
            "modal_response_s_per_case": modal_s,
            "rk4_batch_s": rk4_s,
            "rk4_batch_s_per_trajectory": rk4_s / (N_CASES * len(EPS)),
            "rk4_single_cpu_s": rk4_single_s,
        },
        "wall_s": time.perf_counter() - t_start,
    }
    (OUT / "linear_vs_time_domain.json").write_text(json.dumps(res, indent=2))
    for k, v in summary.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    print(res["amplitude_when_parted_all"], res["states_that_differ_most_at_0.2_pu"], res["cost"])

    # ---------------- figure: waveforms
    stable = int(np.where((sigma < -2) & (sigma > -6))[0][0])
    unstable = int(np.where((sigma > 3) & (sigma < 8))[0][0])
    fig, axes = plt.subplots(
        2, len(EPS), figsize=(15.0, 6.2), sharex=True, facecolor=report.SURFACE
    )
    for r, (i, label) in enumerate(((stable, "decaying"), (unstable, "growing"))):
        for j, eps in enumerate(EPS):
            ax = axes[r, j]
            ax.plot(t, dev[:, j * N_CASES + i, VC[0]], color=report.MUTED, lw=1.4, label="RK4")
            ax.plot(t, eps * lin_unit[i][:, VC[0]], color=report.SERIES[1], lw=0.9, ls="--",
                    label="from the modes")  # fmt: skip
            lim = 1.3 * np.abs(eps * lin_unit[i][:, VC[0]]).max()
            ax.set_ylim(-lim, lim)
            report.style(ax)
            if r == 0:
                ax.set_title(f"Initial offset {eps} pu", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{label} case, {sigma[i]:+.1f} 1/s\ncapacitor voltage (pu)")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL, loc="upper right")
    report.title(fig, "Time-domain response against the response computed from the eigenvalues")
    fig.tight_layout()
    fig.savefig(OUT / "fig38_linear_waveforms.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: where they part
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=report.SURFACE)
    ax = axes[0]
    for kind, colour in (("stable", report.SERIES[0]), ("unstable", report.SERIES[1])):
        med = [summary[f"{e}/{kind}"]["median_error_100ms"] for e in EPS]
        p95 = [summary[f"{e}/{kind}"]["p95_error_100ms"] for e in EPS]
        ax.plot(EPS, med, color=colour, lw=2, marker="o", ms=5, label=f"{kind} cases, median")
        ax.plot(EPS, p95, color=colour, lw=1, ls=":", marker="o", ms=3, label="95th percentile")
    ax.axhline(TOL, color=report.INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Initial offset of capacitor voltage (pu)")
    ax.set_ylabel("Difference over the first 0.1 s, relative to the response")
    ax.legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    report.style(ax)
    ax = axes[1]
    amp = np.array([r["amplitude_when_parted"] for r in parted])
    ax.hist(amp, bins=np.geomspace(0.01, 5.0, 40), color=report.SERIES[2])
    ax.set_xscale("log")
    ax.set_xlabel("Amplitude of the capacitor voltage oscillation when the two part (pu)")
    ax.set_ylabel("Trajectories")
    report.style(ax)
    report.title(fig, f"How far the linear response holds (parted: more than {TOL:.0%} apart)")
    fig.tight_layout()
    fig.savefig(OUT / "fig39_linear_departure.png", dpi=160)
    plt.close(fig)
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
