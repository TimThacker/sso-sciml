"""Rung 4: the full-control model against a second implementation in phase coordinates.

    uv run scripts/second_full_model.py      # about 10 minutes

`physics/dfig_full_abc.py` writes the equations of docs/EQUATIONS.md section 15 again
with different states, in the stationary frame, with no automatic differentiation, and
integrates them with an adaptive method of SciPy. For 12 cases spread over the
parameters of dataset T2 it is compared with `physics/dfig_full.py`:

    trajectory   both nonlinear models from the same small capacitor-voltage offset;
                 the capacitor voltage in the synchronous frame, the largest difference
                 over the run as a share of the largest deviation
    growth rate  matrix pencil on the second model's capacitor voltage, against the
                 eigenvalue of the first model's state matrix

Writes docs/results/validation/second_full_model.json and fig113_second_full_model.png.
Every number in docs/steps/44-second-full-model.md comes from this script.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.data import t2
from sso_sciml.physics import dfig_full_abc as fa
from sso_sciml.physics.dfig_full import Controls, DFIGFull
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "validation"
OFFSET = 1e-3  # pu on the D axis of the capacitor voltage; small, so both stay linear
T_END, DT_RK4, EVERY = 0.4, 5e-5, 10
FIT_FROM = 0.05  # s; the fit skips the first part, where fast modes are still present
GRID = list(product((0.1, 0.3, 0.5), (0.85, 1.15), (5.0, 20.0)))  # k, w_r, rsc_hz


def one(k: float, w_r: float, rsc_hz: float) -> dict:
    # The plant of dataset T2 without the protection of step 33: the second model has no
    # limits or chopper, and the fastest case here would reach them within the run.
    ctl = Controls(rsc_hz=rsc_hz, pll_hz=15.0, outer_hz=t2.OUTER_RATIO * rsc_hz)
    model = DFIGFull(controls=ctl)
    x0, p = model.equilibrium(k * model.data.x_e, w_r=w_r)
    md = modes(jacobian(model.rhs, x0, p))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    eig = complex(md.eigenvalues[j])

    start = x0.clone()
    start[4] += OFFSET
    t0 = time.perf_counter()
    t_a, x_a = rk4(model.rhs, start, p, T_END, DT_RK4, save_every=EVERY)
    rk4_s = time.perf_counter() - t0
    t_a = t_a.numpy()
    vc_a = (x_a[:, 4:6] - x0[4:6]).numpy()

    pm = fa.PhaseFull(p, model.data.wb)
    z0 = pm.from_full_state(start.numpy())
    rest = pm.from_full_state(x0.numpy())
    t0 = time.perf_counter()
    z = fa.simulate(pm, z0, t_a)
    abc_s = time.perf_counter() - t0
    vc_b = pm.capacitor_dq(t_a, z) - (fa.to_dq(0.0) @ rest[9:12])

    gap = np.abs(vc_b - vc_a).max() / np.abs(vc_a).max()
    keep = t_a >= FIT_FROM
    dt = t_a[1] - t_a[0]
    pole = nearest_pole(matrix_pencil(vc_b[keep, 0], dt, 16), abs(eig.imag) / (2 * np.pi))
    # Speed and DC voltage, the slowest and a converter-side state, as further checks.
    w_gap = np.abs(z[:, 12] - x_a[:, 8].numpy()).max()
    vdc_gap = np.abs(z[:, 23] - x_a[:, 19].numpy()).max()
    return {"k": k, "w_r": w_r, "rsc_hz": rsc_hz, "eig_sigma": eig.real,
            "eig_f": abs(eig.imag) / (2 * np.pi), "fit_sigma": pole.real,
            "fit_f": pole.imag / (2 * np.pi), "trajectory_gap": float(gap),
            "speed_gap": float(w_gap), "vdc_gap": float(vdc_gap), "rk4_s": rk4_s,
            "abc_s": abc_s, "t": t_a[::2].tolist(), "vc_first": vc_a[::2, 0].tolist(),
            "vc_second": vc_b[::2, 0].tolist()}  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = []
    for k, w_r, hz in GRID:
        r = one(k, w_r, hz)
        rows.append(r)
        print(f"k {k} w_r {w_r} rsc {hz}: eig {r['eig_sigma']:+.4f} 1/s {r['eig_f']:.3f} Hz,",
              f"fit {r['fit_sigma']:+.4f} {r['fit_f']:.3f},",
              f"trajectory gap {r['trajectory_gap']:.2e}",
              f"speed {r['speed_gap']:.1e} vdc {r['vdc_gap']:.1e} ({r['abc_s']:.0f} s)",
              flush=True)  # fmt: skip
    ds = np.array([r["fit_sigma"] - r["eig_sigma"] for r in rows])
    df = np.array([r["fit_f"] - r["eig_f"] for r in rows])
    res = {"offset": OFFSET, "t_end": T_END, "cases": rows,
           "summary": {"growth_rate_gap_max": float(np.abs(ds).max()),
                       "frequency_gap_max_hz": float(np.abs(df).max()),
                       "trajectory_gap_max": float(max(r["trajectory_gap"] for r in rows)),
                       "unstable_cases": int(sum(r["eig_sigma"] > 0 for r in rows)),
                       "sign_agreement": int(sum((r["eig_sigma"] > 0) == (r["fit_sigma"] > 0)
                                                 for r in rows))}}  # fmt: skip
    (OUT / "second_full_model.json").write_text(json.dumps(res))
    print(res["summary"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=report.SURFACE)
    show = max(rows, key=lambda r: r["eig_sigma"])
    axes[0].plot(show["t"], show["vc_first"], color=report.SERIES[0], lw=2.2,
                 label="full model, D-Q, RK4")  # fmt: skip
    axes[0].plot(show["t"], show["vc_second"], color=report.SERIES[1], lw=1.0, ls="--",
                 label="second model, phases, adaptive")  # fmt: skip
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Capacitor voltage deviation, D (pu)")
    axes[0].set_title(f"The fastest-growing case: k {show['k']}, speed {show['w_r']}, "
                      f"loop {show['rsc_hz']:g} Hz", loc="left", fontsize=10)  # fmt: skip
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.style(axes[0])
    es, fs = [r["eig_sigma"] for r in rows], [r["fit_sigma"] for r in rows]
    axes[1].plot(es, fs, marker="o", ms=7, lw=0, color=report.SERIES[0])
    lim = [min(es + fs) - 1, max(es + fs) + 1]
    axes[1].plot(lim, lim, color=report.INK, lw=0.8, ls="--")
    axes[1].axhline(0, color=report.MUTED, lw=0.6)
    axes[1].axvline(0, color=report.MUTED, lw=0.6)
    axes[1].set_xlabel("Eigenvalue of the full model (1/s)")
    axes[1].set_ylabel("Fitted to the second model's run (1/s)")
    gap = res["summary"]["growth_rate_gap_max"]
    axes[1].set_title(f"Growth rate, 12 cases (largest gap {gap:.1e} 1/s)", loc="left",
                      fontsize=10)  # fmt: skip
    report.style(axes[1])
    report.title(fig, "The full controls written twice: D-Q with loop fluxes, and phase "
                 "coordinates with branch currents")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig113_second_full_model.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
