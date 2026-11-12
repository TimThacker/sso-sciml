"""Phase 2, step 5: assumption studies.

1. Dropping stator and network transients, which is what a positive-sequence tool does.
2. Time step, for RK4 and for the trapezoidal rule.
3. Throughput of batched RK4 on the GPU at the chosen time step.

Writes docs/results/phase2/assumptions.json and two figures. Every number in
docs/steps/06-assumption-studies.md comes from this script.

Run with:  uv run scripts/phase2_assumptions.py
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

from sso_sciml.physics.dfig_full import DFIGFull
from sso_sciml.physics.dfig_meshed import DFIGMeshed
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes, quasi_static
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"

SPEEDS = (0.75, 1.0, 1.2)
# States made algebraic. Fluxes of the two stator-and-network loops, the capacitor
# voltage, and on the meshed network the flux of the loop between the parallel branches.
FAST_ALL = {23: [0, 1, 4, 5, 6, 7], 25: [0, 1, 4, 5, 6, 7, 23, 24]}
FAST_CAP = [4, 5]  # capacitor voltage only: the capacitor becomes a 60 Hz reactance
SLOW_HZ = 5.0  # modes below this are the ones a positive-sequence tool is built for


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(INK)


def linearise(model, k: float, w_r: float):
    x0, p = model.equilibrium(k * model.data.x_e, w_r=w_r)
    a = jacobian(model.rhs, x0, p).numpy()
    md = modes(a)
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return a, md, j, x0, p


def slow_mode_shift(full: np.ndarray, reduced: np.ndarray) -> tuple[int, float]:
    """Largest distance from a slow mode of the full model to its nearest reduced mode."""
    slow = full[np.abs(full) < 2 * np.pi * SLOW_HZ]
    return slow.size, float(max(np.min(np.abs(reduced - lam)) for lam in slow))


def rk4_factor(z: np.ndarray) -> np.ndarray:
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def trap_factor(z: np.ndarray) -> np.ndarray:
    return (1 + z / 2) / (1 - z / 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    systems = {"radial": DFIGFull(), "reference": DFIGMeshed()}

    # ---------------- 1. dropping stator and network transients
    sweep_k = np.round(np.arange(0.10, 0.91, 0.10), 2)
    drop: dict = {}
    for name, model in systems.items():
        fast = FAST_ALL[len(model.equilibrium(0.5 * model.data.x_e)[0])]
        rows = []
        for w in SPEEDS:
            for k in sweep_k:
                a, md, j, _, _ = linearise(model, float(k), w)
                lam_qs = np.linalg.eigvals(quasi_static(a, fast)[0])
                lam_cap = np.linalg.eigvals(quasi_static(a, FAST_CAP)[0])
                n_slow, shift = slow_mode_shift(md.eigenvalues, lam_qs)
                band = (np.abs(lam_qs.imag) > 2 * np.pi * 2.0) & (
                    np.abs(lam_qs.imag) < 2 * np.pi * 58.0
                )
                rows.append({
                    "w_r": w, "k": float(k),
                    "sso_sigma_full": float(md.sigma[j]),
                    "sso_f_full_hz": abs(float(md.freq_hz[j])),
                    "max_sigma_full": float(md.sigma.max()),
                    "max_sigma_quasi_static": float(lam_qs.real.max()),
                    "max_sigma_capacitor_algebraic": float(lam_cap.real.max()),
                    "f_capacitor_algebraic_hz": float(
                        abs(lam_cap[lam_cap.real.argmax()].imag) / (2 * np.pi)
                    ),
                    "quasi_static_modes_2_to_58_hz": int(band.sum() // 2),
                    "n_slow_modes": n_slow,
                    "slow_mode_max_shift": shift,
                })  # fmt: skip
        unstable = [r for r in rows if r["max_sigma_full"] > 1e-6]
        drop[name] = {
            "cases": len(rows),
            "unstable_full": len(unstable),
            "unstable_quasi_static": sum(r["max_sigma_quasi_static"] > 1e-6 for r in rows),
            "unstable_capacitor_algebraic": sum(
                r["max_sigma_capacitor_algebraic"] > 1e-6 for r in rows
            ),
            "missed_by_quasi_static": sum(r["max_sigma_quasi_static"] <= 1e-6 for r in unstable),
            "missed_by_capacitor_algebraic": sum(
                r["max_sigma_capacitor_algebraic"] <= 1e-6 for r in unstable
            ),
            "capacitor_algebraic_frequency_error_hz_max": max(
                (
                    abs(r["f_capacitor_algebraic_hz"] - r["sso_f_full_hz"])
                    for r in unstable
                    if r["max_sigma_capacitor_algebraic"] > 1e-6
                ),
                default=float("nan"),
            ),
            "capacitor_algebraic_frequency_error_hz_min": min(
                (
                    abs(r["f_capacitor_algebraic_hz"] - r["sso_f_full_hz"])
                    for r in unstable
                    if r["max_sigma_capacitor_algebraic"] > 1e-6
                ),
                default=float("nan"),
            ),
            "largest_missed_growth": max((r["sso_sigma_full"] for r in unstable), default=0.0),
            "quasi_static_modes_2_to_58_hz_max": max(
                r["quasi_static_modes_2_to_58_hz"] for r in rows
            ),
            "slow_mode_max_shift": max(r["slow_mode_max_shift"] for r in rows),
            "slow_mode_max_shift_stable_cases": max(
                (r["slow_mode_max_shift"] for r in rows if r["max_sigma_full"] <= 1e-6),
                default=float("nan"),
            ),
            "rows": rows,
        }
    res["dropping_transients"] = drop

    # one case in full, for the figure and the step log
    ref = systems["reference"]
    k_case, w_case = 0.60, 0.75
    a, md, j, x_eq, p_eq = linearise(ref, k_case, w_case)
    lam_qs = np.linalg.eigvals(quasi_static(a, FAST_ALL[25])[0])
    res["example_case"] = {
        "k": k_case, "w_r": w_case,
        "sso_sigma": float(md.sigma[j]), "sso_f_hz": abs(float(md.freq_hz[j])),
        "states_full": int(a.shape[0]), "states_quasi_static": int(lam_qs.size),
        "max_sigma_quasi_static": float(lam_qs.real.max()),
        "fastest_full_rate": float(np.abs(md.eigenvalues).max()),
        "fastest_quasi_static_rate": float(np.abs(lam_qs).max()),
    }  # fmt: skip

    # ---------------- 2. time step
    lam = md.eigenvalues
    lam_sso = lam[j] if lam[j].imag > 0 else np.conj(lam[j])
    steps = np.array([2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3])
    rows = []
    for dt in steps:
        row = {"dt_s": float(dt)}
        for name, factor in (("rk4", rk4_factor), ("trapezoidal", trap_factor)):
            g = factor(lam * dt)
            s_num = np.log(factor(lam_sso * dt)) / dt
            row[name] = {
                "sigma_error": float(s_num.real - lam_sso.real),
                "f_error_hz": float((s_num.imag - lam_sso.imag) / (2 * np.pi)),
                "largest_amplification": float(np.abs(g).max()),
                "spurious_instability": bool(np.any((np.abs(g) > 1 + 1e-12) & (lam.real < 0))),
            }
        rows.append(row)
    dt_grid = np.geomspace(1e-4, 2e-2, 4000)
    bad = [np.any((np.abs(rk4_factor(lam * d)) > 1) & (lam.real < 0)) for d in dt_grid]
    res["time_step"] = {
        "case": {"k": k_case, "w_r": w_case, "sigma": float(lam_sso.real)},
        "rows": rows,
        "rk4_largest_stable_dt_s": float(dt_grid[int(np.argmax(bad)) - 1]),
        "limiting_eigenvalue_rate": float(np.abs(lam).max()),
    }
    for name in ("rk4", "trapezoidal"):
        ok = [r["dt_s"] for r in rows if abs(r[name]["sigma_error"]) < 0.01]
        res["time_step"][f"{name}_largest_dt_within_0p01"] = max(ok) if ok else None

    # confirmation by running RK4: a small kick on the capacitor voltage, growth from the wave
    runs = []
    for dt in (1e-4, 5e-4, 1e-3):
        x0 = x_eq.clone()
        x0[4] += 1e-4
        every = int(round(1e-3 / dt))
        t, traj = rk4(ref.rhs, x0, p_eq, 1.0, dt, save_every=every)
        wave = ref.outputs(traj, p_eq)["p_t"].numpy() - float(ref.outputs(x_eq, p_eq)["p_t"])
        pole = nearest_pole(matrix_pencil(wave[t.numpy() > 0.3], 1e-3, 10), lam_sso.imag / 6.2832)
        predicted = (np.log(rk4_factor(lam_sso * dt)) / dt).real
        runs.append({"dt_s": dt, "sigma_from_run": pole.real, "sigma_predicted": float(predicted)})
    res["time_step"]["rk4_runs"] = runs

    # ---------------- 3. GPU throughput at 0.1 ms
    if torch.cuda.is_available():
        thr = []
        for batch in (1024, 8192, 32768):
            ks = torch.linspace(0.2, 0.8, batch, dtype=torch.float64, device="cuda")
            xs = x_eq.to("cuda").repeat(batch, 1)
            ps = {key: v.to("cuda") for key, v in p_eq.items()}
            ps["x_c"] = ks * ref.data.x_e
            rk4(ref.rhs, xs, ps, 0.002, 1e-4)  # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, tr = rk4(ref.rhs, xs, ps, 0.05, 1e-4, save_every=500)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            thr.append({"batch": batch, "wall_s_per_simulated_s": wall / 0.05,
                        "trajectory_seconds_per_wall_second": batch * 0.05 / wall,
                        "finite": bool(torch.isfinite(tr).all())})  # fmt: skip
        res["gpu_throughput"] = {"device": torch.cuda.get_device_name(0), "dt_s": 1e-4, "rows": thr}

    (OUT / "assumptions.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    ax.scatter(md.sigma, np.abs(md.freq_hz), s=46, color=C1, edgecolor=SURFACE, linewidth=1.0,
               label="Full model (25 states)", zorder=3)  # fmt: skip
    ax.scatter(lam_qs.real, np.abs(lam_qs.imag) / (2 * np.pi), s=46, color=C2, marker="D",
               edgecolor=SURFACE, linewidth=1.0, zorder=4,
               label="Stator and network transients dropped (17 states)")  # fmt: skip
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.annotate("sub-synchronous mode\n(full model only)", (md.sigma[j], abs(md.freq_hz[j])),
                xytext=(-70, 14), textcoords="offset points", fontsize=9, color=LABEL)  # fmt: skip
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Growth rate (1/s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Modes at 60 % compensation, rotor at 0.75 pu, reference network",
                 loc="left", fontsize=11)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_dropping_transients.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    fine = np.geomspace(2e-5, 3e-3, 120)
    for name, factor, colour in (("RK4", rk4_factor, C1), ("Trapezoidal", trap_factor, C2)):
        err = np.abs((np.log(factor(lam_sso * fine)) / fine).real - lam_sso.real)
        ax.loglog(fine * 1e3, err, color=colour, lw=2, label=name)
    ax.axhline(0.01, color=MUTED, lw=0.8, ls="--")
    ax.text(0.021, 0.012, "0.01 1/s", fontsize=9, color=LABEL)
    ax.set_xlabel("Time step (ms)")
    ax.set_ylabel("Error in growth rate (1/s)")
    ax.set_title("Growth-rate error of the sub-synchronous mode against time step",
                 loc="left", fontsize=11)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig14_time_step.png", dpi=160)
    plt.close(fig)

    brief = {k: v for k, v in res.items() if k != "dropping_transients"}
    brief["dropping_transients"] = {
        n: {k: v for k, v in d.items() if k != "rows"} for n, d in drop.items()
    }
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
