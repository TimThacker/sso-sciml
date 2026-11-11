"""Phase 2, step 2: doubly-fed machine with rotor current control.

Writes docs/results/phase2/dfig.json and three figures. Every number quoted in
docs/steps/03-dfig-current-control.md comes from this script.

Run with:  uv run scripts/phase2_dfig.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

from sso_sciml.physics import energy, slip
from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.induction import N_ELECTRICAL, W_R, InductionGenerator
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LABEL = "#52514e"

M = DFIG()
IG = InductionGenerator()
D = M.data
MACH = {key: getattr(D, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}
SPEEDS = (0.75, 1.0, 1.2)  # low, medium and high wind
SPEED_COLOUR = dict(zip(SPEEDS, (C1, C2, C3), strict=True))
KP_REF, KI_REF = gains_from_bandwidth(D, 10.0)
TUNE = KI_REF / KP_REF  # Ki / Kp of the pole-cancellation tuning, 1/s


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


def eigenvalues(k: float, w_r: float, kp: float, ki: float) -> np.ndarray:
    """Eigenvalues of the model. It is linear, so the state it is evaluated at is irrelevant."""
    p = M.params(k * D.x_e, w_r=w_r, kp=kp, ki=ki)
    return np.linalg.eigvals(jacobian(M.rhs, torch.zeros(8, dtype=torch.float64), p).numpy())


def sso_sigma(lam: np.ndarray) -> float:
    """Largest growth rate among oscillatory modes between 2 and 58 Hz in the rotating frame.

    Returns a large negative number when no mode lies in the band.
    """
    f = np.abs(lam.imag) / (2 * np.pi)
    band = lam.real[(f > 2.0) & (f < 58.0)]
    return float(band.max()) if band.size else -1e3


def hand_kp_only(k: float, w_r: float, kp: float) -> slip.SlipResult:
    """Slip calculation with the controller reduced to added rotor resistance Kp."""
    return slip.analyse_loop(
        lambda s: slip.dfig_impedance(s, w_r=w_r, kp=kp, ki=0.0, **MACH),
        D.r_e, D.x_e, k * D.x_e, D.f0, w_r,
    )  # fmt: skip


def loop_impedance(s: complex, k: float, w_r: float, kp: float, ki: float) -> complex:
    z_m = slip.dfig_impedance(s, w_r=w_r, kp=kp, ki=ki, **MACH)
    return D.r_e + s * D.x_e + k * D.x_e / s + z_m


def first_crossing(f, grid: np.ndarray) -> float:
    """Lowest point on the grid where f goes from negative to positive, refined."""
    vals = np.array([f(v) for v in grid])
    idx = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if idx.size == 0:
        return float("nan")
    return float(brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-12))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {
        "reference_tuning": {"bandwidth_hz": 10.0, "kp": KP_REF, "ki": KI_REF, "ki_over_kp": TUNE},
        "gains_30hz": gains_from_bandwidth(D, 30.0),
        "gains_100hz": gains_from_bandwidth(D, 100.0),
    }

    # ---------------- 1. zero gain reproduces the induction machine
    worst = 0.0
    for k in (0.3, 0.6, 0.9):
        for w_r in (0.7, 0.9, 1.0, 1.2):
            lam = eigenvalues(k, w_r, 0.0, 0.0)
            x = torch.zeros(7, dtype=torch.float64)
            x[W_R] = w_r
            a = jacobian(IG.rhs, x, IG.params(k * D.x_e)).numpy()[:N_ELECTRICAL, :N_ELECTRICAL]
            for mu in np.linalg.eigvals(a):
                worst = max(worst, float(np.abs(lam - mu).min()))
    boundary = {}
    for w_r in (0.7, 0.8, 0.9, 1.0):
        boundary[str(w_r)] = first_crossing(
            lambda k, w=w_r: sso_sigma(eigenvalues(k, w, 0.0, 0.0)), np.linspace(0.05, 0.99, 95)
        )
    res["zero_gain"] = {"eigenvalue_max_abs_diff": worst, "boundary_k_by_speed": boundary}

    # ---------------- 2. the circuit with Kp + Ki/S in the rotor is exact
    worst_z, worst_case = 0.0, None
    for k in (0.2, 0.5, 0.8):
        for w_r in SPEEDS:
            for kp in (0.01, 0.05, 0.3):
                for ki in (0.1, 3.0, 100.0):
                    for lam in eigenvalues(k, w_r, kp, ki):
                        z = min(
                            abs(loop_impedance(lm / D.wb + 1j, k, w_r, kp, ki))
                            for lm in (lam, np.conj(lam))
                        )
                        if z > worst_z:
                            worst_z, worst_case = z, [k, w_r, kp, ki]
    res["circuit_exactness"] = {"max_abs_loop_impedance_at_eigenvalues_pu": worst_z,
                                "worst_case_k_wr_kp_ki": worst_case}  # fmt: skip

    # ---------------- 3. mechanism against Kp at one case (low wind, 50 %)
    k_ref, w_ref = 0.5, 0.75
    kps = np.linspace(0.0, 0.1, 41)
    mech = {"kp": kps.tolist(), "slip": [], "r_machine": [], "r_net": [], "sigma_hand": [],
            "sigma_kp_only": [], "sigma_solver": []}  # fmt: skip
    for kp in kps:
        sr = hand_kp_only(k_ref, w_ref, float(kp))
        mech["slip"].append(sr.slip)
        mech["r_machine"].append(sr.r_machine)
        mech["r_net"].append(sr.r_net)
        mech["sigma_hand"].append(sr.sigma_hand)
        mech["sigma_kp_only"].append(sr.sigma_circuit)
        mech["sigma_solver"].append(sso_sigma(eigenvalues(k_ref, w_ref, float(kp), TUNE * kp)))
    mech["k"], mech["w_r"] = k_ref, w_ref
    mech["r_machine_per_unit_kp"] = float(np.polyfit(kps, mech["r_machine"], 1)[0])
    res["mechanism"] = mech

    # ---------------- 4. trends at the reference tuning
    ks = np.linspace(0.05, 0.95, 37)
    trend_k = {"k": ks.tolist()}
    for w_r in SPEEDS:
        trend_k[str(w_r)] = [sso_sigma(eigenvalues(k, w_r, KP_REF, KI_REF)) for k in ks]
    kp_grid = np.linspace(0.0, 0.3, 61)
    trend_kp = {"kp": kp_grid.tolist(), "k": k_ref}
    for w_r in SPEEDS:
        trend_kp[str(w_r)] = [
            sso_sigma(eigenvalues(k_ref, w_r, float(kp), TUNE * kp)) for kp in kp_grid
        ]
    wr_grid = np.linspace(0.7, 1.25, 23)
    trend_w = {
        "w_r": wr_grid.tolist(),
        "k": k_ref,
        "sigma": [sso_sigma(eigenvalues(k_ref, float(w), KP_REF, KI_REF)) for w in wr_grid],
    }
    res["trends"] = {
        "against_compensation": trend_k,
        "against_gain": trend_kp,
        "against_speed": trend_w,
        "rises_with_compensation": {
            str(w): bool((np.diff(trend_k[str(w)]) > 0).all()) for w in SPEEDS
        },
        "rises_with_gain": {str(w): bool((np.diff(trend_kp[str(w)]) > 0).all()) for w in SPEEDS},
        "falls_with_speed": bool((np.diff(trend_w["sigma"]) < 0).all()),
        "peak_against_compensation": {
            str(w): {
                "k": float(ks[int(np.argmax(trend_k[str(w)]))]),
                "sigma": float(np.max(trend_k[str(w)])),
                "sigma_at_k_max": float(trend_k[str(w)][-1]),
            }
            for w in SPEEDS
        },
        "lowest_unstable_k": {
            str(w): float(ks[int(np.argmax(np.array(trend_k[str(w)]) > 0))]) for w in SPEEDS
        },
        "sigma_at_k50": {str(w): sso_sigma(eigenvalues(0.5, w, KP_REF, KI_REF)) for w in SPEEDS},
    }

    # ---------------- 5. boundary: critical Kp against compensation
    k_grid = np.linspace(0.10, 0.95, 18)
    kp_scan = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 80)])
    bnd = {"k": k_grid.tolist(), "ratios": {}}
    for label, ratio in (("tuned", TUNE), ("10x", 10 * TUNE), ("100x", 100 * TUNE)):
        per_speed = {}
        for w_r in SPEEDS:
            hand, solver = [], []
            for k in k_grid:
                hand.append(first_crossing(lambda kp, k=k, w=w_r: -hand_kp_only(k, w, kp).r_net,
                                           kp_scan))  # fmt: skip
                solver.append(
                    first_crossing(
                        lambda kp, k=k, w=w_r, r=ratio: sso_sigma(eigenvalues(k, w, kp, r * kp)),
                        kp_scan[1:],
                    )
                )
            h, s = np.array(hand), np.array(solver)
            per_speed[str(w_r)] = {
                "kp_hand": hand,
                "kp_solver": solver,
                "max_rel_gap": float(np.nanmax(np.abs(h - s) / s)),
                "median_rel_gap": float(np.nanmedian(np.abs(h - s) / s)),
            }
        bnd["ratios"][label] = {"ki_over_kp": ratio, "by_speed": per_speed}
    res["boundary"] = bnd

    # ---------------- 6. time domain and energy budget
    cases = [(0.5, 0.75, 10.0), (0.3, 1.0, 10.0), (0.5, 1.2, 3.0), (0.7, 1.0, 30.0)]
    td, dt, eps = [], 1e-4, 1e-6
    for k, w_r, f_bw in cases:
        kp, ki = gains_from_bandwidth(D, f_bw)
        x0, p = M.equilibrium(k * D.x_e, w_r=w_r, kp=kp, ki=ki)
        md = modes(jacobian(M.rhs, x0, p))
        f = np.abs(md.freq_hz)
        cand = np.where((f > 2.0) & (f < 58.0) & (md.freq_hz > 0))[0]
        j = int(cand[np.argmax(md.sigma[cand])])
        v = md.right[:, j].real
        v = v / np.abs(v).max()
        t_end = float(np.clip(6.0 / abs(md.sigma[j]), 0.2, 1.0))
        _, traj = rk4(M.rhs, x0 + eps * torch.as_tensor(v), p, t_end, dt)
        probe = int(np.argmax(np.abs(v[:6])))
        y = (traj[::5, probe] - x0[probe]).numpy() / eps
        pole = nearest_pole(matrix_pencil(y, 5 * dt, 6), md.freq_hz[j])
        acc = energy.account(M, traj, x0, p, dt)
        fl = energy.flows(M, traj, x0, p, dt)
        s_mode = complex(md.sigma[j], -abs(md.eigenvalues[j].imag)) / D.wb + 1j
        r_exact = slip.dfig_impedance(s_mode, w_r=w_r, kp=kp, ki=ki, **MACH).real
        sr = hand_kp_only(k, w_r, kp)
        td.append(
            {
                "k": k, "w_r": w_r, "bandwidth_hz": f_bw, "kp": kp, "ki": ki,
                "equilibrium_residual": float(M.rhs(x0, p).abs().max()),
                "f_hz": float(md.freq_hz[j]),
                "sigma_eig": float(md.sigma[j]),
                "sigma_td": pole.real,
                "sigma_abs_err": abs(pole.real - float(md.sigma[j])),
                "sigma_hand_kp_only": sr.sigma_hand,
                "sigma_circuit_kp_only": sr.sigma_circuit,
                "r_machine_kp_only_jw": sr.r_machine,
                "r_machine_complex_frequency": r_exact,
                "r_machine_from_waveform": acc.r_effective,
                "waveform_vs_complex_rel": abs(acc.r_effective - r_exact) / abs(r_exact),
                "energy_from_shaft_over_to_network": fl.from_shaft / fl.to_network,
                "energy_from_converter_over_to_network": fl.from_converter / fl.to_network,
                "energy_copper_over_to_network": fl.copper / fl.to_network,
                "energy_stored_over_to_network": fl.stored / fl.to_network,
                "energy_closure": fl.closure,
            }
        )  # fmt: skip
    res["time_domain"] = {
        "cases": td,
        "max_sigma_abs_err": max(c["sigma_abs_err"] for c in td),
        "max_waveform_vs_complex_rel": max(c["waveform_vs_complex_rel"] for c in td),
        "max_energy_closure": max(abs(c["energy_closure"]) for c in td),
    }

    (OUT / "dfig.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, facecolor=SURFACE)
    ax = axes[0]
    ax.plot(kps, mech["r_machine"], color=C2, lw=2, label="Machine")
    ax.plot(kps, np.full_like(kps, D.r_e), color=C1, lw=2, label="Network")
    ax.plot(kps, mech["r_net"], color=C3, lw=2, label="Net")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Resistance at resonance (pu)")
    ax.set_title(
        "Current-loop gain acts as rotor resistance (50 % compensation, rotor at 0.75 pu)",
        loc="left", fontsize=11,
    )  # fmt: skip
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower left")
    ax = axes[1]
    ax.plot(kps, mech["slip"], color=C1, lw=2)
    ax.set_ylabel("Slip at resonance")
    ax = axes[2]
    ax.plot(kps, mech["sigma_solver"], color=C1, lw=2, label="Solver")
    ax.plot(kps, mech["sigma_kp_only"], color=C3, lw=2, ls=(0, (1, 1.5)),
            label="Circuit root, Kp as resistance")  # fmt: skip
    ax.plot(kps, mech["sigma_hand"], color=C2, lw=2, ls=(0, (4, 2)),
            label="Hand estimate (net resistance / reactance slope)")  # fmt: skip
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Growth rate (1/s)")
    ax.set_xlabel("Current-loop proportional gain Kp (pu)")
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower right")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_dfig_mechanism.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), facecolor=SURFACE)
    ax = axes[0]
    for w_r in SPEEDS:
        ax.plot(100 * ks, trend_k[str(w_r)], color=SPEED_COLOUR[w_r], lw=2,
                label=f"Rotor at {w_r:.2f} pu")  # fmt: skip
    ax.set_xlabel("Series compensation (%)")
    ax.set_ylabel("Growth rate (1/s)")
    ax.set_title("Against compensation", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=LABEL, loc="upper left")
    ax = axes[1]
    ax.plot(wr_grid, trend_w["sigma"], color=INK, lw=2)
    ax.set_xlabel("Rotor speed (pu)")
    ax.set_title("Against rotor speed (50 %)", loc="left", fontsize=10)
    ax = axes[2]
    for w_r in SPEEDS:
        ax.plot(kp_grid, trend_kp[str(w_r)], color=SPEED_COLOUR[w_r], lw=2)
    ax.set_xlabel("Gain Kp (pu)")
    ax.set_title("Against current-loop gain (50 %)", loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax)
    fig.suptitle("Doubly-fed machine: growth rate of the sub-synchronous mode", x=0.02,
                 ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig5_dfig_trends.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), sharey=True, facecolor=SURFACE)
    titles = {"tuned": f"Ki/Kp = {TUNE:.0f} 1/s (tuned)", "10x": f"Ki/Kp = {10 * TUNE:.0f} 1/s",
              "100x": f"Ki/Kp = {100 * TUNE:.0f} 1/s"}  # fmt: skip
    for ax, label in zip(axes, ("tuned", "10x", "100x"), strict=True):
        for w_r in SPEEDS:
            b = bnd["ratios"][label]["by_speed"][str(w_r)]
            ax.plot(100 * k_grid, b["kp_hand"], color=SPEED_COLOUR[w_r], lw=2,
                    label=f"Hand, rotor at {w_r:.2f} pu")  # fmt: skip
            ax.plot(100 * k_grid, b["kp_solver"], ls="none", marker="o", ms=5,
                    mfc=SPEED_COLOUR[w_r], mec=SURFACE, mew=1)  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel("Series compensation (%)")
        ax.set_title(titles[label], loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel("Gain Kp at the boundary (pu)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=LABEL, loc="upper right")
    fig.suptitle(
        "Largest stable gain: lines are the hand calculation (Kp as rotor resistance), "
        "dots the solver", x=0.02, ha="left", fontsize=11, color=INK,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig6_dfig_boundary.png", dpi=160)
    plt.close(fig)

    slim = {k: v for k, v in res.items() if k not in ("mechanism", "trends", "boundary")}
    print(json.dumps(slim, indent=1))
    t = res["trends"]
    print({k: t[k] for k in ("rises_with_compensation", "rises_with_gain", "falls_with_speed",
                             "sigma_at_k50")})  # fmt: skip
    print("r_machine per unit kp", mech["r_machine_per_unit_kp"], "slip", mech["slip"][0],
          mech["slip"][-1])  # fmt: skip
    for label, r in bnd["ratios"].items():
        for w, b in r["by_speed"].items():
            print(label, w, "max gap", round(b["max_rel_gap"], 4), "median",
                  round(b["median_rel_gap"], 4),
                  [f"{h:.4f}/{s:.4f}" for h, s in zip(b["kp_hand"][::4], b["kp_solver"][::4],
                                                      strict=True)])  # fmt: skip


if __name__ == "__main__":
    main()
