"""Phase 2, step 1: fixed-speed induction generator on a series-compensated line.

Writes docs/results/phase2/induction.json and three figures. Every number quoted in
docs/steps/02-induction-generator.md comes from this script.

Run with:  uv run scripts/phase2_induction.py
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
from sso_sciml.physics.induction import N_ELECTRICAL, W_R, InductionGenerator
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"
P_GEN = 0.9
K_MIN = 0.30  # below about 0.25 the line cannot carry P_GEN (pull-out)
K_MAX = 0.99  # highest compensation level searched, as a fraction of the line reactance

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LABEL = "#52514e"


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


M = InductionGenerator()
D = M.data
MACHINE_KEYS = ("r_s", "x_ls", "x_m", "r_r", "x_lr")


def hand(k: float, w_r: float, **ov) -> slip.SlipResult:
    """Slip calculation at compensation level k = X_C / X_E and rotor speed w_r."""
    q = {key: ov.get(key, getattr(D, key)) for key in (*MACHINE_KEYS, "r_e", "x_e")}
    return slip.analyse_induction(
        q["r_s"],
        q["x_ls"],
        q["x_m"],
        q["r_r"],
        q["x_lr"],
        q["r_e"],
        q["x_e"],
        k * q["x_e"],
        D.f0,
        w_r,
    )


def full(k: float, **ov) -> tuple[np.ndarray, torch.Tensor, dict]:
    """State matrix at the operating point that delivers P_GEN to the infinite bus."""
    x0, p = M.equilibrium(k * ov.get("x_e", D.x_e), p_gen=P_GEN, **ov)
    return jacobian(M.rhs, x0, p).numpy(), x0, p


def network_mode(lam: np.ndarray, sr: slip.SlipResult) -> complex:
    """The eigenvalue closest to the circuit root, moved into the synchronous frame."""
    target = complex(sr.sigma_circuit, 2 * np.pi * (D.f0 - sr.f_circuit))
    return complex(lam[np.argmin(np.abs(lam - target))])


def locked_matrix(k: float, w_r: float, **ov) -> np.ndarray:
    """Electrical state matrix with the rotor speed imposed. It is linear, so any state does."""
    p = M.params(k * ov.get("x_e", D.x_e), **ov)
    x = torch.zeros(7, dtype=torch.float64)
    x[W_R] = w_r
    return jacobian(M.rhs, x, p).numpy()[:N_ELECTRICAL, :N_ELECTRICAL]


def crossing(f, lo: float, hi: float) -> float:
    """Root of f in [lo, hi], or nan when f does not change sign there."""
    try:
        a, b = f(lo), f(hi)
    except ValueError:
        return float("nan")
    if not np.isfinite(a) or not np.isfinite(b) or a * b > 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"data": {"r_e": D.r_e, "x_e": D.x_e, "p_gen": P_GEN, "plant_mva": D.plant_mva}}

    # ---------------- 1. steady state
    worst = 0.0
    for k in np.linspace(K_MIN, K_MAX, 9):
        x0, p = M.equilibrium(k * D.x_e, p_gen=P_GEN)
        worst = max(worst, float(M.rhs(x0, p).abs().max()))
        i = complex(M.currents(x0, p)["i_D"], M.currents(x0, p)["i_Q"])
        assert abs((float(p["v_b"]) * np.conj(i)).real - P_GEN) < 1e-10
    res["equilibrium_residual_max"] = worst

    # ---------------- 2. mechanism sweep
    ks = np.linspace(K_MIN, K_MAX, 80)
    sweep = {
        key: []
        for key in (
            "k w_r f_er slip r_machine r_net sigma_hand sigma_circuit sigma_locked "
            "f_locked_err sigma_full sigma_max"
        ).split()
    }
    for k in ks:
        a, x0, _ = full(k)
        w_r = float(x0[W_R])
        sr = hand(k, w_r)
        lock = network_mode(np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL]), sr)
        lam = np.linalg.eigvals(a)
        net = network_mode(lam, sr)
        row = {
            "k": k,
            "w_r": w_r,
            "f_er": sr.f_er,
            "slip": sr.slip,
            "r_machine": sr.r_machine,
            "r_net": sr.r_net,
            "sigma_hand": sr.sigma_hand,
            "sigma_circuit": sr.sigma_circuit,
            "sigma_locked": lock.real,
            "f_locked_err": abs(lock.imag / (2 * np.pi) - (D.f0 - sr.f_circuit)),
            "sigma_full": net.real,
            "sigma_max": float(lam.real.max()),
        }
        for key, v in row.items():
            sweep[key].append(float(v))
    sw = {key: np.array(v) for key, v in sweep.items()}
    res["sweep"] = {
        "k_range": [float(ks[0]), float(ks[-1])],
        "slip_range": [float(sw["slip"].min()), float(sw["slip"].max())],
        "r_machine_range": [float(sw["r_machine"].min()), float(sw["r_machine"].max())],
        "r_net_min": float(sw["r_net"].min()),
        "k_where_r_machine_turns_negative": crossing(
            lambda k: hand(k, float(full(k)[1][W_R])).r_machine, K_MIN, K_MAX
        ),
        "circuit_root_vs_locked_eig_max_abs_1_per_s": float(
            np.abs(sw["sigma_circuit"] - sw["sigma_locked"]).max()
        ),
        "circuit_root_vs_locked_eig_max_abs_hz": float(sw["f_locked_err"].max()),
        "hand_vs_locked_max_abs_1_per_s": float(
            np.abs(sw["sigma_hand"] - sw["sigma_locked"]).max()
        ),
        "hand_vs_locked_max_rel": float(
            np.abs((sw["sigma_hand"] - sw["sigma_locked"]) / sw["sigma_locked"]).max()
        ),
        "locked_vs_full_max_abs_1_per_s": float(
            np.abs(sw["sigma_locked"] - sw["sigma_full"]).max()
        ),
        "sigma_full_at_k_max": float(sw["sigma_full"][-1]),
        "locked_vs_full_largest_at_k": float(
            sw["k"][np.argmax(np.abs(sw["sigma_locked"] - sw["sigma_full"]))]
        ),
        "any_unstable": bool((sw["sigma_max"] > 0).any()),
        "unstable_mode_other_than_network": bool(
            ((sw["sigma_max"] > 0) & (sw["sigma_full"] <= 0)).any()
        ),
    }

    # ---------------- 3. boundary: k against rotor resistance
    def k_hand_at_power(**ov):
        return crossing(lambda k: hand(k, float(full(k, **ov)[1][W_R]), **ov).r_net, K_MIN, K_MAX)

    def k_full(**ov):
        def f(k):
            a, x0, _ = full(k, **ov)
            return network_mode(np.linalg.eigvals(a), hand(k, float(x0[W_R]), **ov)).real

        return crossing(f, K_MIN, K_MAX)

    def k_locked_at_power(**ov):
        def f(k):
            a, x0, _ = full(k, **ov)
            lam = np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL])
            return network_mode(lam, hand(k, float(x0[W_R]), **ov)).real

        return crossing(f, K_MIN, K_MAX)

    rr_grid = np.linspace(0.004, 0.03, 27)
    b_rr = {"r_r": rr_grid.tolist(), "k_hand": [], "k_locked": [], "k_full": []}
    for rr in rr_grid:
        b_rr["k_hand"].append(k_hand_at_power(r_r=float(rr)))
        b_rr["k_locked"].append(k_locked_at_power(r_r=float(rr)))
        b_rr["k_full"].append(k_full(r_r=float(rr)))
    b_rr["max_gap_hand_vs_locked_k"] = float(
        np.nanmax(np.abs(np.array(b_rr["k_hand"]) - np.array(b_rr["k_locked"])))
    )
    b_rr["base_machine"] = {"r_r": D.r_r, "k_hand": k_hand_at_power(), "k_full": k_full()}
    gap_rr = np.abs(np.array(b_rr["k_hand"]) - np.array(b_rr["k_full"]))
    b_rr["max_gap_k"] = float(np.nanmax(gap_rr))
    b_rr["r_r_min_with_boundary"] = float(rr_grid[np.isfinite(b_rr["k_hand"])].min())
    res["boundary_rotor_resistance"] = b_rr

    # ---------------- 4. boundary: k against rotor speed (imposed)
    def k_hand_at_speed(w_r, **ov):
        def f(k):
            try:
                return hand(k, w_r, **ov).r_net
            except ValueError:  # no resonance below rotor speed: slip is positive, no negative R
                return 1.0

        return scan_crossing(f)

    def k_locked(w_r, **ov):
        return scan_crossing(lambda k: -np.linalg.eigvals(locked_matrix(k, w_r, **ov)).real.max())

    def scan_crossing(f):
        """Lowest k at which f turns negative, refined by bisection."""
        grid = np.linspace(0.05, K_MAX, 95)
        vals = np.array([f(k) for k in grid])
        idx = np.where((vals[:-1] > 0) & (vals[1:] <= 0))[0]
        if idx.size == 0:
            return float("nan")
        return float(brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-10))

    wr_grid = np.linspace(0.70, 1.30, 31)
    b_wr = {"w_r": wr_grid.tolist(), "k_hand": [], "k_locked": []}
    for w in wr_grid:
        b_wr["k_hand"].append(k_hand_at_speed(float(w)))
        b_wr["k_locked"].append(k_locked(float(w)))
    gap_wr = np.abs(np.array(b_wr["k_hand"]) - np.array(b_wr["k_locked"]))
    b_wr["max_gap_k"] = float(np.nanmax(gap_wr))
    b_wr["highest_speed_with_boundary"] = float(wr_grid[np.isfinite(b_wr["k_hand"])].max())
    b_wr["same_points_have_a_boundary"] = bool(
        (np.isfinite(b_wr["k_hand"]) == np.isfinite(b_wr["k_locked"])).all()
    )
    res["boundary_rotor_speed"] = b_wr

    # ---------------- 5. time domain against eigenvalues
    cases = [(0.5, {}), (0.8, {}), (0.95, {}), (0.85, {"r_r": 0.012}), (0.95, {"r_r": 0.012})]
    td = []
    dt, t_end, eps = 1e-4, 1.0, 1e-6
    for k, ov in cases:
        a, x0, p = full(k, **ov)
        md = modes(a)
        sr = hand(k, float(x0[W_R]), **ov)
        target = complex(sr.sigma_circuit, 2 * np.pi * (D.f0 - sr.f_circuit))
        j = int(np.argmin(np.abs(md.eigenvalues - target)))
        v = md.right[:, j].real
        v = v / np.abs(v).max()
        probe = int(np.argmax(np.abs(v[:N_ELECTRICAL])))
        _, traj = rk4(M.rhs, x0 + eps * torch.as_tensor(v), p, t_end, dt, save_every=5)
        y = (traj[:, probe] - x0[probe]).numpy() / eps
        pole = nearest_pole(matrix_pencil(y, 5 * dt, 6), md.freq_hz[j])
        td.append(
            {
                "k": k,
                "r_r": ov.get("r_r", D.r_r),
                "f_hz": float(md.freq_hz[j]),
                "sigma_eig": float(md.sigma[j]),
                "sigma_td": pole.real,
                "abs_err_1_per_s": abs(pole.real - float(md.sigma[j])),
                "f_err_hz": abs(pole.imag / (2 * np.pi) - float(md.freq_hz[j])),
            }
        )
    res["time_domain"] = {
        "cases": td,
        "max_sigma_abs_err": max(c["abs_err_1_per_s"] for c in td),
        "max_f_err_hz": max(c["f_err_hz"] for c in td),
    }

    # ---------------- 6. energy accounting
    def marginal_r_e(k):
        def f(r_e):
            a, x0, _ = full(k, r_e=r_e)
            return network_mode(np.linalg.eigvals(a), hand(k, float(x0[W_R]), r_e=r_e)).real

        return brentq(f, 1e-4, D.r_e, xtol=1e-14)

    r_e_marg = marginal_r_e(0.8)
    e_cases = [
        ("decaying", 0.8, {}),
        ("sustained", 0.8, {"r_e": r_e_marg}),
        ("growing", 0.95, {"r_r": 0.012}),
    ]
    en, en_plot = [], []
    t_end = 0.5

    def one_account(k, ov):
        a, x0, p = full(k, **ov)
        md = modes(a)
        w_r = float(x0[W_R])
        sr = hand(k, w_r, **ov)
        target = complex(sr.sigma_circuit, 2 * np.pi * (D.f0 - sr.f_circuit))
        j = int(np.argmin(np.abs(md.eigenvalues - target)))
        v = md.right[:, j].real
        v = v / np.abs(v).max()
        _, traj = rk4(M.rhs, x0 + eps * torch.as_tensor(v), p, t_end, dt)
        acc = energy.account(M, traj, x0, p, dt)
        mach = {key: ov.get(key, getattr(D, key)) for key in MACHINE_KEYS}
        # Machine impedance at the mode's own complex frequency (growth included).
        s_mode = complex(md.sigma[j], -abs(md.eigenvalues[j].imag)) / D.wb + 1j
        r_exact = slip.induction_impedance(s_mode, w_r=w_r, **mach).real
        return acc, sr, float(md.sigma[j]), r_exact, float(p["r_e"]), mach["r_r"]

    for name, k, ov in e_cases:
        acc, sr, sigma, r_exact, r_e, r_r = one_account(k, ov)
        # Same case with the rotor speed held by a very large inertia.
        acc_h, _, sigma_h, r_exact_h, _, _ = one_account(k, {**ov, "h": 1e9})
        en.append(
            {
                "case": name,
                "k": k,
                "r_e": r_e,
                "r_r": r_r,
                "sigma_1_per_s": sigma,
                "f_er_hz": sr.f_er,
                "slip": sr.slip,
                "r_machine_hand": sr.r_machine,
                "r_machine_at_complex_frequency": r_exact,
                "r_machine_from_waveform": acc.r_effective,
                "waveform_vs_hand_rel": abs(acc.r_effective - sr.r_machine) / abs(sr.r_machine),
                "waveform_vs_complex_rel": abs(acc.r_effective - r_exact) / abs(r_exact),
                "energy_from_machine_over_network_loss": float(
                    acc.e_machine[-1] / acc.e_network_loss[-1]
                ),
                "speed_held": {
                    "sigma_1_per_s": sigma_h,
                    "r_machine_at_complex_frequency": r_exact_h,
                    "r_machine_from_waveform": acc_h.r_effective,
                    "waveform_vs_complex_rel": abs(acc_h.r_effective - r_exact_h) / abs(r_exact_h),
                },
            }
        )
        en_plot.append((name, acc, sr.r_machine))
    res["energy"] = {
        "marginal_r_e": r_e_marg,
        "cases": en,
        "max_waveform_vs_hand_rel": max(c["waveform_vs_hand_rel"] for c in en),
        "max_waveform_vs_complex_rel": max(c["waveform_vs_complex_rel"] for c in en),
        "max_waveform_vs_complex_rel_speed_held": max(
            c["speed_held"]["waveform_vs_complex_rel"] for c in en
        ),
    }

    (OUT / "induction.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    pct = 100 * sw["k"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True, facecolor=SURFACE)
    ax = axes[0]
    ax.plot(pct, sw["slip"], color=C1, lw=2)
    ax.set_ylabel("Slip at resonance")
    ax.set_title(
        "Induction generator: slip, resistance and growth rate against compensation",
        loc="left",
        fontsize=11,
    )
    ax = axes[1]
    ax.plot(pct, sw["r_machine"], color=C2, lw=2, label="Machine")
    ax.plot(pct, np.full_like(pct, D.r_e), color=C1, lw=2, label="Network")
    ax.plot(pct, sw["r_net"], color=C3, lw=2, label="Net")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Resistance at resonance (pu)")
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower left")
    ax = axes[2]
    ax.plot(pct, sw["sigma_full"], color=C1, lw=2, label="Full model eigenvalue")
    ax.plot(pct, sw["sigma_hand"], color=C2, lw=2, ls=(0, (4, 2)), label="Hand estimate")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Growth rate (1/s)")
    ax.set_xlabel("Series compensation (% of line reactance)")
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_induction_mechanism.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True, facecolor=SURFACE)
    ax = axes[0]
    rr = np.array(b_rr["r_r"])
    ax.plot(rr, 100 * np.array(b_rr["k_hand"]), color=C2, lw=2, label="Hand: net resistance = 0")
    ax.plot(
        rr,
        100 * np.array(b_rr["k_full"]),
        ls="none",
        marker="o",
        ms=5,
        mfc=C1,
        mec=SURFACE,
        mew=1,
        label="Solver: growth rate = 0",
    )
    ax.axvline(D.r_r, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(D.r_r + 0.0004, 22, "base machine", color=MUTED, fontsize=8)
    ax.set_xlabel("Rotor resistance (pu)")
    ax.set_ylabel("Compensation at the boundary (%)")
    ax.set_title("At 0.9 pu power, speed free", loc="left", fontsize=10)
    ax.text(0.022, 62, "self-excited", color=LABEL, fontsize=9, ha="center")
    ax.text(0.009, 40, "stable", color=LABEL, fontsize=9, ha="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper right")
    ax = axes[1]
    wr = np.array(b_wr["w_r"])
    ax.plot(wr, 100 * np.array(b_wr["k_hand"]), color=C2, lw=2)
    ax.plot(
        wr,
        100 * np.array(b_wr["k_locked"]),
        ls="none",
        marker="o",
        ms=5,
        mfc=C1,
        mec=SURFACE,
        mew=1,
    )
    ax.set_xlabel("Rotor speed, imposed (pu)")
    ax.set_title("Base machine, speed held", loc="left", fontsize=10)
    ax.text(0.78, 90, "self-excited", color=LABEL, fontsize=9, ha="center")
    ax.text(0.92, 45, "stable", color=LABEL, fontsize=9, ha="center")
    ax.set_ylim(20, 100)
    for ax in axes:
        style(ax)
    fig.suptitle(
        "Self-excitation boundary: hand calculation against solver",
        x=0.02,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_induction_boundary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), facecolor=SURFACE)
    for ax, (name, acc, r_hand) in zip(axes, en_plot, strict=True):
        scale = 1e12
        ax.plot(acc.t, scale * acc.e_machine, color=C1, lw=2, label="From the waveform")
        ax.plot(
            acc.t,
            scale * (-r_hand) * acc.i_squared,
            color=C2,
            lw=2,
            ls=(0, (4, 2)),
            label="Slip calculation",
        )
        ax.plot(acc.t, scale * acc.e_network_loss, color=C3, lw=2, label="Lost in the network")
        ax.set_title(name.capitalize(), loc="left", fontsize=10)
        ax.set_xlabel("Time (s)")
        style(ax)
    axes[0].set_ylabel("Energy (1e-12 pu·s)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=LABEL, loc="lower right")
    fig.suptitle(
        "Energy the machine feeds into the oscillation", x=0.02, ha="left", fontsize=11, color=INK
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_induction_energy.png", dpi=160)
    plt.close(fig)

    slim = {
        k: v
        for k, v in res.items()
        if k not in ("boundary_rotor_resistance", "boundary_rotor_speed")
    }
    print(json.dumps(slim, indent=2))
    for name, b, key in (("r_r", b_rr, "k_full"), ("w_r", b_wr, "k_locked")):
        print(name, "max gap", b["max_gap_k"])
        for a_, h_, s_ in zip(b[name], b["k_hand"], b[key], strict=True):
            print(f"  {a_:.4f}  hand {h_:.5f}  solver {s_:.5f}")


if __name__ == "__main__":
    main()
