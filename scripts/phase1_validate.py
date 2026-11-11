"""Phase 1 validation of the solver core on the IEEE First Benchmark Model.

Writes docs/results/phase1/results.json and three figures. Every number quoted in
docs/steps/01-solver-core-fbm.md comes from this script.

Run with:  uv run scripts/phase1_validate.py
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
from scipy.optimize import brentq

from sso_sciml.physics import slip, torsional
from sso_sciml.physics.fbm import FBM, GEN, N_ELECTRICAL, stack_cases
from sso_sciml.physics.shaft import torsional_modes
from sso_sciml.solver.integrate import reference, rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase1"
PUBLISHED_HZ = [15.71, 20.21, 25.55, 32.28, 47.46]

# Chart palette (fixed categorical order) and neutral inks.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color("#52514e")
    ax.yaxis.label.set_color("#52514e")
    ax.title.set_color(INK)


def state_matrix(model: FBM, x_c: float, **kw) -> tuple[np.ndarray, torch.Tensor, dict]:
    x0, p = model.equilibrium(x_c, **kw)
    return jacobian(model.rhs, x0, p).numpy(), x0, p


def pick_torsional(md, f_n: float) -> int:
    """Index of the eigenvalue belonging to the torsional mode near f_n."""
    f = md.freq_hz
    shaft = md.participation[N_ELECTRICAL:].sum(axis=0)
    cand = np.where((np.abs(f - f_n) < 1.5) & (f > 0))[0]
    return int(cand[np.argmax(shaft[cand])])


def pick_electrical(md) -> int:
    """Index of the sub-synchronous network mode (dq frame), the least shaft-like one."""
    f = md.freq_hz
    shaft = md.participation[N_ELECTRICAL:].sum(axis=0)
    cand = np.where((f > 1.0) & (f < 59.0))[0]
    return int(cand[np.argmin(shaft[cand])])


def locked_shaft_mode(a: np.ndarray) -> complex:
    """Sub-synchronous eigenvalue of the electrical subsystem with the rotor motion frozen."""
    lam = np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL])
    cand = lam[(lam.imag > 2 * np.pi) & (lam.imag < 2 * np.pi * 59)]
    return complex(cand[np.argmax(cand.real)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = FBM()
    dat = model.data
    h = np.array(dat.inertia_h)
    res: dict = {}

    # ---------------------------------------------------------------- 1. shaft alone
    f_tors, shapes = torsional_modes(h, np.array(dat.stiffness_k), dat.f0)
    res["torsional_hz"] = {
        "computed": [round(float(v), 4) for v in f_tors[1:]],
        "published": PUBLISHED_HZ,
        "max_abs_error_hz": float(np.max(np.abs(f_tors[1:] - PUBLISHED_HZ))),
        "generator_entry_of_mode_shape": [round(float(v), 4) for v in shapes[GEN, 1:]],
    }

    # ---------------------------------------------------------------- 2. equilibrium
    resid = [float(model.rhs(*model.equilibrium(xc)).abs().max()) for xc in (0.1, 0.371, 0.6)]
    res["equilibrium_max_residual"] = max(resid)

    # ---------------------------------------------------------------- 3. compensation sweep
    xcs = np.linspace(0.02, 0.62, 241)
    sweep = {k: [] for k in (
        "slip", "f_er", "r_machine", "r_net", "sigma_hand", "sigma_circuit", "sigma_locked",
        "f_locked", "sigma_elec_full",
    )}
    tors_eig = np.zeros((4, xcs.size))
    tors_pred = np.zeros((4, xcs.size))
    for j, xc in enumerate(xcs):
        a, _, _ = state_matrix(model, float(xc))
        md = modes(a)
        sr = slip.analyse(model.d, model.q, dat.r_e, dat.x_e, float(xc), dat.f0)
        lam_locked = locked_shaft_mode(a)
        sweep["slip"].append(sr.slip)
        sweep["f_er"].append(sr.f_er)
        sweep["r_machine"].append(float(sr.r_machine))
        sweep["r_net"].append(float(sr.r_net))
        sweep["sigma_hand"].append(float(sr.sigma_hand))
        sweep["sigma_circuit"].append(float(sr.sigma_circuit))
        sweep["sigma_locked"].append(lam_locked.real)
        sweep["f_locked"].append(lam_locked.imag / (2 * np.pi))
        sweep["sigma_elec_full"].append(float(md.sigma[pick_electrical(md)]))
        for n in range(4):
            tors_eig[n, j] = md.sigma[pick_torsional(md, f_tors[n + 1])]
            tors_pred[n, j] = torsional.predicted_growth(
                a, h, shapes[:, n + 1], f_tors[n + 1], dat.wb
            )
    sw = {k: np.array(v) for k, v in sweep.items()}

    rel = np.abs(sw["sigma_hand"] - sw["sigma_locked"]) / np.abs(sw["sigma_locked"])
    rel_c = np.abs(sw["sigma_circuit"] - sw["sigma_locked"]) / np.abs(sw["sigma_locked"])
    df = np.abs((dat.f0 - sw["f_er"]) - sw["f_locked"])
    res["induction_generator_effect"] = {
        "x_c_range": [float(xcs[0]), float(xcs[-1])],
        "slip_range": [float(sw["slip"].min()), float(sw["slip"].max())],
        "r_machine_range_pu": [float(sw["r_machine"].min()), float(sw["r_machine"].max())],
        "r_net_min_pu": float(sw["r_net"].min()),
        "self_excited_anywhere": bool((sw["r_net"] < 0).any()),
        "hand_vs_locked_eig_max_rel_err": float(rel.max()),
        "circuit_root_vs_locked_eig_max_rel_err": float(rel_c.max()),
        "resonance_freq_max_abs_err_hz": float(df.max()),
    }

    # Where does the slip effect win by itself? Lower the network resistance until R_net = 0.
    xc0 = 0.371

    def r_net_of(r_e: float) -> float:
        return float(slip.analyse(model.d, model.q, r_e, dat.x_e, xc0, dat.f0).r_net)

    def sigma_locked_of(r_e: float) -> float:
        a, _, _ = state_matrix(model, xc0, r_e=r_e)
        return locked_shaft_mode(a).real

    r_hand = brentq(r_net_of, 1e-4, 0.03)
    r_eig = brentq(sigma_locked_of, 1e-4, 0.03)
    res["self_excitation_threshold"] = {
        "x_c": xc0,
        "benchmark_r_e_pu": dat.r_e,
        "r_e_threshold_hand_pu": r_hand,
        "r_e_threshold_locked_eig_pu": r_eig,
        "rel_diff": abs(r_hand - r_eig) / r_eig,
    }

    # Torsional interaction: peak location and height, eigenvalue against damping-torque estimate.
    ti = []
    for n in range(4):
        j = int(np.argmax(tors_eig[n]))
        jp = int(np.argmax(tors_pred[n]))
        j_align = int(np.argmin(np.abs((dat.f0 - sw["f_er"]) - f_tors[n + 1])))
        ti.append({
            "mode": n + 1,
            "f_n_hz": round(float(f_tors[n + 1]), 3),
            "x_c_at_peak_eig": round(float(xcs[j]), 4),
            "x_c_at_peak_damping_torque": round(float(xcs[jp]), 4),
            "x_c_where_f0_minus_fer_equals_fn": round(float(xcs[j_align]), 4),
            "peak_sigma_eig": round(float(tors_eig[n, j]), 4),
            "peak_sigma_damping_torque": round(float(tors_pred[n, jp]), 4),
        })
    away = np.abs(tors_eig) < 0.2  # weak-coupling region, where first-order theory should hold
    res["torsional_interaction"] = {
        "modes": ti,
        "weak_coupling_max_abs_err_per_s": float(np.abs(tors_eig - tors_pred)[away].max()),
        "sign_agreement_fraction": float(
            np.mean(np.sign(tors_eig[np.abs(tors_eig) > 0.01])
                    == np.sign(tors_pred[np.abs(tors_eig) > 0.01]))
        ),
    }

    # ---------------------------------------------------------------- 4. time domain
    eps, t_end, dt, keep = 1e-6, 2.0, 1e-4, 10
    cases, meta = [], []
    for xc in (0.20, 0.287, 0.371, 0.45, 0.55):
        a, x0, p = state_matrix(model, xc)
        md = modes(a)
        k_el = pick_electrical(md)
        tors_idx = [pick_torsional(md, f_tors[n + 1]) for n in range(4)]
        k_tor = tors_idx[int(np.argmax(md.sigma[tors_idx]))]
        for label, k in (("network", k_el), ("torsional", k_tor)):
            v = md.right[:, k].real
            v = v / np.abs(v).max()
            cases.append((x0 + eps * torch.as_tensor(v), p))
            meta.append({"x_c": xc, "kind": label, "lam": complex(md.eigenvalues[k]),
                         "probe": int(np.argmax(np.abs(v))), "x_eq": x0})
    xb, pb = stack_cases(cases)
    t0 = time.perf_counter()
    t, traj = rk4(model.rhs, xb, pb, t_end, dt, save_every=keep)
    rk4_seconds = time.perf_counter() - t0
    rows = []
    for i, m in enumerate(meta):
        y = (traj[:, i, m["probe"]] - m["x_eq"][m["probe"]]).numpy() / eps
        pole = nearest_pole(matrix_pencil(y, dt * keep, 4), abs(m["lam"].imag) / (2 * np.pi))
        lam = complex(m["lam"].real, abs(m["lam"].imag))
        rows.append({
            "x_c": m["x_c"], "mode": m["kind"],
            "f_eig_hz": round(lam.imag / (2 * np.pi), 4),
            "f_sim_hz": round(pole.imag / (2 * np.pi), 4),
            "sigma_eig": round(lam.real, 5),
            "sigma_sim": round(pole.real, 5),
            "sigma_abs_err": abs(pole.real - lam.real),
            "sigma_rel_err": abs(pole.real - lam.real) / abs(lam.real),
        })
    res["time_domain_vs_eigenvalue"] = {
        "dt_s": dt, "t_end_s": t_end, "perturbation": eps, "rk4_seconds_cpu": rk4_seconds,
        "cases": rows,
        "max_sigma_rel_err": max(r["sigma_rel_err"] for r in rows),
        "max_freq_abs_err_hz": max(abs(r["f_eig_hz"] - r["f_sim_hz"]) for r in rows),
    }

    # RK4 against adaptive implicit references, large disturbance (nonlinear regime).
    a, x0, p = state_matrix(model, 0.371)
    xk = x0.clone()
    xk[N_ELECTRICAL + 6 + GEN] += 0.01  # 1 % generator speed kick
    t_eval = np.linspace(0.0, 0.5, 501)
    ref = reference(model.rhs, xk, p, t_eval, method="Radau", rtol=1e-11, atol=1e-13)
    ref2 = reference(model.rhs, xk, p, t_eval, method="DOP853", rtol=1e-12, atol=1e-14)
    conv = {}
    for step in (5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
        _, tr = rk4(model.rhs, xk, p, 0.5, step, save_every=int(round(1e-3 / step)))
        conv[step] = float(np.abs(tr.numpy() - ref2).max())
    steps = sorted(conv, reverse=True)
    order = [float(np.log2(conv[steps[i]] / conv[steps[i + 1]])) for i in range(len(steps) - 1)]
    res["integrator_check"] = {
        "disturbance": "1 % generator speed step, x_c = 0.371, 0.5 s",
        "radau_vs_dop853_max_abs": float(np.abs(ref - ref2).max()),
        "rk4_max_abs_err_vs_dop853": {f"{k:.3g}": v for k, v in conv.items()},
        "observed_order": order,
    }

    # ---------------------------------------------------------------- 5. batched RK4, CPU vs GPU
    if torch.cuda.is_available():
        bench = {}
        gen = np.random.default_rng(0)
        for n_batch in (256, 4096, 32768):
            xcb = gen.uniform(0.05, 0.6, n_batch)
            big = [model.equilibrium(float(v)) for v in xcb]
            timing = {}
            finals = {}
            for dev in ("cpu", "cuda"):
                xs, ps = stack_cases(big, device=dev)
                xs[:, N_ELECTRICAL + 6 + GEN] += 0.01
                rk4(model.rhs, xs, ps, 10 * 2e-4, 2e-4, save_every=10)  # warm-up
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, tr = rk4(model.rhs, xs, ps, 0.1, 2e-4, save_every=500)
                if dev == "cuda":
                    torch.cuda.synchronize()
                timing[dev] = time.perf_counter() - t0
                finals[dev] = tr[-1].cpu()
            bench[str(n_batch)] = {
                "cpu_s": timing["cpu"], "cuda_s": timing["cuda"],
                "speedup": timing["cpu"] / timing["cuda"],
                "max_abs_diff": float((finals["cpu"] - finals["cuda"]).abs().max()),
            }
        res["batched_rk4_500_steps"] = bench

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

    # Figure 1: the slip mechanism. Three quantities with different units: three panels.
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True, facecolor=SURFACE)
    ax = axes[0]
    ax.plot(xcs, sw["slip"], color=C1, lw=2)
    ax.set_ylabel("slip at resonance")
    ax.set_title("Rotor slip at the electrical resonance", loc="left", fontsize=11)
    ax = axes[1]
    ax.plot(xcs, np.full_like(xcs, dat.r_e), color=C1, lw=2, label="network resistance")
    ax.plot(xcs, -sw["r_machine"], color=C2, lw=2, label="machine negative resistance (magnitude)")
    ax.set_ylabel("resistance (pu)")
    ax.set_ylim(0, 0.035)
    ax.set_title("Resistance the machine cancels, against what the network supplies",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor="#52514e", loc="center left")
    ax = axes[2]
    ax.plot(xcs, sw["sigma_locked"], color=C1, lw=2, label="eigenvalue, shaft locked")
    ax.plot(xcs[::12], sw["sigma_hand"][::12], "o", color=C2, ms=5, mec=SURFACE, mew=1,
            label="hand estimate from net resistance")
    ax.set_ylabel("growth rate (1/s)")
    ax.set_xlabel("series capacitor reactance $X_C$ (pu)")
    ax.set_title("Decay rate of the sub-synchronous network mode", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor="#52514e")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_slip_mechanism.png", dpi=160)
    plt.close(fig)

    # Figure 2: torsional interaction, four modes as small multiples sharing both axes.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, facecolor=SURFACE)
    for n, ax in enumerate(axes.flat):
        ax.plot(xcs, tors_eig[n], color=C1, lw=2, label="eigenvalue, full model")
        ax.plot(xcs[::4], tors_pred[n][::4], "o", color=C2, ms=4, mec=SURFACE, mew=0.8,
                label="damping-torque estimate")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_title(f"Mode {n + 1}: {f_tors[n + 1]:.2f} Hz", loc="left", fontsize=11)
        style(ax)
        if n >= 2:
            ax.set_xlabel("series capacitor reactance $X_C$ (pu)")
        if n % 2 == 0:
            ax.set_ylabel("growth rate (1/s)")
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor="#52514e", loc="lower right")
    fig.suptitle("Torsional mode growth rate against compensation", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_torsional_interaction.png", dpi=160)
    plt.close(fig)

    # Figure 3: time-domain growth at the benchmark compensation against the eigenvalue envelope.
    i = next(k for k, m in enumerate(meta) if m["x_c"] == 0.371 and m["kind"] == "torsional")
    m = meta[i]
    y = (traj[:, i, m["probe"]] - m["x_eq"][m["probe"]]).numpy() / eps
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    ax.plot(t.numpy(), y, color=C1, lw=1.2, label="nonlinear simulation (RK4)")
    env = np.abs(y[0]) * np.exp(m["lam"].real * t.numpy())
    ax.plot(t.numpy(), env, color=C2, lw=2, label="envelope from eigenvalue")
    ax.plot(t.numpy(), -env, color=C2, lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deviation / initial perturbation")
    ax.set_title(f"Torsional mode at $X_C$ = 0.371 pu: {abs(m['lam'].imag) / 2 / np.pi:.2f} Hz, "
                 f"growth {m['lam'].real:+.3f} 1/s", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor="#52514e", loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_time_domain_growth.png", dpi=160)
    plt.close(fig)

    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
