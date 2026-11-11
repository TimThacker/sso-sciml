"""Phase 2, step 4b: the network study confirmed in the time-domain model.

Writes docs/results/phase2/meshed.json and one figure. The numbers in the section
"Confirmation in the time-domain model" of docs/steps/05-network-and-impedance.md come
from this script.

Run with:  uv run scripts/phase2_meshed.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

from sso_sciml.physics import circuit as ct
from sso_sciml.physics.dfig import gains_from_bandwidth
from sso_sciml.physics.dfig_full import Controls, DFIGFull
from sso_sciml.physics.dfig_meshed import DFIGMeshed
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"

BASE = Controls()
D = DFIGFull().data
MACH = {key: getattr(D, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}
SPEEDS = (0.75, 1.0, 1.2)
SPEED_COLOUR = dict(zip(SPEEDS, (C1, C2, C3), strict=True))
OFF = {"pll_hz": 1e-3, "outer_hz": 1e-3, "x_f": 1e3, "dc_hz": 1e-3}
NO_PATH = 1e6  # parallel path this weak is absent
NETWORKS = [
    ("Plant impedance only", 0.010, 0.15, NO_PATH),
    ("Equal parallel path only", 0.0, 0.0, 1.0),
    ("Plant impedance + equal parallel path", 0.010, 0.15, 1.0),
    ("Plant impedance + parallel path at half the line impedance", 0.010, 0.15, 0.5),
]


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


def analyse(model, k: float, w_r: float, **over):
    x0, p = model.equilibrium(k * D.x_e, w_r=w_r, **over)
    md = modes(jacobian(model.rhs, x0, p))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return md, j, x0, p


def sigma(model, k, w_r, **over) -> float:
    md, j, _, _ = analyse(model, k, w_r, **over)
    return float(md.sigma[j])


def impedance_sigma(k, w_r, bw, r_t, x_t, par) -> float:
    kp, ki = gains_from_bandwidth(D, bw)
    z = ct.parallel(ct.branch(D.r_e, D.x_e, k * D.x_e), ct.branch(par * D.r_e, par * D.x_e))
    z = z + ct.branch(r_t, x_t)
    return ct.sso_growth(ct.modes(ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **MACH) + z))


def first_crossing(f, grid: np.ndarray) -> float:
    vals = np.array([f(v) for v in grid])
    idx = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if idx.size == 0:
        return float("nan")
    return float(brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-9))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    # ---------------- 1. radial limit of the new model
    full = DFIGFull()
    radial = DFIGMeshed(r_t=0.0, x_t=0.0, par_ratio=NO_PATH)
    lim = [
        abs(sigma(full, k, w) - sigma(radial, k, w))
        for k, w in ((0.3, 0.75), (0.5, 1.0), (0.7, 1.2))
    ]
    worst_res = 0.0
    for _, r_t, x_t, par in NETWORKS:
        m = DFIGMeshed(r_t=r_t, x_t=x_t, par_ratio=par)
        for w in SPEEDS:
            x0, p = m.equilibrium(0.5 * D.x_e, w_r=w)
            worst_res = max(worst_res, float(m.rhs(x0, p).abs().max()))
    res["radial_limit_max_abs_diff"] = max(lim)
    res["equilibrium_residual_max"] = worst_res

    # ---------------- 2. impedance model against the time-domain model, per network
    rows, worst = [], 0.0
    for label, r_t, x_t, par in NETWORKS:
        m_off = DFIGMeshed(controls=replace(BASE, **OFF), r_t=r_t, x_t=x_t, par_ratio=par)
        m_on = DFIGMeshed(r_t=r_t, x_t=x_t, par_ratio=par)
        m_noff = DFIGMeshed(controls=replace(BASE, ff=0.0), r_t=r_t, x_t=x_t, par_ratio=par)
        row = {"network": label}
        for w in SPEEDS:
            imp = impedance_sigma(0.5, w, BASE.rsc_hz, r_t, x_t, par)
            off = sigma(m_off, 0.5, w, h=1e6)
            worst = max(worst, abs(imp - off))
            row[str(w)] = {"impedance": imp, "state_space_controls_off": off,
                           "full_controls": sigma(m_on, 0.5, w),
                           "full_controls_no_feed_forward": sigma(m_noff, 0.5, w)}  # fmt: skip
        rows.append(row)
    res["growth_k50"] = rows
    res["impedance_vs_state_space_max_abs_diff"] = worst
    res["full_vs_impedance_max_abs_diff"] = max(
        abs(r[str(w)]["full_controls"] - r[str(w)]["impedance"]) for r in rows for w in SPEEDS
    )

    # ---------------- 3. reference network with full controls: boundary and tolerable bandwidth
    _, r_t, x_t, par = NETWORKS[2]
    ref = {}
    k_scan = np.linspace(0.05, 0.99, 24)
    bw_scan = np.geomspace(1.0, 150.0, 22)
    for w in SPEEDS:
        m = DFIGMeshed(r_t=r_t, x_t=x_t, par_ratio=par)
        ref[str(w)] = {
            "boundary_k_full": first_crossing(lambda k, m=m, w=w: sigma(m, k, w), k_scan),
            "boundary_k_impedance": first_crossing(
                lambda k, w=w: impedance_sigma(k, w, BASE.rsc_hz, r_t, x_t, par), k_scan
            ),
            "largest_stable_bw_full_hz": first_crossing(
                lambda f, w=w: sigma(
                    DFIGMeshed(controls=replace(BASE, rsc_hz=f), r_t=r_t, x_t=x_t, par_ratio=par),
                    0.5,
                    w,
                ),
                bw_scan,
            ),  # fmt: skip
            "largest_stable_bw_impedance_hz": first_crossing(
                lambda f, w=w: impedance_sigma(0.5, w, f, r_t, x_t, par), bw_scan
            ),
        }
    res["reference_network"] = ref

    # ---------------- 4. nonlinear run on the reference network
    m = DFIGMeshed(r_t=r_t, x_t=x_t, par_ratio=par)
    k_from, k_to, w_step, dt = 0.45, 0.60, 0.75, 1e-4
    x_a, p_a = m.equilibrium(k_from * D.x_e, w_r=w_step)
    p_b = {**p_a, "x_c": torch.as_tensor(k_to * D.x_e, dtype=torch.float64)}
    t, traj = rk4(m.rhs, x_a, p_b, 0.8, dt, save_every=5)
    md, j, x_b, _ = analyse(m, k_to, w_step)
    wave = m.outputs(traj, p_b)["p_t"].numpy()
    t = t.numpy()
    win = t > 0.15
    pole = nearest_pole(
        matrix_pencil(wave[win] - float(m.outputs(x_b, p_b)["p_t"]), 5 * dt, 10),
        abs(float(md.freq_hz[j])),
    )
    res["compensation_step"] = {
        "k_from": k_from, "k_to": k_to, "w_r": w_step,
        "sigma_eig_after": float(md.sigma[j]),
        "f_eig_after_hz": abs(float(md.freq_hz[j])),
        "sigma_from_power_waveform": pole.real,
        "f_from_power_waveform_hz": pole.imag / (2 * np.pi),
        "power_swing_first_0p1s_pu": float(np.ptp(wave[t < 0.1])),
        "power_swing_last_0p1s_pu": float(np.ptp(wave[t > 0.7])),
    }  # fmt: skip

    (OUT / "meshed.json").write_text(json.dumps(res, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0), sharey=True, facecolor=SURFACE)
    ypos = np.arange(len(rows))[::-1]
    for ax, w in zip(axes, SPEEDS, strict=True):
        ax.barh(ypos + 0.2, [r[str(w)]["impedance"] for r in rows], height=0.36,
                color=MUTED, label="Impedance model, current loop only")  # fmt: skip
        ax.barh(ypos - 0.2, [r[str(w)]["full_controls"] for r in rows], height=0.36,
                color=SPEED_COLOUR[w], label="Time-domain model, full controls")  # fmt: skip
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Rotor at {w:.2f} pu", loc="left", fontsize=10)
        ax.set_xlabel("Growth rate (1/s)")
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, ["Plant impedance", "Equal parallel path", "Both",
                              "Plant impedance,\nstronger parallel path"])  # fmt: skip
    fig.suptitle(
        "Grey: impedance model, current loop only. Colour: time-domain model, full controls. "
        "50 % compensation, 10 Hz current loop",
        x=0.02,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig12_meshed_confirmation.png", dpi=160)
    plt.close(fig)

    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
