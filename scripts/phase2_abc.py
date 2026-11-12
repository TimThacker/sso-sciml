"""Phase 2, step 6: a second time-domain implementation in phase coordinates.

Compares `physics/dfig_abc.py` (phase a, b, c quantities, trapezoidal rule, NumPy) with
`physics/dfig.py` (D and Q quantities, RK4, PyTorch) on the same event.

Writes docs/results/phase2/abc.json and one figure. Every number in
docs/steps/07-phase-coordinates.md comes from this script.

Run with:  uv run scripts/phase2_abc.py
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

from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.dfig_abc import PhaseModel, trapezoidal
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"

K_FROM, K_TO = 0.40, 0.50
T_END = 0.6
SAVE = 1e-3  # s between stored samples
CASES = ((0.75, 10.0), (1.0, 10.0), (1.2, 10.0), (1.0, 3.0))  # rotor speed (pu), loop (Hz)


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


def setup(w_r: float, bw: float):
    """Steady state before the step, parameters after it, and the mode after it."""
    m = DFIG()
    kp, ki = gains_from_bandwidth(m.data, bw)
    x_a, p_a = m.equilibrium(K_FROM * m.data.x_e, w_r=w_r, kp=kp, ki=ki)
    p_b = {**p_a, "x_c": p_a["x_c"] * (K_TO / K_FROM)}
    md = modes(jacobian(m.rhs, x_a, p_b))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return m, x_a, p_b, md.eigenvalues[j]


def run_phase(m, x_a, p_b, dt: float):
    pm = PhaseModel.from_params(p_b, m.data.wb)
    t, x = trapezoidal(pm, PhaseModel.from_dq_state(x_a.numpy()), T_END, dt, int(round(SAVE / dt)))
    return pm, t, x


def phase_growth(pm, t, x, lam) -> complex:
    """Sub-synchronous pole from the phase a stator current, which also holds 60 Hz."""
    f_sub = 60.0 - abs(lam.imag) / (2 * np.pi)
    wave = pm.stator_current(x)[:, 0]
    return nearest_pole(matrix_pencil(wave[t > 0.1], SAVE, 12), f_sub)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"event": {"k_from": K_FROM, "k_to": K_TO, "t_end_s": T_END}}

    # ---------------- 1. the same event in both implementations
    rows, keep = [], None
    for w_r, bw in CASES:
        m, x_a, p_b, lam = setup(w_r, bw)
        t_ref, ref = rk4(m.rhs, x_a, p_b, T_END, 2.5e-5, save_every=40)
        ref = ref.numpy()
        pm, t, x = run_phase(m, x_a, p_b, 2.5e-5)
        back = pm.to_dq_state(x, t)
        pole = phase_growth(pm, t, x, lam)
        swing = np.abs(ref - ref[0]).max()
        rows.append({
            "w_r": w_r, "loop_hz": bw,
            "sigma_eigenvalue": float(lam.real),
            "f_dq_hz": abs(float(lam.imag)) / (2 * np.pi),
            "f_phase_expected_hz": 60.0 - abs(float(lam.imag)) / (2 * np.pi),
            "sigma_phase_waveform": pole.real,
            "f_phase_waveform_hz": pole.imag / (2 * np.pi),
            "max_state_difference_pu": float(np.abs(back[:, :6] - ref[:, :6]).max()),
            "largest_excursion_pu": float(swing),
            "relative_state_difference": float(np.abs(back[:, :6] - ref[:, :6]).max() / swing),
            "zero_sequence_max": float(np.abs(x[:, 0:9].reshape(-1, 3, 3).sum(axis=2)).max()),
        })  # fmt: skip
        if (w_r, bw) == (1.0, 10.0):
            keep = (m, x_a, p_b, lam, t_ref.numpy(), ref, t, x, back, pm)
    res["cases"] = rows
    res["max_relative_state_difference"] = max(r["relative_state_difference"] for r in rows)
    res["max_sigma_difference"] = max(
        abs(r["sigma_phase_waveform"] - r["sigma_eigenvalue"]) for r in rows
    )

    # ---------------- 2. order of the trapezoidal implementation, and its growth-rate error
    m, x_a, p_b, lam, t_ref, ref, *_ = keep
    conv = []
    for dt in (2.5e-5, 5e-5, 1e-4, 2e-4, 5e-4):
        t0 = time.perf_counter()
        pm, t, x = run_phase(m, x_a, p_b, dt)
        wall = time.perf_counter() - t0
        back = pm.to_dq_state(x, t)
        pole = phase_growth(pm, t, x, lam)
        # the same rule applied to the mode at its frequency in each frame
        s_phase = lam.real + 1j * (m.data.wb - abs(lam.imag))
        s_dq = lam.real + 1j * abs(lam.imag)
        est = [
            float((np.log((1 + s * dt / 2) / (1 - s * dt / 2)) / dt).real - lam.real)
            for s in (s_phase, s_dq)
        ]
        conv.append({
            "dt_s": dt,
            "max_state_difference_pu": float(np.abs(back[:, :6] - ref[:, :6]).max()),
            "relative_state_difference": float(
                np.abs(back[:, :6] - ref[:, :6]).max() / np.abs(ref - ref[0]).max()
            ),
            "sigma_error": pole.real - float(lam.real),
            "f_error_hz": pole.imag / (2 * np.pi) - (60.0 - abs(lam.imag) / (2 * np.pi)),
            "sigma_error_predicted_phase_frame": est[0],
            "sigma_error_predicted_rotating_frame": est[1],
            "wall_s": wall,
        })  # fmt: skip
    e = [c["max_state_difference_pu"] for c in conv]
    res["trapezoidal"] = {
        "rows": conv,
        "observed_order_50_to_100_us": float(np.log2(e[2] / e[1])),
        "observed_order_100_to_200_us": float(np.log2(e[3] / e[2])),
    }
    ok = [c["dt_s"] for c in conv if abs(c["sigma_error"]) < 0.01]
    res["trapezoidal"]["largest_dt_within_0p01"] = max(ok) if ok else None

    # ---------------- 3. how well a steady state holds with no event
    mdl = DFIG()
    kp, ki = gains_from_bandwidth(mdl.data, 3.0)
    x_s, p_s = mdl.equilibrium(K_FROM * mdl.data.x_e, w_r=1.0, kp=kp, ki=ki)
    hold = {}
    for dt in (5e-5, 2.5e-5):
        pm = PhaseModel.from_params(p_s, mdl.data.wb)
        t, x = trapezoidal(
            pm, PhaseModel.from_dq_state(x_s.numpy()), 0.1, dt, int(round(SAVE / dt))
        )
        hold[f"{dt * 1e6:.0f}_us"] = float(np.abs(pm.to_dq_state(x, t) - x_s.numpy()).max())
    res["steady_state_departure_pu"] = hold

    (OUT / "abc.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure
    m, x_a, p_b, lam, t_ref, ref, t, x, back, pm = keep
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), facecolor=SURFACE)
    cur = pm.stator_current(x)
    for n, (colour, name) in enumerate(zip((C1, C2, C3), "abc", strict=True)):
        axes[0].plot(t, cur[:, n], color=colour, lw=1.1, label=f"Phase {name}")
    axes[0].set_ylabel("Stator current (pu)")
    axes[0].set_title("Phase-coordinate model: what a recorder at the plant would see",
                      loc="left", fontsize=11)  # fmt: skip
    axes[0].legend(frameon=False, fontsize=9, labelcolor=LABEL, ncol=3, loc="upper left")
    mdl = DFIG()
    i_ref = mdl.currents(torch.as_tensor(ref), p_b)["i_D"].numpy()
    i_back = mdl.currents(torch.as_tensor(back), p_b)["i_D"].numpy()
    axes[1].plot(t_ref, i_ref, color=C1, lw=2.6, label="D-Q model, RK4")
    axes[1].plot(t, i_back, color=C2, lw=1.0, label="Phase model, trapezoidal, turned into D-Q")
    axes[1].set_ylabel("Stator current, D axis (pu)")
    axes[1].set_xlabel("Time after the compensation step (s)")
    axes[1].set_title("The same event in both implementations", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper left")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig15_phase_coordinates.png", dpi=160)
    plt.close(fig)

    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
