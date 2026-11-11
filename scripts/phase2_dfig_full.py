"""Phase 2, step 3: doubly-fed machine with full controls.

Writes docs/results/phase2/dfig_full.json and three figures. Every number quoted in
docs/steps/04-dfig-full-controls.md comes from this script.

Run with:  uv run scripts/phase2_dfig_full.py
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

from sso_sciml.physics import slip
from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.dfig_full import STATE_NAMES, Controls, DFIGFull
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LABEL = "#52514e"

BASE = Controls()
STAGE1 = DFIG()
D = STAGE1.data
MACH = {key: getattr(D, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}
SPEEDS = (0.75, 1.0, 1.2)
SPEED_COLOUR = dict(zip(SPEEDS, (C1, C2, C3), strict=True))
SLOW = 1e-3  # Hz: a loop this slow is switched off for the sub-synchronous mode
H_HELD = 1e6  # s: inertia that holds the rotor speed
X_F_OFF = 1e3  # pu: filter reactance that disconnects the grid-side converter

# The four additions of this step, each as (controller changes, parameter overrides) that
# switch it OFF.
OFF = {
    "free rotor speed": ({}, {"h": H_HELD}),
    "outer power loops": ({"outer_hz": SLOW}, {}),
    "phase-locked loop": ({"pll_hz": SLOW}, {}),
    "grid-side converter": ({"x_f": X_F_OFF, "dc_hz": SLOW}, {}),
}


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


def build(on: tuple[str, ...] | None = None, **ctrl) -> tuple[DFIGFull, dict]:
    """Model with only the additions named in `on` active (all of them when on is None)."""
    changes, overrides = dict(ctrl), {}
    for name, (c_off, p_off) in OFF.items():
        if on is not None and name not in on:
            changes.update(c_off)
            overrides.update(p_off)
    return DFIGFull(D, replace(BASE, **changes)), overrides


def analyse(model: DFIGFull, overrides: dict, k: float, w_r: float):
    x0, p = model.equilibrium(k * D.x_e, w_r=w_r, **overrides)
    return modes(jacobian(model.rhs, x0, p)), x0, p


def sso_index(md) -> int:
    """The sub-synchronous network mode: the mode in 2 to 58 Hz that the series capacitor
    participates in most."""
    f = md.freq_hz
    cand = np.where((f > 2.0) & (f < 58.0))[0]
    cap = md.participation[4:6].sum(axis=0)
    return int(cand[np.argmax(cap[cand])])


def sso_sigma(model, overrides, k, w_r) -> float:
    md, _, _ = analyse(model, overrides, k, w_r)
    return float(md.sigma[sso_index(md)])


def stage1_sigma(k: float, w_r: float, rsc_hz: float) -> float:
    kp, ki = gains_from_bandwidth(D, rsc_hz)
    p = STAGE1.params(k * D.x_e, w_r=w_r, kp=kp, ki=ki)
    lam = np.linalg.eigvals(jacobian(STAGE1.rhs, torch.zeros(8, dtype=torch.float64), p).numpy())
    f = np.abs(lam.imag) / (2 * np.pi)
    return float(lam.real[(f > 2.0) & (f < 58.0)].max())


def hand_r_net(k: float, w_r: float, rsc_hz: float) -> float:
    kp, _ = gains_from_bandwidth(D, rsc_hz)
    return slip.analyse_loop(
        lambda s: slip.dfig_impedance(s, w_r=w_r, kp=kp, ki=0.0, **MACH),
        D.r_e, D.x_e, k * D.x_e, D.f0, w_r,
    ).r_net  # fmt: skip


def first_crossing(f, grid: np.ndarray) -> float:
    vals = np.array([f(v) for v in grid])
    idx = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if idx.size == 0:
        return float("nan")
    return float(brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-10))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, none = build()
    res: dict = {"controls": full.describe()}

    # ---------------- 1. steady state and power balance
    worst_res, worst_bal, ops = 0.0, 0.0, []
    for w_r in SPEEDS:
        for k in (0.1, 0.5, 0.9):
            x0, p = full.equilibrium(k * D.x_e, w_r=w_r)
            worst_res = max(worst_res, float(full.rhs(x0, p).abs().max()))
            o, c = full.outputs(x0, p), full.currents(x0, p)
            cu = (
                float(p["r_s"]) * float(c["i_D"] ** 2 + c["i_Q"] ** 2)
                + float(p["r_r"]) * float(c["ir_D"] ** 2 + c["ir_Q"] ** 2)
                + float(p["r_f"]) * float(c["ig_D"] ** 2 + c["ig_Q"] ** 2)
            )
            worst_bal = max(worst_bal, abs(float(p["p_m"]) - float(o["p_t"]) - cu))
            if k == 0.5:
                ops.append(
                    {
                        "w_r": w_r,
                        "p_terminal": float(o["p_t"]),
                        "q_terminal": float(o["q_t"]),
                        "p_mech": float(p["p_m"]),
                        "p_rotor_to_dc": float(o["p_rsc"]),
                        "theta_deg": float(np.degrees(x0[9])),
                    }
                )
    res["steady_state"] = {"residual_max": worst_res, "power_balance_max_abs": worst_bal,
                           "operating_points_k50": ops}  # fmt: skip

    # ---------------- 2. with every addition switched off, stage 1 comes back
    off_model, off_over = build(on=())
    red = []
    for w_r in SPEEDS:
        for k in (0.2, 0.5, 0.8):
            a, b = sso_sigma(off_model, off_over, k, w_r), stage1_sigma(k, w_r, BASE.rsc_hz)
            red.append({"k": k, "w_r": w_r, "full_all_off": a, "stage1": b, "abs_diff": abs(a - b)})
    res["reduction"] = {"cases": red, "max_abs_diff": max(r["abs_diff"] for r in red)}

    # ---------------- 3. what each addition does to the growth rate
    k_ref = 0.5
    attr = {"k": k_ref, "rows": []}
    configs = (
        [("current loop only", (), {})]
        + [(f"+ {n}", (n,), {}) for n in OFF]
        + [
            ("+ grid-side converter, no feed-forward", ("grid-side converter",), {"ff": 0.0}),
            ("all four", None, {}),
            ("all four, no feed-forward", None, {"ff": 0.0}),
        ]
    )
    for label, on, ctrl in configs:
        mdl, over = build(on=on, **ctrl)
        attr["rows"].append(
            {"config": label, **{str(w): sso_sigma(mdl, over, k_ref, w) for w in SPEEDS}}
        )
    base_row, all_row = attr["rows"][0], attr["rows"][-2]
    attr["sum_of_single_shifts"] = {
        str(w): sum(r[str(w)] - base_row[str(w)] for r in attr["rows"][1:5]) for w in SPEEDS
    }
    attr["total_shift"] = {str(w): all_row[str(w)] - base_row[str(w)] for w in SPEEDS}
    res["attribution"] = attr

    # ---------------- 4. sensitivity of the full model, one setting at a time
    sens = {}
    for name, key, grid in (
        ("rotor current loop bandwidth (Hz)", "rsc_hz", np.geomspace(2, 100, 18)),
        ("PLL bandwidth (Hz)", "pll_hz", np.geomspace(1, 60, 16)),
        ("outer loop bandwidth (Hz)", "outer_hz", np.geomspace(0.1, 20, 16)),
        ("grid-side filter reactance (pu)", "x_f", np.geomspace(0.1, 3, 12)),
        ("feed-forward filter time constant (ms)", "t_ff", np.geomspace(0.5, 50, 14) / 1e3),
    ):
        sens[key] = {"label": name, "x": grid.tolist()}
        for w_r in SPEEDS:
            vals = []
            for g in grid:
                mdl, over = build(**{key: float(g)})
                vals.append(sso_sigma(mdl, over, k_ref, w_r))
            sens[key][str(w_r)] = vals
    res["sensitivity"] = sens

    # ---------------- 5. boundary: critical compensation against rotor speed
    wr_grid = np.linspace(0.70, 1.25, 12)
    k_scan = np.linspace(0.03, 0.99, 49)
    bnd = {"w_r": wr_grid.tolist(), "by_bandwidth": {}}
    for rsc_hz in (5.0, 10.0, 30.0):
        mdl, over = build(rsc_hz=rsc_hz)
        mdl0, over0 = build(rsc_hz=rsc_hz, ff=0.0)
        rows = {"full": [], "full_no_ff": [], "stage1": [], "hand": []}
        for w in wr_grid:
            w = float(w)
            rows["full"].append(
                first_crossing(lambda k, w=w, m=mdl, o=over: sso_sigma(m, o, k, w), k_scan)
            )
            rows["full_no_ff"].append(
                first_crossing(lambda k, w=w, m=mdl0, o=over0: sso_sigma(m, o, k, w), k_scan)
            )
            rows["stage1"].append(
                first_crossing(lambda k, w=w, f=rsc_hz: stage1_sigma(k, w, f), k_scan)
            )
            rows["hand"].append(
                first_crossing(lambda k, w=w, f=rsc_hz: -hand_r_net(k, w, f), k_scan)
            )
        gap = np.array(rows["full"]) - np.array(rows["hand"])
        gap0 = np.array(rows["full_no_ff"]) - np.array(rows["hand"])
        rows["full_no_ff_minus_hand_points"] = {"min": float(np.nanmin(gap0) * 100),
                                                "max": float(np.nanmax(gap0) * 100)}  # fmt: skip
        rows["full_minus_hand_points"] = {"min": float(np.nanmin(gap) * 100),
                                          "max": float(np.nanmax(gap) * 100)}  # fmt: skip
        bnd["by_bandwidth"][str(rsc_hz)] = rows
    res["boundary"] = bnd

    # ---------------- 6. any other unstable mode?
    other = []
    for w_r in SPEEDS:
        for k in np.linspace(0.05, 0.95, 19):
            md, _, _ = analyse(full, none, float(k), w_r)
            j = sso_index(md)
            rest = np.delete(md.sigma, [j, int(np.argmin(np.abs(md.eigenvalues - np.conj(
                md.eigenvalues[j]))))])  # fmt: skip
            if rest.max() > 0:
                other.append({"k": float(k), "w_r": w_r, "sigma": float(rest.max())})
    md, _, _ = analyse(full, none, k_ref, 1.0)
    j = sso_index(md)
    top = np.argsort(md.participation[:, j])[::-1][:6]
    res["modes"] = {
        "other_unstable_cases": other,
        "sso_mode_k50_w1": {"sigma": float(md.sigma[j]), "f_hz": float(abs(md.freq_hz[j])),
                            "participation": {STATE_NAMES[t]: float(md.participation[t, j])
                                              for t in top}},
        "least_damped_other_k50_w1": float(np.sort(md.sigma)[-3]),
    }  # fmt: skip

    # ---------------- 7. time domain: small signal, then a compensation step
    td, dt, eps = [], 1e-4, 1e-6
    for k, w_r in ((0.3, 0.75), (0.5, 1.0), (0.7, 1.2), (0.3, 1.2)):
        md, x0, p = analyse(full, none, k, w_r)
        j = sso_index(md)
        if md.freq_hz[j] < 0:
            j = int(np.argmin(np.abs(md.eigenvalues - np.conj(md.eigenvalues[j]))))
        v = md.right[:, j].real
        v = v / np.abs(v).max()
        t_end = float(np.clip(6.0 / abs(md.sigma[j]), 0.3, 1.0))
        _, traj = rk4(full.rhs, x0 + eps * torch.as_tensor(v), p, t_end, dt, save_every=5)
        y = (traj[:, 4] - x0[4]).numpy() / eps
        pole = nearest_pole(matrix_pencil(y, 5 * dt, 8), md.freq_hz[j])
        td.append({"k": k, "w_r": w_r, "f_hz": float(md.freq_hz[j]),
                   "sigma_eig": float(md.sigma[j]), "sigma_td": pole.real,
                   "abs_err": abs(pole.real - float(md.sigma[j]))})  # fmt: skip
    res["time_domain"] = {"cases": td, "max_abs_err": max(c["abs_err"] for c in td)}

    # Compensation stepped from 20 % to 50 % at rated speed: a large disturbance.
    k_from, k_to, w_step = 0.30, 0.35, 1.0
    x_a, p_a = full.equilibrium(k_from * D.x_e, w_r=w_step)
    p_b = {**p_a, "x_c": torch.as_tensor(k_to * D.x_e, dtype=torch.float64)}
    t_step, traj = rk4(full.rhs, x_a, p_b, 0.6, dt, save_every=5)
    p_wave = full.outputs(traj, p_b)["p_t"].numpy()
    t_step = t_step.numpy()
    md_b, x_b, _ = analyse(full, none, k_to, w_step)
    jb = sso_index(md_b)
    win = (t_step > 0.1) & (t_step < 0.6)
    pole = nearest_pole(
        matrix_pencil(p_wave[win] - float(full.outputs(x_b, p_b)["p_t"]), 5 * dt, 10),
        abs(float(md_b.freq_hz[jb])),
    )
    res["compensation_step"] = {
        "k_from": k_from, "k_to": k_to, "w_r": w_step,
        "sigma_eig_after": float(md_b.sigma[jb]),
        "f_eig_after_hz": abs(float(md_b.freq_hz[jb])),
        "sigma_from_power_waveform": pole.real,
        "f_from_power_waveform_hz": pole.imag / (2 * np.pi),
        "power_swing_first_0p1s_pu": float(np.ptp(p_wave[t_step < 0.1])),
        "power_swing_last_0p1s_pu": float(np.ptp(p_wave[t_step > 0.5])),
    }  # fmt: skip

    (OUT / "dfig_full.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=SURFACE)
    labels = [r["config"] for r in attr["rows"]]
    ypos = np.arange(len(labels))[::-1]
    for n, w_r in enumerate(SPEEDS):
        vals = [r[str(w_r)] for r in attr["rows"]]
        ax.barh(ypos + (1 - n) * 0.26, vals, height=0.24, color=SPEED_COLOUR[w_r],
                label=f"Rotor at {w_r:.2f} pu")  # fmt: skip
        for y, v in zip(ypos + (1 - n) * 0.26, vals, strict=True):
            ax.text(v + (0.3 if v >= 0 else -0.3), y, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=LABEL)  # fmt: skip
    ax.set_yticks(ypos, labels)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Growth rate of the sub-synchronous mode (1/s)")
    ax.set_title("What each control addition does (50 % compensation, 10 Hz current loop)",
                 loc="left", fontsize=11)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="lower right")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_dfig_full_attribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), sharey=True, facecolor=SURFACE)
    for ax, rsc_hz in zip(axes, ("5.0", "10.0", "30.0"), strict=True):
        b = bnd["by_bandwidth"][rsc_hz]
        ax.plot(wr_grid, 100 * np.array(b["hand"]), color=C2, lw=2,
                label="Hand: Kp as rotor resistance")  # fmt: skip
        ax.plot(wr_grid, 100 * np.array(b["stage1"]), color=C3, lw=2, ls=(0, (1, 1.5)),
                label="Solver, current loop only")  # fmt: skip
        ax.plot(wr_grid, 100 * np.array(b["full"]), ls="none", marker="o", ms=5, mfc=C1,
                mec=SURFACE, mew=1, label="Solver, full controls")  # fmt: skip
        ax.plot(wr_grid, 100 * np.array(b["full_no_ff"]), ls="none", marker="s", ms=5, mfc=C4,
                mec=SURFACE, mew=1, label="Solver, full controls, no feed-forward")  # fmt: skip
        ax.set_xlabel("Rotor speed (pu)")
        ax.set_title(f"Current loop {float(rsc_hz):.0f} Hz", loc="left", fontsize=10)
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("Compensation at the boundary (%)")
    axes[2].legend(frameon=False, fontsize=8, labelcolor=LABEL, loc="upper left")
    fig.suptitle("Stability boundary: unstable above each curve", x=0.02, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig8_dfig_full_boundary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True, facecolor=SURFACE)
    axes[0].plot(t_step, p_wave, color=C1, lw=1.2)
    axes[0].set_ylabel("Power at the terminal (pu)")
    axes[0].set_title(
        "Compensation stepped from 30 % to 35 % at rated speed, full nonlinear model",
        loc="left", fontsize=11,
    )  # fmt: skip
    axes[1].plot(t_step, 1e4 * (traj[:, W_R_IDX].numpy() - w_step), color=C2, lw=1.2)
    axes[1].set_ylabel("Rotor speed deviation (1e-4 pu)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_dfig_compensation_step.png", dpi=160)
    plt.close(fig)

    slim = {k: v for k, v in res.items() if k not in ("sensitivity", "boundary", "controls")}
    print(json.dumps(slim, indent=1))
    for key, s in sens.items():
        step = max(1, len(s["x"]) // 6)
        print(key, [round(v, 4) for v in s["x"][::step]])
        for w in SPEEDS:
            print("   ", w, [round(v, 2) for v in s[str(w)][::step]])
    for f_hz, b in bnd["by_bandwidth"].items():
        print(f_hz, b["full_minus_hand_points"], b["full_no_ff_minus_hand_points"])
        for name in ("hand", "stage1", "full", "full_no_ff"):
            print("   ", name, [round(v, 3) for v in b[name]])


W_R_IDX = 8

if __name__ == "__main__":
    main()
