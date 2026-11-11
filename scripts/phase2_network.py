"""Phase 2, step 4: how the network changes the answer, using the impedance formulation.

Writes docs/results/phase2/network.json and two figures. Every number quoted in
docs/steps/05-network-and-impedance.md comes from this script.

Run with:  uv run scripts/phase2_network.py
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

from sso_sciml.physics import circuit as ct
from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.solver.linearize import jacobian

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "phase2"

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
LABEL = "#52514e"

M = DFIG()
D = M.data
MACH = {key: getattr(D, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}
SPEEDS = (0.75, 1.0, 1.2)
SPEED_COLOUR = dict(zip(SPEEDS, (C1, C2, C3), strict=True))
# Plant-side series impedance between the machine terminals and the line: unit and station
# transformers plus collector cables. Representative values, not from a specific plant.
R_PLANT, X_PLANT = 0.010, 0.15


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


def network(k: float, r_mult: float = 1.0, plant: bool = False, par: float | None = None):
    """Network impedance seen from the machine terminals.

    k is the compensation of the line reactance. `par` adds an uncompensated path in
    parallel with the compensated line, with `par` times its impedance.
    """
    z = ct.branch(r_mult * D.r_e, D.x_e, k * D.x_e)
    if par is not None:
        z = ct.parallel(z, ct.branch(par * D.r_e, par * D.x_e))
    if plant:
        z = z + ct.branch(R_PLANT, X_PLANT)
    return z


def growth(k: float, w_r: float, bw_hz: float, online: float = 1.0, **net) -> float:
    kp, ki = gains_from_bandwidth(D, bw_hz)
    z_m = ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **MACH) / online
    return ct.sso_growth(ct.modes(z_m + network(k, **net)))


def first_crossing(f, grid: np.ndarray) -> float:
    vals = np.array([f(v) for v in grid])
    idx = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if idx.size == 0:
        return float("nan")
    return float(brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-10))


K_SCAN = np.linspace(0.03, 0.99, 49)
BW_SCAN = np.geomspace(0.5, 300.0, 60)


def boundary_k(w_r: float, bw_hz: float = 10.0, **kw) -> float:
    """Lowest unstable compensation level."""
    return first_crossing(lambda k: growth(k, w_r, bw_hz, **kw), K_SCAN)


def boundary_bw(k: float, w_r: float, **kw) -> float:
    """Largest stable current-loop bandwidth, Hz."""
    return first_crossing(lambda f: growth(k, w_r, f, **kw), BW_SCAN)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"plant_impedance": {"r": R_PLANT, "x": X_PLANT},
                 "line": {"r": D.r_e, "x": D.x_e, "x_over_r": D.x_e / D.r_e}}  # fmt: skip

    # ---------------- 1. impedance formulation against the state-space model
    worst, n_modes, n_cases = 0.0, set(), 0
    for k in (0.2, 0.5, 0.8):
        for w_r in SPEEDS:
            for bw in (3.0, 10.0, 30.0):
                kp, ki = gains_from_bandwidth(D, bw)
                lam = ct.modes(ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **MACH) + network(k))
                p = M.params(k * D.x_e, w_r=w_r, kp=kp, ki=ki)
                ss = np.linalg.eigvals(
                    jacobian(M.rhs, torch.zeros(8, dtype=torch.float64), p).numpy()
                )
                worst = max(worst, max(float(np.abs(ss - v).min()) for v in lam))
                n_modes.add(len(lam))
                n_cases += 1
    res["cross_check"] = {"cases": n_cases, "max_abs_eigenvalue_diff": worst,
                          "modes_found_per_case": sorted(n_modes)}  # fmt: skip

    # ---------------- 2. reference: radial connection
    ref = {
        "growth_k50": {str(w): growth(0.5, w, 10.0) for w in SPEEDS},
        "boundary_k": {str(w): boundary_k(w) for w in SPEEDS},
        "largest_stable_bw_k50": {str(w): boundary_bw(0.5, w) for w in SPEEDS},
    }
    res["radial"] = ref

    # ---------------- 3. network resistance
    r_grid = np.geomspace(0.5, 8.0, 17)
    r_sweep = {"multiplier": r_grid.tolist(), "x_over_r": (D.x_e / (r_grid * D.r_e)).tolist()}
    for w in SPEEDS:
        r_sweep[str(w)] = {
            "growth_k50": [growth(0.5, w, 10.0, r_mult=float(r)) for r in r_grid],
            "boundary_k": [boundary_k(w, r_mult=float(r)) for r in r_grid],
            "multiplier_for_stability_k50": first_crossing(
                lambda r, w=w: -growth(0.5, w, 10.0, r_mult=r), np.geomspace(0.5, 50.0, 60)
            ),
        }
    res["network_resistance"] = r_sweep

    # ---------------- 4. parallel uncompensated path
    par_grid = np.geomspace(0.25, 32.0, 15)
    par_sweep = {"ratio": par_grid.tolist()}
    for w in SPEEDS:
        par_sweep[str(w)] = {
            "growth_k50": [growth(0.5, w, 10.0, par=float(r)) for r in par_grid],
            "boundary_k": [boundary_k(w, par=float(r)) for r in par_grid],
            "ratio_for_stability_k50": first_crossing(
                lambda r, w=w: growth(0.5, w, 10.0, par=r), np.geomspace(0.05, 64.0, 80)
            ),
        }
    res["parallel_path"] = par_sweep

    # ---------------- 5. units online
    n_grid = np.linspace(0.05, 1.0, 20)
    res["units_online"] = {
        "fraction": n_grid.tolist(),
        **{
            str(w): {
                "growth_k50": [growth(0.5, w, 10.0, online=float(n)) for n in n_grid],
                "boundary_k": [boundary_k(w, online=float(n)) for n in n_grid],
            }
            for w in SPEEDS
        },
    }
    for w in SPEEDS:
        g = res["units_online"][str(w)]["growth_k50"]
        res["units_online"][str(w)]["worst_fraction"] = float(n_grid[int(np.argmax(g))])
        res["units_online"][str(w)]["worst_growth"] = float(np.max(g))

    # ---------------- 6. scenarios: largest stable bandwidth and boundary
    scenarios = [
        ("Radial (reference)", {}),
        ("+ plant transformers and collector", {"plant": True}),
        ("Network resistance x2", {"r_mult": 2.0}),
        ("Parallel path, 2x line impedance", {"par": 2.0}),
        ("Parallel path, equal to the line", {"par": 1.0}),
        ("Parallel path, half the line impedance", {"par": 0.5}),
        ("Plant impedance + equal parallel path", {"plant": True, "par": 1.0}),
    ]
    rows = []
    for label, kw in scenarios:
        rows.append(
            {
                "scenario": label,
                "largest_stable_bw_k50_hz": {str(w): boundary_bw(0.5, w, **kw) for w in SPEEDS},
                "boundary_k_at_10hz": {str(w): boundary_k(w, **kw) for w in SPEEDS},
                "growth_k50_at_10hz": {str(w): growth(0.5, w, 10.0, **kw) for w in SPEEDS},
            }
        )
    res["scenarios"] = rows

    (OUT / "network.json").write_text(json.dumps(res, indent=2))

    # ---------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (x, sweep, xlabel, title) in zip(
        axes,
        (
            (r_grid, r_sweep, "Network resistance (multiple of base)", "Network resistance"),
            (par_grid, par_sweep, "Parallel path impedance / line impedance",
             "Parallel uncompensated path"),
            (100 * n_grid, res["units_online"], "Units online (%)", "Units online"),
        ),
        strict=True,
    ):  # fmt: skip
        for w in SPEEDS:
            ax.plot(x, sweep[str(w)]["growth_k50"], color=SPEED_COLOUR[w], lw=2,
                    label=f"Rotor at {w:.2f} pu")  # fmt: skip
        if title != "Units online":
            ax.set_xscale("log")
            ticks = [0.5, 1, 2, 4, 8] if title == "Network resistance" else [0.25, 1, 4, 16]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel("Growth rate (1/s)")
    axes[0].set_ylim(-30, 25)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=LABEL, loc="lower right")
    fig.suptitle(
        "Sub-synchronous growth rate at 50 % compensation, 10 Hz current loop", x=0.02,
        ha="left", fontsize=11, color=INK,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig10_network_sweeps.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.6), facecolor=SURFACE)
    labels = [r["scenario"] for r in rows]
    ypos = np.arange(len(labels))[::-1]
    for n, w in enumerate(SPEEDS):
        vals = [r["largest_stable_bw_k50_hz"][str(w)] for r in rows]
        shown = [BW_SCAN[0] if np.isnan(v) else v for v in vals]
        ax.barh(ypos + (1 - n) * 0.26, [v - 0.3 for v in shown], height=0.24, color=SPEED_COLOUR[w],
                left=0.3, label=f"Rotor at {w:.2f} pu")  # fmt: skip
        for y, v, sh in zip(ypos + (1 - n) * 0.26, vals, shown, strict=True):
            text = f"below {BW_SCAN[0]:.1f}" if np.isnan(v) else f"{v:.1f}"
            ax.text(sh * 1.06, y, text, va="center", ha="left", fontsize=8, color=LABEL)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Largest stable current-loop bandwidth at 50 % compensation (Hz)")
    ax.set_title("How much controller the network tolerates", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper right")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig11_network_scenarios.png", dpi=160)
    plt.close(fig)

    def r3(v):
        return round(v, 3) if isinstance(v, float) else v

    print(json.dumps({k: res[k] for k in ("cross_check", "radial", "line")}, indent=1))
    for w in SPEEDS:
        print(w, "R mult for stability", r3(r_sweep[str(w)]["multiplier_for_stability_k50"]),
              "parallel ratio for stability", r3(par_sweep[str(w)]["ratio_for_stability_k50"]),
              "units worst", r3(res["units_online"][str(w)]["worst_fraction"]),
              r3(res["units_online"][str(w)]["worst_growth"]))  # fmt: skip
        print("   R growth", [r3(v) for v in r_sweep[str(w)]["growth_k50"][::4]],
              "bnd", [r3(v) for v in r_sweep[str(w)]["boundary_k"][::4]])  # fmt: skip
        print("   par growth", [r3(v) for v in par_sweep[str(w)]["growth_k50"][::2]],
              "bnd", [r3(v) for v in par_sweep[str(w)]["boundary_k"][::2]])  # fmt: skip
        print("   units growth", [r3(v) for v in res["units_online"][str(w)]["growth_k50"][::3]],
              "bnd", [r3(v) for v in res["units_online"][str(w)]["boundary_k"][::3]])  # fmt: skip
    for r in rows:
        print(r["scenario"], {k: {w: r3(v) for w, v in d.items()} for k, d in r.items()
                              if k != "scenario"})  # fmt: skip


if __name__ == "__main__":
    main()
