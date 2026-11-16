"""Rung 3: the doubly-fed model of this repository on a published Type-3 case.

    uv run scripts/published_type3.py

The case is the benchmark of Garcia Arregui, Rimorov and Gagnon, "Comparative Modal
Analysis of Grid-Following and Grid-Forming Control in a Type-3 Wind Power Plant with a
Series Compensated Grid", IEEE PES General Meeting 2026, paper 000654 (Hydro-Quebec).
Its Table I gives the grid, line, transformers and collector, Table II the machine, and
Table III the eigenvalues of the grid-following plant at 20 % series compensation.

What the paper does not give, and how this script handles it:

    rotor current loop gains   one bandwidth is fitted so that the growth rate of the
                               sub-synchronous mode at 20 % matches the paper's 4.17 1/s;
                               every other comparison then uses that bandwidth
    phase-locked loop, outer   the defaults of this repository (15 Hz; outer loops at
    loops, grid-side converter 0.07 of the current loop)
    operating point            rotor speed swept from 1.0 to 1.2 pu
    shaft                      one mass (the paper models several)
    shunt capacitance          left out (line 324 uS, collector 7.17 uF)
    meaning of "compensation   the capacitor's reactance as a share of the line's
    level"                     reactance (50.9 ohm); the share of line plus grid
                               (75.9 ohm) is also reported

Compared: the frequency of the mode at 20 % (no fitting), the compensation at which it
becomes unstable (paper: stable only below 10 %), whether the PLL moves it (paper: no),
and the gain at which it becomes stable as a share of the fitted gain.

Writes docs/results/validation/published_type3.json and fig112_published_type3.png.
Every number in docs/steps/43-published-type3.md comes from this script.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from sso_sciml import report
from sso_sciml.physics.dfig_full import Controls
from sso_sciml.physics.dfig_meshed import DFIGMeshed
from sso_sciml.physics.induction import IGData
from sso_sciml.solver.linearize import jacobian, modes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "validation"

# ---------------------------------------------------------------- the paper's data
UNITS, UNIT_MVA = 66, 1.7  # Table II: Pn = 1.7e6 VA; text: 66 turbines
S_B = UNITS * UNIT_MVA  # MVA, the plant base used here
V_B = 230.0  # kV
Z_B = V_B**2 / S_B  # ohm
Z_B_MV = 34.5**2 / S_B  # ohm at the collector voltage
MACHINE = {"r_s": 0.01028, "x_ls": 0.1543, "r_r": 0.006684, "x_lr": 0.1543, "x_m": 3.085,
           "h": 0.56347}  # Table II, per unit on the machine rating  # fmt: skip
GRID = (2.0, 25.0)  # Table I: grid equivalent, ohm at 230 kV
LINE = (5.96, 50.9)  # Table I: R1, X1, ohm
MAIN_TX = (0.115, 45.0, 115.0)  # Table I: 11.5 %, X/R 45, 115 MVA
TURBINE_TX = (0.057, 15.2, 115.0)  # Table I: 5.7 %, X/R 15.2, 115 MVA
COLLECTOR = (0.22, 0.147)  # Table I: R1, X1, ohm at 34.5 kV
PAPER = {"sigma": 4.1718, "omega": 320.92667, "f_dq_hz": 51.07707, "f_emt_hz": 9.8,
         "stable_below_level": 0.10, "nominal_level": 0.20}  # Table III and text  # fmt: skip
PAR_RATIO = 1e6  # no parallel path


def network(level: float, of: str = "line") -> dict:
    """Per-unit network on the plant base for a compensation level."""

    def tx(pct, xr, mva):
        x = pct * S_B / mva
        return x / xr, x

    r_m, x_m = tx(*MAIN_TX)
    r_t1, x_t1 = tx(*TURBINE_TX)
    r_t = r_m + r_t1 + COLLECTOR[0] / Z_B_MV
    x_t = x_m + x_t1 + COLLECTOR[1] / Z_B_MV
    r_e, x_e = (GRID[0] + LINE[0]) / Z_B, (GRID[1] + LINE[1]) / Z_B
    x_ref = LINE[1] if of == "line" else LINE[1] + GRID[1]
    return {"r_e": r_e, "x_e": x_e, "x_c": level * x_ref / Z_B, "r_t": r_t, "x_t": x_t,
            "r_p": PAR_RATIO * r_e, "x_p": PAR_RATIO * x_e}  # fmt: skip


def sso_mode(level: float, rsc_hz: float, w_r: float = 1.2, pll_hz: float = 15.0,
             of: str = "line") -> tuple[float, float]:  # fmt: skip
    """Growth rate (1/s) and rotating-frame frequency (Hz) of the mode the series
    capacitor takes most part in."""
    data = replace(IGData(), **MACHINE)
    ctl = Controls(rsc_hz=rsc_hz, pll_hz=pll_hz, outer_hz=0.07 * rsc_hz)
    model = DFIGMeshed(data=data, controls=ctl)
    net = network(level, of)
    x0, p = model.equilibrium(net.pop("x_c"), w_r=w_r, **net)
    if float(model.rhs(x0, p).abs().max()) > 1e-8:
        return float("nan"), float("nan")
    md = modes(jacobian(model.rhs, x0, p))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 59.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return float(md.sigma[j]), float(md.freq_hz[j])


KP_STABLE = 0.45  # the paper: "full stability achieved when the gain Kp is reduced to 0.45"


def kp_of(rsc_hz: float) -> float:
    from sso_sciml.physics.dfig import gains_from_bandwidth

    return gains_from_bandwidth(replace(IGData(), **MACHINE), rsc_hz)[0]


def hz_of(kp: float) -> float:
    """The bandwidth whose tuning rule gives this proportional gain."""
    return kp / kp_of(1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"base": {"mva": S_B, "kv": V_B, "ohm": Z_B}, "paper": PAPER,
                 "network_pu": network(PAPER["nominal_level"])}  # fmt: skip
    speeds = (1.0, 1.1, 1.2)
    levels = np.round(np.arange(0.02, 0.61, 0.02), 3)
    res["by_speed"] = {}
    for w_r in speeds:
        # Fit: the bandwidth at which the mode grows at the paper's rate at 20 %.
        def gap(hz, w_r=w_r):
            return sso_mode(PAPER["nominal_level"], hz, w_r)[0] - PAPER["sigma"]

        grid = np.geomspace(1.0, 400.0, 30)
        vals = np.array([gap(h) for h in grid])
        k = np.nonzero(np.diff(np.sign(vals)) > 0)[0]
        if not k.size:
            res["by_speed"][str(w_r)] = {"fitted": None, "scan_hz": grid.tolist(),
                                         "scan_sigma": (vals + PAPER["sigma"]).tolist()}
            print("speed", w_r, "no bandwidth gives the paper's growth rate")
            continue
        hz = brentq(gap, grid[k[0]], grid[k[0] + 1], xtol=1e-6)
        sigma, f = sso_mode(PAPER["nominal_level"], hz, w_r)
        sweep = np.array([sso_mode(lv, hz, w_r) for lv in levels])
        crossing = None
        s = sweep[:, 0]
        ok = np.isfinite(s)  # the steady-state solve fails at a few levels
        lv_ok, s_ok = levels[ok], s[ok]
        up = np.nonzero((s_ok[:-1] < 0) & (s_ok[1:] >= 0))[0]
        if up.size:
            i = up[0]
            # Linear interpolation between the two finite points either side.
            crossing = float(lv_ok[i] - s_ok[i] * (lv_ok[i + 1] - lv_ok[i])
                             / (s_ok[i + 1] - s_ok[i]))  # fmt: skip
        stable_hz = brentq(lambda h, w_r=w_r: sso_mode(PAPER["nominal_level"], h, w_r)[0], 0.5, hz,
                           xtol=1e-6) if gap(0.5) + PAPER["sigma"] < 0 else None  # fmt: skip
        pll = {str(f_pll): sso_mode(PAPER["nominal_level"], hz, w_r, f_pll)
               for f_pll in (5.0, 15.0, 30.0, 60.0)}  # fmt: skip
        other = sso_mode(PAPER["nominal_level"], hz, w_r, of="line+grid")
        # No fitting: the paper's gain at which the plant becomes stable, Kp = 0.45 pu,
        # read as the proportional gain of this model's loop on the same base.
        at_045 = sso_mode(PAPER["nominal_level"], hz_of(KP_STABLE), w_r)
        # The level at which this model's mode has the paper's frequency.
        f_match = brentq(lambda lv, hz=hz, w_r=w_r: sso_mode(lv, hz, w_r)[1] - PAPER["f_dq_hz"],
                         0.02, 0.2,
                         xtol=1e-5)  # fmt: skip
        res["by_speed"][str(w_r)] = {
            "fitted_rsc_hz": hz, "fitted_kp": kp_of(hz), "sigma": sigma, "f_dq_hz": f,
            "f_phase_hz": 60.0 - f, "levels": levels.tolist(), "sweep_sigma": s.tolist(),
            "sweep_f": sweep[:, 1].tolist(), "unstable_from_level": crossing,
            "stable_at_rsc_hz": stable_hz,
            "stable_gain_share": (kp_of(stable_hz) / kp_of(hz)) if stable_hz else None,
            "pll": pll, "level_of_line_plus_grid": {"sigma": other[0], "f_dq_hz": other[1]},
            "at_paper_stable_kp": {"kp": KP_STABLE, "rsc_hz": hz_of(KP_STABLE),
                                   "sigma": at_045[0], "f_dq_hz": at_045[1]},
            "level_matching_frequency": f_match,
        }  # fmt: skip
        r = res["by_speed"][str(w_r)]
        print("speed", w_r, f"fitted {hz:.2f} Hz kp {r['fitted_kp']:.3f}",
              f"f {f:.2f} Hz (paper {PAPER['f_dq_hz']})", f"unstable from {crossing}",
              f"stable at {stable_hz} Hz share {r['stable_gain_share']}",
              {k2: tuple(round(x, 3) for x in v) for k2, v in pll.items()},
              "line+grid", np.round(other, 3).tolist(), "at kp 0.45",
              np.round(at_045, 3).tolist(), f"freq match at level {f_match:.3f}",
              flush=True)  # fmt: skip
    (OUT / "published_type3.json").write_text(json.dumps(res, indent=1))

    # ---------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=report.SURFACE)
    for k, (w_r, r) in enumerate(res["by_speed"].items()):
        if "levels" not in r:
            continue
        c = report.SERIES[k]
        axes[0].plot(100 * np.array(r["levels"]), r["sweep_sigma"], color=c, lw=2, marker="o",
                     ms=3, label=f"this model, {w_r} pu speed")  # fmt: skip
        axes[1].plot(100 * np.array(r["levels"]), r["sweep_f"], color=c, lw=2, marker="o", ms=3,
                     label=f"this model, {w_r} pu speed")  # fmt: skip
    axes[0].plot(100 * PAPER["nominal_level"], PAPER["sigma"], marker="*", ms=14, color=report.INK,
                 lw=0, label="paper, Table III (fitted point)")  # fmt: skip
    axes[0].axvline(100 * PAPER["stable_below_level"], color=report.MUTED, ls="--", lw=1,
                    label="paper: stable only below 10 %")  # fmt: skip
    axes[0].axhline(0, color=report.INK, lw=0.8)
    axes[1].plot(100 * PAPER["nominal_level"], PAPER["f_dq_hz"], marker="*", ms=14,
                 color=report.INK, lw=0, label="paper, Table III")  # fmt: skip
    axes[1].plot(100 * PAPER["nominal_level"], 60 - PAPER["f_emt_hz"], marker="s", ms=8,
                 color=report.MUTED, lw=0, label="paper, EMT (60 - 9.8 Hz)")  # fmt: skip
    axes[0].set_ylabel("Growth rate of the sub-synchronous mode (1/s)")
    axes[1].set_ylabel("Its frequency in the rotating frame (Hz)")
    for ax in axes:
        ax.set_xlabel("Series compensation, share of the line reactance (%)")
        ax.legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
        report.style(ax)
    report.title(fig, "A published Type-3 case: Hydro-Quebec benchmark, grid-following control "
                 "(PES GM 2026, paper 000654)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig112_published_type3.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
