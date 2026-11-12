"""Which limit does a growing oscillation reach first, and can the modes say when?

Needs the dataset from scripts/phase3_generate_t1.py. Unstable and stable cases of the
T1 test fold are simulated for 2 s with the rotor-side converter voltage limited to
0.4 pu, after a 0.01 pu offset of capacitor voltage.

Two limits are watched: the rotor voltage command reaching 0.4 pu, and the DC-link
voltage leaving a band of 0.2 pu around its set point. The time each is reached is
compared with the time found from the modes of the state matrix, with no time stepping.

Writes docs/results/phase3/limits.json and two figures. Every number in
docs/steps/16-converter-limit.md comes from this script.

Run with:  uv run scripts/phase3_limits.py        # about 4 minutes on the GPU
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

from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian
from sso_sciml.solver.modal import linear_response

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / data.VERSION
OUT = ROOT / "docs" / "results" / "phase3"
N_UNSTABLE, N_STABLE = 96, 32
VR_MAX, EPS = 0.4, 0.01
T_END, DT, SAVE = 2.0, 5e-4, 2
W_R, V_DC, VC = 8, 19, [4, 5]
DC_BAND = 0.2  # pu; the DC link is taken to be out of its band beyond this


def command_size(model, x: torch.Tensor, p: dict) -> np.ndarray:
    """Magnitude of the rotor-voltage command at states x of shape (n_t, batch, 25)."""
    out = []
    for chunk in torch.split(x, 200):
        d, q = model.outputs(chunk, p)["vr_command"]
        out.append(torch.hypot(d, q).cpu().numpy())
    return np.concatenate(out)


def first_time(t: np.ndarray, above: np.ndarray) -> np.ndarray:
    """First time each column of `above` (n_t, batch) is true; NaN if never."""
    hit = above.any(axis=0)
    return np.where(hit, t[above.argmax(axis=0)], np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    z = np.load(DATA / "in_distribution.npz")
    ok = (z["solver"][:, 4] > 0) & (z["fold"] == 2)
    pick = np.concatenate([
        np.where(ok & (z["solver"][:, 0] > 0))[0][:N_UNSTABLE],
        np.where(ok & (z["solver"][:, 0] < 0))[0][:N_STABLE],
    ])  # fmt: skip
    theta, sigma = z["theta"][pick], z["solver"][pick, 0]
    n = len(pick)

    ops = [data.operating_point(row) for row in theta]
    a = np.array([jacobian(m.rhs, x, p).numpy() for m, x, p in ops])
    model = ops[0][0]
    x_eq = torch.stack([o[1] for o in ops])
    p = {k: torch.stack([o[2][k] for o in ops]).to(device) for k in ops[0][2]}
    p["vr_max"] = torch.full_like(p["kp"], VR_MAX)
    v0 = command_size(model, x_eq[None].to(device), p)[0]

    # ---------------- time domain with the limit
    x0 = x_eq.clone()
    x0[:, VC[0]] += EPS
    t0 = time.perf_counter()
    t, traj = rk4(model.rhs, x0.to(device), p, T_END, DT, save_every=SAVE)
    rk4_s = time.perf_counter() - t0
    t = t.numpy()
    x = traj.cpu().numpy()
    finite = np.isfinite(x).all(axis=2)
    cmd = np.where(finite, command_size(model, torch.nan_to_num(traj), p), np.inf)
    vdc = np.where(finite, np.abs(x[:, :, V_DC] - x_eq[:, V_DC].numpy()), np.inf)
    speed = np.where(finite, np.abs(x[:, :, W_R] - x_eq[:, W_R].numpy()), np.inf)
    sim = {
        "rotor voltage": first_time(t, cmd >= VR_MAX),
        "DC voltage": first_time(t, vdc >= DC_BAND),
    }
    t_runaway = first_time(t, ~finite | (speed > 0.1))

    # ---------------- the same from the modes, with no time stepping
    t0 = time.perf_counter()
    dx0 = np.zeros(data.N_STATE)
    dx0[VC[0]] = EPS
    lin = np.array([linear_response(a[i], dx0, t) for i in range(n)])  # (n, n_t, 25)
    lin_x = torch.as_tensor(lin.transpose(1, 0, 2) + x_eq.numpy()[None]).to(device)
    cmd_lin = command_size(model, lin_x, p)
    vdc_lin = np.abs(lin[:, :, V_DC]).T
    modal = {
        "rotor voltage": first_time(t, cmd_lin >= VR_MAX),
        "DC voltage": first_time(t, vdc_lin >= DC_BAND),
    }
    modal_s = time.perf_counter() - t0

    unstable = sigma > 0

    def first_of(d):
        return np.fmin(d["rotor voltage"], d["DC voltage"])

    sim["first of the two"], modal["first of the two"] = first_of(sim), first_of(modal)
    limits: dict = {}
    for name in sim:
        reached = unstable & np.isfinite(sim[name])
        both = reached & np.isfinite(modal[name])
        err = modal[name][both] - sim[name][both]
        limits[name] = {
            "reached_in_the_run": int(reached.sum()),
            "reached_from_the_modes": int((unstable & np.isfinite(modal[name])).sum()),
            "both": int(both.sum()),
            "median_time_s": float(np.median(sim[name][reached])),
            "error_median_s": float(np.median(err)),
            "error_median_abs_s": float(np.median(np.abs(err))),
            "error_p95_abs_s": float(np.quantile(np.abs(err), 0.95)),
            "error_relative_median": float(np.median(np.abs(err) / sim[name][both])),
        }
    reached = unstable & np.isfinite(sim["first of the two"])
    dc_first = sim["DC voltage"][reached] <= np.nan_to_num(
        sim["rotor voltage"][reached], nan=np.inf
    )
    lead = (sim["rotor voltage"] - sim["DC voltage"])[
        unstable & np.isfinite(sim["rotor voltage"]) & np.isfinite(sim["DC voltage"])
    ]
    # A hand estimate: exponential growth at the eigenvalue's rate from the size the DC
    # voltage swing has early on to the size of the band.
    k_ref = int(np.searchsorted(t, 0.2))
    early = np.abs(lin[:, : k_ref + 1, V_DC]).max(axis=1)
    hand = np.where(unstable, t[k_ref] + np.log(DC_BAND / early) / np.maximum(sigma, 1e-9), np.nan)
    hb = reached & np.isfinite(sim["DC voltage"]) & (hand < T_END)
    hand_err = hand[hb] - sim["DC voltage"][hb]
    res = {
        "unstable_cases": int(unstable.sum()),
        "stable_cases": int((~unstable).sum()),
        "vr_max_pu": VR_MAX,
        "dc_band_pu": DC_BAND,
        "offset_pu": EPS,
        "window_s": T_END,
        "command_at_steady_state": {
            "min": float(v0.min()),
            "median": float(np.median(v0)),
            "max": float(v0.max()),
        },
        "stable_cases_reaching_any_limit": int(
            np.isfinite(sim["first of the two"][~unstable]).sum()
        ),
        "limits": limits,
        "dc_voltage_first_share": float(dc_first.mean()),
        "dc_leads_rotor_voltage_by_median_s": float(np.median(lead)),
        "hand_estimate_of_dc_time": {
            "cases": int(hb.sum()),
            "error_median_abs_s": float(np.median(np.abs(hand_err))),
            "error_relative_median": float(np.median(np.abs(hand_err) / sim["DC voltage"][hb])),
        },
        "after_the_first_limit": {
            "speed_runs_away_or_run_fails": int((reached & np.isfinite(t_runaway)).sum()),
            "of": int(reached.sum()),
        },
        "cost": {"rk4_batch_s": rk4_s, "modal_all_cases_s": modal_s},
        "wall_s": time.perf_counter() - t_start,
    }
    (OUT / "limits.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "command_at_steady_state"}, indent=1))

    # ---------------- figure: time to each limit, modes against the time-domain run
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), facecolor=report.SURFACE)
    for ax, name, colour in zip(axes, sim, report.SERIES, strict=False):
        both = unstable & np.isfinite(sim[name]) & np.isfinite(modal[name])
        ax.scatter(sim[name][both], modal[name][both], s=18, color=colour, lw=0)
        ax.plot([0, T_END], [0, T_END], color=report.INK, lw=0.8, ls="--")
        ax.set_xlabel("Time-domain run (s)")
        ax.set_title(f"Time until: {name}", loc="left", fontsize=10)
        report.style(ax)
    axes[0].set_ylabel("From the modes (s)")
    report.title(fig, f"When a growing oscillation reaches a limit: rotor voltage {VR_MAX} pu, "
                 f"DC voltage within {DC_BAND} pu of its set point")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig40_limit_time.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: one case
    cand = np.where(reached & np.isfinite(sim["rotor voltage"]) & np.isfinite(sim["DC voltage"]))[0]
    i = int(cand[len(cand) // 2])
    stop = min(len(t) - 1, int(np.searchsorted(t, sim["rotor voltage"][i] + 0.05)))
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True, facecolor=report.SURFACE)
    axes[0].plot(t[:stop], x[:stop, i, V_DC], color=report.MUTED, lw=1.2, label="time-domain run")
    axes[0].plot(t[:stop], 1.0 + lin[i, :stop, V_DC], color=report.SERIES[1], lw=0.9, ls="--",
                 label="from the modes")  # fmt: skip
    for level in (1.0 - DC_BAND, 1.0 + DC_BAND):
        axes[0].axhline(level, color=report.INK, lw=0.8, ls="--")
    axes[0].set_ylabel("DC-link voltage (pu)")
    axes[0].set_ylim(1.0 - 3 * DC_BAND, 1.0 + 3 * DC_BAND)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL, loc="upper left")
    axes[1].plot(t[:stop], cmd[:stop, i], color=report.MUTED, lw=1.2)
    axes[1].plot(t[:stop], cmd_lin[:stop, i], color=report.SERIES[1], lw=0.9, ls="--")
    axes[1].axhline(VR_MAX, color=report.INK, lw=0.8, ls="--")
    axes[1].set_ylabel("Rotor voltage command (pu)")
    axes[1].set_ylim(0, 1.5 * VR_MAX)
    axes[2].plot(t[:stop], x[:stop, i, W_R], color=report.SERIES[0], lw=1.2)
    axes[2].set_ylabel("Rotor speed (pu)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(sim["DC voltage"][i], color=report.SERIES[2], lw=0.8, ls=":")
        ax.axvline(sim["rotor voltage"][i], color=report.INK, lw=0.8, ls=":")
        report.style(ax)
    report.title(fig, f"One case, growth rate {sigma[i]:+.1f} 1/s. Dotted: DC voltage leaves its "
                 f"band at {sim['DC voltage'][i]:.2f} s, rotor voltage limit at "
                 f"{sim['rotor voltage'][i]:.2f} s")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig41_limit_case.png", dpi=160)
    plt.close(fig)
    print(f"done in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
