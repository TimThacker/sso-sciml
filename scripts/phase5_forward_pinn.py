"""Phase 5: a forward physics-informed network on one case of System B1.

The induction generator on a series-compensated line (7 states) is disturbed by a
0.01 pu step in capacitor voltage. A network x_hat(t) is trained on the ODE residual
alone and compared with the RK4 solution. The remedies of the plan are switched on one
at a time at an equal budget of training steps.

Writes docs/results/phase5/forward_pinn.json and two figures. Every number in
docs/steps/12-forward-pinn.md comes from this script.

Run with:  uv run scripts/phase5_forward_pinn.py        # about 50 minutes on the GPU
           uv run scripts/phase5_forward_pinn.py --quick   # 1/10 of the steps, a smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.models.pinn import PinnConfig, train_pinn, trajectory_error
from sso_sciml.physics.induction import InductionGenerator
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "phase5"
INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
T_END, DT, SAVE = 0.5, 1e-4, 5
STEP_V = 0.01  # pu, initial offset of the D-axis capacitor voltage
FOURIER = tuple(float(f) for f in np.linspace(8.0, 128.0, 16))
WINDOW_S = 0.05


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


class Case:
    """One operating point with its disturbance, state scales and RK4 reference."""

    def __init__(self, t_end: float, **machine):
        self.model = InductionGenerator()
        self.t_end = t_end
        self.x_eq, self.p = self.model.equilibrium(0.8 * self.model.data.x_e, **machine)
        md = modes(jacobian(self.model.rhs, self.x_eq, self.p))
        self.x0 = self.x_eq.clone()
        self.x0[4] += STEP_V
        # Size of each state's deviation, from the modes the disturbance excites.
        amp = np.abs(np.linalg.solve(md.right, (self.x0 - self.x_eq).numpy()))
        grow = np.exp(np.maximum(md.sigma, 0.0) * t_end)
        self.scale = torch.as_tensor((np.abs(md.right) * amp * grow).sum(axis=1))
        sub = np.where((md.freq_hz > 5.0) & (md.freq_hz < 55.0))[0]
        self.lam = complex(md.eigenvalues[sub[np.argmax(md.sigma[sub])]])
        self.eigenvalues = [[float(v.real), float(v.imag / 2 / np.pi)] for v in md.eigenvalues]
        t0 = time.perf_counter()
        t, x = rk4(self.model.rhs, self.x0, self.p, t_end, DT, save_every=SAVE)
        self.rk4_s = time.perf_counter() - t0
        self.t, self.x_ref = t.numpy(), x.numpy()

    def run(self, cfg: PinnConfig, device: str) -> tuple[dict, np.ndarray]:
        scale, x_eq = self.scale.numpy(), self.x_eq.numpy()

        def monitor(predict, t_reached):
            """Error against RK4 over the time trained so far. Not used in training."""
            seen = self.t <= t_reached + 1e-9
            return trajectory_error(predict(self.t[seen]), self.x_ref[seen], x_eq, scale)

        predict, info = train_pinn(
            self.model.rhs, self.p, self.x0, self.scale, cfg, device=device, monitor=monitor
        )
        x_hat = predict(self.t)
        fifths = np.array_split(np.arange(self.t.size), 5)
        out = {
            "config": asdict(cfg),
            "error": trajectory_error(x_hat, self.x_ref, x_eq, scale),
            "error_by_fifth": [
                trajectory_error(x_hat[j], self.x_ref[j], x_eq, scale) for j in fifths
            ],
            "final_residual": info["history"]["residual"][-1],
            "train_s": info["train_s"],
            "history": info["history"],
        }
        try:
            pole = nearest_pole(
                matrix_pencil((x_hat[:, 4] - x_eq[4]) / scale[4], DT * SAVE, 8),
                abs(self.lam.imag) / 2 / np.pi,
            )
            out["sigma"], out["f_hz"] = pole.real, pole.imag / 2 / np.pi
        except (ValueError, np.linalg.LinAlgError):
            out["sigma"], out["f_hz"] = float("nan"), float("nan")
        return out, x_hat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the training figure from the results file")  # fmt: skip
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replot:
        training_figure(json.loads((OUT / "forward_pinn.json").read_text()))
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = 1000 if args.quick else 10000
    t_start = time.perf_counter()

    stable = Case(T_END)
    n_win = int(round(T_END / WINDOW_S))
    base = PinnConfig(T_END, steps=steps)
    variants = {
        "plain": replace(base, nondim=False),
        "scaled": base,
        "scaled, Fourier inputs": replace(base, fourier_hz=FOURIER),
        "scaled, sine activations": replace(base, activation="sin"),
        "scaled, Fourier, causal": replace(base, fourier_hz=FOURIER, causal_eps=1.0),
        "scaled, Fourier, marching": replace(base, fourier_hz=FOURIER, windows=n_win),
        "scaled, Fourier, causal, marching": replace(
            base, fourier_hz=FOURIER, causal_eps=1.0, windows=n_win
        ),
    }
    res: dict = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "case": {
            "compensation": 0.8,
            "step_pu": STEP_V,
            "t_end_s": T_END,
            "eigenvalues_sigma_hz": stable.eigenvalues,
            "sub_synchronous": [stable.lam.real, abs(stable.lam.imag) / 2 / np.pi],
            "scale": stable.scale.tolist(),
            "rk4_s": stable.rk4_s,
        },
        "variants": {},
    }
    traces = {}
    for name, cfg in variants.items():
        res["variants"][name], traces[name] = stable.run(cfg, device)
        r = res["variants"][name]
        print(f"{name:36s} error {r['error']:.3f} residual {r['final_residual']:.1e} "
              f"{r['train_s']:.0f} s")  # fmt: skip
    best = min(res["variants"], key=lambda k: res["variants"][k]["error"])
    res["best"] = best

    # ---------------- how far in time does it reach?
    horizon: dict = {}
    for t_end in (0.05, 0.1, 0.2):
        case = Case(t_end)
        for name in ("scaled", "scaled, Fourier inputs"):
            horizon[f"{name}/{t_end}"] = case.run(replace(variants[name], t_end=t_end), device)[0]
            print("horizon", name, t_end, f"{horizon[f'{name}/{t_end}']['error']:.3f}")
    res["horizon"] = horizon

    # ---------------- more training for the best variant
    longer, traces["longer"] = stable.run(replace(variants[best], steps=4 * steps), device)
    res["best_4x_steps"] = longer
    print("4x steps", best, f"{longer['error']:.3f} {longer['train_s']:.0f} s")

    # ---------------- a growing oscillation
    unstable = Case(T_END, r_r=0.010)
    grow, _ = unstable.run(variants[best], device)
    grow["sub_synchronous"] = [unstable.lam.real, abs(unstable.lam.imag) / 2 / np.pi]
    res["unstable"] = grow
    print(
        "unstable", f"{grow['error']:.3f} sigma {grow['sigma']:.2f} against {unstable.lam.real:.2f}"
    )
    res["wall_s"] = time.perf_counter() - t_start
    (OUT / "forward_pinn.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: waveforms
    show = ("scaled", "scaled, Fourier inputs", best)
    fig, axes = plt.subplots(len(show), 1, figsize=(10.6, 6.6), sharex=True, facecolor=SURFACE)
    ref = (stable.x_ref[:, 4] - float(stable.x_eq[4])) / STEP_V
    for ax, name in zip(axes, show, strict=True):
        ax.plot(stable.t, ref, color=MUTED, lw=1.2, label="RK4")
        ax.plot(stable.t, (traces[name][:, 4] - float(stable.x_eq[4])) / STEP_V, color=ORANGE,
                lw=1.2, label="network")  # fmt: skip
        ax.set_title(f"{name}: error {res['variants'][name]['error']:.2f}", loc="left", fontsize=10)
        ax.set_ylim(-1.3, 1.3)
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=LABEL, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Capacitor voltage deviation, D axis (share of the initial step)")
    fig.suptitle(f"Network trained on the equations alone, {steps:,} steps", x=0.02, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig24_forward_pinn_waveforms.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: error by variant and through time
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1]})  # fmt: skip
    names = list(variants)
    ypos = np.arange(len(names))[::-1]
    ax = axes[0]
    ax.barh(ypos, [res["variants"][n]["error"] for n in names], height=0.62,
            color=[GREEN if n == best else BLUE for n in names])  # fmt: skip
    ax.set_xscale("log")
    ax.axvline(1.0, color=INK, lw=1.0, ls="--")
    ax.set_yticks(ypos, names)
    ax.set_xlabel("Error relative to the size of the response (1 = no better than standing still)")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    mid = [(k + 0.5) * T_END / 5 for k in range(5)]
    for name, colour in zip(show, (BLUE, ORANGE, GREEN), strict=True):
        ax.plot(mid, res["variants"][name]["error_by_fifth"], color=colour, lw=2, marker="o",
                ms=5, label=name)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("Time (s), fifths of the window")
    ax.set_ylabel("Error in that fifth")
    ax.legend(frameon=False, fontsize=9, labelcolor=LABEL)
    style(ax)
    fig.suptitle("Trajectory error of each variant, and where in time it arises", x=0.02,
                 ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig25_forward_pinn_errors.png", dpi=160)
    plt.close(fig)

    training_figure(res)
    print(f"done in {res['wall_s']:.0f} s")


def training_figure(res: dict) -> None:
    """Training curves of every variant, from the histories in the results file."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), facecolor=SURFACE)
    for (name, r), colour in zip(res["variants"].items(), report.SERIES, strict=True):
        h = r["history"]
        axes[0].plot(h["step"], h["residual"], color=colour, lw=1.3, label=name)
        axes[1].plot(h["step"], h["monitor"], color=colour, lw=1.3)
    h = res["best_4x_steps"]["history"]
    axes[0].plot(h["step"], h["residual"], color=INK, lw=1.0, ls=":",
                 label=f"{res['best']}, 4 times the steps")  # fmt: skip
    axes[1].plot(h["step"], h["monitor"], color=INK, lw=1.0, ls=":")
    axes[0].set_ylabel("Mean squared residual of the equations (scaled)")
    axes[0].set_title("What the optimiser sees", loc="left", fontsize=10)
    axes[1].set_ylabel("Error against RK4 over the time trained so far")
    axes[1].set_title("What it does not see (1 = no better than standing still)", loc="left",
                      fontsize=10)  # fmt: skip
    axes[1].axhline(1.0, color=INK, lw=0.8, ls="--")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        report.style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=LABEL, loc="lower center",
               ncol=4)  # fmt: skip
    report.title(fig, "Training of each variant. Marching variants start a new network every "
                 "tenth of the steps")  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig26_forward_pinn_training.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
