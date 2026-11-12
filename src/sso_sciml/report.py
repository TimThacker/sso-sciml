"""Plots of model training, performance and metrics.

The project keeps no experiment-tracking store. Each training script writes its numbers
to a JSON file in docs/results/ and draws these figures next to it, so a run is
recorded by files in the repository.
"""

from __future__ import annotations

import numpy as np

INK, MUTED, GRID, SURFACE, LABEL = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#52514e"
ARM_COLOUR = {"N": "#2a78d6", "F": "#eb6834", "R": "#1baf7a"}
ARM_NAME = {"N": "N: no physics", "F": "F: physics as features", "R": "R: physics as baseline"}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#c3a13a", "#8a5cc7", "#d1477a", "#4aa3a3")


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


def title(fig, text: str) -> None:
    fig.suptitle(text, x=0.02, ha="left", fontsize=11, color=INK)


def curves(ax, runs: list[tuple[np.ndarray, np.ndarray]], colour: str, label: str) -> None:
    """One thin line per run and their mean as a thick line. Runs share their x values."""
    for x, y in runs:
        ax.plot(x, y, color=colour, lw=0.7, alpha=0.45)
    ax.plot(runs[0][0], np.mean([y for _, y in runs], axis=0), color=colour, lw=2, label=label)


def parity(ax, true: np.ndarray, pred: np.ndarray, lim: tuple[float, float]) -> None:
    """Predicted against true as a density, with the diagonal and the mean absolute error."""
    ax.hexbin(true, pred, gridsize=70, extent=(*lim, *lim), bins="log", cmap="Blues", mincnt=1,
              linewidths=0)  # fmt: skip
    ax.plot(lim, lim, color=INK, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.text(0.04, 0.93, f"mean error {np.abs(pred - true).mean():.2f} 1/s", transform=ax.transAxes,
            fontsize=9, color=LABEL, va="top")  # fmt: skip
    style(ax)


def error_cdf(ax, errors: dict[str, np.ndarray], colours: dict[str, str]) -> None:
    """Share of cases with an absolute error below each value."""
    for name, err in errors.items():
        e = np.sort(np.abs(err))
        ax.plot(e, (np.arange(e.size) + 1) / e.size, color=colours[name], lw=2, label=name)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    style(ax)


def binned_error(ax, x: np.ndarray, err: np.ndarray, edges: np.ndarray, colour: str, label: str):
    """Mean absolute error in bins of x."""
    mid, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 20:
            mid.append(0.5 * (lo + hi))
            val.append(np.abs(err[m]).mean())
    ax.plot(mid, val, color=colour, lw=2, marker="o", ms=4, label=label)


def reliability(ax, levels: np.ndarray, covered: dict[str, np.ndarray], colours: dict[str, str]):
    """Share of cases inside an interval against the share it was built to hold."""
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--")
    for name, cov in covered.items():
        ax.plot(levels, cov, color=colours[name], lw=2, marker="o", ms=4, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    style(ax)
