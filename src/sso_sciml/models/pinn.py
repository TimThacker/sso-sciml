"""Forward physics-informed network for one trajectory of a known ODE.

A network x_hat(t) is trained on the residual dx_hat/dt - f(x_hat, p) alone: no solver
data. The initial condition is imposed exactly,

    x_hat(t) = x_start + (t - t_start) / T * scale * net(t),

so the trivial solution (staying at the equilibrium) is not available at t = t_start.

The remedies of docs/PLAN.md section 5 are switches of `PinnConfig`:

    nondim        per-state scales on the output and per-equation scales on the residual
    fourier_hz    sin and cos of fixed frequencies as extra inputs
    activation    "tanh" or "sin"
    causal_eps    weight later times only once earlier times have a small residual
    windows       march through time in windows, each a fresh network

Everything runs in float64, like the solver.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

W_REF = 100.0  # rad/s; makes the residual of a 16 Hz oscillation of unit scale about 1


@dataclass(frozen=True)
class PinnConfig:
    t_end: float
    steps: int = 20000
    n_col: int = 2048
    lr: float = 1e-3
    width: int = 128
    depth: int = 4
    activation: str = "tanh"
    nondim: bool = True
    fourier_hz: tuple[float, ...] = ()
    causal_eps: float = 0.0
    causal_bins: int = 32
    windows: int = 1
    log_every: int = 100


class TimeNet(nn.Module):
    """t in [t0, t0 + span] to n outputs."""

    def __init__(self, n_out: int, t0: float, span: float, cfg: PinnConfig):
        super().__init__()
        self.t0, self.span, self.sin = t0, span, cfg.activation == "sin"
        self.register_buffer(
            "omega", 2 * math.pi * torch.tensor(cfg.fourier_hz, dtype=torch.float64)
        )
        n_in = 1 + 2 * len(cfg.fourier_hz)
        sizes = [n_in] + [cfg.width] * cfg.depth
        self.hidden = nn.ModuleList(
            nn.Linear(a, b, dtype=torch.float64) for a, b in zip(sizes[:-1], sizes[1:], strict=True)
        )
        self.out = nn.Linear(cfg.width, n_out, dtype=torch.float64)
        if self.sin:  # initialisation of Sitzmann et al. (2020), first-layer frequency 30
            with torch.no_grad():
                for j, layer in enumerate(self.hidden):
                    bound = (
                        1.0 / layer.in_features if j == 0 else math.sqrt(6.0 / layer.in_features)
                    )
                    layer.weight.uniform_(-bound, bound)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.with_rate(t)[0]

    def with_rate(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output and its time derivative, carried forward through the layers by the
        chain rule. Several times faster than generic forward-mode differentiation."""
        h = 2.0 * (t - self.t0) / self.span - 1.0
        dh = torch.full_like(t, 2.0 / self.span)
        if self.omega.numel():
            s, c = torch.sin(t * self.omega), torch.cos(t * self.omega)
            h = torch.cat([h, s, c], dim=-1)
            dh = torch.cat([dh, self.omega * c, -self.omega * s], dim=-1)
        for j, layer in enumerate(self.hidden):
            z, dz = layer(h), dh @ layer.weight.T
            if self.sin:
                k = 30.0 if j == 0 else 1.0
                h, dh = torch.sin(k * z), k * torch.cos(k * z) * dz
            else:
                h = torch.tanh(z)
                dh = (1.0 - h * h) * dz
        return self.out(h), dh @ self.out.weight.T


class Window:
    """One trained window: x_hat(t) for t in [t0, t0 + span]."""

    def __init__(self, net: TimeNet, x_start: torch.Tensor, scale: torch.Tensor):
        self.net, self.x_start, self.scale = net, x_start, scale

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ramp = (t - self.net.t0) / self.net.span
        return self.x_start + ramp * self.scale * self.net(t)

    def with_rate(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_hat and dx_hat/dt. t has shape (n, 1)."""
        net, dnet = self.net.with_rate(t)
        ramp = (t - self.net.t0) / self.net.span
        x = self.x_start + ramp * self.scale * net
        return x, self.scale * (net / self.net.span + ramp * dnet)


def causal_weights(per_bin: torch.Tensor, eps: float) -> torch.Tensor:
    """Weight of each time bin: exp(-eps * sum of the losses of all earlier bins)."""
    earlier = torch.cumsum(per_bin.detach(), 0) - per_bin.detach()
    return torch.exp(-eps * earlier)


def train_pinn(
    rhs: Callable,
    p: dict,
    x0: torch.Tensor,
    scale: torch.Tensor,
    cfg: PinnConfig,
    seed: int = 0,
    device: str = "cuda",
    monitor: Callable | None = None,
) -> tuple[Callable, dict]:
    """Train on the ODE residual. `scale` is the expected size of each state's deviation.

    Returns a function t (numpy, seconds) -> x_hat (numpy, (n_t, n_state)) and a record
    with the training history and the training time. The history holds the mean squared
    residual every `cfg.log_every` steps. If `monitor(predict, t_reached)` is given, its
    value is logged too; it sees the windows trained so far and plays no part in training.
    The time it takes is included in the training time.
    """
    dev = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    p = {k: v.to(dev) for k, v in p.items()}
    x_start = x0.to(dev)
    s_out = scale.to(dev) if cfg.nondim else torch.ones_like(x_start)
    s_res = scale.to(dev) * W_REF if cfg.nondim else torch.ones_like(x_start)
    span = cfg.t_end / cfg.windows
    steps = cfg.steps // cfg.windows
    windows: list[Window] = []
    history: dict = {"step": [], "residual": [], "t_reached": [], "monitor": []}

    @torch.no_grad()
    def predict(t: np.ndarray, upto: list[Window] | None = None) -> np.ndarray:
        wins = windows if upto is None else upto
        tt = torch.as_tensor(t, dtype=torch.float64, device=dev).reshape(-1, 1)
        out = torch.empty(tt.shape[0], x0.numel(), dtype=torch.float64, device=dev)
        idx = torch.clamp((tt[:, 0] / span).floor().long(), 0, len(wins) - 1)
        for w, win in enumerate(wins):
            mask = idx == w
            if mask.any():
                out[mask] = win(tt[mask])
        return out.cpu().numpy()

    t_begin = time.perf_counter()
    for w in range(cfg.windows):
        t0 = w * span
        win = Window(TimeNet(x_start.numel(), t0, span, cfg).to(dev), x_start, s_out)
        opt = torch.optim.Adam(win.net.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=cfg.lr * 0.01)
        for step in range(1, steps + 1):
            u = torch.rand(cfg.n_col, 1, generator=gen, device=dev, dtype=torch.float64)
            u, _ = torch.sort(u, dim=0)
            x, dx = win.with_rate(t0 + span * u)
            res = ((dx - rhs(x, p)) / s_res) ** 2
            if cfg.causal_eps > 0:
                per_bin = res.reshape(cfg.causal_bins, -1).mean(dim=1)
                weights = causal_weights(per_bin, cfg.causal_eps)
                loss = (weights * per_bin).mean()
            else:
                loss = res.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % cfg.log_every == 0 or step == steps:
                history["step"].append(w * steps + step)
                history["residual"].append(float(res.mean().detach()))
                history["t_reached"].append(t0 + span)
                if monitor is not None:
                    so_far = [*windows, win]
                    history["monitor"].append(
                        float(monitor(lambda t, so_far=so_far: predict(t, so_far), t0 + span))
                    )
        with torch.no_grad():
            x_start = win(torch.full((1, 1), t0 + span, device=dev, dtype=torch.float64))[0]
        windows.append(win)
    train_s = time.perf_counter() - t_begin
    return predict, {"history": history, "train_s": train_s}


def trajectory_error(
    x_hat: np.ndarray, x_ref: np.ndarray, x_eq: np.ndarray, scale: np.ndarray
) -> float:
    """Root-mean-square error over time and states, each state in units of its scale,
    relative to the root-mean-square deviation of the reference from the equilibrium."""
    err = (x_hat - x_ref) / scale
    dev = (x_ref - x_eq) / scale
    return float(np.sqrt((err**2).mean() / (dev**2).mean()))
