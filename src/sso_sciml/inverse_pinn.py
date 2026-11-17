"""Inverse physics-informed network for the four parameters of task T2 (step 47).

From one record of the line current after the capacitor-voltage offset, find the
parameters. Trainable together:

    u         the four parameters in the scaled units of `t2.scale_theta`
    p_m       the mechanical power, which the steady state fixes and which is unknown
    x_eq      the steady state (23 values; the rotor speed is taken from u)
    net(t)    the deviation of every state from x_eq over the window

    x_hat(t) = x_eq + offset + (t / T) * S * net(t)

so the initial state is the steady state plus the known offset. The loss is the misfit
of the line current to the record, plus the residual of the full equations of
`dfig_full` along x_hat, plus their residual at x_eq. Every parameter of those equations
that depends on u is written here in torch, so the residual is differentiable in u.
The others take the values of `t2.plant`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import t2
from .models.pinn import PinnConfig, TimeNet

RATE = 100.0  # 1/s; residuals are divided by this times the state's size


@dataclass(frozen=True)
class Config:
    steps: int = 15000
    n_col: int = 1024
    lr_net: float = 1e-3
    lr_u: float = 3e-3
    lr_eq: float = 1e-4
    w_res: float = 1.0
    w_eq: float = 10.0
    fourier_hz: tuple = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    log_every: int = 500


def unscale_torch(u: torch.Tensor) -> torch.Tensor:
    """Inverse of `t2.scale_theta`, in torch."""
    out = []
    for j, par in enumerate(t2.SPEC):
        lo, hi = (math.log(par.low), math.log(par.high)) if par.log else (par.low, par.high)
        v = 0.5 * (u[j] * (hi - lo) + lo + hi)
        out.append(torch.exp(v) if par.log else v)
    return torch.stack(out)


def parameters(theta: torch.Tensor, p_m: torch.Tensor, base: dict) -> dict:
    """The parameter dictionary of the full model for theta = (k, w_r, rsc_hz, pll_hz),
    built from `base` (any case of T2) with every entry that depends on theta replaced."""
    model, _, _ = t2.plant(np.array([0.2, 1.0, 10.0, 10.0]))
    d, c = model.data, model.controls
    k, _, rsc, pll = theta
    alpha = 2 * math.pi * rsc
    x_rr = d.x_lr + d.x_m
    sigma_x = x_rr - d.x_m**2 / (d.x_ls + d.x_m)
    w_out = 2 * math.pi * t2.OUTER_RATIO * rsc
    wn = 2 * math.pi * pll
    q = dict(base)
    q.update({"x_c": k * d.x_e, "kp": alpha * sigma_x / d.wb, "ki": alpha * d.r_r,
              "pll_kp": 2 * c.pll_zeta * wn, "pll_ki": wn**2, "out_kp": w_out * c.t_f,
              "out_ki": w_out, "p_m": p_m})  # fmt: skip
    return q


def fit(record: np.ndarray, u0: np.ndarray, x_eq0: np.ndarray, p_m0: float, base: dict,
        size: np.ndarray, cfg: Config | None = None, seed: int = 0,
        device: str = "cuda") -> dict:  # fmt: skip
    """`record` is the line current deviation (N_T, 2) in pu, with noise. `u0`, `x_eq0`
    and `p_m0` are the start; `base` a parameter dictionary of T2 (floats); `size` the
    size of each state's deviation. Returns the parameters found and the history."""
    cfg = cfg or Config()
    dev = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    f64 = {"dtype": torch.float64, "device": dev}
    model, _, _ = t2.plant(np.array([0.2, 1.0, 10.0, 10.0]))
    base_t = {k: torch.as_tensor(v, **f64) for k, v in base.items()}
    s = torch.as_tensor(size, **f64)
    off = torch.zeros(t2.N_STATE, **f64)
    off[t2.VC] = t2.OFFSET
    y = torch.as_tensor(record, **f64)
    t_rec = torch.as_tensor(t2.times(), **f64)[:, None]
    span = t2.T_END

    u = torch.nn.Parameter(torch.as_tensor(u0, **f64))
    p_m = torch.nn.Parameter(torch.as_tensor(p_m0, **f64))
    x_eq = torch.nn.Parameter(torch.as_tensor(x_eq0, **f64))
    pc = PinnConfig(t_end=span, width=128, depth=4, fourier_hz=cfg.fourier_hz)
    net = TimeNet(t2.N_STATE, 0.0, span, pc).to(dev)
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": cfg.lr_net},
                            {"params": [u], "lr": cfg.lr_u},
                            {"params": [p_m, x_eq], "lr": cfg.lr_eq}])  # fmt: skip
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=0.0)
    noise = float(y.pow(2).mean().sqrt())

    def state(t):
        n, dn = net.with_rate(t)
        ramp = t / span
        eq = x_eq.clone()
        eq = torch.cat([eq[:8], unscale_torch(torch.clamp(u, -1.6, 1.6))[1:2], eq[9:]])
        return eq, eq + off + ramp * s * n, s * (n / span + ramp * dn)

    hist: dict = {"step": [], "data": [], "residual": [], "equilibrium": [], "u": []}
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        theta = unscale_torch(torch.clamp(u, -1.6, 1.6))
        q = parameters(theta, p_m, base_t)
        tc = span * torch.rand(cfg.n_col, 1, generator=gen, **f64)
        eq, x, dx = state(tc)
        res = ((dx - model.rhs(x, q)) / (s * RATE)).pow(2).mean()
        r_eq = (model.rhs(eq, q) / (s * RATE)).pow(2).mean()
        _, xr, _ = state(t_rec)
        il = t2.observe(model, xr, q)[:, :2] - t2.observe(model, eq[None], q)[:, :2]
        data = ((il - y) / noise).pow(2).mean()
        loss = data + cfg.w_res * res + cfg.w_eq * r_eq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*net.parameters(), u, p_m, x_eq], 10.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps:
            hist["step"].append(step)
            hist["data"].append(float(data.detach()))
            hist["residual"].append(float(res.detach()))
            hist["equilibrium"].append(float(r_eq.detach()))
            hist["u"].append(torch.clamp(u, -1.6, 1.6).detach().cpu().tolist())
    return {"u": np.array(hist["u"][-1]), "history": hist, "train_s": time.perf_counter() - t0}
