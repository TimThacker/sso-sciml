"""Arm L for task T1: the physics as a penalty during training.

The network predicts an eigenvalue and an eigenvector of the state matrix A(theta),

    theta -> (sigma_hat, f_hat, v_hat),

and is penalised by the eigen-residual ||A v_hat - lambda_hat v_hat||^2 with
||v_hat|| = 1. The residual needs A(theta) but no eigenvalue label. Labelled cases add
an ordinary data loss on sigma and f.

The residual is zero at every eigenpair of A, not only at the sub-synchronous one, so
without labels nothing says which of the 25 the network should find.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .t1 import SIGMA_SCALE

F_SCALE = 60.0  # Hz
RATE = 100.0  # 1/s; A and lambda are divided by this inside the residual


class EigenNet(nn.Module):
    def __init__(self, n_in: int, n_state: int, width: int = 256, depth: int = 4):
        super().__init__()
        layers, last = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(width, 2)
        self.vector = nn.Linear(width, 2 * n_state)
        self.n_state = n_state

    def forward(self, x: torch.Tensor):
        """sigma / SIGMA_SCALE, f / F_SCALE, and the real and imaginary parts of a
        unit-norm eigenvector."""
        h = self.trunk(x)
        y = self.value(h)
        v = self.vector(h)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return y[..., 0], y[..., 1], v[..., : self.n_state], v[..., self.n_state :]


def eigen_residual(a, y_sigma, y_f, v_r, v_i) -> torch.Tensor:
    """Squared eigen-residual of each case, in units of RATE^2. `a` is A / RATE."""
    sigma = y_sigma * (SIGMA_SCALE / RATE)
    omega = y_f * (2 * math.pi * F_SCALE / RATE)
    av_r = torch.einsum("bij,bj->bi", a, v_r)
    av_i = torch.einsum("bij,bj->bi", a, v_i)
    r_r = av_r - (sigma[:, None] * v_r - omega[:, None] * v_i)
    r_i = av_i - (sigma[:, None] * v_i + omega[:, None] * v_r)
    return (r_r**2 + r_i**2).sum(dim=-1)


def singular_residual(a, y_sigma, y_f) -> torch.Tensor:
    """Squared smallest singular value of A - lambda_hat I for each case, in units of
    RATE^2. Zero exactly when lambda_hat is an eigenvalue. Needs no eigenvector: it is
    the eigen-residual already minimised over unit vectors. `a` is A / RATE."""
    sigma = y_sigma * (SIGMA_SCALE / RATE)
    omega = y_f * (2 * math.pi * F_SCALE / RATE)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    shifted = torch.complex(a - sigma[:, None, None] * eye, -omega[:, None, None] * eye)
    return torch.linalg.svdvals(shifted)[:, -1] ** 2


@dataclass(frozen=True)
class EigenConfig:
    steps: int = 4000
    batch: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 256
    depth: int = 4
    eval_every: int = 200
    weight: float = 1.0  # of the eigen-residual against the data loss; 0 switches it off
    form: str = "vector"  # "vector": predicted eigenvector; "singular": no eigenvector


def train_eigen(
    labelled: dict | None,
    held_out: dict | None,
    pool: dict | None,
    seed: int,
    cfg: EigenConfig | None = None,
    device: str = "cuda",
) -> tuple[EigenNet, dict]:
    """Train on a data loss, an eigen-residual, or both.

    `labelled` and `held_out` hold "x", "sigma" (1/s) and "f" (Hz). `pool` holds "x" and
    "a" (state matrices in 1/s) and needs no labels. With held-out cases the weights
    with the lowest held-out error in sigma are kept; without, the last ones.
    """
    cfg = cfg or EigenConfig()
    torch.manual_seed(seed)
    dev = torch.device(device)

    def to(v):
        return torch.as_tensor(v, dtype=torch.float32, device=dev)

    use_res = pool is not None and cfg.weight > 0
    n_in = (labelled or pool)["x"].shape[1]
    n_state = pool["a"].shape[-1] if pool is not None else 1
    model = EigenNet(n_in, n_state, cfg.width, cfg.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    gen = torch.Generator(device=dev).manual_seed(seed)
    if labelled is not None:
        xl, sl, fl = (
            to(labelled["x"]),
            to(labelled["sigma"] / SIGMA_SCALE),
            to(labelled["f"] / F_SCALE),
        )
    if held_out is not None:
        xh, sh = to(held_out["x"]), to(held_out["sigma"])
    if use_res:
        xp, ap = to(pool["x"]), to(pool["a"] / RATE)
    history: dict = {"step": [], "data_loss": [], "residual": [], "held_out_mae": []}
    best, best_state = float("inf"), None
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        loss = torch.zeros((), device=dev)
        data_loss = res = float("nan")
        if labelled is not None:
            idx = torch.randint(0, xl.shape[0], (min(cfg.batch, xl.shape[0]),), device=dev,
                                generator=gen)  # fmt: skip
            ys, yf, _, _ = model(xl[idx])
            d = nn.functional.huber_loss(ys, sl[idx]) + nn.functional.huber_loss(yf, fl[idx])
            loss = loss + d
            data_loss = float(d.detach())
        if use_res:
            idx = torch.randint(0, xp.shape[0], (min(cfg.batch, xp.shape[0]),), device=dev,
                                generator=gen)  # fmt: skip
            out = model(xp[idx])
            if cfg.form == "singular":
                r = singular_residual(ap[idx], out[0], out[1]).mean()
            else:
                r = eigen_residual(ap[idx], *out).mean()
            loss = loss + cfg.weight * r
            res = float(r.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            held = float("nan")
            if held_out is not None:
                model.eval()
                with torch.no_grad():
                    held = float((model(xh)[0] * SIGMA_SCALE - sh).abs().mean())
                model.train()
                if held < best:
                    best = held
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["step"].append(step)
            history["data_loss"].append(data_loss)
            history["residual"].append(res)
            history["held_out_mae"].append(held)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "train_s": time.perf_counter() - t0}


@torch.no_grad()
def predict(model: EigenNet, x: np.ndarray, device: str = "cuda") -> dict:
    """sigma (1/s), f (Hz) and the eigenvector (complex, unit norm) for each row of x."""
    model.eval()
    ys, yf, v_r, v_i = model(torch.as_tensor(x, dtype=torch.float32, device=torch.device(device)))
    return {
        "sigma": (ys * SIGMA_SCALE).double().cpu().numpy(),
        "f": (yf * F_SCALE).double().cpu().numpy(),
        "v": (v_r.double() + 1j * v_i.double()).cpu().numpy(),
    }


def residual_norm(a: np.ndarray, sigma: np.ndarray, f: np.ndarray, v: np.ndarray) -> np.ndarray:
    """||A v - lambda v|| in 1/s for unit v, in float64."""
    lam = sigma + 2j * np.pi * f
    return np.linalg.norm(np.einsum("bij,bj->bi", a, v) - lam[:, None] * v, axis=1)
