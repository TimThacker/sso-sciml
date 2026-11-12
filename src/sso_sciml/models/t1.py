"""Task T1 models: growth rate of the sub-synchronous mode from plant parameters.

Three ways of using the hand calculation, with everything else held equal:

    arm N   parameters in, growth rate out
    arm F   parameters and the hand-calculation quantities in, growth rate out
    arm R   parameters in, correction to the hand-calculated growth rate out

Input scaling is fixed by the parameter ranges in `data.t1.SPEC`, not fitted to data, so
it is the same at every data budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..data import t1

ARMS = ("N", "F", "R")
SIGMA_SCALE = 10.0  # 1/s
_M = {name: j for j, name in enumerate(t1.MECHANISM_COLUMNS)}


def scale_theta(theta: np.ndarray) -> np.ndarray:
    """Parameters to about [-1, 1] over the training range, in log where sampled in log."""
    out = np.empty_like(theta)
    for j, par in enumerate(t1.SPEC):
        v, lo, hi = theta[:, j], par.low, par.high
        if par.log:
            v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        out[:, j] = (2 * v - (lo + hi)) / (hi - lo)
    return out


def scale_mechanism(mech: np.ndarray) -> np.ndarray:
    """Hand-calculation quantities with heavy tails compressed. Fixed transforms."""
    return np.stack(
        [
            mech[:, _M["has_resonance"]],
            mech[:, _M["f_er"]] / 60.0,
            np.clip(mech[:, _M["slip_res"]], -4.0, 0.0),
            np.arcsinh(mech[:, _M["r_machine"]] / 0.05),
            np.arcsinh(mech[:, _M["r_net"]] / 0.05),
            np.arcsinh(mech[:, _M["sigma_hand"]] / SIGMA_SCALE),
            np.arcsinh(mech[:, _M["sigma_circuit"]] / SIGMA_SCALE),
            mech[:, _M["f_circuit"]] / 60.0,
            np.arcsinh(mech[:, _M["sigma_impedance"]] / SIGMA_SCALE),
            mech[:, _M["f_impedance"]] / 60.0,
        ],
        axis=1,
    )


def baseline(mech: np.ndarray) -> np.ndarray:
    """Hand-calculated growth rate used by arm R: the least-damped root of the impedance
    model, which exists in every case."""
    return np.clip(mech[:, _M["sigma_impedance"]], -40.0, 40.0)


def inputs(arm: str, theta: np.ndarray, mech: np.ndarray) -> np.ndarray:
    x = scale_theta(theta)
    return np.hstack([x, scale_mechanism(mech)]) if arm == "F" else x


def targets(arm: str, sigma: np.ndarray, mech: np.ndarray) -> np.ndarray:
    return ((sigma - baseline(mech)) if arm == "R" else sigma) / SIGMA_SCALE


def to_sigma(arm: str, y: np.ndarray, mech: np.ndarray) -> np.ndarray:
    return y * SIGMA_SCALE + (baseline(mech) if arm == "R" else 0.0)


class MLP(nn.Module):
    def __init__(self, n_in: int, width: int = 256, depth: int = 4):
        super().__init__()
        layers, last = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 4000
    batch: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 256
    depth: int = 4
    eval_every: int = 200


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    cfg: TrainConfig | None = None,
    device: str = "cuda",
) -> tuple[MLP, dict]:
    """Train on (x, y); keep the weights with the lowest validation error."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(seed)
    dev = torch.device(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=dev)
    model = MLP(x.shape[1], cfg.width, cfg.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n, batch = xt.shape[0], min(cfg.batch, xt.shape[0])
    best, best_state, history = float("inf"), None, []
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, n, (batch,), device=dev, generator=gen)
        loss = nn.functional.huber_loss(model(xt[idx]), yt[idx], delta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                val = float((model(xv) - yv).abs().mean())
            history.append((step, float(loss.detach()), val))
            if val < best:
                best = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"best_val_mae_scaled": best, "history": history}


@torch.no_grad()
def predict(model: MLP, x: np.ndarray, device: str = "cuda") -> np.ndarray:
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    return model(xt).double().cpu().numpy()


def metrics(sigma_true: np.ndarray, sigma_pred: np.ndarray) -> dict:
    """Errors in 1/s and screening errors. `missed_unstable` is the share of truly unstable
    cases predicted stable."""
    err = np.abs(sigma_pred - sigma_true)
    near = np.abs(sigma_true) < 2.0
    unstable = sigma_true > 0
    return {
        "n": int(sigma_true.size),
        "mae": float(err.mean()),
        "p95": float(np.quantile(err, 0.95)),
        "mae_near_boundary": float(err[near].mean()) if near.any() else float("nan"),
        "missed_unstable": float((sigma_pred[unstable] <= 0).mean())
        if unstable.any()
        else float("nan"),
        "false_alarm": float((sigma_pred[~unstable] > 0).mean())
        if (~unstable).any()
        else float("nan"),
    }
