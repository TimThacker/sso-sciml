"""Task T2 models: the response in time after a capacitor-voltage offset, from the
parameters of the case.

Six ways of using the physics, with one network size and one schedule:

    N    parameters and time in, the five observed quantities out
    F    as N, with the response of the reduced model at that time and its
         sub-synchronous mode as further inputs
    R    as N, and the network learns the correction to the reduced model's response
    L    parameters and time in, all 23 states out; besides the data loss, the residual
         of the full equations dx/dt - f(x, theta) at times and cases with no labels
    FL   L with the inputs of F
    A    a linear system whose state matrix is that of the reduced model plus a learned
         correction, with four further learned states; the initial state is exact
    AN   A with a learned nonlinear term added to its dynamics: a neural ODE that
         starts as arm A (step 46)

L0 and FL0 are L and FL with the residual switched off: the same network and data loss.

Everything is in deviations from the steady state. The observed quantities are divided
by fixed sizes (CHANNEL), and the error of a case is relative to the size of its own
response over the times before any trip.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..data import t2
from ..physics.dfig_full import DFIGFull

ARMS = ("N", "F", "R", "L0", "L", "FL0", "FL", "A", "AN")
CHANNEL = np.array([0.05, 0.05, 0.01, 0.01, 0.001])  # size of each observed quantity, pu
FREQS = (10.0, 20.0, 30.0, 40.0, 50.0)  # Hz, time features
RATE = 100.0  # 1/s; the residual is divided by this
N_LATENT = 4
A_RATE = 0.25  # arm A trains at this share of the learning rate; at the full one it blows up
_STATES = ("L0", "L", "FL0", "FL")
_SYSTEMS = ("A", "AN")  # arms built on the reduced model's state matrix
_FEATURES = ("F", "FL0", "FL")


@dataclass(frozen=True)
class Config:
    steps: int = 4000
    batch: int = 4096  # (case, time) pairs; arm A takes batch // 64 whole cases
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 256
    depth: int = 4
    eval_every: int = 200
    weight: float = 1.0  # of the residual against the data loss
    data: float = 1.0  # of the data loss; 0 trains on the residual alone (a parametric PINN)


def _mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    layers, last = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(last, width), nn.SiLU()]
        last = width
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class Response(nn.Module):
    """u(theta, t) = tau * core(theta, features of t, extra) * exp(a(theta) tau), with
    tau = t / T_END. It is zero at t = 0, so a known initial value can be added outside.
    The factor exp(a tau) lets one network cover responses that decay and responses that
    grow a hundredfold."""

    def __init__(self, n_theta: int, n_extra: int, n_out: int, width: int, depth: int):
        super().__init__()
        self.core = _mlp(n_theta + 1 + 2 * len(FREQS) + n_extra, n_out, width, depth)
        self.rate = _mlp(n_theta, 1, 64, 2)
        self.register_buffer("omega", 2 * math.pi * t2.T_END * torch.tensor(FREQS))

    def forward(self, theta: torch.Tensor, tau: torch.Tensor, extra: torch.Tensor | None = None):
        phase = tau[:, None] * self.omega
        parts = [theta, tau[:, None], torch.sin(phase), torch.cos(phase)]
        if extra is not None:
            parts.append(extra)
        a = torch.clamp(self.rate(theta)[:, 0], -4.0, 9.0)
        return (tau * torch.exp(a * tau))[:, None] * self.core(torch.cat(parts, dim=1))


class LinearSystem(nn.Module):
    """Arm A. z holds the 8 states of the reduced model and N_LATENT learned ones:

        dz/dt = (blockdiag(A_red(theta), 0) + M(theta)) z,   z(0) known exactly.

    Stator current and capacitor voltage are read off the reduced states as the reduced
    model does; the DC voltage is a learned combination. M starts near zero, so before
    training this is the reduced model."""

    def __init__(self, n_theta: int, width: int, depth: int, nonlinear: bool = False):
        super().__init__()
        self.n = 8 + N_LATENT
        self.net = _mlp(n_theta, self.n * self.n + self.n, width, depth)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        # Arm AN: dz/dt = (A_red + M) z + RATE * OFFSET * (g(z / OFFSET, theta) - g(0, theta)).
        # The last layer of g starts at zero, so the model starts as arm A, and g(0) is
        # taken off so the steady state stays a steady state.
        self.g = None
        if nonlinear:
            self.g = _mlp(self.n + n_theta, self.n, 64, 2)
            with torch.no_grad():
                self.g[-1].weight.zero_()
                self.g[-1].bias.zero_()
        self.register_buffer("c_red", torch.as_tensor(t2.reduced_outputs(), dtype=torch.float32))
        z0 = torch.zeros(self.n)
        z0[t2.VC] = t2.OFFSET
        self.register_buffer("z0", z0)

    def forward(self, theta: torch.Tensor, a_red: torch.Tensor) -> torch.Tensor:
        """Observed quantities at every saved time, divided by CHANNEL: (batch, N_T, 5)."""
        out = self.net(theta)
        b, n = theta.shape[0], self.n
        total = RATE * out[:, : n * n].reshape(b, n, n)
        total = total + nn.functional.pad(a_red, (0, N_LATENT, 0, N_LATENT))
        h = t2.DT * t2.EVERY
        step = torch.linalg.matrix_exp(total * h)
        z = self.z0.expand(b, n)
        zs = [z]
        if self.g is not None:
            g0 = self.g(torch.cat([torch.zeros_like(z), theta], dim=1))
        for _ in range(t2.N_T - 1):
            if self.g is not None:
                # Lie splitting: an Euler step of the nonlinear term, then the exact
                # step of the linear part.
                kick = self.g(torch.cat([z / t2.OFFSET, theta], dim=1)) - g0
                z = z + h * RATE * t2.OFFSET * kick
            z = torch.einsum("bij,bj->bi", step, z)
            zs.append(z)
        zs = torch.stack(zs, dim=1)
        network = torch.einsum("oj,btj->bto", self.c_red, zs[:, :, :8])
        # The DC voltage starts at its steady value, so it reads the change of z.
        dc = torch.einsum("bj,btj->bt", out[:, n * n :], zs - zs[:, :1])
        scale = torch.as_tensor(CHANNEL, dtype=zs.dtype, device=zs.device)
        return torch.cat([network / scale[:4], dc[:, :, None] / scale[4]], dim=2)


def targets(d: dict) -> np.ndarray:
    """Observed deviations divided by CHANNEL."""
    return d["observed"] / CHANNEL.astype(np.float32)


def mask(d: dict) -> np.ndarray:
    return np.arange(t2.N_T)[None, :] < d["valid"][:, None]


def case_error(pred: np.ndarray, d: dict) -> np.ndarray:
    """Relative error of each case: root of the summed squared error of the five scaled
    quantities over the times before any trip, over the same sum of the truth."""
    y, m = targets(d), mask(d)[:, :, None]
    return np.sqrt((((pred - y) ** 2) * m).sum(axis=(1, 2)) / ((y**2) * m).sum(axis=(1, 2)))


def metrics(pred: np.ndarray, d: dict) -> dict:
    err = case_error(pred, d)
    return {"n": int(err.size), "median": float(np.median(err)), "mean": float(err.mean()),
            "p90": float(np.quantile(err, 0.9)),
            "within_10_percent": float((err < 0.1).mean())}  # fmt: skip


def reduced_alone(d: dict) -> np.ndarray:
    """The reduced model as a prediction: its four quantities, and no DC voltage."""
    out = np.zeros_like(d["observed"])
    out[:, :, :4] = d["reduced"] / CHANNEL[:4].astype(np.float32)
    return out


def unlabelled(theta: np.ndarray, processes: int = 1) -> dict:
    """Cases for the residual: steady state, parameters and the reduced model, with no
    run of the full model."""
    rows = [r for r in t2.setups(theta, processes)]
    keep = [j for j, r in enumerate(rows) if r["solved"] > 0]
    return {
        "theta": theta[keep],
        "x_eq": np.array([rows[j]["x_eq"] for j in keep]),
        "reduced": np.array([rows[j]["reduced"] for j in keep], dtype=np.float32),
        "a_red": np.array([rows[j]["a_red"] for j in keep]),
        "p": {k: np.array([rows[j]["p"][k] for j in keep]) for k in rows[keep[0]]["p"]},
    }


def _theta_inputs(arm: str, d: dict) -> np.ndarray:
    x = t2.scale_theta(d["theta"])
    if arm in _FEATURES:
        mode = t2.reduced_mode(d["a_red"])
        x = np.hstack([x, mode[:, :1] / 10.0, mode[:, 1:] / 60.0])
    return x.astype(np.float32)


def _extra(d: dict) -> np.ndarray:
    return np.arcsinh(d["reduced"] / CHANNEL[:4].astype(np.float32))


def train(
    arm: str,
    labelled: dict,
    held: dict,
    seed: int,
    cfg: Config | None = None,
    pool: dict | None = None,
    device: str = "cuda",
):
    """Train one arm. `labelled` and `held` are cases of the dataset. `pool` holds cases
    used without their labels by the residual: theta, x_eq, reduced, a_red and the
    parameter arrays `p`. Returns a function from cases to predictions (n, N_T, 5) in
    scaled units, and a record of the training."""
    cfg = cfg or Config()
    torch.manual_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def to(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)

    n = labelled["theta"].shape[0]
    y_obs = targets(labelled)
    m = mask(labelled)
    valid = to(labelled["valid"])
    c_obs = t2.observation_matrix()
    if arm in _STATES:
        # Size of each state: its largest deviation, at the median over the labelled cases.
        size = np.maximum(np.median(np.abs(labelled["states"]).max(axis=1), axis=0), 1e-4)
        y = labelled["states"] / size.astype(np.float32)
        y0 = np.zeros(t2.N_STATE, dtype=np.float32)
        y0[t2.VC] = t2.OFFSET / size[t2.VC]
        to_obs = to((c_obs * size[None, :]) / CHANNEL[:, None])
    else:
        y = y_obs - reduced_alone(labelled) if arm == "R" else y_obs
        # The value at t = 0 is the same in every case: the capacitor offset of T2, or
        # zero for a disturbance that starts from rest (`t2_events`).
        y0 = np.median(y[:, 0], axis=0).astype(np.float32)
    scale = np.sqrt(((y if arm != "R" else y_obs) ** 2 * m[:, :, None]).sum(axis=(1, 2))
                    / m.sum(axis=1))  # fmt: skip
    th, yt, st, y0t = to(_theta_inputs(arm, labelled)), to(y), to(scale), to(y0)
    extra = to(_extra(labelled)) if arm in _FEATURES else None
    if arm in _SYSTEMS:
        model = LinearSystem(th.shape[1], cfg.width, cfg.depth, nonlinear=arm == "AN").to(dev)
        a_red = to(labelled["a_red"])
        mt = to(m)
    else:
        model = Response(th.shape[1], 4 if arm in _FEATURES else 0, yt.shape[2], cfg.width,
                         cfg.depth).to(dev)  # fmt: skip

    use_res = arm in ("L", "FL") and cfg.weight > 0
    if use_res:
        plant = DFIGFull()
        p_pool = {k: to(v) for k, v in pool["p"].items()}
        x_eq, th_pool = to(pool["x_eq"]), to(_theta_inputs(arm, pool))
        size_t = to(size)
        if arm == "FL":
            feat = _extra(pool)
            extra_pool = to(feat)
            slope_pool = to(np.gradient(feat, 1.0 / (t2.N_T - 1), axis=1))

    def predict(d: dict, chunk: int = 512) -> np.ndarray:
        model.eval()
        out = []
        tau = torch.linspace(0.0, 1.0, t2.N_T, device=dev)
        with torch.no_grad():
            for lo in range(0, d["theta"].shape[0], chunk):
                part = {k: v[lo : lo + chunk] for k, v in d.items() if k != "p"}
                x = to(_theta_inputs(arm, part))
                b = x.shape[0]
                if arm in _SYSTEMS:
                    out.append(model(x, to(part["a_red"])).cpu().numpy())
                    continue
                ex = to(_extra(part)).reshape(b * t2.N_T, 4) if arm in _FEATURES else None
                u = model(x.repeat_interleave(t2.N_T, 0), tau.repeat(b), ex) + y0t
                u = u.reshape(b, t2.N_T, -1)
                if arm in _STATES:
                    u = u @ to_obs.T
                pred = u.cpu().numpy()
                if arm == "R":
                    pred = pred + reduced_alone(part)
                out.append(pred)
        model.train()
        return np.concatenate(out)

    lr = cfg.lr * (A_RATE if arm in _SYSTEMS else 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=cfg.steps)
    history: dict = {"step": [], "data_loss": [], "residual": [], "held_out": []}
    best, best_state, skipped = float("inf"), None, 0
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        res = float("nan")
        if arm in _SYSTEMS:
            i = torch.randint(0, n, (min(max(cfg.batch // 64, 1), n),), device=dev, generator=gen)
            err = (model(th[i], a_red[i]) - yt[i]) * mt[i][:, :, None]
            loss = ((err**2).sum(dim=(1, 2)) / (mt[i].sum(dim=1) * st[i] ** 2)).mean()
        else:
            i = torch.randint(0, n, (cfg.batch,), device=dev, generator=gen)
            k = (torch.rand(cfg.batch, device=dev, generator=gen) * valid[i]).long()
            u = model(th[i], k / (t2.N_T - 1), extra[i, k] if extra is not None else None) + y0t
            loss = (((u - yt[i, k]) / st[i, None]) ** 2).sum(dim=1).mean()
        data_loss = float(loss.detach())
        loss = cfg.data * loss
        if use_res:
            b = cfg.batch // 4
            i = torch.randint(0, x_eq.shape[0], (b,), device=dev, generator=gen)
            k = torch.randint(0, t2.N_T, (b,), device=dev, generator=gen)
            tau = k / (t2.N_T - 1)
            x_in = th_pool[i]
            if arm == "FL":
                args, tangents = (tau, extra_pool[i, k]), (torch.ones_like(tau), slope_pool[i, k])
                u, du = torch.func.jvp(lambda a, e, x_in=x_in: model(x_in, a, e), args, tangents)
            else:
                u, du = torch.func.jvp(lambda a, x_in=x_in: model(x_in, a), (tau,),
                                       (torch.ones_like(tau),))  # fmt: skip
            x = x_eq[i] + (u + y0t) * size_t
            f = plant.rhs(x, {key: v[i] for key, v in p_pool.items()})
            r = ((du * size_t / t2.T_END - f) / (size_t * RATE)) ** 2
            r = r.mean()
            loss = loss + cfg.weight * r
            res = float(r.detach())
        opt.zero_grad(set_to_none=True)
        # Arm A can step into a matrix whose response overflows within the window. Such a
        # step is left out, so the weights stay finite. It is caught before the backward
        # pass: the gradient of the matrix exponential does not return on such an input.
        norm = None
        if torch.isfinite(loss):
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if norm is not None and torch.isfinite(norm):
            opt.step()
        else:
            skipped += 1
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            score = float(np.median(case_error(predict(held), held)))
            history["step"].append(step)
            history["data_loss"].append(data_loss)
            history["residual"].append(res)
            history["held_out"].append(score)
            if np.isfinite(score) and score < best:
                best = score
                best_state = {key: v.detach().clone() for key, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict, {"history": history, "train_s": time.perf_counter() - t0,
                     "best_held_out": best, "skipped_steps": skipped}  # fmt: skip
