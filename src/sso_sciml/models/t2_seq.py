"""Task T2 with operators and sequence models (Phase 7).

Four model families that give the whole response of a case at once:

    deeponet     a branch network on the case, a trunk network on time, and their product
    fno          a Fourier neural operator along time
    tcn          dilated convolutions along time
    transformer  self-attention over the saved times

Each is built with or without the physics as inputs. Without, the inputs are the
parameters and features of time. With (`features=True`), the response of the reduced
model and its sub-synchronous mode are added, so the model maps the reduced response to
the full one.

The output of every family goes through the same envelope as the perceptron of
`models.t2`: it is multiplied by time and by a learned exponential in time, and the
known initial value is added. The loss and the error measure are those of `models.t2`.

Also here: the error against time within the window, and the growth rate read off a
response, for the evaluation of Phase 7.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..data import t2
from . import t2 as mt

ARCHS = ("deeponet", "fno", "tcn", "transformer")
RATE = {"deeponet": 1.0, "fno": 1.0, "tcn": 1.0, "transformer": 0.25}  # share of cfg.lr
PAD = 23  # zero padding of the window for the Fourier layers, which assume a period


@dataclass(frozen=True)
class Config:
    steps: int = 8000
    cases: int = 64  # whole cases per step
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 64
    eval_every: int = 400


def _mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    layers, last = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(last, width), nn.SiLU()]
        last = width
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, n_case: int, n_time: int, n_seq: int, width: int):
        super().__init__()
        self.p = 2 * width
        self.branch = _mlp(n_case + n_seq * t2.N_T, 5 * self.p, 4 * width, 3)
        self.trunk = _mlp(n_time, self.p, 4 * width, 3)

    def forward(self, case: torch.Tensor, when: torch.Tensor, seq: torch.Tensor | None):
        x = case if seq is None else torch.cat([case, seq.flatten(1)], dim=1)
        b = self.branch(x).reshape(-1, 5, self.p)
        return torch.einsum("bcp,tp->btc", b, self.trunk(when)) / math.sqrt(self.p)


class _Spectral(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / width
        self.weight = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, width, time)
        f = torch.fft.rfft(x, dim=2)
        out = torch.zeros_like(f)
        out[:, :, : self.modes] = torch.einsum("bim,iom->bom", f[:, :, : self.modes], self.weight)
        return torch.fft.irfft(out, n=x.shape[2], dim=2)


class FNO(nn.Module):
    def __init__(self, n_in: int, width: int, modes: int = 40, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(n_in, width)
        self.spectral = nn.ModuleList(_Spectral(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.out = _mlp(width, 5, 2 * width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, n_in)
        h = nn.functional.pad(self.lift(x).transpose(1, 2), (0, PAD))
        for k, (spec, loc) in enumerate(zip(self.spectral, self.local, strict=True)):
            h = spec(h) + loc(h)
            if k < len(self.spectral) - 1:
                h = nn.functional.silu(h)
        return self.out(h[:, :, : x.shape[1]].transpose(1, 2))


class TCN(nn.Module):
    def __init__(self, n_in: int, width: int, dilations: tuple = (1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.lift = nn.Conv1d(n_in, width, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.SiLU(),
                          nn.Conv1d(width, width, 3, padding=d, dilation=d))
            for d in dilations
        )  # fmt: skip
        self.out = nn.Conv1d(width, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x.transpose(1, 2))
        for block in self.blocks:
            h = nn.functional.silu(h + block(h))
        return self.out(h).transpose(1, 2)


class Transformer(nn.Module):
    def __init__(self, n_in: int, width: int, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(n_in, width)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")  # fmt: skip
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(self.lift(x)))


class Sequence(nn.Module):
    """One family behind the shared envelope. `case` is (batch, n_case); `seq` is the
    reduced response (batch, N_T, 4) or None. Returns (batch, N_T, 5), zero at t = 0."""

    def __init__(self, arch: str, n_case: int, n_seq: int, width: int):
        super().__init__()
        self.arch = arch
        tau = torch.linspace(0.0, 1.0, t2.N_T)
        phase = tau[:, None] * (2 * math.pi * t2.T_END * torch.tensor(mt.FREQS))
        self.register_buffer("tau", tau)
        self.register_buffer("when", torch.cat([tau[:, None], phase.sin(), phase.cos()], dim=1))
        n_time = self.when.shape[1]
        n_in = n_case + n_time + n_seq
        if arch == "deeponet":
            self.core = DeepONet(n_case, n_time, n_seq, width)
        elif arch == "fno":
            self.core = FNO(n_in, width)
        elif arch == "tcn":
            self.core = TCN(n_in, int(1.5 * width))
        elif arch == "transformer":
            self.core = Transformer(n_in, int(1.5 * width))
        else:
            raise ValueError(arch)
        self.rate = _mlp(n_case, 1, 64, 2)

    def forward(self, case: torch.Tensor, seq: torch.Tensor | None = None) -> torch.Tensor:
        if self.arch == "deeponet":
            u = self.core(case, self.when, seq)
        else:
            b = case.shape[0]
            parts = [case[:, None, :].expand(b, t2.N_T, -1), self.when.expand(b, -1, -1)]
            if seq is not None:
                parts.append(seq)
            u = self.core(torch.cat(parts, dim=2))
        a = torch.clamp(self.rate(case), -4.0, 9.0)
        return (self.tau * torch.exp(a * self.tau))[:, :, None] * u


def train(arch: str, features: bool, labelled: dict, held: dict, seed: int,
          cfg: Config | None = None, device: str = "cuda"):  # fmt: skip
    """Train one family on whole cases. Returns a function from cases to predictions
    (n, N_T, 5) in scaled units, and a record of the training."""
    cfg = cfg or Config()
    torch.manual_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    arm = "F" if features else "N"

    def to(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)

    n = labelled["theta"].shape[0]
    y, m = mt.targets(labelled), mt.mask(labelled)
    scale = np.sqrt((y**2 * m[:, :, None]).sum(axis=(1, 2)) / m.sum(axis=1))
    case, yt, st, mk = to(mt._theta_inputs(arm, labelled)), to(y), to(scale), to(m)
    seq = to(mt._extra(labelled)) if features else None
    y0 = torch.zeros(5, device=dev)
    y0[2] = t2.OFFSET / mt.CHANNEL[2]
    model = Sequence(arch, case.shape[1], 4 if features else 0, cfg.width).to(dev)

    def predict(d: dict, chunk: int = 512) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, d["theta"].shape[0], chunk):
                part = {k: v[lo : lo + chunk] for k, v in d.items() if k != "p"}
                s = to(mt._extra(part)) if features else None
                out.append((model(to(mt._theta_inputs(arm, part)), s) + y0).cpu().numpy())
        model.train()
        return np.concatenate(out)

    lr = cfg.lr * RATE[arch]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=cfg.steps)
    history: dict = {"step": [], "data_loss": [], "held_out": []}
    best, best_state, skipped = float("inf"), None, 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        i = torch.randint(0, n, (min(cfg.cases, n),), device=dev, generator=gen)
        u = model(case[i], seq[i] if seq is not None else None) + y0
        err = (u - yt[i]) * mk[i][:, :, None]
        loss = ((err**2).sum(dim=(1, 2)) / (mk[i].sum(dim=1) * st[i] ** 2)).mean()
        opt.zero_grad(set_to_none=True)
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
            score = float(np.median(mt.case_error(predict(held), held)))
            history["step"].append(step)
            history["data_loss"].append(float(loss.detach()))
            history["held_out"].append(score)
            if np.isfinite(score) and score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"history": history, "train_s": train_s, "best_held_out": best,
            "skipped_steps": skipped,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "peak_memory_mb": (torch.cuda.max_memory_allocated(dev) / 2**20
                               if dev.type == "cuda" else float("nan"))}  # fmt: skip
    return predict, info


# ---------------------------------------------------------------- evaluation over time


def horizon_error(pred: np.ndarray, d: dict, edges: np.ndarray) -> np.ndarray:
    """Relative error of each case within each stretch of time: (cases, len(edges) - 1).
    `edges` are indices of saved times. A stretch a case does not reach before its trip
    is NaN."""
    y, m = mt.targets(d), mt.mask(d)[:, :, None]
    out = np.full((y.shape[0], len(edges) - 1), np.nan)
    for k in range(len(edges) - 1):
        s = slice(int(edges[k]), int(edges[k + 1]))
        full = d["valid"] >= edges[k + 1]
        num = (((pred[:, s] - y[:, s]) ** 2) * m[:, s]).sum(axis=(1, 2))
        den = ((y[:, s] ** 2) * m[:, s]).sum(axis=(1, 2))
        out[full, k] = np.sqrt(num[full] / den[full])
    return out


def growth(y: np.ndarray) -> np.ndarray:
    """Growth rate of each response, 1/s: the log of the ratio of its size over the last
    third of the window to its size over the middle third, over the time between them."""
    third = (t2.N_T - 1) // 3
    mid = np.sqrt((y[:, third : 2 * third] ** 2).mean(axis=(1, 2)))
    late = np.sqrt((y[:, 2 * third : 3 * third] ** 2).mean(axis=(1, 2)))
    return np.log(late / mid) / (third * t2.T_END / (t2.N_T - 1))


def growth_scores(pred: np.ndarray, d: dict) -> dict:
    """Over the cases that do not trip: the error of the growth rate read off the
    prediction, the share of decaying cases predicted to grow, and the share of growing
    cases predicted to decay."""
    whole = d["valid"] >= t2.N_T
    true, got = growth(mt.targets(d)[whole]), growth(pred[whole])
    down, up = true < 0, true > 0
    return {
        "cases": int(whole.sum()),
        "growth_error_median": float(np.median(np.abs(got - true))),
        "growth_error_p90": float(np.quantile(np.abs(got - true), 0.9)),
        "false_divergence": float((got[down] > 0).mean()) if down.any() else float("nan"),
        "missed_growth": float((got[up] < 0).mean()) if up.any() else float("nan"),
    }
