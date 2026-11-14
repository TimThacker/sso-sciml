"""Task T3: from a noisy phase current to the mode and the plant parameters.

A one-dimensional convolutional network. Its input is the recorded deviation of the
phase current from the fundamental, divided by its own size, and the log of that size.
Its outputs are the growth rate and frequency of the sub-synchronous mode and the four
parameters of dataset T2 in scaled units. It is trained on clean waveforms with fresh
white noise added at every step, at a signal-to-noise ratio drawn evenly in decibels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

SIGMA, F_MID, F_SPAN = 10.0, 35.0, 10.0  # scales of the growth rate (1/s) and frequency (Hz)


@dataclass(frozen=True)
class Config:
    steps: int = 6000
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 32
    snr_db: tuple = (0.0, 60.0)
    eval_every: int = 500


class WaveNet(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        chans = (1, width, 2 * width, 3 * width, 4 * width)
        layers = []
        for a, b in zip(chans[:-1], chans[1:], strict=True):
            layers += [nn.Conv1d(a, b, 9, stride=2, padding=4), nn.SiLU(),
                       nn.Conv1d(b, b, 5, padding=2), nn.SiLU()]  # fmt: skip
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(8 * width + 1, 4 * width), nn.SiLU(),
                                  nn.Linear(4 * width, 6))  # fmt: skip

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        size = torch.sqrt((y**2).mean(dim=1, keepdim=True))
        h = self.conv((y / size)[:, None, :])
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2), torch.log(size)], dim=1)
        return self.head(pooled)


def targets(sigma: np.ndarray, f: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.column_stack([sigma / SIGMA, (f - F_MID) / F_SPAN, u]).astype(np.float32)


def decode(out: np.ndarray) -> dict:
    return {"sigma": out[:, 0] * SIGMA, "f": out[:, 1] * F_SPAN + F_MID, "u": out[:, 2:]}


def noisy(clean: torch.Tensor, snr_db: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    size = torch.sqrt((clean**2).mean(dim=1, keepdim=True))
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    return clean + size * 10 ** (-snr_db[:, None] / 20) * noise


def train(clean: np.ndarray, target: np.ndarray, held: tuple, seed: int,
          cfg: Config | None = None, device: str = "cuda"):  # fmt: skip
    """`clean` is (cases, samples), `target` from `targets`. `held` is a pair of the same
    for picking the best step, scored at 30 dB. Returns a function from waveforms to the
    decoded outputs, and a record of the training."""
    cfg = cfg or Config()
    torch.manual_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def to(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)

    x, t = to(clean), to(target)
    hx = noisy(to(held[0]), torch.full((held[0].shape[0],), 30.0, device=dev), gen)
    ht = to(held[1])
    model = WaveNet(cfg.width).to(dev)

    def run(y: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(y[lo : lo + chunk]) for lo in range(0, y.shape[0], chunk)])
        model.train()
        return out

    def predict(y: np.ndarray) -> dict:
        return decode(run(to(y)).cpu().numpy())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    history: dict = {"step": [], "loss": [], "held_out": []}
    best, best_state = float("inf"), None
    t0 = time.perf_counter()
    lo, hi = cfg.snr_db
    for step in range(1, cfg.steps + 1):
        i = torch.randint(0, x.shape[0], (cfg.batch,), device=dev, generator=gen)
        snr = lo + (hi - lo) * torch.rand(cfg.batch, device=dev, generator=gen)
        loss = ((model(noisy(x[i], snr, gen)) - t[i]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            score = float(((run(hx) - ht) ** 2).mean())
            history["step"].append(step)
            history["loss"].append(float(loss.detach()))
            history["held_out"].append(score)
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"history": history, "train_s": time.perf_counter() - t0, "best_held_out": best,
            "parameters": int(sum(p.numel() for p in model.parameters()))}  # fmt: skip
    return predict, info
