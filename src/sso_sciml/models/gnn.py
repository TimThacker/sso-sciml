"""A message-passing network over a network of plants (step 51).

Nodes are the buses, with the infinite bus as node 0. Node features: whether the node
is the source, whether it has a plant, and the plant's rotor speed, current-loop
bandwidth and size. Edge features: the line's reactance, X/R and compensation. Edges
go both ways. After `layers` rounds of

    m_i = sum over neighbours j of  phi([h_j, e_ji])
    h_i = h_i + psi([h_i, m_i])

the case's growth rate is read from the maximum and the mean of h over the nodes. The
same weights serve a network of any size, which is what is being tested.

Plain PyTorch: a batch is the union of its cases' graphs, with an index from each node
to its case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

NODE, EDGE = 6, 4
SIGMA_SCALE = 10.0


def _mlp(n_in: int, n_out: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, n_out))  # fmt: skip


def graph(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node features (n + 1, NODE), edges (2, 2n) and edge features (2n, EDGE)."""
    n = case["n"]
    pl, ln = case["plant"], case["line"]
    nodes = np.zeros((n + 1, NODE), dtype=np.float32)
    nodes[0, 0] = 1.0
    for i in range(n):
        if pl["has"][i]:
            nodes[i + 1, 1:] = [1.0, (pl["w_r"][i] - 1.0) / 0.2,
                                np.log(pl["rsc_hz"][i] / 7.75) / 1.35, 2 * pl["size"][i] - 1.2,
                                0.0]  # fmt: skip
    src = np.arange(1, n + 1)
    dst = np.asarray(case["parent"])
    feat = np.column_stack([np.log(ln["x"] / 0.14) / 0.7, np.log(ln["xr"] / 17.3) / 0.55,
                            ln["k"] / 0.35, (ln["k"] > 0).astype(float)])
    feat = feat.astype(np.float32)
    edges = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    return nodes, edges, np.concatenate([feat, feat])


class GNN(nn.Module):
    def __init__(self, width: int = 128, layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(NODE, width)
        self.phi = nn.ModuleList(_mlp(width + EDGE, width, width) for _ in range(layers))
        self.psi = nn.ModuleList(_mlp(2 * width, width, width) for _ in range(layers))
        self.head = _mlp(2 * width, 1, width)

    def forward(self, nodes, edges, efeat, batch, n_cases):
        h = self.embed(nodes)
        for phi, psi in zip(self.phi, self.psi, strict=True):
            msg = phi(torch.cat([h[edges[0]], efeat], dim=1))
            agg = torch.zeros_like(h).index_add_(0, edges[1], msg)
            h = h + psi(torch.cat([h, agg], dim=1))
        mx = torch.full((n_cases, h.shape[1]), -1e9, device=h.device, dtype=h.dtype)
        mx = mx.scatter_reduce(0, batch[:, None].expand_as(h), h, reduce="amax")
        ones = torch.ones_like(batch, dtype=h.dtype)
        count = torch.zeros(n_cases, device=h.device).index_add_(0, batch, ones)
        mean = torch.zeros(n_cases, h.shape[1], device=h.device).index_add_(0, batch, h)
        return self.head(torch.cat([mx, mean / count[:, None]], dim=1))[:, 0]


def collate(graphs: list, device) -> tuple:
    nodes, edges, efeat, batch = [], [], [], []
    offset = 0
    for k, (nd, ed, ef) in enumerate(graphs):
        nodes.append(nd)
        edges.append(ed + offset)
        efeat.append(ef)
        batch.append(np.full(len(nd), k))
        offset += len(nd)

    def to(a, dtype, axis=0):
        return torch.as_tensor(np.concatenate(a, axis=axis), dtype=dtype, device=device)

    return (to(nodes, torch.float32), to(edges, torch.long, axis=1), to(efeat, torch.float32),
            to(batch, torch.long), len(graphs))  # fmt: skip


@dataclass(frozen=True)
class Config:
    steps: int = 6000
    batch: int = 256
    lr: float = 2e-3
    width: int = 128
    layers: int = 4
    eval_every: int = 250


def train(graphs: list, sigma: np.ndarray, held: tuple, seed: int, cfg: Config | None = None,
          device: str = "cuda"):  # fmt: skip
    """`graphs` from `graph`, `sigma` the growth rates (1/s). `held` a pair of the same
    for picking the best step. Returns a prediction function and a record."""
    cfg = cfg or Config()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    model = GNN(cfg.width, cfg.layers).to(dev)
    y = torch.as_tensor(np.clip(sigma, -40, 40) / SIGMA_SCALE, dtype=torch.float32, device=dev)
    hb = collate(held[0], dev)
    hy = np.asarray(held[1])

    def predict(gs: list, chunk: int = 2048) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, len(gs), chunk):
                out.append(model(*collate(gs[lo : lo + chunk], dev)).cpu().numpy())
        model.train()
        return np.concatenate(out) * SIGMA_SCALE

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    hist: dict = {"step": [], "held_out": []}
    best, best_state = float("inf"), None
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(graphs), cfg.batch)
        b = collate([graphs[i] for i in idx], dev)
        loss = nn.functional.huber_loss(model(*b), y[idx], delta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            with torch.no_grad():
                score = float(np.abs(model(*hb).cpu().numpy() * SIGMA_SCALE - hy).mean())
            model.train()
            hist["step"].append(step)
            hist["held_out"].append(score)
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict, {"history": hist, "train_s": time.perf_counter() - t0, "best": best}
