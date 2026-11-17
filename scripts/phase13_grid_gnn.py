"""Step 51: several plants on a network, and a graph network that takes the topology.

    uv run scripts/phase13_grid_gnn.py            # generate and train, about 50 minutes
    uv run scripts/phase13_grid_gnn.py --reuse    # figures from the saved numbers

Dataset G1 (`data.grid`): trees of 2 to 4 buses for training, with test sets of the
same sizes and of 5, 6 and 8 buses, which no model sees in training. Truth: the growth
rate of the least-damped sub-synchronous mode. Models, trained on 2 to 4 buses:

    alone      no training: each plant judged on its own network with the others left
               out, and the worst taken. What a plant-by-plant study would say.
    padded     a perceptron on the features of up to 8 buses laid side by side, zeros
               where there is no bus
    GNN-N      the graph network of `models.gnn`
    GNN-F      the same with each plant's own growth rate (as in "alone") as a node
               feature: the physics as inputs

Three seeds for the learned models. Scored by mean absolute error, the share of cases
with the sign wrong, and as a screen: the margin set on the validation fold so that no
unstable case there is cleared, and the share of test cases cleared at it, with the
unstable ones among them.

Writes docs/results/phase13/grid_gnn.json and fig123. Every number in
docs/steps/51-graph-network.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import report
from sso_sciml.data import grid
from sso_sciml.models import compare as cmp
from sso_sciml.models import gnn
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / grid.VERSION
OUT = ROOT / "docs" / "results" / "phase13"
SETS = {"train": ((2, 3, 4), 18000, 0), "validation": ((2, 3, 4), 2000, 100000),
        "test_2_4": ((2, 3, 4), 3000, 200000), "test_5": ((5,), 2000, 300000),
        "test_6": ((6,), 2000, 400000), "test_8": ((8,), 1000, 500000)}  # fmt: skip
TESTS = ("test_2_4", "test_5", "test_6", "test_8")
SEEDS = (0, 1, 2)
MAX_BUS = 8
MODELS = ("alone", "padded", "GNN-N", "GNN-F")


def make(name: str) -> list:
    path = DATA / f"{name}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    sizes, n, seed0 = SETS[name]
    t0 = time.perf_counter()
    rows = grid.generate([sizes[k % len(sizes)] for k in range(n)], seed0)
    rows = [(c, e) for c, e in rows if np.isfinite(e["sigma"]) and np.isfinite(e["alone_max"])]
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(rows))
    print(name, len(rows), f"{time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def node_alone(case: dict) -> np.ndarray:
    """Each plant's own growth rate, as `evaluate` computes it, by bus."""
    out = np.zeros(case["n"])
    for i in range(case["n"]):
        if case["plant"]["has"][i]:
            s = grid.growth(case, only=i)[0]
            out[i] = s if np.isfinite(s) else 0.0
    return out


def graph_f(case: dict, alone: np.ndarray) -> tuple:
    nodes, edges, efeat = gnn.graph(case)
    nodes[1:, 5] = np.arcsinh(alone / 5.0)
    return nodes, edges, efeat


def padded(case: dict) -> np.ndarray:
    nodes, _, efeat = gnn.graph(case)
    n = case["n"]
    x = np.zeros((MAX_BUS, gnn.NODE - 1 + gnn.EDGE + 1), dtype=np.float32)
    x[:n, : gnn.NODE - 1] = nodes[1:, 1:]
    x[:n, gnn.NODE - 1 : gnn.NODE - 1 + gnn.EDGE] = efeat[:n]
    x[:n, -1] = (np.asarray(case["parent"]) + 1) / MAX_BUS  # which bus it hangs from
    return x.ravel()


def scores(pred: np.ndarray, sigma: np.ndarray, margin: float) -> dict:
    cleared = pred < -margin
    return {"mae": float(np.abs(pred - sigma).mean()),
            "wrong_sign": float(((pred > 0) != (sigma > 0)).mean()),
            "cleared": float(cleared.mean()),
            "unstable_cleared": int((cleared & (sigma > 0)).sum()),
            "unstable": int((sigma > 0).sum()), "cases": int(sigma.size)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "grid_gnn.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.reuse:
        data = {k: make(k) for k in SETS}
        sig = {k: np.array([e["sigma"] for _, e in v]) for k, v in data.items()}
        alone = {k: np.array([e["alone_max"] for _, e in v]) for k, v in data.items()}
        t0 = time.perf_counter()
        nodal = {k: [node_alone(c) for c, _ in v] for k, v in data.items()}
        alone_s = (time.perf_counter() - t0) / sum(len(v) for v in data.values())
        g_n = {k: [gnn.graph(c) for c, _ in v] for k, v in data.items()}
        g_f = {k: [graph_f(c, a) for (c, _), a in zip(v, nodal[k], strict=True)]
               for k, v in data.items()}  # fmt: skip
        xp = {k: np.array([padded(c) for c, _ in v]) for k, v in data.items()}
        res: dict = {"cases": {k: len(v) for k, v in data.items()},
                     "unstable_share": {k: float((v > 0).mean()) for k, v in sig.items()},
                     "interaction": {k: float(np.median(np.abs(sig[k] - alone[k]))) for k in sig},
                     "alone_s_per_case": alone_s, "models": {}}  # fmt: skip
        val = sig["validation"]

        def record(name, preds_by_seed):
            runs = []
            for preds in preds_by_seed:
                margin = cmp.safe_margin(preds["validation"], val)
                runs.append({"margin": margin, "sets": {k: scores(preds[k], sig[k], margin)
                                                        for k in TESTS}})  # fmt: skip
            res["models"][name] = {"runs": runs, "summary": {
                k: {m: float(np.mean([r["sets"][k][m] for r in runs]))
                    for m in ("mae", "wrong_sign", "cleared", "unstable_cleared")}
                for k in TESTS}}  # fmt: skip
            summ = res["models"][name]["summary"]
            print(name, {k: {m: round(v, 3) for m, v in s.items()} for k, s in summ.items()},
                  flush=True)

        record("alone", [alone])
        preds = []
        for seed in SEEDS:
            y = np.clip(sig["train"], -40, 40) / mt.SIGMA_SCALE
            h = len(y) // 10
            net, _ = mt.train_mlp(xp["train"][h:], y[h:], xp["train"][:h], y[:h], seed,
                                  mt.TrainConfig(steps=6000), device=device)  # fmt: skip
            preds.append({k: mt.predict(net, v, device) * mt.SIGMA_SCALE for k, v in xp.items()})
        record("padded", preds)
        hist = {}
        for name, g in (("GNN-N", g_n), ("GNN-F", g_f)):
            preds = []
            for seed in SEEDS:
                predict, info = gnn.train(g["train"], sig["train"],
                                          (g["validation"], val), seed, device=device)  # fmt: skip
                preds.append({k: predict(v) for k, v in g.items()})
                hist.setdefault(name, []).append(info["history"])
                print(name, seed, f"{info['train_s']:.0f} s best {info['best']:.3f}", flush=True)
            record(name, preds)
        res["history"] = hist
        res["example"] = {"true": sig["test_6"][:1500].tolist(),
                          "gnn_f": preds[0]["test_6"][:1500].tolist(),
                          "alone": alone["test_6"][:1500].tolist()}  # fmt: skip
        path.write_text(json.dumps(res))
    res = json.loads(path.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    xs = np.arange(len(TESTS))
    names = [t.replace("test_", "").replace("_", " to ") + " buses" for t in TESTS]
    for k, m in enumerate(MODELS):
        s = res["models"][m]["summary"]
        axes[0].bar(xs + (k - 1.5) * 0.2, [s[t]["mae"] for t in TESTS], 0.18,
                    color=report.SERIES[k], label=m)  # fmt: skip
        axes[1].bar(xs + (k - 1.5) * 0.2, [100 * s[t]["cleared"] for t in TESTS], 0.18,
                    color=report.SERIES[k])  # fmt: skip
        for x, t in zip(xs, TESTS, strict=True):
            axes[1].text(x + (k - 1.5) * 0.2, 100 * s[t]["cleared"] + 1,
                         f"{s[t]['unstable_cleared']:.0f}", ha="center", fontsize=7,
                         color=report.LABEL)  # fmt: skip
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean absolute error (1/s)")
    axes[1].set_ylabel("Cases cleared at the margin set on validation (%)")
    for ax in axes[:2]:
        ax.set_xticks(xs, names, fontsize=8)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    axes[1].set_title("Numbers: unstable cases cleared (mean of seeds)", loc="left", fontsize=9)
    ex = res["example"]
    axes[2].plot(ex["true"], ex["alone"], ls="", marker="o", ms=2, color=report.MUTED, alpha=0.5,
                 label="each plant alone, worst")  # fmt: skip
    axes[2].plot(ex["true"], ex["gnn_f"], ls="", marker="o", ms=2, color=report.SERIES[3],
                 alpha=0.6, label="GNN-F")  # fmt: skip
    axes[2].plot([-40, 30], [-40, 30], color=report.INK, lw=0.8, ls="--")
    axes[2].set_xlim(-40, 30)
    axes[2].set_ylim(-40, 30)
    axes[2].set_xlabel("Growth rate of the network (1/s)")
    axes[2].set_ylabel("Predicted (1/s)")
    axes[2].set_title("6 buses, never seen in training", loc="left", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=3)
    report.style(axes[2])
    report.title(fig, "Several plants on a network: trained on 2 to 4 buses, tested on up to 8")
    fig.tight_layout()
    fig.savefig(OUT / "fig123_graph_network.png", dpi=160)
    plt.close(fig)
    print(res["cases"], {k: round(v, 3) for k, v in res["unstable_share"].items()},
          "interaction", {k: round(v, 2) for k, v in res["interaction"].items()},
          "alone s/case", round(res["alone_s_per_case"], 4))  # fmt: skip
    for m in MODELS:
        print(m, {t: {k: round(v, 3) for k, v in s.items()}
                  for t, s in res["models"][m]["summary"].items()},
              "margins", [round(r["margin"], 2) for r in res["models"][m]["runs"]])  # fmt: skip


if __name__ == "__main__":
    main()
