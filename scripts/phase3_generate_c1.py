"""Generate dataset C1: a generic converter on a series-compensated or weak network.

    uv run scripts/phase3_generate_c1.py            # full dataset
    uv run scripts/phase3_generate_c1.py --small    # one sixteenth, for a quick look

Writes data/c1-v1/<family>_<split>.npz (not committed) and
docs/results/converter/c1_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sso_sciml.data import c1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / c1.VERSION
OUT = ROOT / "docs" / "results" / "converter"


def summarise(out: np.ndarray) -> dict:
    col = {name: out[:, j] for j, name in enumerate(c1.COLUMNS)}
    ok = col["solved"] > 0
    sigma = col["sigma"][ok]
    return {
        "rows": int(len(out)),
        "no_steady_state": int((~ok).sum()),
        "unstable": float((sigma > 0).mean()),
        "within_1_of_boundary": float((np.abs(sigma) < 1.0).mean()),
        "sigma_max": float(sigma.max()),
        "converter_alone_unstable": float((col["poles"][ok] > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    scale = 16 if args.small else 1
    sizes = {"pll": 32768 // scale, "follow": 8192 // scale}
    n_ex = 4096 // scale

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"version": c1.VERSION, "spec_hash": c1.spec_hash(), "sets": {}}
    t_start = time.perf_counter()
    first = None
    for family in c1.FAMILIES:
        for split in c1.SPLITS:
            if family == "follow" and split == "fast_pll":
                continue  # this family has no phase-locked loop
            n = sizes[family] if split == "in_distribution" else n_ex
            theta = c1.sample(n, family, split)
            t0 = time.perf_counter()
            out = c1.generate(theta, family, args.processes)
            info = summarise(out)
            info["wall_s"] = time.perf_counter() - t0
            extra = {"fold": c1.folds(n)} if split == "in_distribution" else {}
            np.savez_compressed(DATA / f"{family}_{split}.npz", theta=theta, out=out, **extra)
            report["sets"][f"{family}_{split}"] = info
            print(family, split, json.dumps(info))
            if first is None:
                first = (theta, out)

    again = c1.generate(first[0][:128], "pll", 1)
    report["regenerated_rows_max_abs_diff"] = float(np.nanmax(np.abs(again - first[1][:128])))
    report["wall_s_total"] = time.perf_counter() - t_start
    report["columns"] = {"theta": c1.NAMES, "out": c1.COLUMNS}
    if not args.small:
        (OUT / "c1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, indent=1))


if __name__ == "__main__":
    main()
