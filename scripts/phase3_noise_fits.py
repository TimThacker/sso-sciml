"""Fit noisy scans of both plant types by least squares and sort the modes by frequency.

    uv run scripts/phase3_noise_fits.py            # every 4th case, about 25,000 cases
    uv run scripts/phase3_noise_fits.py --small    # every 64th, for a quick look

Needs data/t1-v2/scan_<set>.npz from scripts/phase3_scan_t1.py and dataset C1. Writes
data/noise-v1/<kind>_<set>.npz (not committed) and docs/results/scan/noise_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sso_sciml import scanfit
from sso_sciml.data import c1, t1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "noise-v1"
OUT = ROOT / "docs" / "results" / "scan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    step = 64 if args.small else 4
    DATA.mkdir(parents=True, exist_ok=True)
    report: dict = {"columns": scanfit.COLUMNS, "every": step, "sets": {}}
    t_start = time.perf_counter()
    jobs = []
    for name in ("in_distribution", "boundary", *list(t1.SPLITS)[1:]):
        rows = np.load(ROOT / "data" / t1.VERSION / f"scan_{name}.npz")["rows"]
        jobs.append(
            ("t1", name, np.load(ROOT / "data" / t1.VERSION / f"{name}.npz")["theta"][rows])
        )
    for family in c1.FAMILIES:
        for split in c1.SPLITS:
            if not (family == "follow" and split == "fast_pll"):
                theta = np.load(ROOT / "data" / c1.VERSION / f"{family}_{split}.npz")["theta"]
                jobs.append((family, split, theta))
    for kind, name, theta in jobs:
        theta = theta[::step]
        t0 = time.perf_counter()
        out = scanfit.cases(kind, theta, args.processes)
        np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta, out=out)
        info = {"cases": int(np.isfinite(out[:, 0]).sum()), "wall_s": time.perf_counter() - t0}
        report["sets"][f"{kind}_{name}"] = info
        print(kind, name, info)
    report["wall_s_total"] = time.perf_counter() - t_start
    if not args.small:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "noise_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
