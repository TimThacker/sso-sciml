"""Fit a rational function to thinned and to noisy scans of every case of both plant
types, and read the growth rate of the least-damped mode from its roots.

    uv run scripts/phase3_fit_scans.py            # 100,000 cases, about 30 minutes
    uv run scripts/phase3_fit_scans.py --small    # one sixteenth, for a quick look

Needs data/t1-v2/scan_<set>.npz from scripts/phase3_scan_t1.py and dataset C1 from
scripts/phase3_generate_c1.py. Writes data/<version>/fit_<set>.npz (not committed) and
docs/results/scan/fit_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sso_sciml import admittance as adm
from sso_sciml.data import c1, t1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "scan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    step = 16 if args.small else 1
    report: dict = {"columns": adm.FIT_COLUMNS, "sets": {}}
    t_start = time.perf_counter()

    for name in ("in_distribution", "boundary", *list(t1.SPLITS)[1:]):
        folder = ROOT / "data" / t1.VERSION
        rows = np.load(folder / f"scan_{name}.npz")["rows"][::step]
        theta = np.load(folder / f"{name}.npz")["theta"][rows]
        t0 = time.perf_counter()
        out = t1.fits(theta, args.processes)
        np.savez_compressed(folder / f"fit_{name}.npz", rows=rows, out=out)
        report["sets"][f"t1_{name}"] = {"cases": len(rows), "wall_s": time.perf_counter() - t0}
        print("t1", name, report["sets"][f"t1_{name}"])

    for family in c1.FAMILIES:
        for split in c1.SPLITS:
            if family == "follow" and split == "fast_pll":
                continue
            folder = ROOT / "data" / c1.VERSION
            rows = np.arange(len(np.load(folder / f"{family}_{split}.npz")["theta"]))[::step]
            theta = np.load(folder / f"{family}_{split}.npz")["theta"][rows]
            t0 = time.perf_counter()
            out = c1.fits(theta, family, args.processes)
            np.savez_compressed(folder / f"fit_{family}_{split}.npz", rows=rows, out=out)
            key = f"c1_{family}_{split}"
            report["sets"][key] = {"cases": len(rows), "wall_s": time.perf_counter() - t0}
            print("c1", family, split, report["sets"][key])

    report["wall_s_total"] = time.perf_counter() - t_start
    if not args.small:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fit_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
