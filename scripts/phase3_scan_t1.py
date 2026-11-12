"""Scan the doubly-fed plant of dataset T1: its 2 x 2 admittance against the impedance
of its network, for the test fold and every set outside the training range.

    uv run scripts/phase3_scan_t1.py            # 38,511 cases, about 6 minutes
    uv run scripts/phase3_scan_t1.py --small    # one sixteenth, for a quick look

Needs the dataset from scripts/phase3_generate_t1.py. Writes data/t1-v2/scan_<set>.npz
(not committed) and docs/results/scan/t1_scan_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sso_sciml.data import t1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t1.VERSION
OUT = ROOT / "docs" / "results" / "scan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    col = {name: j for j, name in enumerate(t1.SCAN_COLUMNS)}
    report: dict = {"version": t1.VERSION, "columns": t1.SCAN_COLUMNS, "sets": {}}
    t_start = time.perf_counter()
    for name in ("in_distribution", "boundary", *list(t1.SPLITS)[1:]):
        z = np.load(DATA / f"{name}.npz")
        keep = z["solver"][:, 4] > 0
        if name == "in_distribution":
            keep &= z["fold"] == 2
        rows = np.where(keep)[0]
        if args.small:
            rows = rows[::16]
        t0 = time.perf_counter()
        out = t1.scans(z["theta"][rows], args.processes)
        ok = out[:, col["solved"]] > 0
        sigma = out[ok, col["sigma"]]
        info = {
            "cases": int(ok.sum()),
            "unstable": int((sigma > 0).sum()),
            "unstable_sub_synchronous_mode": int((z["solver"][rows, 0][ok] > 0).sum()),
            "largest_growth_differs_from_dataset": float(
                np.abs(sigma - z["solver"][rows, 2][ok]).max()
            ),
            "plant_alone_unstable": int((out[ok, col["poles"]] > 0).sum()),
            "count_agrees_with_eigenvalues": int(
                ((out[ok, col["count"]] > 0) == (sigma > 0)).sum()
            ),
            "wall_s": time.perf_counter() - t0,
        }
        np.savez_compressed(DATA / f"scan_{name}.npz", rows=rows, out=out)
        report["sets"][name] = info
        print(name, json.dumps(info))
    report["wall_s_total"] = time.perf_counter() - t_start
    if not args.small:
        (OUT / "t1_scan_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
