"""Step 49: System C, the generic converter, as a learned task.

    uv run scripts/phase13_system_c.py            # 10 training runs, about 2 minutes
    uv run scripts/phase13_system_c.py --reuse    # figures from the saved numbers

Needs the dataset from scripts/phase3_generate_c1.py. The target is the largest real
part of any mode of the converter on its network (1/s), the truth of step 17. Both
families (with and without a phase-locked loop) are pooled, with the family as an input.

    arm N   the 12 parameters
    arm F   the parameters and what the converter's 2 x 2 admittance scan against the
            network gives without solving the model (step 17): the largest loop gain
            where the converter is not passive, the spectral radius of the loop matrix,
            the most negative conductance, the highest frequency at which the converter
            is not passive, the growth rate from the positive-sequence reactance
            crossing, and the operating point (terminal voltage, reactive current)

There is no arm R: no quantity available without a solve is close enough to the growth
rate to serve as a baseline (the best, the positive-sequence crossing, has a correlation
of 0.15 with it).

Each arm is an ensemble of five perceptrons of step 09 (4,000 steps). The fallback rule
of step 40 is set on half of the validation fold (factor 1.5) and applied to the test
fold and the three outside sets of step 17, without a range check.

Writes docs/results/phase13/system_c.json and fig121. Every number in
docs/steps/49-system-c.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import fallback as fb
from sso_sciml import report
from sso_sciml.data import c1
from sso_sciml.models import t1 as mt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / c1.VERSION
OUT = ROOT / "docs" / "results" / "phase13"
C = {c: i for i, c in enumerate(c1.COLUMNS)}
MEMBERS, FACTOR, CLIP = 5, 1.5, 40.0
SPLITS = {"pll": ("in_distribution", "fast_pll", "fast_dc_loop", "very_weak"),
          "follow": ("in_distribution", "fast_dc_loop", "very_weak")}  # fmt: skip


def scale(theta: np.ndarray) -> np.ndarray:
    out = np.empty_like(theta)
    for j, par in enumerate(c1.SPEC):
        v, lo, hi = theta[:, j], par.low, par.high
        if par.log:
            v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        out[:, j] = (2 * v - (lo + hi)) / (hi - lo)
    return out


def features(out: np.ndarray) -> np.ndarray:
    f = np.column_stack([
        np.log(out[:, C["small_gain"]]), np.log(out[:, C["radius"]]),
        np.arcsinh(out[:, C["g_min"]]), np.log(out[:, C["f_passive"]]) / 3.0,
        np.arcsinh(out[:, C["scan_sigma"]] / 10.0) / 3.0,
        out[:, C["v_t"]] - 1.0, out[:, C["iq"]],
    ])  # fmt: skip
    return np.nan_to_num(f, nan=0.0, posinf=5.0, neginf=-5.0)


def load(family: str, split: str) -> dict:
    z = np.load(DATA / f"{family}_{split}.npz")
    ok = z["out"][:, 0] > 0
    d = {"theta": z["theta"][ok], "out": z["out"][ok],
         "family": np.full(int(ok.sum()), 1.0 if family == "pll" else -1.0)}  # fmt: skip
    if "fold" in z.files:
        d["fold"] = z["fold"][ok]
    d["sigma"] = d["out"][:, C["sigma"]]
    return d


def cat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k in
            ("theta", "out", "family", "sigma", "fold")}  # fmt: skip


def take(d: dict, m) -> dict:
    return {k: v[m] for k, v in d.items()}


def inputs(arm: str, d: dict) -> np.ndarray:
    x = np.column_stack([scale(d["theta"]), d["family"]])
    return np.hstack([x, features(d["out"])]) if arm == "F" else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "system_c.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.reuse:
        inside = cat([load(f, "in_distribution") for f in SPLITS])
        train, val, test = (take(inside, inside["fold"] == j) for j in range(3))
        half = val["sigma"].size // 2
        tune = take(val, np.arange(val["sigma"].size) >= half)
        tests = {"in_distribution": test}
        for split in ("fast_pll", "fast_dc_loop", "very_weak"):
            tests[split] = cat([load(f, split) for f in SPLITS if split in SPLITS[f]])
        res: dict = {"cases": {k: int(v["sigma"].size) for k, v in
                               {"train": train, "tune": tune, **tests}.items()},
                     "unstable": {k: int((v["sigma"] > 0).sum()) for k, v in tests.items()},
                     "clip": CLIP, "arms": {}}  # fmt: skip
        y = np.clip(train["sigma"], -CLIP, CLIP) / mt.SIGMA_SCALE
        h = train["sigma"].size // 5
        for arm in ("N", "F"):
            x = inputs(arm, train)
            t0 = time.perf_counter()
            nets = [mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed, device=device)[0]
                    for seed in range(MEMBERS)]  # fmt: skip
            fit_s = time.perf_counter() - t0

            def ens(d, arm=arm, nets=nets):
                p = np.array([mt.predict(n, inputs(arm, d), device) * mt.SIGMA_SCALE
                              for n in nets])  # fmt: skip
                return p.mean(axis=0), p.std(axis=0)

            pt, st = ens(tune)
            e: dict = {"fit_s": fit_s, "rules": {}}
            for det in ("margin only", "ensemble spread"):
                d_t = np.zeros_like(st) if det == "margin only" else st
                floor = 1.0 if det == "margin only" else float(np.median(st))
                level = FACTOR * fb.threshold(fb.confidence(pt, d_t, floor), pt, tune["sigma"])
                e["rules"][det] = {"floor": floor, "level": level, "sets": {}}
                for k, d in tests.items():
                    p, s = ens(d)
                    dd = np.zeros_like(s) if det == "margin only" else s
                    conf = fb.confidence(p, dd, floor)
                    truth = np.clip(d["sigma"], -CLIP, CLIP)
                    e["rules"][det]["sets"][k] = {
                        "mae": float(np.abs(p - truth).mean()),
                        "wrong_sign": int(((p > 0) != (d["sigma"] > 0)).sum()),
                        "policy": fb.outcome(fb.decide(conf, p, level), d["sigma"]),
                        "share_for_no_miss": fb.share_for_no_miss(conf, p, d["sigma"]),
                    }  # fmt: skip
                    if det == "ensemble spread" and k == "in_distribution":
                        keep = np.random.default_rng(0).permutation(p.size)[:2500]
                        e["parity"] = {"true": truth[keep].tolist(), "pred": p[keep].tolist()}
            res["arms"][arm] = e
            for det, r in e["rules"].items():
                print(arm, det, {k: (round(v["mae"], 3),
                                     round(100 * v["policy"]["solver_share"], 2),
                                     v["policy"]["missed_unstable"], v["policy"]["unstable"])
                                 for k, v in r["sets"].items()}, flush=True)  # fmt: skip
        path.write_text(json.dumps(res))
    res = json.loads(path.read_text())

    tests = list(res["unstable"])
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), facecolor=report.SURFACE)
    xs = np.arange(len(tests))
    for k, arm in enumerate(("N", "F")):
        r = res["arms"][arm]["rules"]["ensemble spread"]["sets"]
        axes[0].bar(xs + (k - 0.5) * 0.35, [r[t]["policy"]["solver_share"] * 100 for t in tests],
                    0.33, color=report.ARM_COLOUR[arm], label=report.ARM_NAME[arm])  # fmt: skip
        missed = [r[t]["policy"]["missed_unstable"] + 0.1 for t in tests]
        axes[1].bar(xs + (k - 0.5) * 0.35, missed, 0.33, color=report.ARM_COLOUR[arm])
    for ax, lab in zip(axes[:2], ("Cases sent to the solver (%)", "Unstable cases called stable "
                                  "(+0.1)"), strict=True):  # fmt: skip
        ax.set_xticks(xs, [t.replace("_", " ") for t in tests], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    par = res["arms"]["F"]["parity"]
    report.parity(axes[2], np.array(par["true"]), np.array(par["pred"]), (-15.0, 40.0))
    axes[2].set_xlabel("Largest real part from the solver (1/s)")
    axes[2].set_ylabel("Ensemble, arm F (1/s)")
    axes[2].set_title("Test fold", loc="left", fontsize=10)
    report.title(fig, "System C as a learned screen: ensembles of five, with the fallback rule "
                 "of step 40 (no range check)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig121_system_c.png", dpi=160)
    plt.close(fig)
    print(res["cases"], res["unstable"])
    for arm, e in res["arms"].items():
        for det, r in e["rules"].items():
            print(arm, det, "level", round(r["level"], 3), {k: (round(v["mae"], 3), v["wrong_sign"],
                  round(100 * v["policy"]["solver_share"], 2), v["policy"]["missed_unstable"],
                  v["policy"]["false_unstable"], round(100 * v["share_for_no_miss"], 2))
                  for k, v in r["sets"].items()})  # fmt: skip


if __name__ == "__main__":
    main()
