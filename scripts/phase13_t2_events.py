"""Step 45: task T2 with three more disturbances.

    uv run scripts/phase13_t2_events.py            # generate and train, about an hour
    uv run scripts/phase13_t2_events.py --reuse    # figures from the saved numbers

For each disturbance of `data.t2_events` (compensation step, voltage dip, power ramp):
8,192 cases in distribution and 1,024 at high compensation are generated (data/t2e-v1,
not committed). The perceptron without physics (N) and with the reduced model's
response as inputs (F) are trained as in step 38 (16,000 steps), on the training fold,
three seeds each, and scored on the test fold and the high-compensation set. The
reduced model alone is scored too.

Writes docs/results/phase13/t2_events.json and fig114, fig115. Every number in
docs/steps/45-t2-events.md comes from this script.
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

from sso_sciml import report
from sso_sciml.data import t2
from sso_sciml.data import t2_events as te
from sso_sciml.models import t2 as mt2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / te.VERSION
OUT = ROOT / "docs" / "results" / "phase13"
SIZES = {"in_distribution": 8192, "high_compensation": 1024}
SEEDS = (0, 1, 2)
STEPS = 16000
ARMS = ("N", "F")
NAME = {"compensation_step": "Compensation step (+5 % of the line)",
        "voltage_dip": "Voltage dip (15 % for 50 ms)",
        "power_ramp": "Power ramp (+0.1 pu over 0.2 s)", "offset": "Capacitor offset (T2)"}


def load_or_make(event: str, split: str, device: str) -> dict:
    path = DATA / f"{event}_{split}.npz"
    if path.is_file():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    theta = t2.sample(SIZES[split], split)
    t0 = time.perf_counter()
    d = te.generate(theta, event, device=device)
    keep = {k: d[k] for k in ("theta", "solved", "observed", "valid", "reduced", "a_red",
                              "sigma", "f")}  # fmt: skip
    if split == "in_distribution":
        keep["fold"] = t2.folds(len(theta))
    keep["wall_s"] = np.array(time.perf_counter() - t0)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **keep)
    return keep


def usable(d: dict) -> dict:
    """The cases that ran, with only the per-case arrays."""
    ok = d["solved"].astype(bool)
    return {k: v[ok] for k, v in d.items() if np.ndim(v) and len(v) == len(ok)}


def take(d: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in d.items() if np.ndim(v) and len(v) == len(m)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "t2_events.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.reuse:
        res = json.loads(path.read_text())
    else:
        res = {"events": {}, "steps": STEPS, "seeds": list(SEEDS)}
        if path.is_file():  # keep the events already done, if run with the same settings
            old = json.loads(path.read_text())
            if old.get("steps") == STEPS and old.get("seeds") == list(SEEDS):
                res["events"] = old["events"]
        for event in te.EVENTS:
            if event in res["events"]:
                continue
            raw = load_or_make(event, "in_distribution", device)
            gen_s = float(raw["wall_s"])
            full = usable(raw)
            high = usable(load_or_make(event, "high_compensation", device))
            high.pop("fold", None)
            train, held, test = (take(full, full["fold"] == j) for j in range(3))
            sets = {"in_distribution": test, "high_compensation": high}
            e: dict = {
                "cases": {"train": int(len(train["theta"])), "held": int(len(held["theta"])),
                          "test": int(len(test["theta"])), "high": int(len(high["theta"]))},
                "generate_s": gen_s,
                "tripped_share": float((full["valid"] < t2.N_T).mean()),
                "size_median": float(np.median(np.abs(full["observed"][:, :, 2]).max(axis=1))),
                "reduced_alone": {k: mt2.metrics(mt2.reduced_alone(v), v) for k, v in sets.items()},
                "runs": [],
            }  # fmt: skip
            for arm in ARMS:
                for seed in SEEDS:
                    predict, info = mt2.train(arm, train, held, seed, mt2.Config(steps=STEPS),
                                              device=device)  # fmt: skip
                    run = {"arm": arm, "seed": seed, "train_s": info["train_s"],
                           "scores": {k: mt2.metrics(predict(v), v) for k, v in sets.items()},
                           "history": {k: info["history"][k]
                                       for k in ("step", "held_out")}}  # fmt: skip
                    if seed == SEEDS[0]:  # the first test case
                        first = take(test, np.arange(len(test["theta"])) == 0)
                        run["example"] = {"true": mt2.targets(first)[0, :, 2].tolist(),
                                          "pred": predict(first)[0, :, 2].tolist()}  # fmt: skip
                    e["runs"].append(run)
                    s = run["scores"]
                    print(event, arm, seed, {k: round(v["median"], 4) for k, v in s.items()},
                          f"{info['train_s']:.0f} s", flush=True)  # fmt: skip
            e["summary"] = {
                arm: {k: {"mean": float(np.mean([r["scores"][k]["median"] for r in e["runs"]
                                                 if r["arm"] == arm])),
                          "sd": float(np.std([r["scores"][k]["median"] for r in e["runs"]
                                              if r["arm"] == arm], ddof=1))}
                      for k in sets}
                for arm in ARMS
            }  # fmt: skip
            res["events"][event] = e
            path.write_text(json.dumps(res))
        # The capacitor offset of T2 for comparison, from step 38.
        prev = json.loads((ROOT / "docs/results/phase7/t2_models.json").read_text())["summary"]
        res["offset_step38"] = {arm: {k: prev[f"mlp/{arm}@1000"][k]["median"]["mean"]
                                      for k in ("in_distribution", "high_compensation")}
                                for arm in ARMS}  # fmt: skip
        path.write_text(json.dumps(res))

    # ---------------- figure: error by disturbance
    events = list(res["events"])
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=report.SURFACE, sharey=True)
    for ax, split in zip(axes, ("in_distribution", "high_compensation"), strict=True):
        xs = np.arange(len(events))
        for k, arm in enumerate(ARMS):
            v = [res["events"][e]["summary"][arm][split] for e in events]
            ax.bar(xs + (k - 0.5) * 0.28, [x["mean"] for x in v], 0.26,
                   yerr=[x["sd"] for x in v], color=report.ARM_COLOUR[arm],
                   label=report.ARM_NAME[arm])  # fmt: skip
        ax.plot(xs, [res["events"][e]["reduced_alone"][split]["median"] for e in events],
                marker="_", ms=34, lw=0, color=report.INK, label="reduced model alone")  # fmt: skip
        ax.set_xticks(xs, [NAME[e] for e in events], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(split.replace("_", " "), loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Median relative error of the response")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, markerscale=0.4)
    report.title(fig, "Three more disturbances: perceptrons trained on about 6,500 trajectories "
                 "of each (three seeds)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig114_t2_events.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: one case of each, and training
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.0), facecolor=report.SURFACE)
    tt = t2.times()
    for col, e in enumerate(events):
        runs = res["events"][e]["runs"]
        for arm in ARMS:
            r = next(x for x in runs if x["arm"] == arm and "example" in x)
            if arm == "N":
                axes[0, col].plot(tt, r["example"]["true"], color=report.INK, lw=2.2,
                                  label="full model")  # fmt: skip
            axes[0, col].plot(tt, r["example"]["pred"], color=report.ARM_COLOUR[arm], lw=1.1,
                              ls="--", label=report.ARM_NAME[arm])  # fmt: skip
            hist = [(np.array(x["history"]["step"]), np.array(x["history"]["held_out"]))
                    for x in runs if x["arm"] == arm]  # fmt: skip
            report.curves(axes[1, col], hist, report.ARM_COLOUR[arm], report.ARM_NAME[arm])
        axes[0, col].set_title(NAME[e], loc="left", fontsize=10)
        axes[0, col].set_xlabel("Time (s)")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Training step")
        for ax in axes[:, col]:
            report.style(ax)
    axes[0, 0].set_ylabel("Capacitor voltage D, scaled")
    axes[1, 0].set_ylabel("Held-out median relative error")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    report.title(fig, "One test case of each disturbance (first seed), and training")
    fig.tight_layout()
    fig.savefig(OUT / "fig115_t2_events_cases.png", dpi=160)
    plt.close(fig)

    for e, v in res["events"].items():
        print(e, v["cases"], "tripped", round(v["tripped_share"], 3), "reduced",
              {k: round(x["median"], 3) for k, x in v["reduced_alone"].items()},
              {a: {k: (round(x["mean"], 4), round(x["sd"], 4)) for k, x in s.items()}
               for a, s in v["summary"].items()})  # fmt: skip
    print("offset, step 38 at 1,000 trajectories", res.get("offset_step38"))


if __name__ == "__main__":
    main()
