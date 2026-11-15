"""Phase 11: build the screening model, register it and write its card.

    uv run scripts/phase11_build.py            # 5 minutes on the GPU
    uv run scripts/phase11_build.py --reuse    # card and figures from the registered model

Needs the dataset from scripts/phase3_generate_t1.py. Trains the screen of step 40 (arm
R, five members, 100,000 cases) and a network for the frequency of the mode, sets the
interval and the fallback rule on the validation fold, scores the model on every test
set, and saves it under registry/ (not committed). Then it answers the test fold
through the service's own code path and times it.

Writes docs/results/phase11/registry.json, docs/model_cards/t1-screen.md and figures
fig109 and fig110. Every number in docs/steps/42-service.md comes from this script.
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

from sso_sciml import registry, report, service
from sso_sciml.data import t1
from sso_sciml.models import compare as cmp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / t1.VERSION
OUT = ROOT / "docs" / "results" / "phase11"
CARDS = ROOT / "docs" / "model_cards"
VERSION = "1.0.0"
N_SERVICE = 500


def card_text(card: dict, extra: dict) -> str:
    m = card["metrics"]
    ind = m["in_distribution"]
    lines = [
        f"# Model card: {card['name']} {card['version']}",
        "",
        "Written by `scripts/phase11_build.py` from the card in the registry. Do not edit by hand.",
        "",
        "## What it is",
        "",
        f"- **Task:** {card['task']}.",
        "- **Output:** growth rate (1/s) with an interval, frequency (Hz), and where the answer "
        "came from.",
        "- **Form:** the hand calculation's growth rate plus a learned correction (arm R), mean "
        f"of {card['architecture']['members']} perceptrons of {card['architecture']['depth']} "
        f"layers of {card['architecture']['width']}; one more perceptron for the frequency.",
        f"- **Trained on:** dataset {card['dataset']['version']} (specification hash "
        f"`{card['dataset']['spec_hash']}`), {card['dataset']['train_cases']:,} cases, with "
        f"{card['dataset']['held_out_cases']:,} held out to pick the weights and "
        f"{card['dataset']['validation_cases']:,} validation cases for the interval and the rule.",
        f"- **Training commit:** `{card['training_commit']}`.",
        f"- **Weights:** SHA-256 `{card['weights_sha256']}`. Not in the repository; rebuilt by the "
        "script above.",
        "",
        "## Valid inputs",
        "",
        "Outside any of these ranges the service answers from the solver, not from this model.",
        "",
        "| Parameter | From | To | Meaning |",
        "|---|---|---|---|",
        *[f"| `{p['name']}` | {p['low']:g} | {p['high']:g} | {p['meaning']} |"
          for p in card["inputs"]],
        "",
        "## The fallback rule",
        "",
        f"Confidence = |growth rate| / (spread of the members + {card['rule']['floor']:.4f} 1/s). "
        f"At or below {card['rule']['level']:.2f} the solver is called. The level is "
        f"{card['rule']['factor']:g} times the largest confidence at which a validation case had "
        "the wrong sign.",
        "",
        f"The interval of the growth rate is the prediction ± {card['interval']['quantile']:.2f} × "
        f"(spread + floor), built to hold {100 * card['interval']['coverage']:.0f} % of cases. The "
        f"frequency is within {card['interval']['f_half_width_hz']:.2f} Hz in "
        f"{100 * card['interval']['coverage']:.0f} % of validation cases.",
        "",
        "## Measured",
        "",
        "| Test set | Cases | Growth-rate error (1/s) | Frequency error (Hz) | Inside the interval "
        "| Sent to the solver | Unstable called stable | Stable called unstable |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {name.replace('_', ' ')} | {v['cases']:,} | {v['growth_rate_mae']:.3f} | "
          f"{v['frequency_mae_hz']:.3f} | {100 * v['interval_coverage']:.1f} % | "
          f"{100 * v['solver_share']:.1f} % | {v['missed_unstable']} of {v['unstable']:,} | "
          f"{v['false_unstable']} |" for name, v in m.items()],
        "",
        "Only the first row is inside the valid inputs. The others show what the model would do "
        "if the range check were removed: it must not be.",
        "",
        "## Limits",
        "",
        f"- In distribution the rule called {ind['missed_unstable']} of {ind['unstable']:,} "
        "unstable test cases stable. That is a count, not a guarantee.",
        "- Cases within the model's error of the stability boundary cannot be called by sign; "
        "the rule sends most of them to the solver and still misses some (row `boundary`).",
        "- Above 80 % compensation and below 0.8 pu rotor speed the model is wrong without "
        "warning (step 40). The range check is what protects those cases.",
        "- The plant is System B2 of this repository: one aggregated doubly-fed plant on a "
        "series-compensated line with a parallel path. It is not a model of any real plant.",
        "- The hand calculation is returned with every answer as a second opinion that does not "
        "depend on the training data.",
        "",
        "## Cost",
        "",
        f"- Answer from the model, through the service code: {1e3 * extra['surrogate_s']:.1f} ms a "
        "case, nearly all of it the hand calculation.",
        f"- Answer from the solver: {1e3 * extra['solver_s']:.1f} ms a case.",
        f"- Training: {extra['build_s']:.0f} s on {extra['device']}.",
        "",
    ]  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CARDS.mkdir(parents=True, exist_ok=True)
    path = OUT / "registry.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, tests = cmp.sets(DATA)
    if args.reuse:
        res = json.loads(path.read_text())
        bundle = registry.load(version=VERSION)
    else:
        sub = {k: v[:100000] for k, v in train.items() if k != "fold"}
        vals = {k: v for k, v in val.items() if k != "fold"}
        t0 = time.perf_counter()
        bundle = registry.build(sub, vals, VERSION, device=device,
                                tests={k: {kk: vv for kk, vv in v.items() if kk != "fold"}
                                       for k, v in tests.items()})  # fmt: skip
        build_s = time.perf_counter() - t0
        bundle.card["dataset"]["source"] = f"{t1.VERSION}, training fold"
        saved = registry.save(bundle)
        bundle = registry.load(version=VERSION)  # what the service will run, on the CPU
        # The test fold through the service's own code, one case at a time.
        d = tests["in_distribution"]
        rows, t_sur, t_sol = [], [], []
        for theta, sigma in zip(d["theta"][:N_SERVICE], d["sigma"][:N_SERVICE], strict=True):
            t0 = time.perf_counter()
            out = service.screen(bundle, theta)
            (t_sur if out.source == "surrogate" else t_sol).append(time.perf_counter() - t0)
            rows.append({"true": float(sigma), "answer": out.growth_rate, "source": out.source,
                         "lower": out.interval.lower, "upper": out.interval.upper,
                         "hand": out.hand_calculation.growth_rate})  # fmt: skip
        t0 = time.perf_counter()
        for theta in d["theta"][:50]:
            t1.solve(theta)
        res = {
            "card": bundle.card, "path": str(saved.relative_to(ROOT)), "build_s": build_s,
            "device": device, "service_cases": N_SERVICE, "rows": rows,
            "surrogate_s": float(np.mean(t_sur)) if t_sur else float("nan"),
            "solver_answer_s": float(np.mean(t_sol)) if t_sol else float("nan"),
            "solver_s": (time.perf_counter() - t0) / 50,
            "service_solver_share": float(np.mean([r["source"] == "solver" for r in rows])),
            "service_wrong_sign": int(sum((r["answer"] > 0) != (r["true"] > 0) for r in rows)),
        }  # fmt: skip
        path.write_text(json.dumps(res))
    card = res["card"]
    (CARDS / "t1-screen.md").write_text(card_text(card, res), encoding="utf-8", newline="\r\n")

    # ---------------- figure: the model on every test set
    m = card["metrics"]
    names = [n.replace("_", " ") for n in m]
    xs = np.arange(len(m))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), facecolor=report.SURFACE)
    axes[0].bar(xs, [v["growth_rate_mae"] for v in m.values()], 0.6, color=report.SERIES[0])
    axes[0].set_ylabel("Mean absolute error of the growth rate (1/s)")
    axes[1].bar(xs, [100 * v["interval_coverage"] for v in m.values()], 0.6,
                color=report.SERIES[2])  # fmt: skip
    axes[1].axhline(100 * card["interval"]["coverage"], color=report.INK, lw=0.8, ls="--")
    axes[1].set_ylabel("Cases inside the interval (%)")
    axes[2].bar(xs - 0.2, [100 * v["solver_share"] for v in m.values()], 0.4,
                color=report.SERIES[3], label="sent to the solver (%)")  # fmt: skip
    axes[2].bar(xs + 0.2, [100 * v["missed_unstable_rate"] for v in m.values()], 0.4,
                color=report.SERIES[5], label="unstable called stable (%)")  # fmt: skip
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for ax in axes:
        ax.set_xticks(xs, names, rotation=20, ha="right", fontsize=8)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    report.title(fig, f"{card['name']} {card['version']} on every test set (only the first is "
                 "inside the valid inputs)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig109_registered_model.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: answers of the service
    rows = res["rows"]
    true = np.array([r["true"] for r in rows])
    ans = np.array([r["answer"] for r in rows])
    sol = np.array([r["source"] == "solver" for r in rows])
    lo, hi = (np.array([r[k] for r in rows]) for k in ("lower", "upper"))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor=report.SURFACE)
    axes[0].errorbar(true[~sol], ans[~sol], yerr=[ans[~sol] - lo[~sol], hi[~sol] - ans[~sol]],
                     ls="", marker="o", ms=2.5, lw=0.5, color=report.SERIES[0],
                     label="from the model, with its interval")  # fmt: skip
    axes[0].plot(true[sol], ans[sol], ls="", marker="s", ms=5, color=report.SERIES[1],
                 label="from the solver")  # fmt: skip
    axes[0].plot([-20, 15], [-20, 15], color=report.INK, lw=0.8, ls="--")
    axes[0].axhline(0, color=report.MUTED, lw=0.6)
    axes[0].axvline(0, color=report.MUTED, lw=0.6)
    axes[0].set_xlabel("Growth rate from the solver (1/s)")
    axes[0].set_ylabel("Answer of the service (1/s)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    axes[0].set_title(f"{len(rows)} test cases through POST /screen", loc="left", fontsize=10)
    report.style(axes[0])
    hand = np.array([r["hand"] for r in rows])
    report.error_cdf(axes[1], {"model, where it answered": (ans - true)[~sol],
                               "hand calculation": hand - true},
                     {"model, where it answered": report.SERIES[0],
                      "hand calculation": report.MUTED})  # fmt: skip
    axes[1].set_xlabel("Absolute error of the growth rate (1/s)")
    axes[1].set_ylabel("Share of cases below it")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    axes[1].set_title("Error of the answer and of the second opinion", loc="left", fontsize=10)
    report.title(fig, "What the service returns on the in-distribution test fold")
    fig.tight_layout()
    fig.savefig(OUT / "fig110_service_answers.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: v for k, v in card.items() if k not in ("inputs", "metrics")}, indent=1))
    for name, v in m.items():
        print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    print({k: v for k, v in res.items() if k not in ("card", "rows")})


if __name__ == "__main__":
    main()
