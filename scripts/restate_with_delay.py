"""The converter screens of steps 17 to 24 restated with a control delay.

    uv run scripts/restate_with_delay.py            # a random quarter of the converters, 15,034
    uv run scripts/restate_with_delay.py --small    # every 64th, for a quick look
    uv run scripts/restate_with_delay.py --reuse    # numbers and figure from the saved pass

Needs dataset C1. Writes data/restate-v1/<family>_<set>.npz (not committed),
docs/results/hf/restate_with_delay.json and figure fig72. Every number in
docs/steps/29-restated-with-delay.md comes from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sso_sciml import report, restate
from sso_sciml.data import c1
from sso_sciml.sampling import part

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "restate-v1"
OUT = ROOT / "docs" / "results" / "hf"
PLANTS = {"pll": "converter with PLL", "follow": "converter without PLL"}
COL = {name: j for j, name in enumerate(restate.COLUMNS)}
SCREENS = {
    "loop gain below 1 where not passive (step 17)": lambda d, v: d[f"norm_{v}"] < 1,
    "gain margin, guard 30 deg (step 18)": lambda d, v: d[f"guard_30_{v}"] < 1,
    "Nyquist count, 4,001 points (step 18)": lambda d, v: d[f"count_{v}"] == 0,
    "positive sequence only, Nyquist count": lambda d, v: d[f"count_pos_{v}"] == 0,
    "rational fit, 251 clean points (step 19)": lambda d, v: d[f"fit_{v}"] < 0,
    "1 % scan, network as given within 10 %": lambda d, v: d[f"ref_given_{v}"] < 0,
    "1 % scan, 32 ends of the network ranges (step 24)": lambda d, v: d[f"ref_ends_{v}"] < 0,
}


def sets_of(kind: str) -> list[str]:
    return [s for s in c1.SPLITS if not (kind == "follow" and s == "fast_pll")]


def load(kind: str, names: list[str]) -> dict:
    out = np.concatenate([np.load(DATA / f"{kind}_{n}.npz")["out"] for n in names])
    ok = (out[:, COL["solved_base"]] > 0) & (out[:, COL["solved_delay"]] > 0)
    return {name: out[ok, j] for name, j in COL.items()}


def study(d: dict) -> dict:
    res: dict = {"cases": int(d["t_d"].size), "variants": {}}
    for v in restate.VARIANTS:
        unstable = d[f"top_{v}"] > 0
        alone = d[f"poles_{v}"] == 0
        entry = {
            "unstable": int(unstable.sum()),
            "unstable_share": float(unstable.mean()),
            "plant_alone_unstable": int((~alone).sum()),
            "count_agrees_with_modes": int(((d[f"count_{v}"] > 0) == unstable).sum()),
            "screens": {},
        }
        for name, rule in SCREENS.items():
            cleared = rule(d, v) & alone
            entry["screens"][name] = {
                "cleared": float(cleared.mean()),
                "unstable_cleared": int((cleared & unstable).sum()),
                "stable_cleared": float(cleared[~unstable].mean()),
            }
        res["variants"][v] = entry
    both = (d["top_base"] > 0), (d["top_delay"] > 0)
    res["verdict"] = {
        "stable_to_unstable": int((~both[0] & both[1]).sum()),
        "unstable_to_stable": int((both[0] & ~both[1]).sum()),
    }
    shift = d["top_delay"] - d["top_base"]
    near = np.abs(d["top_base"]) < 5.0
    res["growth_shift_near_boundary"] = [
        float(q) for q in np.quantile(shift[near], [0.05, 0.25, 0.5, 0.75, 0.95])
    ]
    res["delay_ms_quartiles"] = [float(q) for q in 1e3 * np.quantile(d["t_d"], [0.25, 0.5, 0.75])]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.reuse:
        for kind in PLANTS:
            for name in sets_of(kind):
                theta = np.load(ROOT / "data" / c1.VERSION / f"{kind}_{name}.npz")["theta"]
                theta = theta[part(len(theta), 64 if args.small else 4)]
                out = restate.cases(kind, theta, args.processes)
                np.savez_compressed(DATA / f"{kind}_{name}.npz", theta=theta, out=out)
                print("done", kind, name, len(theta))

    res: dict = {"noise": restate.NOISE, "bound": restate.BOUND, "plants": {}}
    for kind, plant in PLANTS.items():
        names = sets_of(kind)
        res["plants"][plant] = r = {
            "in range": study(load(kind, names[:1])),
            "outside": study(load(kind, names[1:])),
            "all": study(load(kind, names)),
        }
        for where, e in r.items():
            print(plant, where, e["cases"], "cases; verdict", e["verdict"], "shift",
                  [round(q, 3) for q in e["growth_shift_near_boundary"]])  # fmt: skip
            for v, x in e["variants"].items():
                print(f"  {v:5s} unstable {x['unstable']} ({x['unstable_share']:.1%}), alone "
                      f"unstable {x['plant_alone_unstable']}, count agrees "
                      f"{x['count_agrees_with_modes']}")  # fmt: skip
                for name, s in x["screens"].items():
                    print(f"      {name:52s} {s['cleared']:.1%} cleared, "
                          f"{s['unstable_cleared']} unstable")  # fmt: skip
    if not args.small:
        (OUT / "restate_with_delay.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: each screen without and with the delay
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.0), facecolor=report.SURFACE, sharey=True)
    ypos = np.arange(len(SCREENS))[::-1]
    for ax, plant in zip(axes, PLANTS.values(), strict=True):
        e = res["plants"][plant]["all"]
        for k, (v, colour) in enumerate(zip(restate.VARIANTS, report.SERIES, strict=False)):
            s = e["variants"][v]["screens"]
            vals = [s[n]["cleared"] for n in SCREENS]
            ax.barh(ypos + (0.5 - k) * 0.38, vals, 0.36, color=colour,
                    label="as in steps 17 to 24" if v == "base" else "with the delay")  # fmt: skip
            for y, n, val in zip(ypos, SCREENS, vals, strict=True):
                ax.text(val + 0.01, y + (0.5 - k) * 0.38, f"{val:.0%}, {s[n]['unstable_cleared']} "
                        "unstable", va="center", fontsize=8, color=report.LABEL)  # fmt: skip
        ax.set_xlim(0, 1.35)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Share of cases cleared")
        ax.set_title(f"{plant[0].upper() + plant[1:]}, {e['cases']:,} cases in and outside the "
                     "range", loc="left", fontsize=10)  # fmt: skip
        report.style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, list(SCREENS), fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL, loc="upper right")
    report.title(fig, "The screens of steps 17 to 24 without and with a control delay. Label: "
                 "share cleared, unstable cases cleared")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig72_restated.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
