"""Failure analysis: the info-rate planner trusts the satellite prior too much.

When the hotspot product is stale or mislocated, cells outside the flagged
pixels carry a prior of only 0.002, their expected information gain is
negligible, and a greedy planner never goes there.  This script sweeps the
background prior ("tempering") and measures the trade-off.

    python experiments/ablation_prior_tempering.py --seeds 100
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from argus_autonomy import NOMINAL, run_mission
from argus_autonomy.mission import make_scenario

ROOT = Path(__file__).resolve().parents[1]
DEGRADED = replace(NOMINAL, prior=replace(NOMINAL.prior, geoloc_sigma=6.0, latency_steps=12))
BACKGROUNDS = (0.002, 0.005, 0.01, 0.02, 0.05)


def _job(args):
    cond, bg, seed = args
    base = NOMINAL if cond == "nominal" else DEGRADED
    r = run_mission(replace(base, prior=replace(base.prior, p_background=bg)), "info-rate", seed)
    return cond, bg, r.f1, r.t_confirm, r.false_confirmations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    a = ap.parse_args()

    # how often is the prior simply wrong?
    no_detection, outside = 0, 0
    for s in range(a.seeds):
        sc = make_scenario(DEGRADED, s)
        no_detection += int(not sc.hot.any())
        inside = ((sc.prior >= DEGRADED.prior.p_ring) & sc.fire.affected).sum() / sc.fire.affected.sum()
        outside += int(inside == 0)

    jobs = [(c, bg, s) for c in ("nominal", "degraded-prior") for bg in BACKGROUNDS for s in range(a.seeds)]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_job, jobs, chunksize=10))
    res = {"degraded_prior_no_detection": no_detection,
           "degraded_prior_fire_entirely_outside_prior": outside, "seeds": a.seeds, "sweep": {}}
    lines = ["| condition | background prior | F1 mean | runs with F1 < 0.7 | first confirmation, median (min) | false confirmations, mean |",
             "|---|---|---|---|---|---|"]
    for c in ("nominal", "degraded-prior"):
        for bg in BACKGROUNDS:
            rs = [o for o in out if o[0] == c and o[1] == bg]
            d = {"f1_mean": float(np.mean([o[2] for o in rs])),
                 "f1_below_0.7": int(sum(o[2] < 0.7 for o in rs)),
                 "t_confirm_median_min": float(np.nanmedian([o[3] for o in rs])),
                 "false_confirmations_mean": float(np.mean([o[4] for o in rs]))}
            res["sweep"][f"{c}|{bg}"] = d
            lines.append(f"| {c} | {bg} | {d['f1_mean']:.3f} | {d['f1_below_0.7']} | "
                         f"{d['t_confirm_median_min']:.1f} | {d['false_confirmations_mean']:.1f} |")
    (ROOT / "results" / "prior_tempering.json").write_text(json.dumps(res, indent=2))
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "prior_tempering.md").write_text(md)
    print(f"degraded prior: no hotspot detected in {no_detection}/{a.seeds}, "
          f"fire entirely outside hotspot+ring in {outside}/{a.seeds}")
    print(md)


if __name__ == "__main__":
    main()
