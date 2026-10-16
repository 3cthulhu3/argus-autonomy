"""v1.1 milestone (iii): sweep runs for the threshold sweep and calibration.

    python experiments/run_sweep.py --split select     # B 0-999, A/C 0-499
    python experiments/run_sweep.py --split heldout    # B 1000-1999, A/C 500-999
    python experiments/run_sweep.py --split select --quick

Every mission flies to endurance without deciding (design doc §13) and logs its
log-odds trajectory; experiments/analyse_sweep.py turns the logs into the decision
of every threshold pair. Writes results/sweep_<split>.jsonl.gz (one mission per line).
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from argus_autonomy import NOMINAL
from argus_autonomy.config import SPRTConfig
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("M1", "M2", "M3")
STEER = -4.595
GRID = (2.0, 3.0, 4.595, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0)
SPLITS = {"select": {"B": range(0, 1000), "A": range(0, 500), "C": range(0, 500)},
          "heldout": {"B": range(1000, 2000), "A": range(500, 1000), "C": range(500, 1000)}}


def _job(args):
    method, family, seed = args
    r = run_decision_mission(NOMINAL, method, family, seed, sprt=SPRTConfig(steer_clear=STEER),
                             sweep=True, alert_levels=GRID)
    return {
        "method": method, "family": family, "seed": seed,
        "trace": [[int(t), round(float(l), 4), bool(g)] for t, l, g in r.trace],
        "crossings": {str(k): v for k, v in r.crossings.items()},
        "geofence_violations": r.geofence_violations, "returned_to_base": bool(r.returned_to_base),
        "n_close_looks": r.n_close_looks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    jobs = [(m, f, s) for m in METHODS for f, seeds in SPLITS[args.split].items()
            for s in (list(seeds)[:10] if args.quick else seeds)]
    out = ROOT / "results" / f"sweep_{args.split}{'_quick' if args.quick else ''}.jsonl.gz"
    n = 0
    with ProcessPoolExecutor(args.workers) as ex, gzip.open(out, "wt") as fh:
        for row in ex.map(_job, jobs, chunksize=8):
            fh.write(json.dumps(row) + "\n")
            n += 1
    print(f"{n} missions -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
