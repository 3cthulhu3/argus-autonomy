"""Choose the Chernoff planner's one free parameter on seeds disjoint from the benchmark.

    python experiments/screen_min_drift.py

min_drift is the expected log-odds change (nats) below which a look is not
worth making and the planner searches for heat instead.  Writes
results/screen_min_drift.md.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from argus_autonomy import NOMINAL
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 5000
N = {"A": 200, "B": 400}
GRID = (0.05, 0.2, 0.5, 1.0, 2.0)
METHODS = ("M2", "M3")


def _job(a):
    m, md, fam, seed = a
    r = run_decision_mission(NOMINAL, m, fam, seed, min_drift=md)
    return m, md, fam, r.decision, bool(r.alert_correct), r.t_decision_min


def main():
    jobs = [(m, md, f, SEED0 + s) for m in METHODS for md in GRID for f, n in N.items() for s in range(n)]
    with ProcessPoolExecutor(2) as ex:
        out = list(ex.map(_job, jobs, chunksize=8))
    L = ["# Screening min_drift (seeds 5000+, disjoint from the benchmark)", "",
         "| method | min_drift (nats) | B false alerts | B undecided | B t_clear | A missed | A undecided | A t_alert |",
         "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for md in GRID:
            b = [o for o in out if o[0] == m and o[1] == md and o[2] == "B"]
            a = [o for o in out if o[0] == m and o[1] == md and o[2] == "A"]
            tb = [o[5] for o in b if o[3] == "CLEAR"]
            ta = [o[5] for o in a if o[3] == "ALERT" and o[4]]
            L.append(f"| {m} | {md} | {sum(o[3] == 'ALERT' for o in b)}/{len(b)} | "
                     f"{sum(o[3] == 'UNDECIDED' for o in b)} | {np.median(tb):.2f} | "
                     f"{sum(o[3] == 'CLEAR' for o in a)}/{len(a)} | {sum(o[3] == 'UNDECIDED' for o in a)} | "
                     f"{np.median(ta):.2f} |")
    (ROOT / "results" / "screen_min_drift.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
