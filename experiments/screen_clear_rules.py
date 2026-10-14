"""Screen candidate fixes for M1's "never looked" misses on seeds disjoint from the benchmark.

    python experiments/screen_clear_rules.py

Seeds 5000+ are used so that the rule finally adopted is not selected on the
seeds it is reported on.  Writes results/screen_clear_rules.md and .json.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import beta

from argus_autonomy import NOMINAL
from argus_autonomy.config import SPRTConfig
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 5000
N = {"A": 500, "B": 1000, "C": 300}
RULES = {
    "M1 (milestone i)": SPRTConfig(),
    "gate: hotspot seen": SPRTConfig(clear_gate="hotspot"),
    "gate: hotspot + ring seen": SPRTConfig(clear_gate="cue"),
    "H1 cue split phi = 0.5": SPRTConfig(phi=0.5),
    "phi = 0.5 + hotspot gate": SPRTConfig(phi=0.5, clear_gate="hotspot"),
}


def _job(a):
    rule, fam, seed = a
    r = run_decision_mission(NOMINAL, "M1", fam, seed, sprt=RULES[rule])
    return rule, fam, r.decision, bool(r.alert_correct), r.t_decision_min


def ci(k, n):
    return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def main():
    jobs = [(r, f, SEED0 + s) for r in RULES for f, n in N.items() for s in range(n)]
    with ProcessPoolExecutor(2) as ex:
        out = list(ex.map(_job, jobs, chunksize=8))
    res = {}
    for rule in RULES:
        d = {}
        for fam in N:
            o = [x for x in out if x[0] == rule and x[1] == fam]
            n = len(o)
            bad = sum(x[2] == ("ALERT" if fam == "B" else "CLEAR") for x in o)
            und = sum(x[2] == "UNDECIDED" for x in o)
            t = [x[4] for x in o if (x[2] == "CLEAR" if fam == "B" else (x[2] == "ALERT" and x[3]))]
            d[fam] = {"n": n, "errors": bad, "undecided": und, "error_ci95": ci(bad, n),
                      "t_median_min": float(np.median(t)) if t else None}
        res[rule] = d
    (ROOT / "results" / "screen_clear_rules.json").write_text(json.dumps(res, indent=1))
    L = ["# Screening CLEAR rules (seeds 5000+, disjoint from the benchmark)", "",
         ("Errors: B = false ALERT, A/C = real fire CLEARed. Times: B = median time to CLEAR, "
          "A/C = median time to correct ALERT (min)."), "",
         ("| rule | B false alerts | B undecided | B t_clear | A fires missed | A undecided | A t_alert "
          "| C fires missed | C t_alert |"), "|---|---|---|---|---|---|---|---|---|"]
    for rule, d in res.items():
        a, b, c = d["A"], d["B"], d["C"]
        L.append(f"| {rule} | {b['errors']}/{b['n']} | {b['undecided']} | {b['t_median_min']:.2f} | "
                 f"{a['errors']}/{a['n']} | {a['undecided']} | {a['t_median_min']:.2f} | "
                 f"{c['errors']}/{c['n']} | {c['t_median_min']:.2f} |")
    (ROOT / "results" / "screen_clear_rules.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
