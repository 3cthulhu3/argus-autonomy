"""v1.1 milestone (ii): CLEAR gate, Chernoff planner, close look, oracle.

    python experiments/run_decoy_benchmark_ii.py            # B 1000 seeds, A/C 500
    python experiments/run_decoy_benchmark_ii.py --quick    # CI smoke run

Same seeds as milestone (i), so rows are paired across milestones.  Every
free choice made in this milestone (the CLEAR gate, min_drift) was screened
on seeds 5000+ (experiments/screen_*.py), not on these.  Thresholds are still
the a-priori +-log 99.

Pre-registered before this run (design doc §6.5, §12):
  E3  M2 is faster than M1 at no worse error rates.
  E4  M3 shortens time to CLEAR when the cue holds only warm decoys, not when it holds hot ones.
  E6  M3 with phi = 0.5 has fewer false alerts than M3 with phi = 1 (a hot reading from a
      hot decoy no longer counts as evidence for fire once H1 also admits decoys in the cue).

Writes results/decoys_m2_runs.csv, results/decoys_m2.json, results/decoys_m2.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from run_decoy_benchmark import med_ci, pct, rate

from argus_autonomy import NOMINAL
from argus_autonomy.config import HOT, WARM, DecoyConfig, SPRTConfig
from argus_autonomy.decision import make_decoy_scenario, run_decision_mission

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "M0 (v1.0)": ("M0", SPRTConfig(clear_gate=None)),
    "M1, no gate (milestone i)": ("M1", SPRTConfig(clear_gate=None)),
    "M1": ("M1", SPRTConfig()),
    "M2": ("M2", SPRTConfig()),
    "M3": ("M3", SPRTConfig()),
    "M3, phi = 0.5": ("M3", SPRTConfig(phi=0.5)),
    "Oracle": ("OR", SPRTConfig()),
}


def cue_decoys(family: str, seed: int) -> str:
    """Which decoy classes lie inside the flagged satellite pixels: none / warm / hot / mixed."""
    sc = make_decoy_scenario(NOMINAL, DecoyConfig(), family, seed)
    hs = sc.prior >= NOMINAL.prior.p_hotspot
    w, h = bool((hs & (sc.decoys == WARM)).any()), bool((hs & (sc.decoys == HOT)).any())
    return {(False, False): "none", (True, False): "warm", (False, True): "hot"}.get((w, h), "mixed")


def _job(args):
    label, family, seed = args
    method, sprt = VARIANTS[label]
    r = run_decision_mission(NOMINAL, method, family, seed, sprt=sprt)
    row = r.summary() | {"variant": label}
    row["cue_decoys"] = cue_decoys(family, seed) if family != "C" else "none"
    return row


def summarise(rows):
    rng = np.random.default_rng(0)
    out = {}
    for label in VARIANTS:
        out[label] = {}
        for fam in ("A", "B", "C"):
            rs = [r for r in rows if r["variant"] == label and r["family"] == fam]
            if not rs:
                continue
            n = len(rs)
            dec = [r["decision"] for r in rs]
            s = {
                "n": n,
                "decisions": {d: dec.count(d) for d in ("ALERT", "CLEAR", "UNDECIDED")},
                "geofence_violations": int(sum(r["geofence_violations"] for r in rs)),
                "returned_to_base": int(sum(bool(r["returned_to_base"]) for r in rs)),
                "min_reserve_ticks": int(min(r["reserve_left"] for r in rs)),
                "close_looks_mean": float(np.mean([r["n_close_looks"] for r in rs])),
            }
            if fam == "B":
                s["P_FA"] = rate(dec.count("ALERT"), n)
                s["P_U"] = rate(dec.count("UNDECIDED"), n)
                s["t_clear_min"] = med_ci([r["t_decision_min"] for r in rs if r["decision"] == "CLEAR"], rng)
                s["by_cue"] = {}
                for cue in ("warm", "hot", "mixed"):
                    sub = [r for r in rs if r["cue_decoys"] == cue]
                    if sub:
                        s["by_cue"][cue] = {
                            "n": len(sub),
                            "P_FA": rate(sum(r["decision"] == "ALERT" for r in sub), len(sub)),
                            "t_clear_min": med_ci([r["t_decision_min"] for r in sub
                                                   if r["decision"] == "CLEAR"], rng),
                        }
            else:
                ok = [r for r in rs if r["decision"] == "ALERT" and r["alert_correct"]]
                mis = [r for r in rs if r["decision"] == "ALERT" and not r["alert_correct"]]
                s["P_D"] = rate(len(ok), n)
                s["P_MIS"] = rate(len(mis), n)
                s["P_CLEAR_on_fire"] = rate(dec.count("CLEAR"), n)
                s["P_U"] = rate(dec.count("UNDECIDED"), n)
                s["t_correct_alert_min"] = med_ci([r["t_decision_min"] for r in ok], rng)
                f1 = np.array([r["f1"] for r in rs])
                s["f1_mean"] = float(f1.mean())
                s["f1_ci95"] = float(1.96 * f1.std(ddof=1) / np.sqrt(n))
            out[label][fam] = s
    return out


def mins(m):
    if not m or m["median"] is None:
        return "—"
    return f"{m['median']:.2f} [{m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}]"


def write_md(summary, path):
    L = ["# v1.1 milestone (ii): CLEAR gate, Chernoff planner, close look", "",
         ("Generated by `experiments/run_decoy_benchmark_ii.py`. Same seeds as milestone (i). "
          "Thresholds ±log 99, a priori. Rates: percentage (count; Clopper–Pearson 95 % CI). "
          "Times: median [bootstrap 95 % CI], minutes after launch."), "",
         "## Family B — decoys only, the satellite cue is a decoy", "",
         "| method | false alerts P_FA | undecided | time to CLEAR | close looks / mission |",
         "|---|---|---|---|---|"]
    for v, s in summary.items():
        b = s.get("B")
        if b:
            L.append(f"| {v} | {pct(b['P_FA'])} | {pct(b['P_U'])} | {mins(b['t_clear_min'])} | "
                     f"{b['close_looks_mean']:.2f} |")
    L += ["", "### Family B by what the flagged pixels contain (E4)", "",
          ("| method | warm decoys only: P_FA | time to CLEAR | hot decoys only: P_FA | time to CLEAR "
           "| mixed: P_FA | time to CLEAR |"), "|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        b = s.get("B")
        if b:
            cells = []
            for cue in ("warm", "hot", "mixed"):
                c = b["by_cue"].get(cue)
                cells += [pct(c["P_FA"]), mins(c["t_clear_min"])] if c else ["—", "—"]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
    for fam, title in (("A", "Family A — fire plus decoys"), ("C", "Family C — fire only (v1.0 scenario)")):
        L += ["", f"## {title}", "",
              ("| method | correct alert P_D | alert in wrong place | real fire CLEARed | undecided "
               "| time to correct alert | map F1 at landing |"),
              "|---|---|---|---|---|---|---|"]
        for v, s in summary.items():
            a = s.get(fam)
            if a:
                L.append(f"| {v} | {pct(a['P_D'])} | {pct(a['P_MIS'])} | {pct(a['P_CLEAR_on_fire'])} | "
                         f"{pct(a['P_U'])} | {mins(a['t_correct_alert_min'])} | "
                         f"{a['f1_mean']:.3f} ± {a['f1_ci95']:.3f} |")
    L += ["", "## Safety", "",
          "| method | family | geofence violations | returned to base | min reserve (ticks) |",
          "|---|---|---|---|---|"]
    for v, s in summary.items():
        for fam, a in s.items():
            L.append(f"| {v} | {fam} | {a['geofence_violations']} | {a['returned_to_base']}/{a['n']} | "
                     f"{a['min_reserve_ticks']} |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    nb, nac = (20, 10) if args.quick else (1000, 500)
    jobs = [(v, f, s) for v in VARIANTS for f, n in (("B", nb), ("A", nac), ("C", nac)) for s in range(n)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
    summary = summarise(rows)
    out = ROOT / "results"
    suffix = "_quick" if args.quick else ""
    with open(out / f"decoys_m2_runs{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"decoys_m2{suffix}.json").write_text(json.dumps(summary, indent=1))
    write_md(summary, out / f"decoys_m2{suffix}.md")
    print((out / f"decoys_m2{suffix}.md").read_text())


if __name__ == "__main__":
    main()
