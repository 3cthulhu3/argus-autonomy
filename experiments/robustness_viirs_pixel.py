"""Robustness of the v1.1 operating point under 400 m satellite pixels
(docs/design/robustness-viirs-pixel.md).

    python experiments/robustness_viirs_pixel.py

M3 on held-out seeds, on-line, 400 m pixels, at +-log 99 and at the operating
point; paired against 200 m first-passage decisions from the v1.1 sweep log.
Writes results/robustness_viirs_pixel.json and .md.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from analyse_sweep import APRIORI, Mission, load
from run_decoy_benchmark import med_ci, pct, rate
from run_sweep import SPLITS, STEER
from scipy.stats import binomtest

from argus_autonomy import NOMINAL
from argus_autonomy.config import SPRTConfig
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
CFG400 = replace(NOMINAL, prior=replace(NOMINAL.prior, coarse_factor=16, detect_fraction=3.2 / 256))
OP = json.loads((ROOT / "results" / "sweep_select.json").read_text())["operating_points"]["M3"]
PAIRS = {"a priori": APRIORI, "operating point": (OP["a"], OP["b"])}


def _job(args):
    label, fam, seed = args
    a, b = PAIRS[label]
    r = run_decision_mission(CFG400, "M3", fam, seed,
                             sprt=replace(SPRTConfig(steer_clear=STEER), alert=a, clear=b))
    return {"pair": label} | r.summary()


def p2(x: int, y: int) -> float:
    return 1.0 if x + y == 0 else float(binomtest(x, x + y, 0.5).pvalue)


def summarise(recs: dict, keys: list, rng) -> dict:
    """recs[(fam, seed)] = (decision, minutes, correct)"""
    B = [recs[k] for k in keys if k[0] == "B"]
    F = [recs[k] for k in keys if k[0] != "B"]
    return {
        "P_FA": rate(sum(d == "ALERT" for d, _, _ in B), len(B)),
        "P_U_B": rate(sum(d == "UNDECIDED" for d, _, _ in B), len(B)),
        "fire_cleared": rate(sum(d == "CLEAR" for d, _, _ in F), len(F)),
        "fire_alert_wrong_place": rate(sum(d == "ALERT" and ok is False for d, _, ok in F), len(F)),
        "P_U_fire": rate(sum(d == "UNDECIDED" for d, _, _ in F), len(F)),
        "t_clear": med_ci([t for d, t, _ in B if d == "CLEAR"], rng),
        "t_alert": med_ci([t for d, t, ok in F if d == "ALERT" and ok], rng),
    }


def paired(base: dict, new: dict, keys: list) -> dict:
    out = {}
    for name, fams, event in (("false alert", "B", "ALERT"), ("fire CLEARed", "AC", "CLEAR")):
        ks = [k for k in keys if k[0] in fams]
        x = sum(base[k][0] == event and new[k][0] != event for k in ks)
        y = sum(new[k][0] == event and base[k][0] != event for k in ks)
        out[name] = {"200m_only": x, "400m_only": y, "p": p2(x, y)}
    for name, fams, event in (("time to CLEAR", "B", "CLEAR"), ("time to correct alert", "AC", "ALERT")):
        ks = [k for k in keys if k[0] in fams and base[k][0] == event and new[k][0] == event
              and (event != "ALERT" or (base[k][2] and new[k][2]))]
        slower = sum(new[k][1] > base[k][1] for k in ks)
        faster = sum(new[k][1] < base[k][1] for k in ks)
        out[name] = {"n_both": len(ks), "400m_slower": slower, "400m_faster": faster, "p": p2(slower, faster),
                     "median_diff_min": float(np.median([new[k][1] - base[k][1] for k in ks])) if ks else None}
    return out


def main():
    keys = [(f, s) for f, seeds in SPLITS["heldout"].items() for s in seeds]
    jobs = [(lab, f, s) for lab in PAIRS for f, s in keys]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
    sweep = {(r["family"], r["seed"]): Mission(r) for r in load("heldout") if r["method"] == "M3"}
    rng = np.random.default_rng(0)
    out = {"operating_point": OP, "pixel_m": 400, "detect_fraction": 3.2 / 256, "pairs": {}}
    for lab, (a, b) in PAIRS.items():
        new = {(r["family"], r["seed"]): (r["decision"], r["t_decision_min"], r["alert_correct"])
               for r in rows if r["pair"] == lab}
        base = {k: m.decide(a, b) for k, m in sweep.items()}
        out["pairs"][lab] = {
            "a": a, "b": b,
            "200m": summarise(base, keys, rng), "400m": summarise(new, keys, rng),
            "paired": paired(base, new, keys),
            "safety_400m": {"geofence_violations": int(sum(r["geofence_violations"] for r in rows if r["pair"] == lab)),
                            "returned_to_base": int(sum(bool(r["returned_to_base"]) for r in rows if r["pair"] == lab)),
                            "n": len(keys)},
        }
    op = out["pairs"]["operating point"]
    d_clear = op["400m"]["t_clear"]["median"] - op["200m"]["t_clear"]["median"]
    d_alert = op["400m"]["t_alert"]["median"] - op["200m"]["t_alert"]["median"]
    out["verdicts"] = {
        "E14": d_clear >= 1.0, "E15": op["400m"]["P_FA"]["p"] <= 0.02,
        "E16": op["400m"]["fire_cleared"]["k"] <= 2, "E17": d_alert >= 0.25,
        "rule_undecided_ok": op["400m"]["P_U_B"]["p"] <= 0.05 and op["400m"]["P_U_fire"]["p"] <= 0.05,
    }

    def mm(m):
        return "—" if m["median"] is None else f"{m['median']:.2f} [{m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}]"

    L = ["# Robustness: v1.1 operating point under 400 m satellite pixels", "",
         ("Generated by `experiments/robustness_viirs_pixel.py` following `docs/design/robustness-viirs-pixel.md` "
          "(protocol committed before the run). M3, held-out seeds (B 1000–1999, A and C 500–999), on-line. "
          "200 m rows are first-passage decisions from the v1.1 sweep log. Rates: % (count; Clopper–Pearson 95 % CI); "
          "times: median [bootstrap 95 % CI], minutes."), ""]
    for lab, d in out["pairs"].items():
        L += [f"## {lab}: a = {d['a']:g}, b = {d['b']:g}", "", "| | 200 m pixels | 400 m pixels |", "|---|---|---|"]
        for key, name in (("P_FA", "false alerts on a decoy cue (B)"), ("P_U_B", "undecided (B)"),
                          ("fire_cleared", "real fire CLEARed (A ∪ C)"),
                          ("fire_alert_wrong_place", "alert in the wrong place (A ∪ C)"),
                          ("P_U_fire", "undecided (A ∪ C)")):
            L.append(f"| {name} | {pct(d['200m'][key])} | {pct(d['400m'][key])} |")
        L.append(f"| time to CLEAR (B) | {mm(d['200m']['t_clear'])} | {mm(d['400m']['t_clear'])} |")
        L.append(f"| time to correct alert (A ∪ C) | {mm(d['200m']['t_alert'])} | {mm(d['400m']['t_alert'])} |")
        s = d["safety_400m"]
        L.append(f"| geofence violations / returns to base (400 m) | | {s['geofence_violations']} / {s['returned_to_base']} of {s['n']} |")
        L += ["", "Paired on identical seeds:", ""]
        for name, p in d["paired"].items():
            if "200m_only" in p:
                L.append(f"- {name}: 200 m only {p['200m_only']}, 400 m only {p['400m_only']} (exact McNemar p = {p['p']:.2g})")
            else:
                L.append(f"- {name}: 400 m slower in {p['400m_slower']}, faster in {p['400m_faster']} of {p['n_both']} "
                         f"decided in both (sign test p = {p['p']:.2g}); median paired difference "
                         f"{p['median_diff_min']:+.2f} min")
        L.append("")
    v = out["verdicts"]
    L += ["## Pre-registered expectations (at the operating point)", "",
          f"- E14 time to CLEAR rises by ≥ 1.0 min: {'met' if v['E14'] else 'not met'} ({d_clear:+.2f} min)",
          f"- E15 false alerts ≤ 2 %: {'met' if v['E15'] else 'not met'}",
          f"- E16 fires CLEARed ≤ 2/1000: {'met' if v['E16'] else 'not met'}",
          f"- E17 time to correct alert rises by ≥ 0.25 min: {'met' if v['E17'] else 'not met'} ({d_alert:+.2f} min)",
          f"- Undecided ≤ 5 % in B and in A ∪ C (selection rule): {'holds' if v['rule_undecided_ok'] else 'violated'}"]
    (ROOT / "results" / "robustness_viirs_pixel.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results" / "robustness_viirs_pixel.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
