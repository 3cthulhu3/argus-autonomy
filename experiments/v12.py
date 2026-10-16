"""v1.2: ring-aware CLEAR rules against coverage misses (docs/design/v1.2-coverage-misses.md).

    python experiments/v12.py sweep              # selection seeds, all variants, both geometries
    python experiments/v12.py select             # step 1 + 2: operating points and the chosen rule
    python experiments/v12.py heldout            # step 3: chosen rule and C0, fresh held-out seeds, on-line

The sweep stores, per mission and per threshold level, only the first tick at which the
log-odds reached it (alert side) or reached it with the gate open (clear side); that is
all a first-passage decision needs (see first_passage in decision.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from run_decoy_benchmark import med_ci, pct, rate
from run_sweep import GRID, STEER
from scipy.stats import binomtest

from argus_autonomy import NOMINAL
from argus_autonomy.config import SPRTConfig
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
MIN_PER_TICK = NOMINAL.dt_s / 60.0
TARGET, MAX_UNDECIDED = 0.02, 0.05
GEOMS = {
    "G200": NOMINAL,
    "G400": replace(NOMINAL, prior=replace(NOMINAL.prior, coarse_factor=16, detect_fraction=3.2 / 256)),
}
TICKS_PER_MIN = round(1 / MIN_PER_TICK)
VARIANTS = {"C0": {}}
VARIANTS |= {f"D{k}": {"dwell_ticks": k * TICKS_PER_MIN} for k in (1, 2, 3, 5)}
VARIANTS |= {f"R{f:g}": {"ring_cover": f} for f in (0.25, 0.5, 0.75, 1.0)}
ORDER = list(VARIANTS)          # tie-break order of step 2: C0 < D (small k first) < R (small f first)
SEEDS = {"select": {"B": range(0, 1000), "A": range(0, 500), "C": range(0, 500)},
         "heldout": {"B": range(2000, 3000), "A": range(1000, 1500), "C": range(1000, 1500)}}


def sprt_for(variant: str, a: float | None = None, b: float | None = None) -> SPRTConfig:
    s = SPRTConfig(steer_clear=STEER, **VARIANTS[variant])
    return s if a is None else replace(s, alert=a, clear=b)


# ------------------------------------------------------------------ step 0: sweep
def _sweep_job(args):
    variant, geom, fam, seed = args
    r = run_decision_mission(GEOMS[geom], "M3", fam, seed, sprt=sprt_for(variant), sweep=True,
                             alert_levels=GRID)
    llr = np.array([l for _, l, _ in r.trace])
    tick = np.array([t for t, _, _ in r.trace])
    gate = np.array([g for _, _, g in r.trace], bool)
    ta, tb = [], []
    for lvl in GRID:
        i = np.flatnonzero(llr >= lvl)
        ta.append(int(tick[i[0]]) if i.size else -1)
        j = np.flatnonzero((llr <= -lvl) & gate)
        tb.append(int(tick[j[0]]) if j.size else -1)
    ok = [r.crossings[l][1] if l in r.crossings else None for l in GRID]
    return {"v": variant, "g": geom, "f": fam, "s": seed, "ta": ta, "tb": tb, "ok": ok,
            "gv": r.geofence_violations, "rtb": bool(r.returned_to_base)}


def cmd_sweep(args):
    seeds = {f: range(5000, 5004) for f in "BAC"} if args.quick else SEEDS["select"]   # quick: screening seeds
    jobs = [(v, g, f, s) for v in VARIANTS for g in GEOMS for f, ss in seeds.items() for s in ss]
    out = ROOT / "results" / f"v12_sweep{'_quick' if args.quick else ''}.jsonl.gz"
    n = 0
    with ProcessPoolExecutor(args.workers) as ex, gzip.open(out, "wt") as fh:
        for row in ex.map(_sweep_job, jobs, chunksize=8):
            fh.write(json.dumps(row) + "\n")
            n += 1
    print(f"{n} missions -> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ step 1 + 2: selection
def decide(row: dict, ia: int, ib: int):
    ta, tb = row["ta"][ia], row["tb"][ib]
    if ta < 0 and tb < 0:
        return "UNDECIDED", float("nan"), None
    if tb < 0 or (0 <= ta < tb):
        return "ALERT", (ta + 1) * MIN_PER_TICK, row["ok"][ia]
    return "CLEAR", (tb + 1) * MIN_PER_TICK, None


def evaluate(rows: list[dict], ia: int, ib: int) -> dict:
    B = [decide(r, ia, ib) for r in rows if r["f"] == "B"]
    F = [decide(r, ia, ib) for r in rows if r["f"] != "B"]
    t_ok = [t for d, t, ok in F if d == "ALERT" and ok]
    t_cl = [t for d, t, _ in B if d == "CLEAR"]
    return {
        "a": GRID[ia], "b": -GRID[ib],
        "P_FA": rate(sum(d == "ALERT" for d, _, _ in B), len(B)),
        "P_U_B": rate(sum(d == "UNDECIDED" for d, _, _ in B), len(B)),
        "fire_cleared": rate(sum(d == "CLEAR" for d, _, _ in F), len(F)),
        "fire_alert_wrong_place": rate(sum(d == "ALERT" and ok is False for d, _, ok in F), len(F)),
        "P_U_fire": rate(sum(d == "UNDECIDED" for d, _, _ in F), len(F)),
        "t_alert_median": float(np.median(t_ok)) if t_ok else float("inf"),
        "t_clear_median": float(np.median(t_cl)) if t_cl else float("inf"),
    }


def feasible(e: dict) -> bool:
    return (e["P_FA"]["ci95"][1] <= TARGET and e["P_U_B"]["p"] <= MAX_UNDECIDED
            and e["P_U_fire"]["p"] <= MAX_UNDECIDED)


def cmd_select(args):
    q = "_quick" if args.quick else ""
    with gzip.open(ROOT / "results" / f"v12_sweep{q}.jsonl.gz", "rt") as fh:
        rows = [json.loads(line) for line in fh]
    table, ops = {}, {}
    for v in VARIANTS:
        for g in GEOMS:
            rs = [r for r in rows if r["v"] == v and r["g"] == g]
            evs = [evaluate(rs, ia, ib) for ia in range(len(GRID)) for ib in range(len(GRID))]
            ok = [e for e in evs if feasible(e)]
            best = min(ok, key=lambda e: (e["fire_cleared"]["k"], e["t_alert_median"], e["t_clear_median"], e["a"])) if ok else None
            # closest-to-feasible pair, reported for infeasible variants: fewest undecided decoy missions
            # among pairs meeting the false-alert bound
            fa_ok = [e for e in evs if e["P_FA"]["ci95"][1] <= TARGET]
            near = min(fa_ok, key=lambda e: (e["P_U_B"]["k"] + e["P_U_fire"]["k"], e["fire_cleared"]["k"])) if fa_ok else None
            table[(v, g)] = {"n_feasible": len(ok), "chosen": best, "nearest_if_infeasible": None if best else near,
                             "safety": {"geofence_violations": sum(r["gv"] for r in rs),
                                        "returned_to_base": sum(r["rtb"] for r in rs), "n": len(rs)}}
            if best:
                ops[(v, g)] = {"a": best["a"], "b": best["b"]}
    eligible = [v for v in VARIANTS if all((v, g) in ops for g in GEOMS)]

    def key(v):
        miss = sum(table[(v, g)]["chosen"]["fire_cleared"]["k"] for g in GEOMS)
        slow = max(table[(v, g)]["chosen"]["t_clear_median"] for g in GEOMS)
        return (miss, slow, ORDER.index(v))
    chosen = min(eligible, key=key) if eligible else None
    out = {"chosen": chosen, "eligible": eligible,
           "operating_points": {f"{v}|{g}": op for (v, g), op in ops.items()},
           "table": {f"{v}|{g}": d for (v, g), d in table.items()}}
    (ROOT / "results" / f"v12_select{q}.json").write_text(json.dumps(out, indent=1, default=float))

    L = ["# v1.2 selection: ring-aware CLEAR rules on selection seeds", "",
         ("Generated by `experiments/v12.py select` following `docs/design/v1.2-coverage-misses.md` "
          "(protocol committed before the sweep). M3; B seeds 0–999, A and C 0–499; per variant and geometry, "
          "the operating point by the v1.1 rule. Rates: % (count; Clopper–Pearson 95 % CI)."), "",
         "| variant | geometry | feasible pairs | a / b | false alerts (B) | undecided B / A ∪ C | real fire CLEARed (A ∪ C) | median time to correct alert | median time to CLEAR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for g in GEOMS:
            d = table[(v, g)]
            e = d["chosen"]
            if e:
                L.append(f"| {v} | {g} | {d['n_feasible']} | {e['a']:g} / {e['b']:g} | {pct(e['P_FA'])} | "
                         f"{e['P_U_B']['k']} / {e['P_U_fire']['k']} | {pct(e['fire_cleared'])} | "
                         f"{e['t_alert_median']:.2f} | {e['t_clear_median']:.2f} |")
            else:
                n = d["nearest_if_infeasible"]
                extra = (f"nearest: {n['a']:g} / {n['b']:g}, undecided {n['P_U_B']['k']} / {n['P_U_fire']['k']}"
                         if n else "no pair meets the false-alert bound")
                L.append(f"| {v} | {g} | 0 | infeasible ({extra}) | | | | | |")
    L += ["", f"Eligible (feasible in both geometries): {', '.join(eligible) or 'none'}.", "",
          f"**Chosen rule (step 2): {chosen}.**" if chosen else "**No rule eligible.**"]
    if chosen:
        L.append("Total real fires CLEARed over both geometries, eligible variants: " + ", ".join(
            f"{v} {key(v)[0]}" for v in eligible) + ".")
    L += ["", "## Safety (sweep runs fly to endurance)", "", "| variant | geometry | geofence violations | returned to base |",
          "|---|---|---|---|"] + [f"| {v} | {g} | {d['safety']['geofence_violations']} | {d['safety']['returned_to_base']}/{d['safety']['n']} |"
                                  for (v, g), d in table.items()]
    (ROOT / "results" / f"v12_select{q}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ step 3: held-out
def _online_job(args):
    variant, geom, fam, seed, a, b = args
    r = run_decision_mission(GEOMS[geom], "M3", fam, seed, sprt=sprt_for(variant, a, b))
    return {"v": variant, "g": geom} | r.summary()


def p2(x, y):
    return 1.0 if x + y == 0 else float(binomtest(x, x + y, 0.5).pvalue)


def cmd_heldout(args):
    sel = json.loads((ROOT / "results" / "v12_select.json").read_text())
    chosen = sel["chosen"]
    variants = ["C0"] if chosen in (None, "C0") else ["C0", chosen]
    jobs = []
    for v in variants:
        for g in GEOMS:
            op = sel["operating_points"][f"{v}|{g}"]
            jobs += [(v, g, f, s, op["a"], op["b"]) for f, seeds in SEEDS["heldout"].items() for s in seeds]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_online_job, jobs, chunksize=8))
    rng = np.random.default_rng(0)
    res = {}
    for v in variants:
        for g in GEOMS:
            rs = [r for r in rows if r["v"] == v and r["g"] == g]
            B = [r for r in rs if r["family"] == "B"]
            F = [r for r in rs if r["family"] != "B"]
            res[f"{v}|{g}"] = {
                **sel["operating_points"][f"{v}|{g}"],
                "P_FA": rate(sum(r["decision"] == "ALERT" for r in B), len(B)),
                "P_U_B": rate(sum(r["decision"] == "UNDECIDED" for r in B), len(B)),
                "fire_cleared": rate(sum(r["decision"] == "CLEAR" for r in F), len(F)),
                "fire_alert_wrong_place": rate(sum(r["decision"] == "ALERT" and not r["alert_correct"] for r in F), len(F)),
                "P_U_fire": rate(sum(r["decision"] == "UNDECIDED" for r in F), len(F)),
                "t_clear": med_ci([r["t_decision_min"] for r in B if r["decision"] == "CLEAR"], rng),
                "t_alert": med_ci([r["t_decision_min"] for r in F if r["decision"] == "ALERT" and r["alert_correct"]], rng),
                "geofence_violations": sum(r["geofence_violations"] for r in rs),
                "returned_to_base": sum(bool(r["returned_to_base"]) for r in rs), "n": len(rs),
            }
    paired = {}
    if len(variants) == 2:
        by = {(r["v"], r["g"], r["family"], r["seed"]): r for r in rows}
        for g in GEOMS:
            ks = [(f, s) for f, seeds in SEEDS["heldout"].items() for s in seeds]
            def ev(fams, d):
                x = sum(by[("C0", g, f, s)]["decision"] == d and by[(chosen, g, f, s)]["decision"] != d for f, s in ks if f in fams)
                y = sum(by[(chosen, g, f, s)]["decision"] == d and by[("C0", g, f, s)]["decision"] != d for f, s in ks if f in fams)
                return {"C0_only": x, "chosen_only": y, "p": p2(x, y)}
            bk = [(f, s) for f, s in ks if f == "B" and by[("C0", g, f, s)]["decision"] == "CLEAR"
                  and by[(chosen, g, f, s)]["decision"] == "CLEAR"]
            slower = sum(by[(chosen, g, f, s)]["t_decision_min"] > by[("C0", g, f, s)]["t_decision_min"] for f, s in bk)
            faster = sum(by[(chosen, g, f, s)]["t_decision_min"] < by[("C0", g, f, s)]["t_decision_min"] for f, s in bk)
            paired[g] = {"fire_cleared": ev("AC", "CLEAR"), "false_alert": ev("B", "ALERT"),
                         "t_clear": {"slower": slower, "faster": faster, "n": len(bk), "p": p2(slower, faster)}}
    v = {}
    if len(variants) == 2:
        tot = lambda var: sum(res[f"{var}|{g}"]["fire_cleared"]["k"] for g in GEOMS)
        v["E20"] = tot(chosen) * 2 <= tot("C0") and all(res[f"{chosen}|{g}"]["P_FA"]["p"] <= TARGET for g in GEOMS)
        v["E21"] = all(res[f"{chosen}|{g}"]["t_clear"]["median"] - res[f"C0|{g}"]["t_clear"]["median"] <= 2.0 for g in GEOMS)
        v["totals"] = {"C0": tot("C0"), chosen: tot(chosen)}
    out = {"chosen": chosen, "results": res, "paired": paired, "verdicts": v}
    (ROOT / "results" / "v12_heldout.json").write_text(json.dumps(out, indent=1, default=float))

    def mm(m):
        return "—" if m["median"] is None else f"{m['median']:.2f} [{m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}]"
    L = ["# v1.2 held-out: chosen rule against C0", "",
         ("Generated by `experiments/v12.py heldout`. Fresh seeds never flown before (B 2000–2999, A and C 1000–1499), "
          "on-line, each variant at its own per-geometry operating point from `results/v12_select.json`."), "",
         "| variant | geometry | a / b | false alerts (B) | undecided B / A ∪ C | real fire CLEARed (A ∪ C) | alert in wrong place | time to correct alert | time to CLEAR (B) | geofence / RTB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        vv, g = k.split("|")
        L.append(f"| {vv} | {g} | {r['a']:g} / {r['b']:g} | {pct(r['P_FA'])} | {r['P_U_B']['k']} / {r['P_U_fire']['k']} | "
                 f"{pct(r['fire_cleared'])} | {r['fire_alert_wrong_place']['k']} | {mm(r['t_alert'])} | {mm(r['t_clear'])} | "
                 f"{r['geofence_violations']} / {r['returned_to_base']} of {r['n']} |")
    for g, p in paired.items():
        L += ["", f"Paired, {g}: fires CLEARed C0 only {p['fire_cleared']['C0_only']}, {chosen} only {p['fire_cleared']['chosen_only']} "
              f"(exact McNemar p = {p['fire_cleared']['p']:.2g}); false alerts C0 only {p['false_alert']['C0_only']}, "
              f"{chosen} only {p['false_alert']['chosen_only']} (p = {p['false_alert']['p']:.2g}); time to CLEAR: {chosen} slower in "
              f"{p['t_clear']['slower']}, faster in {p['t_clear']['faster']} of {p['t_clear']['n']} (sign test p = {p['t_clear']['p']:.2g})."]
    if v:
        L += ["", "## Pre-registered expectations", "",
              f"- E20 (fires CLEARed at most half of C0's, both geometries; false alerts ≤ 2 % each): {'met' if v['E20'] else 'not met'} "
              f"(C0 {v['totals']['C0']}, {chosen} {v['totals'][chosen]})",
              f"- E21 (time to CLEAR ≤ C0 + 2.0 min in each geometry): {'met' if v['E21'] else 'not met'}"]
    (ROOT / "results" / "v12_heldout.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sweep", "select", "heldout"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    {"sweep": cmd_sweep, "select": cmd_select, "heldout": cmd_heldout}[args.cmd](args)


if __name__ == "__main__":
    main()
