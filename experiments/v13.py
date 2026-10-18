"""v1.3: object-level fire hypothesis against C0 and D5 (docs/design/v1.3-object-level-fire.md, §5).

    python experiments/v13.py sweep      # O on selection seeds, both geometries (C0, D5: v1.2 sweep)
    python experiments/v13.py select     # corrected v1.2b rule, all three variants
    python experiments/v13.py heldout    # fresh seeds, on-line, each at its own operating point
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
from v12 import GEOMS, SEEDS, evaluate, p2
from v12b import pick

from argus_autonomy.config import SPRTConfig
from argus_autonomy.decision import run_decision_mission

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"C0": {}, "D5": {"dwell_ticks": 30}, "O": {"h1_model": "object"}}
HELDOUT = {"B": range(4000, 5000), "A": range(2000, 2500), "C": range(2000, 2500)}
ADOPT = {"undecided": 0.05, "p_fa": 0.02, "t_clear_slack_min": 1.0}


def sprt_for(v: str, a: float | None = None, b: float | None = None) -> SPRTConfig:
    s = SPRTConfig(steer_clear=STEER, **VARIANTS[v])
    return s if a is None else replace(s, alert=a, clear=b)


def _sweep_job(args):
    g, fam, seed = args
    r = run_decision_mission(GEOMS[g], "M3", fam, seed, sprt=sprt_for("O"), sweep=True, alert_levels=GRID)
    llr = np.array([l for _, l, _ in r.trace])
    tick = np.array([t for t, _, _ in r.trace])
    gate = np.array([x for _, _, x in r.trace], bool)
    ta, tb = [], []
    for lvl in GRID:
        i = np.flatnonzero(llr >= lvl)
        ta.append(int(tick[i[0]]) if i.size else -1)
        j = np.flatnonzero((llr <= -lvl) & gate)
        tb.append(int(tick[j[0]]) if j.size else -1)
    ok = [r.crossings[l][1] if l in r.crossings else None for l in GRID]
    return {"v": "O", "g": g, "f": fam, "s": seed, "ta": ta, "tb": tb, "ok": ok,
            "gv": r.geofence_violations, "rtb": bool(r.returned_to_base)}


def _salvage(path: Path) -> list[dict]:
    """Rows readable from a (possibly truncated) gzip or plain jsonl file."""
    rows = []
    if not path.exists():
        return rows
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, OSError):
        pass
    return rows


def cmd_sweep(args):
    """Resumable: every finished mission is appended and flushed to a part file, and a
    restart (the container can be reclaimed mid-run) flies only the missing seeds.
    Missions are deterministic per seed, so a resumed sweep equals an uninterrupted one."""
    jobs = [(g, f, s) for g in GEOMS for f, ss in SEEDS["select"].items() for s in ss]
    out = ROOT / "results" / "v13_sweep.jsonl.gz"
    part = ROOT / "results" / "v13_sweep.part.jsonl"
    done = {(r["g"], r["f"], r["s"]): r for r in _salvage(out) + _salvage(part)}
    with open(part, "w") as fh:                      # rewrite the part file from what survived
        for r in done.values():
            fh.write(json.dumps(r) + "\n")
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} missions recovered, {len(todo)} to fly", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(part, "a") as fh:
        for row in ex.map(_sweep_job, todo, chunksize=4):
            done[(row["g"], row["f"], row["s"])] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    with gzip.open(out, "wt") as fh:
        for j in jobs:
            fh.write(json.dumps(done[j]) + "\n")
    part.unlink()
    print(f"{len(jobs)} missions -> {out.relative_to(ROOT)}")


def cmd_select(args):
    rows = []
    for name, keep in (("v12_sweep.jsonl.gz", ("C0", "D5")), ("v13_sweep.jsonl.gz", ("O",))):
        with gzip.open(ROOT / "results" / name, "rt") as fh:
            rows += [r for r in map(json.loads, fh) if r["v"] in keep]
    table, ops = {}, {}
    for v in VARIANTS:
        for g in GEOMS:
            rs = [r for r in rows if r["v"] == v and r["g"] == g]
            evs = [evaluate(rs, ia, ib) for ia in range(len(GRID)) for ib in range(len(GRID))]
            best, n_ok = pick(evs)
            table[f"{v}|{g}"] = {"n_feasible": n_ok, "chosen": best,
                                 "safety": {"geofence_violations": sum(r["gv"] for r in rs),
                                            "returned_to_base": sum(r["rtb"] for r in rs), "n": len(rs)}}
            if best:
                ops[f"{v}|{g}"] = {"a": best["a"], "b": best["b"]}
    out = {"operating_points": ops, "table": table,
           "E26_O_feasible_both": all(f"O|{g}" in ops for g in GEOMS)}
    (ROOT / "results" / "v13_select.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# v1.3 selection: C0, D5 and the object-level H1 (O)", "",
         ("Generated by `experiments/v13.py select` following `docs/design/v1.3-object-level-fire.md` §5 "
          "(protocol committed before the O sweep). Corrected v1.2b selection rule; selection seeds."), "",
         "| variant | geometry | feasible pairs | a / b | false alerts (B) | undecided B / A ∪ C | real fire CLEARed (A ∪ C) | median time to correct alert | median time to CLEAR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, d in table.items():
        v, g = k.split("|")
        e = d["chosen"]
        if e:
            L.append(f"| {v} | {g} | {d['n_feasible']} | {e['a']:g} / {e['b']:g} | {pct(e['P_FA'])} | "
                     f"{e['P_U_B']['k']} / {e['P_U_fire']['k']} | {pct(e['fire_cleared'])} | "
                     f"{e['t_alert_median']:.2f} | {e['t_clear_median']:.2f} |")
        else:
            L.append(f"| {v} | {g} | 0 | infeasible | | | | | |")
    L += ["", f"E26 (O feasible in both geometries): {'met' if out['E26_O_feasible_both'] else 'not met'}.", "",
          "Safety (sweeps fly to endurance): " + "; ".join(
              f"{k} {d['safety']['geofence_violations']} violations, {d['safety']['returned_to_base']}/{d['safety']['n']} returns"
              for k, d in table.items())]
    (ROOT / "results" / "v13_select.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _online_job(args):
    v, g, fam, seed, a, b = args
    r = run_decision_mission(GEOMS[g], "M3", fam, seed, sprt=sprt_for(v, a, b))
    return {"v": v, "g": g} | r.summary()


def cmd_heldout(args):
    sel = json.loads((ROOT / "results" / "v13_select.json").read_text())
    ops = sel["operating_points"]
    variants = [v for v in VARIANTS if all(f"{v}|{g}" in ops for g in GEOMS)]
    jobs = [(v, g, f, s, ops[f"{v}|{g}"]["a"], ops[f"{v}|{g}"]["b"])
            for v in variants for g in GEOMS for f, ss in HELDOUT.items() for s in ss]
    jobs.sort(key=lambda j: j[0] != "O")          # slow jobs first, for load balance
    part = ROOT / "results" / "v13_heldout.part.jsonl"
    done = {(r["v"], r["g"], r["family"], r["seed"]): r for r in _salvage(part)}   # resumable, as the sweep
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"{len(done)} missions recovered, {len(todo)} to fly", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(part, "a") as fh:
        for r in ex.map(_online_job, todo, chunksize=4):
            done[(r["v"], r["g"], r["family"], r["seed"])] = r
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
    rows = [done[(j[0], j[1], j[2], j[3])] for j in jobs]
    with gzip.open(ROOT / "results" / "v13_heldout_runs.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    part.unlink(missing_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    for v in variants:
        for g in GEOMS:
            rs = [r for r in rows if r["v"] == v and r["g"] == g]
            B = [r for r in rs if r["family"] == "B"]
            F = [r for r in rs if r["family"] != "B"]
            res[f"{v}|{g}"] = {
                **ops[f"{v}|{g}"],
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
    by = {(r["v"], r["g"], r["family"], r["seed"]): r for r in rows}
    ks = [(f, s) for f, ss in HELDOUT.items() for s in ss]
    paired = {}
    for other in [v for v in variants if v != "O"]:
        for g in GEOMS:
            fires = [(f, s) for f, s in ks if f != "B"]
            x = sum(by[(other, g, f, s)]["decision"] == "CLEAR" and by[("O", g, f, s)]["decision"] != "CLEAR" for f, s in fires)
            y = sum(by[("O", g, f, s)]["decision"] == "CLEAR" and by[(other, g, f, s)]["decision"] != "CLEAR" for f, s in fires)
            bk = [(f, s) for f, s in ks if f == "B" and by[(other, g, f, s)]["decision"] == "CLEAR" and by[("O", g, f, s)]["decision"] == "CLEAR"]
            sl = sum(by[("O", g, f, s)]["t_decision_min"] > by[(other, g, f, s)]["t_decision_min"] for f, s in bk)
            fa = sum(by[("O", g, f, s)]["t_decision_min"] < by[(other, g, f, s)]["t_decision_min"] for f, s in bk)
            paired[f"O vs {other}|{g}"] = {"fires_cleared_other_only": x, "fires_cleared_O_only": y, "p": p2(x, y),
                                           "t_clear_O_slower": sl, "t_clear_O_faster": fa, "n": len(bk), "p_t": p2(sl, fa)}
    o, d = (lambda g: res[f"O|{g}"]), (lambda g: res[f"D5|{g}"])
    miss = {v: sum(res[f"{v}|{g}"]["fire_cleared"]["k"] for g in GEOMS) for v in variants}
    tc = {g: o(g)["t_clear"]["median"] - d(g)["t_clear"]["median"] for g in GEOMS}
    ta = {g: o(g)["t_alert"]["median"] - d(g)["t_alert"]["median"] for g in GEOMS}
    crit = {
        "i_undecided": all(o(g)["P_U_B"]["p"] <= ADOPT["undecided"] and o(g)["P_U_fire"]["p"] <= ADOPT["undecided"] for g in GEOMS),
        "ii_false_alerts": all(o(g)["P_FA"]["p"] <= ADOPT["p_fa"] for g in GEOMS),
        "iii_misses_not_more": miss["O"] <= miss["D5"],
        "iv_t_clear_slack": all(tc[g] <= ADOPT["t_clear_slack_min"] for g in GEOMS),
        "v_strict": miss["O"] < miss["D5"] or all(tc[g] < 0 for g in GEOMS),
    }
    verdict = {"E27": miss["O"] <= miss["D5"], "E28": tc["G400"] > 0, "E29": all(abs(ta[g]) <= 0.5 for g in GEOMS),
               "misses": miss, "t_clear_minus_D5": tc, "t_alert_minus_D5": ta, "criteria": crit,
               "adopt": all(crit.values())}
    out = {"results": res, "paired": paired, "verdict": verdict}
    (ROOT / "results" / "v13_heldout.json").write_text(json.dumps(out, indent=1, default=float))

    def mm(m):
        return "—" if m["median"] is None else f"{m['median']:.2f} [{m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}]"
    L = ["# v1.3 held-out: C0, D5 and the object-level H1 (O), fresh seeds", "",
         ("Generated by `experiments/v13.py heldout`. Seeds never flown before (B 4000–4999, A and C 2000–2499), "
          "on-line, each variant at its own operating point from `results/v13_select.json`."), "",
         "| variant | geometry | a / b | false alerts (B) | undecided B / A ∪ C | real fire CLEARed (A ∪ C) | alert in wrong place | time to correct alert | time to CLEAR (B) | geofence / RTB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        vv, g = k.split("|")
        L.append(f"| {vv} | {g} | {r['a']:g} / {r['b']:g} | {pct(r['P_FA'])} | {r['P_U_B']['k']} / {r['P_U_fire']['k']} | "
                 f"{pct(r['fire_cleared'])} | {r['fire_alert_wrong_place']['k']} | {mm(r['t_alert'])} | {mm(r['t_clear'])} | "
                 f"{r['geofence_violations']} / {r['returned_to_base']} of {r['n']} |")
    L += ["", "Paired on identical seeds:", ""]
    for k, p in paired.items():
        L.append(f"- {k}: fires CLEARed by the other only {p['fires_cleared_other_only']}, by O only {p['fires_cleared_O_only']} "
                 f"(exact McNemar p = {p['p']:.2g}); time to CLEAR, O slower in {p['t_clear_O_slower']}, faster in "
                 f"{p['t_clear_O_faster']} of {p['n']} (sign test p = {p['p_t']:.2g})")
    L += ["", "## Pre-registered expectations and adoption", "",
          f"- E27 O CLEARs no more real fires than D5: {'met' if verdict['E27'] else 'not met'} ({miss})",
          f"- E28 O slower to CLEAR than D5 at 400 m: {'met' if verdict['E28'] else 'not met'} ({tc['G400']:+.2f} min; 200 m {tc['G200']:+.2f})",
          f"- E29 O's alert time within 0.5 min of D5's: {'met' if verdict['E29'] else 'not met'} ({ta})",
          "- Adoption criteria: " + ", ".join(f"{k} {'yes' if v_ else 'no'}" for k, v_ in crit.items()),
          f"- **{'ADOPT O' if verdict['adopt'] else 'D5 STAYS'}**"]
    (ROOT / "results" / "v13_heldout.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sweep", "select", "heldout"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    {"sweep": cmd_sweep, "select": cmd_select, "heldout": cmd_heldout}[args.cmd](args)


if __name__ == "__main__":
    main()
