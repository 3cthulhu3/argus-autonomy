"""v2.0: detection and response layers (docs/design/v2.0-detect-respond.md, §6).

    python experiments/v20.py patrol      # D1-D3: patrol latency, false alerts, slow fire
    python experiments/v20.py forecast    # R1: responder forecasts after a correct ALERT
    python experiments/v20.py envelope    # R2: suppression envelope and end-to-end probability
    python experiments/v20.py report      # tables + verdict on E30-E40

Every run is appended to a .jsonl.gz file as it finishes, so an interrupted run
resumes where it stopped. Missions are deterministic per seed.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from argus_autonomy.config import NOMINAL, RECOMMENDED, WIND_MISMATCH, DecoyConfig, PatrolConfig
from argus_autonomy.decision import make_decoy_scenario, run_decision_mission
from argus_autonomy.detection import PATROL_PLANNERS, run_patrol, scaled, slow_fire
from argus_autonomy.response import forecast_ensemble, forecast_meanfield, forecast_scores
from argus_autonomy.suppression import Airframe, attack, sample_cases

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ------------------------------------------------------------------ fixed before any held-out run
PATROL_SEEDS = {"F": range(10000, 10200), "N": range(11000, 11200)}
FORECAST_SEEDS = range(6000, 6200)
SIZES = (32, 48, 64)
RECHARGE = {"1 drone, 30 min recharge": 180, "continuous coverage": 0}
R_CLASSES = (1.0, 2.0, 3.0)              # m/min, quasi-steady head rate of spread
DISTS_M = (500, 1000, 1500, 2000, 3000)
T_ALERT = tuple(range(0, 65, 5))
MC = 400
E_PACK = (180.0, 300.0, 400.0)


def patrol_jobs():
    jobs = []
    for n in SIZES:                       # D1: size x duty x planner, nominal fire
        for rk, rc in RECHARGE.items():
            for pl in PATROL_PLANNERS:
                jobs += [("D1", n, rk, pl, "F", s) for s in PATROL_SEEDS["F"]]
    for pl in PATROL_PLANNERS:            # D2: fire-free maps
        jobs += [("D2", 64, "1 drone, 30 min recharge", pl, "N", s) for s in PATROL_SEEDS["N"]]
    for rk in RECHARGE:                   # D3: slow fire
        for pl in ("P0", "P1"):
            jobs += [("D3", 64, rk, pl, "F", s) for s in PATROL_SEEDS["F"]]
    return jobs


def _patrol(job):
    cond, n, rk, pl, fam, s = job
    cfg = scaled(NOMINAL, n) if n != 64 else NOMINAL
    if cond == "D3":
        cfg = slow_fire(cfg)
    r = run_patrol(cfg, PatrolConfig(recharge_ticks=RECHARGE[rk]), pl, fam, s)
    out = r.summary()
    out.update({"cond": cond, "n": n, "duty": rk, "events": [(e.decision, e.correct, e.minutes_spent)
                                                                for e in r.events]})
    return out


def _forecast(job):
    cond, fam, s = job
    cfg = NOMINAL if cond == "matched wind" else WIND_MISMATCH
    sc = make_decoy_scenario(cfg, DecoyConfig(), fam, s)
    r = run_decision_mission(cfg, "M3", fam, s, sprt=RECOMMENDED["NOMINAL"], scenario=sc,
                             return_on_decision=True)
    row = {"cond": cond, "f": fam, "s": s, "decision": r.decision, "ok": r.alert_correct}
    if r.decision != "ALERT" or not r.alert_correct:
        return row
    b = np.clip(r.h1_fire, 1e-6, 1 - 1e-6)
    for h in (15, 30):
        truth_fire = copy.deepcopy(sc.fire)
        truth_fire.advance(h)
        truth = truth_fire.affected
        rng = np.random.default_rng(s)
        row[f"ens{h}"] = forecast_scores(forecast_ensemble(b, cfg, sc.wind_dir, h, k=200, rng=rng), truth)
        row[f"mf{h}"] = forecast_scores(forecast_meanfield(b, cfg, sc.wind_dir, h), truth)
        row[f"persist{h}"] = forecast_scores(b, truth)   # no-growth reference
    return row


def _run(name: str, jobs, fn, key):
    path = RES / f"v20_{name}_runs.jsonl.gz"
    done = set()
    if path.exists():
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    done.add(key(json.loads(line)))
        except (EOFError, OSError, json.JSONDecodeError):
            pass
    todo = [j for j in jobs if tuple(j) not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to run")
    with ProcessPoolExecutor(2) as ex, gzip.open(path, "at") as fh:
        for i, row in enumerate(ex.map(fn, todo, chunksize=4)):
            fh.write(json.dumps(row, default=_json) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}", flush=True)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load(name):
    with gzip.open(RES / f"v20_{name}_runs.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


# ------------------------------------------------------------------ statistics
def cp(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    ci = binomtest(k, n).proportion_ci(0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def cmed(x, rng, b=2000):
    """Median with missing values (nan = never alerted) counted as +inf; bootstrap CI."""
    x = np.where(np.isnan(np.asarray(x, float)), np.inf, np.asarray(x, float))
    boots = np.median(rng.choice(x, size=(b, x.size)), axis=1)
    return float(np.median(x)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def sign_test(a, b):
    """Paired one-sided sign test that a < b (nan = +inf, ties dropped)."""
    a = np.where(np.isnan(a), np.inf, a)
    b = np.where(np.isnan(b), np.inf, b)
    wins, losses = int((a < b).sum()), int((a > b).sum())
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return wins, losses, float(p)


def fmt(x, d=1):
    return "∞" if x == math.inf else ("—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}")


# ------------------------------------------------------------------ envelope (analytic)
def envelope():
    rng = np.random.default_rng(2026)
    out = {"grid": [], "battery": [], "drones": [], "tau0": [], "fl24": []}
    cases = {r: sample_cases(r, MC, rng) for r in R_CLASSES}
    for r in R_CLASSES:
        for d in DISTS_M:
            for t in T_ALERT:
                o = [attack(c, t, d) for c in cases[r]]
                out["grid"].append({"r": r, "d": d, "t": t,
                                    "p_first": float(np.mean([x.knockdown_first for x in o])),
                                    "p_cont": float(np.mean([math.isfinite(x.contained_min) for x in o])),
                                    "p_fl": float(np.mean([x.flame_m <= 1.2 for x in o])),
                                    "need_med": float(np.median([x.need_l for x in o])),
                                    "payload": o[0].payload_l})
    return out, cases


def e2e(cases, patrol_rows, ac=Airframe(), n_drones=1, fl_max=1.2):
    """Mean over patrol seeds of P(contained) given that seed's alert latency and dock distance."""
    p = []
    for row in patrol_rows:
        t = row["t_alert_min"]
        if t is None or not math.isfinite(t):
            p.append(0.0)               # never alerted inside the horizon: nothing to attack
            continue
        o = [attack(c, t, row["ign_dist_m"], ac, n_drones, fl_max=fl_max) for c in cases]
        p.append(float(np.mean([math.isfinite(x.contained_min) for x in o])))
    return float(np.mean(p)), p


# ------------------------------------------------------------------ report
def report():
    rng = np.random.default_rng(0)
    P = load("patrol")
    lines = ["# v2.0 held-out results", "",
             "Generated by `experiments/v20.py report`. Protocol: "
             "[`docs/design/v2.0-detect-respond.md`](../docs/design/v2.0-detect-respond.md).", ""]
    verdict = {}

    def sel(cond, n=64, duty="1 drone, 30 min recharge", pl=None, fam="F"):
        return sorted([r for r in P if r["cond"] == cond and r["n"] == n and r["duty"] == duty
                       and (pl is None or r["planner"] == pl) and r["family"] == fam], key=lambda r: r["seed"])

    # D1 table
    lines += ["## D1. Ignition to correct ALERT, nominal fire (~20 m/min), 200 fires per cell", "",
              "Median minutes from ignition to the first correct ALERT (95 % bootstrap CI); "
              "fires never alerted within the horizon count as ∞.", "",
              "| map | duty | P0 lawnmower | P1u info-rate, uniform hazard | P1 info-rate, risk map | P1: cue → ALERT, median |",
              "|---|---|---|---|---|---|"]
    d1 = {}
    for n in SIZES:
        for rk in RECHARGE:
            cells = []
            for pl in PATROL_PLANNERS:
                rows = sel("D1", n, rk, pl)
                ta = np.array([r["t_alert_min"] if r["t_alert_min"] is not None else np.nan for r in rows], float)
                m, ci = cmed(ta, rng)
                d1[(n, rk, pl)] = ta
                miss = int(np.isnan(ta).sum())
                cells.append(f"{fmt(m)} ({fmt(ci[0])}–{fmt(ci[1])}){'' if not miss else f', {miss} missed'}")
            rows = sel("D1", n, rk, "P1")
            gap = [r["t_alert_min"] - r["t_first_cue_min"] for r in rows
                   if r["t_alert_min"] is not None and r["t_first_cue_min"] is not None
                   and math.isfinite(r["t_alert_min"]) and math.isfinite(r["t_first_cue_min"])]
            area = (n * 25 / 1000) ** 2
            lines.append(f"| {n}×{n} ({area:.2f} km²) | {rk} | " + " | ".join(cells)
                         + f" | {fmt(float(np.median(gap)) if gap else float('nan'))} |")
    lines.append("")

    w, l, p = sign_test(d1[(64, "1 drone, 30 min recharge", "P1")], d1[(64, "1 drone, 30 min recharge", "P0")])
    verdict["E30"] = (p < 0.05, f"P1 earlier than P0 in {w}, later in {l} of 200 paired fires (sign test p = {p:.2g})")
    w2, l2, p2 = sign_test(d1[(64, "1 drone, 30 min recharge", "P1")], d1[(64, "1 drone, 30 min recharge", "P1u")])
    mp1 = cmed(d1[(64, "1 drone, 30 min recharge", "P1")], rng)[0]
    mp1u = cmed(d1[(64, "1 drone, 30 min recharge", "P1u")], rng)[0]
    verdict["E31"] = (p2 >= 0.05 and abs(mp1 - mp1u) < 3,
                      f"P1 vs P1u: earlier {w2}, later {l2} (p = {p2:.2g}); medians {fmt(mp1)} vs {fmt(mp1u)} min")
    kpi = {(n, rk): cmed(d1[(n, rk, "P1")], rng)[0] for n in SIZES for rk in RECHARGE}
    one = all(kpi[(n, "1 drone, 30 min recharge")] > 10 for n in SIZES)
    cont = [n for n in SIZES if kpi[(n, "continuous coverage")] <= 10]
    verdict["E32"] = (one and cont == [32],
                      "P1 medians, 1 drone: " + ", ".join(f"{n}: {fmt(kpi[(n, '1 drone, 30 min recharge')])}" for n in SIZES)
                      + "; continuous: " + ", ".join(f"{n}: {fmt(kpi[(n, 'continuous coverage')])}" for n in SIZES)
                      + f"; ≤ 10 min with continuous coverage at sizes {cont}")

    # D2
    lines += ["## D2. Fire-free maps (decoys only), 64×64, 1 drone, 200 maps per planner", "",
              "| planner | hand-offs | CLEAR | UNDECIDED | false ALERT | false ALERT per hand-off (95 % CI) | flight hours | unresolved sources |",
              "|---|---|---|---|---|---|---|---|"]
    fa_ok = True
    for pl in PATROL_PLANNERS:
        rows = sel("D2", pl=pl, fam="N")
        ev = [e for r in rows for e in r["events"]]
        k = sum(1 for e in ev if e[0] == "ALERT")
        ci = cp(k, len(ev))
        fa_ok &= ci[1] <= 0.02 if pl == "P1" else True
        lines.append(f"| {pl} | {len(ev)} | {sum(1 for e in ev if e[0] == 'CLEAR')} | "
                     f"{sum(1 for e in ev if e[0] == 'UNDECIDED')} | {k} | "
                     f"{100 * k / max(len(ev), 1):.1f} % ({100 * ci[0]:.1f}–{100 * ci[1]:.1f}) | "
                     f"{sum(r['flight_hours'] for r in rows):.0f} | {sum(r['n_unresolved'] for r in rows)} |")
        if pl == "P1":
            verdict["E33"] = (ci[1] <= 0.02, f"P1: {k} false ALERTs in {len(ev)} hand-offs, upper bound {100 * ci[1]:.1f} %")
    lines.append("")

    viol = sum(r["geofence_violations"] for r in P)
    unlanded = sum(r["sorties"] - r["landed_safely"] for r in P)
    open_end = sum(1 for r in P if r["sorties"] - r["landed_safely"] == 1)
    verdict["E34"] = (viol == 0 and unlanded == open_end,
                      f"{viol} geofence violations in {len(P)} patrols; {sum(r['sorties'] for r in P)} sorties, "
                      f"{unlanded} not on the dock at the end of the run, all of them the sortie in progress")

    # D3
    lines += ["## D3. Slow fire (~2 m/min), 64×64, 200 fires per cell", "",
              "| duty | planner | ignition → first contact (cue on the fire, or ALERT) | ignition → correct ALERT | missed | real fires CLEARed |",
              "|---|---|---|---|---|---|"]
    for rk in RECHARGE:
        for pl in ("P0", "P1"):
            rows = sel("D3", 64, rk, pl)
            ta = np.array([r["t_alert_min"] if r["t_alert_min"] is not None else np.nan for r in rows], float)
            # a fire can also be found during a hand-off on another cue: first contact = min(cue, alert)
            tc = np.fmin(np.array([r["t_first_cue_min"] if r["t_first_cue_min"] is not None else np.nan
                                   for r in rows], float), ta)
            lines.append(f"| {rk} | {pl} | {fmt(cmed(tc, rng)[0])} | {fmt(cmed(ta, rng)[0])} | "
                         f"{int(np.isnan(ta).sum())} | {sum(r['fire_cleared'] for r in rows)} |")
    lines.append("")
    slow = cmed(np.array([r["t_alert_min"] if r["t_alert_min"] is not None else np.nan
                          for r in sel("D3", 64, "1 drone, 30 min recharge", "P1")], float), rng)[0]
    verdict["E35"] = (slow > 1.5 * mp1, f"slow {fmt(slow)} min vs nominal {fmt(mp1)} min (ratio {slow / mp1:.2f})")

    # R1
    F = load("forecast")
    lines += ["## R1. Forecasts after a correct ALERT (held-out v1.x seeds 6000–6199, families A and C)", "",
              "Brier score on the relevant cells (lower is better); `cover` = share of truly affected cells "
              "inside the forecast's P ≥ 0.1 envelope; `area` = forecast / true affected area. Medians over alerts.", "",
              "| estimator wind | horizon | alerts | persistence Brier | mean-field Brier | **ensemble Brier** | ensemble better (sign test) | mean-field cover | ensemble cover | mean-field area | ensemble area |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in ("matched wind", "calm-air assumption"):
        rows = [r for r in F if r["cond"] == cond and "ens15" in r]
        for h in (15, 30):
            e = np.array([r[f"ens{h}"]["brier"] for r in rows])
            m = np.array([r[f"mf{h}"]["brier"] for r in rows])
            ps = np.array([r[f"persist{h}"]["brier"] for r in rows])
            w_, l_, p_ = sign_test(e, m)
            ce = np.median([r[f"ens{h}"]["cover10"] for r in rows])
            cm = np.median([r[f"mf{h}"]["cover10"] for r in rows])
            lines.append(f"| {cond} | +{h} min | {len(rows)} | {np.median(ps):.3f} | {np.median(m):.3f} | "
                         f"**{np.median(e):.3f}** | {w_}:{l_}, p = {p_:.2g} | {cm:.2f} | {ce:.2f} | "
                         f"{np.median([r[f'mf{h}']['area_ratio'] for r in rows]):.2f} | "
                         f"{np.median([r[f'ens{h}']['area_ratio'] for r in rows]):.2f} |")
            if cond == "matched wind" and h == 30:
                verdict["E36"] = (p_ < 0.05 and np.median(e) < np.median(m),
                                  f"ensemble better in {w_} of {w_ + l_} alerts (p = {p_:.2g}); medians {np.median(e):.3f} vs {np.median(m):.3f}")
            if h == 15:
                verdict.setdefault("_cover15", {})[cond] = ce
    c15 = verdict.pop("_cover15")
    verdict["E37"] = (c15["matched wind"] >= 0.8 and c15["calm-air assumption"] < c15["matched wind"],
                      f"+15 min envelope cover: matched {c15['matched wind']:.2f}, calm-air {c15['calm-air assumption']:.2f}")
    lines.append("")

    # R2
    env, cases = envelope()
    lines += ["## R2. Can one dock-launched drone contain it? (model, not simulation)", "",
              "P(contained within 6 sorties, 1 drone) by quasi-steady head spread rate, dock distance and "
              f"minutes from ignition to ALERT; {MC} draws of the assumption ranges per cell "
              "(`suppression.Assumptions`). Reference airframe: 40 L, ~12 min fully loaded.", ""]
    for r in R_CLASSES:
        lines += [f"**R_ss = {r:g} m/min**", "", "| dock distance | " + " | ".join(f"{t}′" for t in T_ALERT) + " |",
                  "|---|" + "---|" * len(T_ALERT)]
        for d in DISTS_M:
            row = [g for g in env["grid"] if g["r"] == r and g["d"] == d]
            lines.append(f"| {d / 1000:.1f} km | " + " | ".join(f"{g['p_cont']:.2f}" for g in row) + " |")
        lines.append("")

    slow_rows = {rk: sel("D3", 64, rk, "P1") for rk in RECHARGE}
    lines += ["### End to end: slow-fire patrol latency × suppression model", "",
              "For each held-out slow fire (D3, P1), the alert latency and dock distance it actually had, "
              "then the model's P(contained); fires never alerted count as 0. Mean over the 200 fires.", "",
              "| R_ss | duty | 1 drone | 2 drones in rotation | 1 drone, 400 Wh/kg pack | 1 drone, flame limit 2.4 m |",
              "|---|---|---|---|---|---|"]
    e2e_tab = {}
    for r in R_CLASSES:
        for rk in RECHARGE:
            a1, _ = e2e(cases[r], slow_rows[rk])
            a2, _ = e2e(cases[r], slow_rows[rk], n_drones=2)
            a4, _ = e2e(cases[r], slow_rows[rk], ac=Airframe(e_pack=400.0))
            a5, _ = e2e(cases[r], slow_rows[rk], fl_max=2.4)
            e2e_tab[(r, rk)] = (a1, a2, a4, a5)
            lines.append(f"| {r:g} m/min | {rk} | {a1:.2f} | {a2:.2f} | {a4:.2f} | {a5:.2f} |")
    lines.append("")
    one_d, two_c = e2e_tab[(2.0, "1 drone, 30 min recharge")][0], e2e_tab[(2.0, "continuous coverage")][1]
    verdict["E38"] = (one_d < 0.25 and two_c < 0.5,
                      f"R_ss = 2: 1 drone with recharge {one_d:.2f}; continuous coverage, 2 drones {two_c:.2f}")
    b180, b400 = e2e_tab[(2.0, "1 drone, 30 min recharge")][0], e2e_tab[(2.0, "1 drone, 30 min recharge")][2]
    verdict["E39"] = (b400 - b180 < 0.10, f"R_ss = 2, 1 drone: 180 Wh/kg {b180:.2f} → 400 Wh/kg {b400:.2f}")

    lines += ["## Verdicts", "", "| | outcome | evidence |", "|---|---|---|"]
    for k in sorted(verdict):
        ok, why = verdict[k]
        lines.append(f"| {k} | {'held' if ok else '**not held**'} | {why} |")
    (RES / "v20_heldout.md").write_text("\n".join(lines) + "\n")
    (RES / "v20_heldout.json").write_text(json.dumps(
        {"verdicts": {k: {"held": bool(v[0]), "evidence": v[1]} for k, v in verdict.items()},
         "envelope": env["grid"],
         "e2e": {f"{r}|{rk}": v for (r, rk), v in e2e_tab.items()}}, indent=1, default=_json))
    print("\n".join(lines[-len(verdict) - 3:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["patrol", "forecast", "envelope", "report"])
    a = ap.parse_args()
    if a.what == "patrol":
        _run("patrol", patrol_jobs(), _patrol,
             key=lambda r: (r["cond"], r["n"], r["duty"], r["planner"], r["family"], r["seed"]))
    elif a.what == "forecast":
        jobs = [(c, f, s) for c in ("matched wind", "calm-air assumption") for f in ("C", "A") for s in FORECAST_SEEDS]
        _run("forecast", jobs, _forecast, key=lambda r: (r["cond"], r["f"], r["s"]))
    elif a.what == "envelope":
        env, _ = envelope()
        print(len(env["grid"]), "envelope cells (computed again inside `report`)")
    else:
        report()


if __name__ == "__main__":
    main()
