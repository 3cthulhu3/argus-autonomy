"""v2.1 held-out report: tables and the verdict on E40-E47 (design doc §8).

Reads results/v21_heldout_runs.jsonl.gz, writes results/v21_heldout.md and .json.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from v21 import HELDOUT_SEEDS, ISSUES, LEADS, RES, load, nc_log_err, pooled, pooled_issue, sign_test, summary

METHODS = ("OL", "SO", "JE")


def rel_sse(rows, m, ld):
    return pooled(rows, m, "sse_m2", leads=(ld,)) / pooled(rows, m, "true_m2", leads=(ld,))


def verdicts(rows, s) -> list[tuple[str, bool, str]]:
    out = []

    def st(a, b, metric, ld):
        return s[f"{a}<{b} {metric}+{ld}"]

    t = st("JE", "OL", "sse_m2", 30)
    out.append(("E40", t["p"] < 0.01, f"JE < OL on summed squared error at +30 min in {t['wins']} of "
                f"{t['wins'] + t['losses']} non-tied pairs (p = {t['p']:.1e})"))
    t = st("JE", "SO", "sse_m2", 30)
    out.append(("E41", t["p"] < 0.05, f"JE < SO at +30 min in {t['wins']} of {t['wins'] + t['losses']} "
                f"(p = {t['p']:.1e})"))
    t = st("SO", "OL", "sse_m2", 15)
    out.append(("E42", t["p"] < 0.05, f"SO < OL at +15 min in {t['wins']} of {t['wins'] + t['losses']} "
                f"(p = {t['p']:.1e})"))
    we, wo = s["JE:wind_err_med"], s["OL:wind_err_med"]
    out.append(("E43", we <= 0.6 * wo, f"median wind error JE {we:.2f} m/s vs prior {wo:.2f} m/s "
                f"(ratio {we / wo:.2f}; bound 0.60)"))
    me = pooled_issue(rows, "JE", "m_err")
    mo = pooled_issue(rows, "OL", "m_err")
    w, l_, p = sign_test(me, mo)
    out.append(("E44", p < 0.05, f"moisture error JE < prior in {w} of {w + l_} (p = {p:.1e}); medians "
                f"{100 * np.median(me):.2f} vs {100 * np.median(mo):.2f} points"))
    ar, cv = s["JE+30"]["area_ratio_med"], s["JE+30"]["cover10_med"]
    out.append(("E45", 0.8 <= ar <= 1.25 and cv >= 0.9, f"JE +30 min: median area ratio {ar:.2f} "
                f"(bound 0.8-1.25), median envelope cover {cv:.2f} (bound 0.9)"))
    a, b = nc_log_err(rows, "JE"), nc_log_err(rows, "OL")
    w, l_, p = sign_test(a, b)
    out.append(("E46", p < 0.05, f"|log10 N_c error| JE < prior in {w} of {w + l_} (p = {p:.1e}); medians "
                f"{np.median(a):.2f} vs {np.median(b):.2f}"))
    t = st("JE", "SO", "brier", 30)          # SO ranks above JE where JE's Brier is higher: JE's losses
    out.append(("E47", t["losses"] > t["wins"], f"v2.0 relevant-set Brier ranks SO above JE at +30 min in "
                f"{t['losses']} of {t['wins'] + t['losses']} pairs"))
    return out


def report():
    rows = sorted(load(RES / "v21_heldout_runs.jsonl.gz"), key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == list(HELDOUT_SEEDS), "held-out run incomplete"
    s = summary(rows)
    v = verdicts(rows, s)
    L = []
    L.append("# v2.1 held-out results — assimilation and regime signal\n")
    L.append(f"{len(rows)} scenarios (seeds {HELDOUT_SEEDS.start}–{HELDOUT_SEEDS.stop - 1}), forecasts issued at "
             f"minutes {', '.join(map(str, ISSUES))}, leads {' and '.join(f'+{x}' for x in LEADS)} min; "
             f"{len(rows) * len(ISSUES)} paired forecasts per lead. Configuration: {rows[0]['overrides'] or 'defaults'}.\n")
    L.append(f"Wind shift in {sum(r['shift'] for r in rows)} scenarios; truth fire reached the domain edge "
             f"by minute 70 in {sum(r['edge_hit'] for r in rows)}.\n")
    L.append("## Verdicts\n")
    L.append("| | outcome | evidence |\n|---|---|---|")
    for e, ok, txt in v:
        L.append(f"| {e} | {'held' if ok else '**not held**'} | {txt} |")
    L.append(f"\n**{sum(ok for _, ok, _ in v)} of {len(v)} held.**\n")
    L.append("## Forecast skill\n")
    L.append("Summed squared error over the domain divided by the true burnt area (lower is better; "
             "a deterministic forecast's value is its misclassified area over the true area). "
             "v2.0's relevant-set Brier and the other v2.0 scores are reported for continuity.\n")
    L.append("| lead | method | SSE / true area, median (mean) | Brier (v2.0), median | IoU of P ≥ 0.5, median | "
             "area ratio, median [IQR] | envelope cover, median |")
    L.append("|---|---|---|---|---|---|---|")
    for ld in LEADS:
        for m in METHODS:
            x = s[f"{m}+{ld}"]
            L.append(f"| +{ld} | {m} | {x['sse_rel_med']:.3f} ({x['sse_rel_mean']:.3f}) | {x['brier_med']:.3f} | "
                     f"{x['iou50_med']:.2f} | {x['area_ratio_med']:.2f} [{x['area_ratio_iqr'][0]:.2f}–"
                     f"{x['area_ratio_iqr'][1]:.2f}] | {x['cover10_med']:.2f} |")
    L.append("\nPaired sign tests (wins : losses, one-sided p):\n")
    L.append("| comparison | SSE +15 | SSE +30 | Brier (v2.0) +15 | Brier (v2.0) +30 |\n|---|---|---|---|---|")
    for a, b in (("JE", "OL"), ("JE", "SO"), ("SO", "OL")):
        cells = []
        for metric in ("sse_m2", "brier"):
            for ld in LEADS:
                t = s[f"{a}<{b} {metric}+{ld}"]
                cells.append(f"{t['wins']}:{t['losses']} ({t['p']:.0e})")
        L.append(f"| {a} < {b} | " + " | ".join(cells) + " |")
    # relative improvement
    imp = {}
    for ld in LEADS:
        for b in ("OL", "SO"):
            imp[f"JE vs {b} +{ld}"] = float(1 - np.median(rel_sse(rows, "JE", ld)) / np.median(rel_sse(rows, b, ld)))
    L.append("\nMedian SSE reduction of JE: " + "; ".join(f"{k} {100 * x:.0f} %" for k, x in imp.items()) + ".\n")
    # by wind shift (post hoc, not tested)
    L.append("### By wind shift (post hoc, not tested)\n")
    L.append("| scenarios | n | JE < SO, SSE +30 | JE < OL, SSE +30 | SO < OL, SSE +30 |\n|---|---|---|---|---|")
    for flag, name in ((False, "no shift"), (True, "shift")):
        sel = [r for r in rows if r["shift"] == flag]
        cells = []
        for a, b in (("JE", "SO"), ("JE", "OL"), ("SO", "OL")):
            w, l_, p = sign_test(pooled(sel, a, "sse_m2", leads=(30,)), pooled(sel, b, "sse_m2", leads=(30,)))
            cells.append(f"{w}:{l_} ({p:.0e})")
        L.append(f"| {name} | {len(sel)} | " + " | ".join(cells) + " |")
    L.append("\n## Inputs\n")
    L.append("| method | wind error, median (m/s) | moisture error, median (points) | log α at issue, 10–90 % |")
    L.append("|---|---|---|---|")
    for m in METHODS:
        la = pooled_issue(rows, m, "la_mean")
        L.append(f"| {m} | {s[f'{m}:wind_err_med']:.2f} | {100 * s[f'{m}:m_err_med']:.2f} | "
                 f"{np.percentile(la, 10):.2f} to {np.percentile(la, 90):.2f} |")
    L.append("\n## Regime signal\n")
    tc = Counter(int(x) for x in pooled_issue(rows, "JE", "regime_true"))
    L.append(f"True regime at the issue times: wind-driven {tc.get(0, 0)}, transitional {tc.get(1, 0)}, "
             f"plume-dominated {tc.get(2, 0)} (of {len(rows) * len(ISSUES)}).\n")
    L.append("| method | median abs. log10 N_c error | regime Brier, mean | argmax class correct | lights G / A / R |")
    L.append("|---|---|---|---|---|")
    for m in METHODS:
        lights = Counter(r[f"{m}@{t}:light"] for r in rows for t in ISSUES)
        L.append(f"| {m} | {s[f'{m}:nc_logerr_med']:.2f} | {s[f'{m}:regime_brier_mean']:.3f} | "
                 f"{100 * s[f'{m}:regime_hit']:.1f} % | {lights['GREEN']} / {lights['AMBER']} / {lights['RED']} |")
    (RES / "v21_heldout.md").write_text("\n".join(L) + "\n")
    (RES / "v21_heldout.json").write_text(json.dumps(
        {"summary": s, "verdicts": [{"id": e, "held": bool(ok), "evidence": t} for e, ok, t in v],
         "sse_reduction_median": imp}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    report()
