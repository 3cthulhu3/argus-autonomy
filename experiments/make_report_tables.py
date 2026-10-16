"""Generate the LaTeX tables of the report directly from results/*.json,
so the PDF can never drift from the data.

    python experiments/make_report_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report" / "tables"
NAMES = {"lawnmower": "Lawnmower", "prior-lawnmower": "Prior-lawnmower", "info-rate": "Info-rate"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    rows = []
    for cond, d in s.items():
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{6}}{{l}}{{\emph{{{cond}}}}}\\")
        best_f1 = max(d[p]["f1_mean"] for p in NAMES)
        for p, label in NAMES.items():
            x = d[p]
            t50 = "--" if x["t_recall50_median_min"] is None else f"{x['t_recall50_median_min']:.1f} ({x['recall50_reached']})"
            f1 = f"{x['f1_mean']:.3f} $\\pm$ {x['f1_ci95']:.3f}"
            if x["f1_mean"] == best_f1:
                f1 = r"\textbf{" + f1 + "}"
            rows.append(f"{label} & {x['t_confirm_median_min']:.1f} [{x['t_confirm_iqr_min'][0]:.1f}, {x['t_confirm_iqr_min'][1]:.1f}] "
                        f"& {t50} & {f1} & {x['boundary_m_median']:.0f} & {x['false_confirmations_mean']:.1f} \\\\")
    (OUT / "main.tex").write_text("\n".join(rows) + "\n")

    paired = []
    for cond, d in s.items():
        t = d["paired_vs_prior_lawnmower"]
        def fmt(k, unit):
            p = t[k]["wilcoxon_p"]
            ps = f"{p:.0e}".replace("e-0", "e-").replace("e-", r"\times 10^{-") + "}" if p < 1e-3 else f"{p:.3f}"
            v = f"{t[k]['median_diff']:+.{1 if unit else 3}f}"
            v = f"${v[0]}${v[1:]}"   # proper math-mode sign
            return (f"{v}{unit} & {100 * t[k]['info_rate_better_frac']:.0f}\\% & ${ps}$")
        paired.append(f"{cond} & {fmt('t_confirm', ' min')} & {fmt('f1', '')} & {fmt('boundary_m', ' m')} \\\\")
    (OUT / "paired.tex").write_text("\n".join(paired) + "\n")

    c = json.loads((ROOT / "results" / "prediction_calibration.json").read_text())
    rows = []
    for name, d in c.items():
        rows.append(name + " & " + " & ".join(
            f"{d[k]['area_ratio_median']:.2f} & {d[k]['iou_median']:.2f}" for k in ("10 min", "20 min", "30 min")) + r" \\")
    (OUT / "calibration.tex").write_text("\n".join(rows) + "\n")

    t = json.loads((ROOT / "results" / "prior_tempering.json").read_text())
    rows = []
    for key, d in t["sweep"].items():
        cond, bg = key.split("|")
        rows.append(f"{cond} & {bg} & {d['f1_mean']:.3f} & {d['f1_below_0.7']} & {d['t_confirm_median_min']:.1f} & {d['false_confirmations_mean']:.1f} \\\\")
        if key == "nominal|0.05":
            rows.append(r"\midrule")
    (OUT / "tempering.tex").write_text("\n".join(rows) + "\n")
    (OUT / "tempering_facts.tex").write_text(
        f"\\newcommand{{\\nodetect}}{{{t['degraded_prior_no_detection']}}}\n"
        f"\\newcommand{{\\outsideprior}}{{{t['degraded_prior_fire_entirely_outside_prior']}}}\n")
    # ---- v1.1: decoys ----------------------------------------------------------
    def pc(r):
        return f"{100 * r['p']:.1f}\\,\\%"

    def md(m):
        return "--" if not m or m["median"] is None else f"{m['median']:.2f}"

    m2 = json.loads((ROOT / "results" / "decoys_m2.json").read_text())
    names = {"M0 (v1.0)": "M0 (v1.0)", "M1, no gate (milestone i)": "M1, no gate", "M1": "M1",
             "M2": "M2", "M3": "M3", "Oracle": "Oracle"}
    rows = []
    for key, label in names.items():
        a, b, c = m2[key]["A"], m2[key]["B"], m2[key]["C"]
        rows.append(f"{label} & {pc(b['P_FA'])} & {md(b['t_clear_min'])} & {pc(a['P_CLEAR_on_fire'])} & "
                    f"{pc(a['P_MIS'])} & {md(c['t_correct_alert_min'])} & {c['f1_mean']:.3f} \\\\")
    (OUT / "v11_ladder.tex").write_text("\n".join(rows) + "\n")

    ap = json.loads((ROOT / "results" / "sweep_heldout.json").read_text())["results"]["M3"]["apriori"]
    op = json.loads((ROOT / "results" / "operating_point.json").read_text())

    def ci(r):
        return f"{pc(r)} [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]"

    rows = [
        f"False alerts on a decoy cue (B) & {ci(ap['P_FA'])} & {ci(op['P_FA'])} \\\\",
        f"Real fire CLEARed (A $\\cup$ C) & {ci(ap['fire_cleared'])} & {ci(op['fire_cleared'])} \\\\",
        f"Alert in the wrong place (A $\\cup$ C) & {ci(ap['fire_alert_wrong_place'])} & {ci(op['fire_alert_wrong_place'])} \\\\",
        f"Undecided (B) & {ap['P_U_B']['k']} & {op['P_U_B']['k']} \\\\",
        f"Time to correct alert, median (min) & {ap['t_alert_median']:.2f} & {op['t_alert']['median']:.2f} \\\\",
        f"Time to CLEAR a decoy cue, median (min) & {ap['t_clear_median']:.2f} & {op['t_clear']['median']:.2f} \\\\",
        f"Geofence violations / returns to base & -- & {op['geofence_violations']} / {op['returned_to_base']} of {op['n']} \\\\",
    ]
    (OUT / "v11_op.tex").write_text("\n".join(rows) + "\n")
    (OUT / "v11_facts.tex").write_text(
        f"\\newcommand{{\\opa}}{{{op['a']:g}}}\n\\newcommand{{\\opb}}{{{op['b']:g}}}\n"
        f"\\newcommand{{\\opmismatch}}{{{op['first_passage_mismatches']}}}\n")
    print("wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
