"""Monte Carlo benchmark: every planner on the same seeds, in three conditions.

    python experiments/run_benchmark.py --seeds 100

Writes results/runs.csv, results/summary.json, results/summary.md and the
figures in media/.  All numbers quoted in README.md and the report come from
these files.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from argus_autonomy import NOMINAL, PLANNERS, WIND_MISMATCH, run_mission

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "nominal": NOMINAL,
    # estimator assumes calm air while the real fire is wind-driven
    "wind-mismatch": WIND_MISMATCH,
    # satellite product is older and worse geolocated
    "degraded-prior": replace(NOMINAL, prior=replace(NOMINAL.prior, geoloc_sigma=6.0, latency_steps=12)),
}

COLORS = {"info-rate": "#2a78d6", "prior-lawnmower": "#eb6834", "lawnmower": "#1baf7a"}


def _job(args):
    cond, planner, seed = args
    r = run_mission(CONDITIONS[cond], planner, seed)
    row = r.summary() | {"condition": cond}
    rec = np.asarray(r.recall_confirmed)
    hit = np.nonzero(rec >= 0.5)[0]
    row["t_recall50"] = float((hit[0] + 1) * CONDITIONS[cond].dt_s / 60) if hit.size else float("nan")
    return row, r.entropy, r.recall_confirmed


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def summarise(rows):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for p in PLANNERS:
            rs = [r for r in rows if r["condition"] == cond and r["planner"] == p]
            tc = np.array([r["t_confirm"] for r in rs])
            t50 = np.array([r["t_recall50"] for r in rs])
            out[cond][p] = {
                "n": len(rs),
                "t_confirm_median_min": float(np.nanmedian(tc)),
                "t_confirm_iqr_min": [float(np.nanpercentile(tc, 25)), float(np.nanpercentile(tc, 75))],
                "never_confirmed": int(np.isnan(tc).sum()),
                "t_recall50_median_min": float(np.nanmedian(t50)) if np.isfinite(t50).any() else None,
                "recall50_reached": int(np.isfinite(t50).sum()),
                "f1_mean": float(np.mean([r["f1"] for r in rs])),
                "f1_ci95": ci95([r["f1"] for r in rs]),
                "boundary_m_median": float(np.nanmedian([r["boundary_m"] for r in rs])),
                "recall_final_mean": float(np.mean([r["recall_final"] for r in rs])),
                "false_confirmations_mean": float(np.mean([r["false_confirmations"] for r in rs])),
                "geofence_violations_total": int(sum(r["geofence_violations"] for r in rs)),
                "returned_to_base": int(sum(bool(r["returned_to_base"]) for r in rs)),
                "min_reserve_ticks": int(min(r["reserve_left"] for r in rs)),
            }
        # paired tests, info-rate vs the strongest baseline, on identical seeds
        a = {r["seed"]: r for r in rows if r["condition"] == cond and r["planner"] == "info-rate"}
        b = {r["seed"]: r for r in rows if r["condition"] == cond and r["planner"] == "prior-lawnmower"}
        seeds = sorted(set(a) & set(b))
        tests = {}
        for key in ("t_confirm", "f1", "boundary_m", "t_recall50"):
            x = np.array([a[s][key] for s in seeds], float)
            y = np.array([b[s][key] for s in seeds], float)
            ok = np.isfinite(x) & np.isfinite(y)
            d = x[ok] - y[ok]
            p = float(wilcoxon(x[ok], y[ok]).pvalue) if ok.sum() > 5 and np.any(d != 0) else float("nan")
            tests[key] = {"pairs": int(ok.sum()), "median_diff": float(np.median(d)) if d.size else float("nan"),
                          "wilcoxon_p": p,
                          "info_rate_better_frac": float(np.mean(d < 0) if key in ("t_confirm", "boundary_m", "t_recall50") else np.mean(d > 0))}
        out[cond]["paired_vs_prior_lawnmower"] = tests
    return out


def markdown(summary):
    lines = ["| condition | planner | first confirmation, median [IQR] (min) | 50 % of fire confirmed, median (min) | F1 at landing (mean ± 95 % CI) | perimeter error, median (m) | geofence violations | returned to base |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, d in summary.items():
        for p in PLANNERS:
            s = d[p]
            t50 = "—" if s["t_recall50_median_min"] is None else f"{s['t_recall50_median_min']:.1f} ({s['recall50_reached']}/{s['n']})"
            lines.append(
                f"| {cond} | {p} | {s['t_confirm_median_min']:.1f} [{s['t_confirm_iqr_min'][0]:.1f}–{s['t_confirm_iqr_min'][1]:.1f}] "
                f"| {t50} | {s['f1_mean']:.3f} ± {s['f1_ci95']:.3f} | {s['boundary_m_median']:.0f} "
                f"| {s['geofence_violations_total']} | {s['returned_to_base']}/{s['n']} |")
    return "\n".join(lines) + "\n"


def figures(rows, traces, summary, cfg_dt_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "legend.frameon": False})
    media = ROOT / "media"
    media.mkdir(exist_ok=True)

    # 1. confirmed-recall and entropy over time, per condition
    for metric, ylabel, fname in (("recall", "share of burnt area confirmed (P ≥ 0.95)", "recall_vs_time.png"),
                                  ("entropy", "map uncertainty (bits / cell)", "entropy_vs_time.png")):
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(9.5, 2.9), sharey=True)
        for ax, cond in zip(axes, CONDITIONS):
            for p in PLANNERS:
                tr = [t[metric] for t in traces if t["condition"] == cond and t["planner"] == p]
                L = max(len(x) for x in tr)
                M = np.array([np.pad(x, (0, L - len(x)), mode="edge") for x in tr])
                t = np.arange(1, L + 1) * cfg_dt_s / 60
                mu = M.mean(0)
                lo, hi = np.percentile(M, 25, 0), np.percentile(M, 75, 0)
                ax.fill_between(t, lo, hi, color=COLORS[p], alpha=0.12, lw=0)
                ax.plot(t, mu, color=COLORS[p], lw=1.6, label=p)
            ax.set_title(cond, fontsize=9, color="#0b0b0b")
            ax.set_xlabel("mission time (min)")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(media / fname, dpi=200)
        plt.close(fig)

    # 2. time to first confirmation, per seed, as a strip/box per planner and condition
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(9.5, 2.8), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        data = [[r["t_confirm"] for r in rows if r["condition"] == cond and r["planner"] == p
                 and np.isfinite(r["t_confirm"])] for p in PLANNERS]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "#0b0b0b", "lw": 1.4})
        for patch, p in zip(bp["boxes"], PLANNERS):
            patch.set_facecolor(COLORS[p])
            patch.set_alpha(0.35)
            patch.set_edgecolor(COLORS[p])
        rng = np.random.default_rng(0)
        for i, (d, p) in enumerate(zip(data, PLANNERS), start=1):
            ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=5, color=COLORS[p], alpha=0.6, lw=0)
        ax.set_yticks(range(1, len(PLANNERS) + 1), PLANNERS)
        ax.set_title(cond, fontsize=9, color="#0b0b0b")
        ax.set_xlabel("time to first confirmation (min)")
    fig.tight_layout()
    fig.savefig(media / "time_to_confirmation.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    jobs = [(c, p, s) for c in CONDITIONS for p in PLANNERS for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        out = list(ex.map(_job, jobs, chunksize=8))
    rows = [o[0] for o in out]
    traces = [{"condition": o[0]["condition"], "planner": o[0]["planner"],
               "entropy": o[1], "recall": o[2]} for o in out]

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    with open(res / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    (res / "summary.md").write_text(markdown(summary))
    figures(rows, traces, summary, NOMINAL.dt_s)
    print(markdown(summary))
    for c in CONDITIONS:
        print(c, json.dumps(summary[c]["paired_vs_prior_lawnmower"], indent=None))


if __name__ == "__main__":
    main()
