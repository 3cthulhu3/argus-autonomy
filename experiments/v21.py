"""v2.1: assimilating drone observations into a Rothermel level-set forecast, and the regime signal
(docs/design/v2.1-assimilation-regime.md).

    python experiments/v21.py screen --tag base                 # screening seeds 0-39, default filter
    python experiments/v21.py screen --tag loc250 --set loc_m=250
    python experiments/v21.py screen-report                     # compare all screening tags
    python experiments/v21.py heldout                           # held-out seeds, frozen configuration
    python experiments/v21.py report                            # tables + verdict on E40-E47

Every scenario is appended to a .jsonl.gz file as it finishes, so an interrupted run
resumes where it stopped. Scenarios are deterministic per seed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from argus_autonomy.assimilation import METHODS, ExperimentConfig, run_scenario

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ------------------------------------------------------------------ fixed before any held-out run
SCREEN_SEEDS = range(0, 40)
HELDOUT_SEEDS = range(21000, 21200)
FROZEN = {"rw_wind_ms": 0.8}   # chosen on screening seeds (design doc §7); frozen before the held-out run
ISSUES = (20, 30, 40)
LEADS = (15, 30)


def _cfg(overrides: dict) -> ExperimentConfig:
    return ExperimentConfig().with_filter(**overrides)


def _job(args):
    seed, overrides, methods = args
    rec = run_scenario(_cfg(overrides), seed, methods=methods)
    rec["overrides"] = overrides
    return rec


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _run(path: Path, seeds, overrides: dict, workers: int = 2, methods=METHODS):
    done = set()
    if path.exists():
        try:
            with gzip.open(path, "rt") as fh:
                done = {json.loads(line)["seed"] for line in fh}
        except (EOFError, OSError, json.JSONDecodeError):
            pass
    todo = [(s, overrides, tuple(methods)) for s in seeds if s not in done]
    print(f"{path.name}: {len(done)} done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(workers) as ex, gzip.open(path, "at") as fh:
        for i, row in enumerate(ex.map(_job, todo)):
            fh.write(json.dumps(row, default=_json) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


# ------------------------------------------------------------------ statistics
def sign_test(a, b):
    """Paired one-sided sign test that a < b (ties dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wins, losses = int((a < b).sum()), int((a > b).sum())
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return wins, losses, float(p)


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def pooled(rows, meth, metric, leads=LEADS, issues=ISSUES):
    """One value per (scenario, issue time, lead), in a fixed order so methods pair up."""
    return np.array([r[f"{meth}@{t}+{ld}:{metric}"] for r in rows for t in issues for ld in leads], float)


def pooled_issue(rows, meth, metric, issues=ISSUES):
    return np.array([r[f"{meth}@{t}:{metric}"] for r in rows for t in issues], float)


def nc_log_err(rows, meth, issues=ISSUES):
    """|log10 N_c(estimate) - log10 N_c(truth)|, infinite values capped at 1e9 on both sides."""
    est = np.clip(pooled_issue(rows, meth, "nc_med", issues), 1e-6, 1e9)
    tru = np.clip(pooled_issue(rows, meth, "nc_true", issues), 1e-6, 1e9)
    return np.abs(np.log10(est) - np.log10(tru))


def summary(rows) -> dict:
    out = {"n": len(rows)}
    for m in METHODS:
        for ld in LEADS:
            b = pooled(rows, m, "brier", leads=(ld,))
            sse = pooled(rows, m, "sse_m2", leads=(ld,)) / pooled(rows, m, "true_m2", leads=(ld,))
            ar = pooled(rows, m, "area_ratio", leads=(ld,))
            out[f"{m}+{ld}"] = {
                "sse_rel_med": float(np.nanmedian(sse)), "sse_rel_mean": float(np.nanmean(sse)),
                "brier_med": float(np.nanmedian(b)), "brier_mean": float(np.nanmean(b)),
                "area_ratio_med": float(np.nanmedian(ar)),
                "area_ratio_iqr": [float(np.nanpercentile(ar, 25)), float(np.nanpercentile(ar, 75))],
                "cover10_med": float(np.nanmedian(pooled(rows, m, "cover10", leads=(ld,)))),
                "iou50_med": float(np.nanmedian(pooled(rows, m, "iou50", leads=(ld,)))),
            }
        out[f"{m}:wind_err_med"] = float(np.median(pooled_issue(rows, m, "wind_err")))
        out[f"{m}:m_err_med"] = float(np.median(pooled_issue(rows, m, "m_err")))
        out[f"{m}:nc_logerr_med"] = float(np.median(nc_log_err(rows, m)))
        out[f"{m}:regime_brier_mean"] = float(np.mean(pooled_issue(rows, m, "regime_brier")))
        out[f"{m}:regime_hit"] = float(np.mean(pooled_issue(rows, m, "regime_hit")))
    for a, b in (("JE", "OL"), ("JE", "SO"), ("SO", "OL")):
        for ld in LEADS:
            for metric in ("sse_m2", "brier"):
                w, l_, p = sign_test(pooled(rows, a, metric, leads=(ld,)), pooled(rows, b, metric, leads=(ld,)))
                out[f"{a}<{b} {metric}+{ld}"] = {"wins": w, "losses": l_, "p": p}
    return out


# ------------------------------------------------------------------ commands
def screen(tag: str, sets: list[str], n: int, methods):
    ov = {}
    for s in sets:
        k, v = s.split("=")
        ov[k] = float(v)
    _run(RES / f"v21_screen_{tag}.jsonl.gz", list(SCREEN_SEEDS)[:n], ov, methods=methods)


def screen_report():
    print("| tag | overrides | n | SSE JE<OL +30 | SSE JE<SO +30 | Brier JE<SO +30 | SSE/true area +30 OL / SO / JE (median) | IoU50 +30 OL / SO / JE | area ratio +30 JE | wind err OL / JE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in sorted(RES.glob("v21_screen_*.jsonl.gz")):
        rows = load(path)
        if not all(f"{m}@20+30:sse_m2" in rows[0] for m in METHODS):
            continue            # round-0 files: other code version, other metrics (see design doc)
        s = summary(rows)
        tag = path.name[len("v21_screen_"):-len(".jsonl.gz")]
        wl = lambda k: f"{s[k]['wins']}:{s[k]['losses']}"  # noqa: E731
        print(f"| {tag} | {rows[0]['overrides']} | {s['n']} | {wl('JE<OL sse_m2+30')} | {wl('JE<SO sse_m2+30')} | "
              f"{wl('JE<SO brier+30')} | "
              f"{s['OL+30']['sse_rel_med']:.3f} / {s['SO+30']['sse_rel_med']:.3f} / {s['JE+30']['sse_rel_med']:.3f} | "
              f"{s['OL+30']['iou50_med']:.2f} / {s['SO+30']['iou50_med']:.2f} / {s['JE+30']['iou50_med']:.2f} | "
              f"{s['JE+30']['area_ratio_med']:.2f} | {s['OL:wind_err_med']:.2f} / {s['JE:wind_err_med']:.2f} |")


def heldout():
    _run(RES / "v21_heldout_runs.jsonl.gz", HELDOUT_SEEDS, FROZEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["screen", "screen-report", "heldout", "report"])
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=len(SCREEN_SEEDS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    if a.what == "screen":
        screen(a.tag, a.set, a.n, a.methods)
    elif a.what == "screen-report":
        screen_report()
    elif a.what == "heldout":
        heldout()
    else:
        from v21_report import report  # noqa: PLC0415 - written with the protocol, see design doc
        report()


if __name__ == "__main__":
    main()
