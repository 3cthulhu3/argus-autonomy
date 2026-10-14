"""v1.1 milestone (iii): threshold sweep, operating points, calibration (design doc §13).

    python experiments/analyse_sweep.py --split select    # choose operating points
    python experiments/analyse_sweep.py --split heldout   # report them, untouched

Reads results/sweep_<split>.jsonl.gz from experiments/run_sweep.py.
Writes results/sweep_<split>.json and .md; the select split also writes
media/det_curves.png and media/reliability.png.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from run_decoy_benchmark import med_ci, pct, rate
from run_sweep import GRID, METHODS
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
MIN_PER_TICK = 10.0 / 60.0          # NOMINAL.dt_s
TARGET, MAX_UNDECIDED = 0.02, 0.05
APRIORI = (4.595, -4.595)
CAL_TIMES = (1.0, 2.0, 3.0, 4.0, 5.0)


def load(split: str, quick: bool = False) -> list[dict]:
    path = ROOT / "results" / f"sweep_{split}{'_quick' if quick else ''}.jsonl.gz"
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


class Mission:
    """First-passage ticks of one sweep run for every grid level."""

    def __init__(self, row: dict):
        self.family, self.seed = row["family"], row["seed"]
        tr = row["trace"]
        self.tick = np.array([t for t, _, _ in tr])
        self.llr = np.array([l for _, l, _ in tr])
        self.gate = np.array([g for _, _, g in tr], bool)
        self.cross = {float(k): v for k, v in row["crossings"].items()}
        self.safe = (row["geofence_violations"] == 0, row["returned_to_base"])

    def t_alert(self, a: float):
        i = np.flatnonzero(self.llr >= a)
        return int(self.tick[i[0]]) if i.size else None

    def t_clear(self, b: float):
        i = np.flatnonzero((self.llr <= b) & self.gate)
        return int(self.tick[i[0]]) if i.size else None

    def decide(self, a: float, b: float):
        ta, tb = self.t_alert(a), self.t_clear(b)
        if ta is None and tb is None:
            return "UNDECIDED", float("nan"), None
        if tb is None or (ta is not None and ta < tb):
            ok = self.cross[a][1] if a in self.cross else None
            return "ALERT", (ta + 1) * MIN_PER_TICK, ok
        return "CLEAR", (tb + 1) * MIN_PER_TICK, None

    def llr_at(self, minutes: float) -> float:
        i = np.searchsorted(self.tick, minutes / MIN_PER_TICK - 1, side="right") - 1
        return float(self.llr[max(i, 0)])


def evaluate(ms: list[Mission], a: float, b: float, rng=None) -> dict:
    B = [m for m in ms if m.family == "B"]
    F = [m for m in ms if m.family != "B"]
    dB = [m.decide(a, b) for m in B]
    dF = [m.decide(a, b) for m in F]
    out = {
        "a": a, "b": b,
        "P_FA": rate(sum(d == "ALERT" for d, _, _ in dB), len(B)),
        "P_U_B": rate(sum(d == "UNDECIDED" for d, _, _ in dB), len(B)),
        "fire_cleared": rate(sum(d == "CLEAR" for d, _, _ in dF), len(F)),
        "fire_alert_wrong_place": rate(sum(d == "ALERT" and ok is False for d, _, ok in dF), len(F)),
        "P_U_fire": rate(sum(d == "UNDECIDED" for d, _, _ in dF), len(F)),
        "P_D": rate(sum(d == "ALERT" and ok for d, _, ok in dF), len(F)),
    }
    t_ok = [t for d, t, ok in dF if d == "ALERT" and ok]
    t_cl = [t for d, t, _ in dB if d == "CLEAR"]
    out["t_alert_median"] = float(np.median(t_ok)) if t_ok else float("inf")
    out["t_clear_median"] = float(np.median(t_cl)) if t_cl else float("inf")
    if rng is not None:
        out["t_alert"] = med_ci(t_ok, rng)
        out["t_clear"] = med_ci(t_cl, rng)
    return out


def feasible(e: dict) -> bool:
    return (e["P_FA"]["ci95"][1] <= TARGET and e["P_U_B"]["p"] <= MAX_UNDECIDED
            and e["P_U_fire"]["p"] <= MAX_UNDECIDED)


def choose(evals: list[dict]):
    ok = [e for e in evals if feasible(e)]
    if not ok:
        return None
    return min(ok, key=lambda e: (e["fire_cleared"]["k"], e["t_alert_median"], e["t_clear_median"], e["a"]))


def platt(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic fit P(y=1 | x) = sigmoid(alpha + kappa x) by maximum likelihood."""
    s = max(np.abs(x).max(), 1.0)

    def nll(p):
        z = p[0] + p[1] * x / s
        return float(np.sum(np.logaddexp(0, z) - y * z))

    r = minimize(nll, np.zeros(2), method="BFGS")
    return float(r.x[0]), float(r.x[1] / s)


def calibration(ms: list[Mission]) -> dict:
    B = [m for m in ms if m.family == "B"]
    F = [m for m in ms if m.family != "B"]
    wald = []
    for a in GRID:
        k = sum(m.t_alert(a) is not None for m in B)
        wald.append({"a": a, "nominal": float(np.exp(-a)), "measured": rate(k, len(B))})
    rel = {}
    edges = np.array([-np.inf, -100, -20, -4.595, 0, 4.595, 20, 100, np.inf])
    for t in CAL_TIMES:
        x = np.array([m.llr_at(t) for m in B + F])
        y = np.array([0] * len(B) + [1] * len(F))
        alpha, kappa = platt(x, y)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x > lo) & (x <= hi)
            if sel.any():
                bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                             "claimed": float(expit(x[sel]).mean()), "observed": float(y[sel].mean())})
        rel[str(t)] = {"alpha": alpha, "kappa": kappa, "bins": bins,
                       "median_abs_llr": float(np.median(np.abs(x)))}
    return {"wald": wald, "reliability": rel}


def det_figure(curves: dict, chosen: dict, path: Path) -> None:
    """Delay against false alerts, best pair per false-alert level. Missed fires are
    not plotted: among deciding pairs they floor at the same count at every level
    (results/sweep_select.md), so that panel would be a single point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for (meth, evs), col in zip(curves.items(), ("#4a6fa5", "#c1666b", "#2e8b57")):
        # only pairs that decide: a pair that leaves > 5 % undecided trades errors for silence
        evs = [e for e in evs if e["P_U_B"]["p"] <= MAX_UNDECIDED and e["P_U_fire"]["p"] <= MAX_UNDECIDED]
        pts = sorted({(e["P_FA"]["p"], e["t_alert_median"]) for e in evs if np.isfinite(e["t_alert_median"])})
        front, best = [], np.inf
        for p, t in pts:
            if t < best:
                front.append((p, t))
                best = t
        ax.step([100 * p for p, _ in front], [t for _, t in front], where="post", color=col, label=meth)
        c = chosen.get(meth)
        if c:
            ax.plot(100 * c["P_FA"]["p"], c["t_alert_median"], "o", color=col)
    ax.axvline(100 * TARGET, color="0.5", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("false alerts on decoy cues, % (family B)")
    ax.set_ylabel("median time to correct alert, min")
    ax.set_title("Delay vs false alerts (pairs with ≤ 5 % undecided)\ndots: chosen operating points", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def rel_figure(cal: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1)
    for t, col in zip(("1.0", "3.0", "5.0"), ("#4a6fa5", "#c1666b", "#2e8b57")):
        bins = cal["reliability"][t]["bins"]
        ax.plot([b["claimed"] for b in bins], [b["observed"] for b in bins], "o-", color=col,
                label=f"t = {float(t):.0f} min, κ = {cal['reliability'][t]['kappa']:.3g}")
    ax.set_xlabel("claimed P(fire) = σ(Λ)")
    ax.set_ylabel("observed fraction with a fire")
    ax.set_title("M3 reliability (selection seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def row_md(name: str, e: dict) -> str:
    return (f"| {name} | {e['a']:g} / {e['b']:g} | {pct(e['P_FA'])} | {pct(e['P_U_B'])} | "
            f"{pct(e['fire_cleared'])} | {pct(e['fire_alert_wrong_place'])} | {pct(e['P_U_fire'])} | "
            f"{e['t_alert_median']:.2f} | {e['t_clear_median']:.2f} |")


HEADER = ["| method | a / b | false alerts (B) | undecided (B) | real fire CLEARed (A ∪ C) | "
          "alert in wrong place (A ∪ C) | undecided (A ∪ C) | median time to correct alert, min | "
          "median time to CLEAR, min |", "|---|---|---|---|---|---|---|---|---|"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("select", "heldout"), required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows = load(args.split, args.quick)
    by = {m: [Mission(r) for r in rows if r["method"] == m] for m in METHODS}
    rng = np.random.default_rng(0)
    q = "_quick" if args.quick else ""
    out: dict = {"split": args.split, "n": {m: len(v) for m, v in by.items()}}
    safety = {m: {"geofence_violations": int(sum(not x.safe[0] for x in v)),
                  "returned_to_base": int(sum(x.safe[1] for x in v)), "n": len(v)} for m, v in by.items()}
    out["safety"] = safety

    if args.split == "select":
        curves, chosen = {}, {}
        for m, ms in by.items():
            evs = [evaluate(ms, a, -b) for a in GRID for b in GRID]
            curves[m] = evs
            c = choose(evs)
            chosen[m] = c
            out.setdefault("n_feasible", {})[m] = sum(feasible(e) for e in evs)
        out["apriori"] = {m: evaluate(ms, *APRIORI) for m, ms in by.items()}
        out["operating_points"] = {m: (None if c is None else {"a": c["a"], "b": c["b"]}) for m, c in chosen.items()}
        out["chosen"] = chosen
        out["calibration"] = {m: calibration(ms) for m, ms in by.items()}
        if not args.quick:
            det_figure(curves, chosen, ROOT / "media" / "det_curves.png")
            rel_figure(out["calibration"]["M3"], ROOT / "media" / "reliability.png")
        L = ["# v1.1 milestone (iii): threshold sweep on selection seeds", "",
             ("Generated by `experiments/analyse_sweep.py --split select` from `run_sweep.py` logs. "
              "B seeds 0–999, A and C 0–499. Decisions are first-passage times of one logged run per "
              "mission (design doc §13). Rates: percentage (count; Clopper–Pearson 95 % CI)."), "",
             "## Operating points (rule fixed in design doc §13 before this run)", "", *HEADER]
        for m in METHODS:
            L.append(row_md(f"{m}, a priori", out["apriori"][m]))
            c = chosen[m]
            L.append(row_md(f"**{m}, chosen**", c) if c else f"| {m}, chosen | no feasible pair | | | | | | | |")
        L += ["", "Feasible pairs out of 256: " + ", ".join(f"{m} {out['n_feasible'][m]}" for m in METHODS), "",
              "## Calibration", "", "### Wald gap: P(Λ ever ≥ a | decoy cue) against the nominal e^(−a)", "",
              "| a | nominal | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for i, a in enumerate(GRID):
            L.append(f"| {a:g} | {np.exp(-a):.2g} | " + " | ".join(
                pct(out["calibration"][m]["wald"][i]["measured"]) for m in METHODS) + " |")
        L += ["", "### Platt slope κ: P(fire | Λ_t) = σ(α + κ Λ_t); κ = 1 if Λ were calibrated", "",
              "| t, min | " + " | ".join(f"{m} κ | {m} median abs Λ" for m in METHODS) + " |",
              "|---|" + "---|---|" * len(METHODS)]
        for t in CAL_TIMES:
            L.append(f"| {t:g} | " + " | ".join(
                f"{out['calibration'][m]['reliability'][str(t)]['kappa']:.3g} | "
                f"{out['calibration'][m]['reliability'][str(t)]['median_abs_llr']:.0f}" for m in METHODS) + " |")
        L += ["", "### M3 reliability at t = 3 min", "", "| Λ bin | missions | claimed P(fire) | observed |",
              "|---|---|---|---|"]
        for b in out["calibration"]["M3"]["reliability"]["3.0"]["bins"]:
            L.append(f"| ({b['lo']:g}, {b['hi']:g}] | {b['n']} | {b['claimed']:.3f} | {b['observed']:.3f} |")
    else:
        sel = json.loads((ROOT / "results" / "sweep_select.json").read_text())["operating_points"]
        res = {}
        for m, ms in by.items():
            res[m] = {"apriori": evaluate(ms, *APRIORI, rng=rng)}
            if sel.get(m):
                res[m]["chosen"] = evaluate(ms, sel[m]["a"], sel[m]["b"], rng=rng)
        out["results"] = res
        L = ["# v1.1 milestone (iii): operating points on held-out seeds", "",
             ("Generated by `experiments/analyse_sweep.py --split heldout`. B seeds 1000–1999, A and C "
              "500–999, never used for any choice. Operating points read from `results/sweep_select.json`."),
             "", *HEADER]
        for m in METHODS:
            L.append(row_md(f"{m}, a priori", res[m]["apriori"]))
            if "chosen" in res[m]:
                L.append(row_md(f"**{m}, chosen**", res[m]["chosen"]))
    L += ["", "## Safety (sweep runs fly to endurance)", "", "| method | geofence violations | returned to base |",
          "|---|---|---|"] + [f"| {m} | {s['geofence_violations']} | {s['returned_to_base']}/{s['n']} |"
                               for m, s in safety.items()]
    (ROOT / "results" / f"sweep_{args.split}{q}.json").write_text(json.dumps(out, indent=1, default=float))
    (ROOT / "results" / f"sweep_{args.split}{q}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
