"""Motivating probe for v1.1: does the v1.0 estimator confirm a decoy as fire?

    python experiments/probe_decoys.py --seeds 100

A decoy is a persistent, non-spreading heat source (sunlit rock, vehicle,
contained burn). The v1.0 sensor model treats false alarms as independent per
look, so a source that answers "hot" on every pass is, to v1.0, simply fire.

Setup (deliberately mild): nominal v1.0 scenario, info-rate planner, plus three
static 2x2 decoys that the thermal camera sees exactly like fire. Decoys sit
in allowed airspace, at least 12 cells (300 m) from the fire at launch, and
are NOT in the satellite cue, so the planner has no reason to favour them.
Any confirmation of a decoy cell is a false fire confirmation.

Writes results/probe_decoys.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from argus_autonomy import NOMINAL, mission
from argus_autonomy.perception import observe as observe_v10
from argus_autonomy.safety import Geofence

ROOT = Path(__file__).resolve().parents[1]
N_DECOYS, SIZE, MIN_SEP = 3, 2, 12


def place_decoys(cfg, affected, fence, rng) -> np.ndarray:
    decoy = np.zeros_like(affected)
    rr, cc = np.nonzero(affected)
    placed = 0
    while placed < N_DECOYS:
        r, c = (int(x) for x in rng.integers(4, cfg.n - 4 - SIZE, size=2))
        if not fence.allowed[r:r + SIZE, c:c + SIZE].all():
            continue
        if np.min(np.hypot(rr - r, cc - c)) < MIN_SEP or decoy[r - 1:r + SIZE + 1, c - 1:c + SIZE + 1].any():
            continue
        decoy[r:r + SIZE, c:c + SIZE] = True
        placed += 1
    return decoy


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    cfg, fence = NOMINAL, Geofence(NOMINAL.n, NOMINAL.nfz)

    per_seed = []
    try:
        for seed in range(args.seeds):
            sc = mission.make_scenario(cfg, seed)
            decoy = place_decoys(cfg, sc.fire.affected, fence, np.random.default_rng(10_000 + seed))
            mission.observe = lambda truth, pos, s, g, d=decoy: observe_v10(truth | d, pos, s, g)
            r = mission.run_mission(cfg, "info-rate", seed, record_every=10**9)
            belief = r.frames[-1]["belief"]
            n_conf = int(np.sum((belief >= cfg.confirm_threshold) & decoy))
            per_seed.append({"seed": seed, "decoy_cells_confirmed": n_conf, "f1": r.f1})
    finally:
        mission.observe = observe_v10

    k = sum(s["decoy_cells_confirmed"] > 0 for s in per_seed)
    n = len(per_seed)
    out = {
        "seeds": n,
        "decoy_cells_per_mission": N_DECOYS * SIZE * SIZE,
        "missions_with_false_confirmation": k,
        "rate": k / n,
        "rate_ci95_clopper_pearson": clopper_pearson(k, n),
        "mean_decoy_cells_confirmed": float(np.mean([s["decoy_cells_confirmed"] for s in per_seed])),
        "per_seed": per_seed,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "probe_decoys.json").write_text(json.dumps(out, indent=1))
    lo, hi = out["rate_ci95_clopper_pearson"]
    print(f"{k}/{n} missions confirm >= 1 decoy cell as fire "
          f"({100 * k / n:.0f} %, 95 % CI {100 * lo:.0f}-{100 * hi:.0f} %); "
          f"mean {out['mean_decoy_cells_confirmed']:.1f} of {out['decoy_cells_per_mission']} decoy cells")


if __name__ == "__main__":
    main()
