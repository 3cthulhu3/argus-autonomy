"""Command-line entry point: run one mission and optionally render it.

    argus-demo --planner info-rate --seed 7
    argus-demo --planner info-rate --seed 7 --condition wind-mismatch --gif mission.gif
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import NOMINAL, WIND_MISMATCH
from .mission import PLANNERS, run_mission

CONDITIONS = {
    "nominal": NOMINAL,
    "wind-mismatch": WIND_MISMATCH,
    "degraded-prior": replace(NOMINAL, prior=replace(NOMINAL.prior, geoloc_sigma=6.0, latency_steps=12)),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run one simulated wildfire-mapping mission.")
    ap.add_argument("--planner", choices=PLANNERS, default="info-rate")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", help="write an animation of the mission to this path")
    a = ap.parse_args(argv)
    cfg = CONDITIONS[a.condition]
    r = run_mission(cfg, a.planner, a.seed, record_every=2 if a.gif else 0)
    print(json.dumps(r.summary(), indent=2, default=float))
    if a.gif:
        from .viz import save_gif
        save_gif(cfg, r, a.gif, title=f"{a.planner} · {a.condition} · seed {a.seed}  ·  ")
        print(f"wrote {a.gif}")


if __name__ == "__main__":
    main()
