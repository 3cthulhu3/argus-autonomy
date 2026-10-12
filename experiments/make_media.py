"""Demo GIF and the side-by-side snapshot figure.

The demo seed is *not* hand-picked: it is the nominal-condition seed whose
info-rate F1 is closest to the median over the benchmark (results/runs.csv).

    python experiments/make_media.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from argus_autonomy import NOMINAL, run_mission
from argus_autonomy.metrics import perimeter
from argus_autonomy.viz import _decorate, _mpl, belief_cmap, save_gif, BURNING, DRONE, INK, SURFACE

ROOT = Path(__file__).resolve().parents[1]


def median_seed() -> int:
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "runs.csv"))
            if r["condition"] == "nominal" and r["planner"] == "info-rate"]
    f1 = np.array([float(r["f1"]) for r in rows])
    return int(rows[int(np.argmin(np.abs(f1 - np.median(f1))))]["seed"])


def snapshot(seed: int) -> None:
    plt, _, _ = _mpl()
    planners = ("prior-lawnmower", "info-rate")
    runs = {p: run_mission(NOMINAL, p, seed, record_every=1) for p in planners}
    times = (5.0, 15.0, None)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), facecolor=SURFACE)
    for i, p in enumerate(planners):
        r = runs[p]
        for j, tm in enumerate(times):
            fr = r.frames[-1] if tm is None else min(
                r.frames, key=lambda f: abs((f["t"] + 1) * NOMINAL.dt_s / 60 - tm))
            ax = axes[i, j]
            ax.imshow(fr["belief"], cmap=belief_cmap(), vmin=0, vmax=1, interpolation="nearest")
            rr, cc = np.nonzero(perimeter(fr["truth"] != 0))
            ax.scatter(cc, rr, s=1.0, color=BURNING, lw=0)
            tr = np.array(r.path[: fr["t"] + 2])
            ax.plot(tr[:, 1], tr[:, 0], color=DRONE, lw=0.7, alpha=0.6)
            _decorate(ax, NOMINAL)
            minutes = (fr["t"] + 1) * NOMINAL.dt_s / 60
            ax.set_title(f"{p}, t = {minutes:.1f} min", fontsize=8, color=INK)
    fig.suptitle(f"Belief P(fire-affected) (blue) vs. true perimeter (orange), nominal seed {seed}", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "media" / "snapshot.png", dpi=200)
    plt.close(fig)


def main():
    seed = median_seed()
    print("median-F1 seed:", seed)
    r = run_mission(NOMINAL, "info-rate", seed, record_every=2)
    save_gif(NOMINAL, r, str(ROOT / "media" / "demo.gif"), fps=10, title=f"info-rate · seed {seed}  ·  ")
    snapshot(seed)


if __name__ == "__main__":
    main()
