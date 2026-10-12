"""How good is the filter's prediction step on its own (no observations)?

Starts the filter from the *true* burnt area at launch and propagates it for
30 fire steps (30 min) without any measurements, for four variants:
{gated, ungated mean-field} x {wind known, calm-air assumption}.

    python experiments/calibrate_prediction.py --seeds 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from argus_autonomy import NOMINAL
from argus_autonomy.mission import make_scenario
from argus_autonomy.perception import OccupancyFilter

ROOT = Path(__file__).resolve().parents[1]
CHECK = (10, 20, 30)


def run(seed: int, gate, wind_known: bool):
    cfg = NOMINAL
    sc = make_scenario(cfg, seed)
    start = np.where(sc.fire.affected, 0.999, 1e-6)
    f = OccupancyFilter(start, cfg.sensor, replace(cfg.model, wind_known=wind_known), sc.wind_dir,
                        cfg.fire.wind_gain, cfg.fire.burn_steps, gate=gate)
    out = {}
    for k in range(1, max(CHECK) + 1):
        sc.fire.step()
        f.predict()
        if k in CHECK:
            truth, est = sc.fire.affected, f.prob >= 0.5
            out[k] = (est.sum() / truth.sum(), (est & truth).sum() / (est | truth).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    variants = {"gated, wind known": (0.5, True), "gated, calm-air model": (0.5, False),
                "ungated, wind known": (None, True), "ungated, calm-air model": (None, False)}
    res = {}
    for name, (gate, wk) in variants.items():
        runs = [run(s, gate, wk) for s in range(a.seeds)]
        res[name] = {f"{k} min": {"area_ratio_median": float(np.median([r[k][0] for r in runs])),
                                  "iou_median": float(np.median([r[k][1] for r in runs]))} for k in CHECK}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "prediction_calibration.json").write_text(json.dumps(res, indent=2))
    lines = ["| variant | " + " | ".join(f"{k} min: area ratio / IoU" for k in CHECK) + " |",
             "|---|" + "---|" * len(CHECK)]
    for name, d in res.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{d[f'{k} min']['area_ratio_median']:.2f} / {d[f'{k} min']['iou_median']:.2f}" for k in CHECK) + " |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "prediction_calibration.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
