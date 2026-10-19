"""v2.0 response layer, part 1: what the drone hands to the people who respond.

After an ALERT the estimator holds a map of P(fire) (the H1 filter of the decision
test). This module turns it into a responder brief -- where, how big, which way it
is heading, how far from the dock -- and into short-horizon forecasts of the
affected area, so that a crew is sent to where the fire *will* be.

Two forecasts are compared (design doc v2.0, E34-E35):
  mean-field  the estimator's own prediction step (v1.0 ``OccupancyFilter.predict``),
              iterated without observations. Known to under-predict growth because
              of its source gate (results/prediction_calibration.md).
  ensemble    K stochastic fires sampled from the belief and advanced with the
              estimator's spread kernel. Cells are sampled independently (the belief
              carries no correlations) and each sampled cell is given a uniformly
              random remaining burn time: both are stated assumptions, not facts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from .config import MissionConfig
from .perception import OccupancyFilter, spread_kernel


# --------------------------------------------------------------------------- brief
@dataclass
class ResponderBrief:
    alert_min: float                 # mission clock at the alert
    east_m: float                    # alert location relative to the dock
    north_m: float
    dock_dist_m: float
    area_m2: float                   # cells with P(fire) >= 0.5
    expected_area_m2: float          # sum of P(fire)
    length_m: float                  # 2-sigma extent of the belief blob along its major axis
    width_m: float
    heading_deg: float               # direction the fire is pushed towards (wind), compass
    forecast_area_m2: dict           # minutes ahead -> expected affected area (ensemble)
    envelope_cells: dict             # minutes ahead -> boolean map, ensemble P >= 0.1


def _blob(belief: np.ndarray) -> np.ndarray:
    lab, k = label(belief >= 0.5, structure=np.ones((3, 3), bool))
    if k == 0:
        return belief >= belief.max() * 0.5
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(np.argmax(sizes)))


def compass_from_rowcol(angle_rc: float) -> float:
    """Grid angle (0 = +col = east, pi/2 = +row = south on screen) -> compass bearing."""
    dx, dy = np.cos(angle_rc), -np.sin(angle_rc)      # east, north
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def responder_brief(belief: np.ndarray, cfg: MissionConfig, wind_dir: float, alert_min: float,
                    horizons_min: tuple[int, ...] = (15, 30), k: int = 200,
                    rng: np.random.Generator | None = None) -> ResponderBrief:
    m = _blob(belief)
    w = belief * m
    rr, cc = np.indices(belief.shape)
    r0, c0 = (rr * w).sum() / w.sum(), (cc * w).sum() / w.sum()
    cov = np.cov(np.stack([rr[m], cc[m]]), aweights=belief[m]) if m.sum() > 1 else np.zeros((2, 2))
    ev = np.sqrt(np.clip(np.linalg.eigvalsh(np.atleast_2d(cov)), 0, None))
    cell = cfg.cell_m
    east = (c0 - cfg.base[1]) * cell
    north = -(r0 - cfg.base[0]) * cell
    rng = rng or np.random.default_rng(0)
    fa, env = {}, {}
    for h in horizons_min:
        f = forecast_ensemble(belief, cfg, wind_dir, h, k=k, rng=rng)
        fa[h] = float(f.sum() * cell * cell)
        env[h] = f >= 0.1
    return ResponderBrief(
        alert_min=alert_min, east_m=float(east), north_m=float(north),
        dock_dist_m=float(np.hypot(east, north)),
        area_m2=float(m.sum() * cell * cell), expected_area_m2=float(belief.sum() * cell * cell),
        length_m=float(4 * ev.max() * cell + cell), width_m=float(4 * ev.min() * cell + cell),
        heading_deg=compass_from_rowcol(wind_dir), forecast_area_m2=fa, envelope_cells=env,
    )


# --------------------------------------------------------------------------- forecasts
def forecast_meanfield(belief: np.ndarray, cfg: MissionConfig, wind_dir: float, minutes: int) -> np.ndarray:
    """The estimator's own prediction, iterated ``minutes`` fire steps (1 step = 1 min)."""
    f = OccupancyFilter(belief, cfg.sensor, cfg.model, wind_dir, cfg.fire.wind_gain, cfg.fire.burn_steps)
    for _ in range(int(round(minutes * 60 / (cfg.fire.fire_step_ticks * cfg.dt_s)))):
        f.predict()
    return f.prob


def _shift3(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(a)
    n, m = a.shape[1:]
    out[:, max(dr, 0):n + min(dr, 0), max(dc, 0):m + min(dc, 0)] = \
        a[:, max(-dr, 0):n + min(-dr, 0), max(-dc, 0):m + min(-dc, 0)]
    return out


def forecast_ensemble(belief: np.ndarray, cfg: MissionConfig, wind_dir: float, minutes: int,
                      k: int = 200, rng: np.random.Generator | None = None) -> np.ndarray:
    """P(affected) after ``minutes`` from K stochastic fires drawn from the belief."""
    rng = rng or np.random.default_rng(0)
    q = spread_kernel(cfg.model, wind_dir, cfg.fire.wind_gain)
    burn = cfg.fire.burn_steps
    aff = rng.random((k,) + belief.shape) < belief[None]
    state = np.where(aff, rng.integers(1, burn + 1, size=aff.shape), 0).astype(np.int16)
    steps = int(round(minutes * 60 / (cfg.fire.fire_step_ticks * cfg.dt_s)))
    for _ in range(steps):
        src = (state > 0).astype(float)
        log_not = np.zeros(state.shape)
        for (dr, dc), qd in q.items():
            log_not += np.log1p(-qd * _shift3(src, dr, dc))
        ignite = (state == 0) & (rng.random(state.shape) < 1.0 - np.exp(log_not))
        was = state > 0
        state[was] -= 1
        state[was & (state == 0)] = -1
        state[ignite] = burn
    return (state != 0).mean(axis=0)


def forecast_scores(fc: np.ndarray, truth: np.ndarray) -> dict:
    """Brier score on the relevant set (forecast >= 0.01 or truly affected), the share of
    truly affected cells inside the P >= 0.1 envelope, and forecast/true area."""
    rel = (fc >= 0.01) | truth
    y = truth.astype(float)
    brier = float(np.mean((fc[rel] - y[rel]) ** 2)) if rel.any() else float("nan")
    cover = float((fc[truth] >= 0.1).mean()) if truth.any() else float("nan")
    ratio = float(fc.sum() / truth.sum()) if truth.any() else float("nan")
    return {"brier": brier, "cover10": cover, "area_ratio": ratio}


__all__ = ["ResponderBrief", "compass_from_rowcol", "forecast_ensemble", "forecast_meanfield",
           "forecast_scores", "responder_brief"]
