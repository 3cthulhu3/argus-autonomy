"""Coverage baselines and the information-rate planner.

Every planner implements ``plan(ctx) -> waypoint``; the mission loop handles
routing (geodesic, geofence-aware) and the safety monitor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from .config import SensorConfig
from .perception import mutual_information


@dataclass
class PlanContext:
    pos: tuple[int, int]
    belief: np.ndarray        # P(affected) per cell
    dist: np.ndarray          # geodesic distance from pos, in cells
    feasible: np.ndarray      # waypoints the safety monitor would accept
    speed: float
    t: int


class Planner:
    name = "planner"

    def plan(self, ctx: PlanContext) -> tuple[int, int]:  # pragma: no cover - interface
        raise NotImplementedError


def serpentine(box: tuple[int, int, int, int], radius: int, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Boustrophedon lane endpoints covering box=(r0, c0, r1, c1) with footprint `radius`."""
    r0, c0, r1, c1 = box
    spacing = 2 * radius + 1
    rows = list(range(r0 + radius, r1, spacing))
    if not rows or rows[-1] < r1 - 1 - radius:
        rows.append(min(r1 - 1, (rows[-1] + spacing) if rows else r0 + radius))
    wps = []
    for i, r in enumerate(rows):
        ends = [(r, c0 + radius), (r, c1 - 1 - radius)]
        wps += ends if i % 2 == 0 else ends[::-1]
    return [w for w in wps if allowed[w]]


class Lawnmower(Planner):
    """Systematic sweep of a box; ignores the belief entirely."""

    name = "lawnmower"

    def __init__(self, box, radius: int, allowed: np.ndarray, start=None):
        wps = serpentine(box, radius, allowed)
        if start is not None and wps:
            # begin from the lane end closest to the launch point
            d_first = np.hypot(*np.subtract(wps[0], start))
            d_last = np.hypot(*np.subtract(wps[-1], start))
            if d_last < d_first:
                wps = wps[::-1]
        self.wps = wps
        self.i = 0

    def plan(self, ctx: PlanContext) -> tuple[int, int]:
        for _ in range(2 * len(self.wps) + 1):
            if self.i >= len(self.wps):        # sweep finished: run it backwards
                self.wps = self.wps[::-1]
                self.i = 1 if len(self.wps) > 1 else 0
            if not self.wps:
                break
            wp = self.wps[self.i]
            if wp != ctx.pos and np.isfinite(ctx.dist[wp]):
                return wp
            self.i += 1
        return ctx.pos


def full_map_lawnmower(n: int, radius: int, allowed: np.ndarray, base) -> Lawnmower:
    p = Lawnmower((0, 0, n, n), radius, allowed, start=base)
    p.name = "lawnmower"
    return p


def prior_lawnmower(prior: np.ndarray, p_min: float, radius: int, allowed: np.ndarray, base) -> Lawnmower:
    """Sweep only the bounding box of the satellite hotspot + ring."""
    rr, cc = np.nonzero(prior >= p_min)
    n = prior.shape[0]
    if rr.size == 0:
        box = (0, 0, n, n)
    else:
        box = (max(rr.min() - radius, 0), max(cc.min() - radius, 0),
               min(rr.max() + radius + 1, n), min(cc.max() + radius + 1, n))
    p = Lawnmower(box, radius, allowed, start=base)
    p.name = "prior-lawnmower"
    return p


class InfoRate(Planner):
    """Greedy information-rate planner.

    For every candidate footprint centre c, the expected information gain of
    one observation is G(c) = sum over the footprint of I(X_i; Z_i).  The
    planner flies to argmax G(c) / (travel_ticks(c) + tau), restricted to the
    waypoints the safety monitor marks as feasible.
    """

    name = "info-rate"

    def __init__(self, sensor: SensorConfig, tau: float = 1.0):
        self.sensor = sensor
        self.tau = tau

    def gain_map(self, belief: np.ndarray) -> np.ndarray:
        mi = mutual_information(belief, self.sensor)
        k = 2 * self.sensor.radius + 1
        return uniform_filter(mi, size=k, mode="constant") * k * k

    def plan(self, ctx: PlanContext) -> tuple[int, int]:
        g = self.gain_map(ctx.belief)
        travel = np.ceil(ctx.dist / ctx.speed)
        score = np.where(ctx.feasible, g / (travel + self.tau), -np.inf)
        best = np.unravel_index(int(np.argmax(score)), score.shape)
        if not np.isfinite(score[best]):
            return ctx.pos
        return (int(best[0]), int(best[1]))


# --------------------------------------------------------------------------- v1.1
def _kl_bern(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = np.clip(p, 1e-9, 1 - 1e-9), np.clip(q, 1e-9, 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _kl_cat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = np.clip(p, 1e-9, 1.0), np.clip(q, 1e-9, 1.0)
    return np.sum(p * np.log(p / q), axis=0)


def _window_sum(a: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return uniform_filter(a, size=k, mode="constant") * k * k


class Chernoff:
    """Active hypothesis testing: maximise the expected drift of the log-odds per tick.

    Given H, cells are independent, so the expected log-likelihood-ratio
    increment of a look is additive over its footprint:

        D(c) = P1 sum_i KL(p_i^1 || p_i^0) + P0 sum_i KL(p_i^0 || p_i^1)

    with p_i^k the predictive distribution of cell i's reading under H_k
    (Chernoff 1959; Naghshvar & Javidi 2013).  A wide look costs its travel
    time; a close look costs travel plus hover time.  When no action is
    expected to move the statistic by at least ``min_drift`` nats, the
    hypotheses do not yet disagree anywhere the drone can reach and the
    planner falls back to information-rate search for heat.  min_drift = 2
    was chosen on seeds disjoint from the benchmark
    (``experiments/screen_min_drift.py``).
    """

    name = "chernoff"

    def __init__(self, sensor: SensorConfig, close_lik: np.ndarray | None = None,
                 close_radius: int = 1, t_close: int = 2, tau: float = 1.0, min_drift: float = 2.0):
        self.sensor = sensor
        self.close_lik = None if close_lik is None else np.asarray(close_lik, float)
        self.close_radius, self.t_close = close_radius, t_close
        self.tau, self.min_drift = tau, min_drift
        self.fallback = InfoRate(sensor, tau)
        self.last_mode = "wide"

    def drift_maps(self, test) -> tuple[np.ndarray, np.ndarray | None]:
        p1h = test.p_h1
        p1, p0 = test.h1.p_hit(), test.h0.p_hit()
        wide = _window_sum(p1h * _kl_bern(p1, p0) + (1 - p1h) * _kl_bern(p0, p1), self.sensor.radius)
        close = None
        if self.close_lik is not None:
            q1 = np.tensordot(self.close_lik.T, test.h1.probs, axes=1)
            q0 = np.tensordot(self.close_lik.T, test.h0.probs, axes=1)
            close = _window_sum(p1h * _kl_cat(q1, q0) + (1 - p1h) * _kl_cat(q0, q1), self.close_radius)
        return wide, close

    def plan(self, ctx: PlanContext, test, feasible_close: np.ndarray | None = None
             ) -> tuple[tuple[int, int], str]:
        wide, close = self.drift_maps(test)
        travel = np.ceil(ctx.dist / ctx.speed)
        best_val, best, mode = -np.inf, ctx.pos, "wide"
        cands = [("wide", wide, ctx.feasible, 0)]
        if close is not None and feasible_close is not None:
            cands.append(("close", close, feasible_close, self.t_close))
        for m, d, feas, extra in cands:
            ok = feas & (d >= self.min_drift)
            if not ok.any():
                continue
            score = np.where(ok, d / (travel + self.tau + extra), -np.inf)
            i = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[i] > best_val:
                best_val, best, mode = float(score[i]), (int(i[0]), int(i[1])), m
        if not np.isfinite(best_val):
            self.last_mode = "search"
            return self.fallback.plan(ctx), "wide"
        self.last_mode = mode
        return best, mode
