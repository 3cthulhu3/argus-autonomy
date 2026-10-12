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
