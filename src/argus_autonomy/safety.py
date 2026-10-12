"""Geofence, geodesic navigation and the return-to-base (RTB) runtime monitor.

The monitor is deliberately simple and checkable: the vehicle may only occupy
cells inside the geofence, and at every tick it must be able to reach the base
within the remaining endurance minus a fixed reserve.  Planners propose
waypoints; the monitor has the final word.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SQRT2 = float(np.sqrt(2.0))


class Geofence:
    def __init__(self, n: int, nfz: tuple[int, int, int, int] | None):
        self.n = n
        self.allowed = np.ones((n, n), dtype=bool)
        if nfz is not None:
            r0, c0, r1, c1 = nfz
            self.allowed[r0:r1, c0:c1] = False
        self._graph = self._build_graph()

    def contains(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.n and 0 <= c < self.n and bool(self.allowed[r, c])

    def _build_graph(self):
        n = self.n
        idx = np.arange(n * n).reshape(n, n)
        rows, cols, w = [], [], []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            r0, r1 = max(0, -dr), n - max(0, dr)
            c0, c1 = max(0, -dc), n - max(0, dc)
            a = idx[r0:r1, c0:c1]
            b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = self.allowed[r0:r1, c0:c1] & self.allowed[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if dr and dc:
                # no corner cutting through the no-fly zone
                ok &= self.allowed[r0 + dr:r1 + dr, c0:c1] & self.allowed[r0:r1, c0 + dc:c1 + dc]
            cost = SQRT2 if (dr and dc) else 1.0
            rows += [a[ok], b[ok]]
            cols += [b[ok], a[ok]]
            w += [np.full(ok.sum(), cost)] * 2
        rows, cols, w = map(np.concatenate, (rows, cols, w))
        return coo_matrix((w, (rows, cols)), shape=(n * n, n * n)).tocsr()

    def distances_from(self, cell: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Geodesic distance (cells) from `cell` to every cell, plus predecessors."""
        src = cell[0] * self.n + cell[1]
        d, pred = dijkstra(self._graph, indices=src, return_predecessors=True)
        return d.reshape(self.n, self.n), pred

    def path(self, pred: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Cells from start (exclusive) to goal (inclusive), using predecessors from `start`."""
        n = self.n
        s, g = start[0] * n + start[1], goal[0] * n + goal[1]
        out = []
        cur = g
        while cur != s:
            if cur < 0:
                return []
            out.append((cur // n, cur % n))
            cur = pred[cur]
        return out[::-1]


def advance(pos: tuple[int, int], path: list[tuple[int, int]], speed: float,
            carry: float = 0.0) -> tuple[tuple[int, int], list, float]:
    """Move along `path` by at most `speed` (+ unused `carry`) cells of path length.

    Returns the new cell, the remaining path and the unused distance, which is
    carried into the next tick so that diagonal moves do not slow the vehicle.
    """
    budget = speed + carry
    cur = pos
    i = 0
    while i < len(path):
        nxt = path[i]
        step = SQRT2 if (nxt[0] != cur[0] and nxt[1] != cur[1]) else 1.0
        if step > budget + 1e-9:
            break
        budget -= step
        cur = nxt
        i += 1
    rest = path[i:]
    return cur, rest, (min(budget, SQRT2) if rest else 0.0)


class RTBMonitor:
    """Forces return-to-base when remaining endurance only just covers the way home."""

    def __init__(self, fence: Geofence, base: tuple[int, int], speed: float, reserve_ticks: int):
        self.fence = fence
        self.base = base
        self.speed = speed
        self.reserve = reserve_ticks
        self.home_dist, self.home_pred = fence.distances_from(base)

    def ticks_home(self, cell) -> float:
        return float(np.ceil(self.home_dist[cell] / self.speed))

    def must_return(self, cell, remaining_ticks: int) -> bool:
        # +1: after this tick's move we must still be able to make it
        return remaining_ticks - 1 - self.reserve <= self.ticks_home(cell) + 1

    def feasible_mask(self, dist_from_drone: np.ndarray, remaining_ticks: int) -> np.ndarray:
        """Waypoints that can be reached and still allow RTB with reserve."""
        t = np.ceil(dist_from_drone / self.speed) + np.ceil(self.home_dist / self.speed)
        return self.fence.allowed & np.isfinite(t) & (t <= remaining_ticks - self.reserve - 2)

    def path_home(self, cell) -> list[tuple[int, int]]:
        # predecessors are rooted at base; reverse the base->cell path
        p = self.fence.path(self.home_pred, self.base, cell)
        return (p[::-1][1:] + [self.base]) if p else []
