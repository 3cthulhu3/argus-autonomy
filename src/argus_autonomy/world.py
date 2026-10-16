"""Ground-truth wildfire: a stochastic cellular automaton with wind and fuel.

Cell states: 0 = unburnt, 1..burn_steps = burning (countdown), -1 = burnt out.
"Affected" (burning or burnt) is the quantity the drone has to map: it is
monotone in time, which is what makes the estimator's prediction step simple.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import EMPTY, FIRE, HOT, WARM, FireConfig

# 8-neighbourhood offsets (dr, dc)
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shift(a: np.ndarray, dr: int, dc: int, fill=0) -> np.ndarray:
    """Shift array so that out[r, c] = a[r - dr, c - dc]; no wrap-around."""
    out = np.full_like(a, fill)
    n, m = a.shape
    rs = slice(max(dr, 0), n + min(dr, 0))
    cs = slice(max(dc, 0), m + min(dc, 0))
    rd = slice(max(-dr, 0), n + min(-dr, 0))
    cd = slice(max(-dc, 0), m + min(-dc, 0))
    out[rs, cs] = a[rd, cd]
    return out


def spread_factors(wind_dir: float, wind_gain: float) -> dict[tuple[int, int], float]:
    """Multiplicative spread factor for fire moving in direction (dr, dc).

    wind_dir is the direction the wind blows *towards*, in radians, measured
    in (row, col) coordinates: 0 = +col (east on screen), pi/2 = +row (down).
    Diagonal neighbours are 1/sqrt(2) as likely (longer distance).
    """
    w = np.array([np.sin(wind_dir), np.cos(wind_dir)])
    out = {}
    for dr, dc in NEIGHBOURS:
        d = np.array([dr, dc], dtype=float)
        norm = np.linalg.norm(d)
        cos = float(d @ w / norm)
        out[(dr, dc)] = float(np.exp(wind_gain * cos) / norm)
    return out


def random_fuel(n: int, rng: np.random.Generator, cfg: FireConfig) -> np.ndarray:
    """Spatially correlated fuel load in [fuel_min, fuel_max]."""
    f = gaussian_filter(rng.standard_normal((n, n)), sigma=4)
    f = (f - f.min()) / (np.ptp(f) + 1e-12)
    return cfg.fuel_min + (cfg.fuel_max - cfg.fuel_min) * f


class Fire:
    def __init__(self, n: int, ignition: tuple[int, int], wind_dir: float,
                 cfg: FireConfig, rng: np.random.Generator):
        self.n = n
        self.cfg = cfg
        self.rng = rng
        self.wind_dir = wind_dir
        self.fuel = random_fuel(n, rng, cfg)
        self.factors = spread_factors(wind_dir, cfg.wind_gain)
        self.state = np.zeros((n, n), dtype=np.int16)
        self.state[ignition] = cfg.burn_steps
        self.steps = 0

    @property
    def burning(self) -> np.ndarray:
        return self.state > 0

    @property
    def affected(self) -> np.ndarray:
        return self.state != 0

    def step(self) -> None:
        burning = self.burning.astype(float)
        log_not = np.zeros((self.n, self.n))
        for (dr, dc), f in self.factors.items():
            # neighbour at (r - dr, c - dc) spreads in direction (dr, dc) into (r, c)
            q = np.clip(self.cfg.p_spread * f * self.fuel, 0.0, 0.999)
            src = shift(burning, dr, dc)
            log_not += np.log1p(-q * src)
        p_ign = 1.0 - np.exp(log_not)
        ignite = (self.state == 0) & (self.rng.random((self.n, self.n)) < p_ign)
        s = self.state
        was_burning = s > 0
        s[was_burning] -= 1
        s[was_burning & (s == 0)] = -1   # burnt out
        s[ignite] = self.cfg.burn_steps
        self.steps += 1

    def advance(self, k: int) -> None:
        for _ in range(k):
            self.step()


# --------------------------------------------------------------------------- v1.1
def class_map(affected: np.ndarray, decoy_class: np.ndarray) -> np.ndarray:
    """Per-cell latent class: fire wins over a decoy it has burnt into.

    `decoy_class` holds EMPTY / WARM / HOT; the result also uses FIRE.
    """
    out = decoy_class.astype(np.int8).copy()
    out[affected] = FIRE
    return out


def place_decoys(n: int, allowed: np.ndarray, keep_out: np.ndarray, n_warm: int, n_hot: int,
                 max_side: int, rng: np.random.Generator) -> list[tuple[int, int, int, int, int]]:
    """Rectangles (r, c, h, w, cls) in allowed airspace, clear of `keep_out`,
    not touching each other.  `keep_out` is a boolean mask (already dilated
    by the caller to the required separation)."""
    taken = keep_out.copy()
    out = []
    for cls in [WARM] * n_warm + [HOT] * n_hot:
        for _ in range(10_000):
            h, w = (int(x) for x in rng.integers(1, max_side + 1, size=2))
            r = int(rng.integers(2, n - 2 - h))
            c = int(rng.integers(2, n - 2 - w))
            if not allowed[r:r + h, c:c + w].all() or taken[r - 1:r + h + 1, c - 1:c + w + 1].any():
                continue
            taken[r:r + h, c:c + w] = True
            out.append((r, c, h, w, cls))
            break
        else:  # pragma: no cover - only on absurdly crowded maps
            raise RuntimeError("could not place decoy")
    return out


def rasterise_decoys(n: int, rects) -> np.ndarray:
    m = np.full((n, n), EMPTY, dtype=np.int8)
    for r, c, h, w, cls in rects:
        m[r:r + h, c:c + w] = cls
    return m
