"""Satellite hotspot prior, thermal sensor model and Bayesian occupancy filter."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .config import ModelConfig, PriorConfig, SensorConfig
from .world import NEIGHBOURS, shift, spread_factors

EPS = 1e-6


# --------------------------------------------------------------------------- prior
def satellite_prior(affected_at_overpass: np.ndarray, cfg: PriorConfig,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Coarse, stale, mislocated hotspot product turned into a per-cell prior.

    Returns (prior probability map, boolean mask of flagged cells after the
    geolocation error), the latter only for plotting.
    """
    n = affected_at_overpass.shape[0]
    k = cfg.coarse_factor
    m = n // k
    frac = affected_at_overpass[: m * k, : m * k].reshape(m, k, m, k).mean(axis=(1, 3))
    flagged = frac >= cfg.detect_fraction
    ring = binary_dilation(flagged, structure=np.ones((3, 3), bool)) & ~flagged

    coarse = np.full((m, m), cfg.p_background)
    coarse[ring] = cfg.p_ring
    coarse[flagged] = cfg.p_hotspot
    prior = np.kron(coarse, np.ones((k, k)))
    hot = np.kron(flagged, np.ones((k, k), bool))

    dr, dc = np.rint(rng.normal(0.0, cfg.geoloc_sigma, size=2)).astype(int)
    prior = shift(prior, dr, dc, fill=cfg.p_background)
    hot = shift(hot, dr, dc, fill=False)
    return prior, hot


# --------------------------------------------------------------------------- sensor
def footprint(pos: tuple[int, int], radius: int, n: int) -> tuple[slice, slice]:
    r, c = pos
    return (slice(max(r - radius, 0), min(r + radius + 1, n)),
            slice(max(c - radius, 0), min(c + radius + 1, n)))


def observe(truth: np.ndarray, pos: tuple[int, int], cfg: SensorConfig,
            rng: np.random.Generator) -> tuple[tuple[slice, slice], np.ndarray]:
    """Nadir thermal camera: independent binary detections over a square footprint."""
    fp = footprint(pos, cfg.radius, truth.shape[0])
    t = truth[fp]
    p = np.where(t, cfg.p_detect, cfg.p_false_alarm)
    z = rng.random(t.shape) < p
    return fp, z


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mutual_information(p: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """I(X; Z) in bits for a binary cell with P(X=1)=p observed once by the sensor."""
    pz = p * cfg.p_detect + (1 - p) * cfg.p_false_alarm
    h_cond = p * binary_entropy(np.asarray(cfg.p_detect)) + \
        (1 - p) * binary_entropy(np.asarray(cfg.p_false_alarm))
    return binary_entropy(pz) - h_cond


# --------------------------------------------------------------------------- filter
class OccupancyFilter:
    """Per-cell Bayes filter over 'affected by fire', in log-odds.

    Update: exact Bayes for the independent-cell sensor model.
    Predict: mean-field propagation through the estimator's own spread model,
    b' = b + (1 - b) * (1 - prod_j (1 - q_j s_j)), where the spreading source
    s_j = b_j(now) - b_j(burn_steps ago) is the probability that cell j became
    affected recently enough to still be burning.

    Mean-field treats neighbours as independent.  Left alone, that makes any
    diffuse low-probability mass grow exponentially (each cell sees "fresh"
    neighbours); measured from the true initial state it over-predicts
    the burnt area 4.7x after 30 fire steps, and with a diffuse prior the whole
    map saturates.  We
    therefore only let cells the filter currently believes are affected
    (b_j >= ``gate``) act as sources.  This removes the blow-up at the price
    of under-predicting growth between observations; ``experiments/
    calibrate_prediction.py`` measures both effects.
    """

    def __init__(self, prior: np.ndarray, sensor: SensorConfig, model: ModelConfig,
                 wind_dir: float, true_wind_gain: float, burn_steps: int = 8,
                 gate: float | None = 0.5):
        self.gate = gate
        p = np.clip(prior, EPS, 1 - EPS)
        self.L = np.log(p / (1 - p))
        # beliefs at the last `burn_steps` predict calls; zeros = "everything is fresh"
        self.history = [np.zeros_like(p)] * burn_steps
        self.sensor = sensor
        self.l_hit = np.log(sensor.p_detect / sensor.p_false_alarm)
        self.l_miss = np.log((1 - sensor.p_detect) / (1 - sensor.p_false_alarm))
        gain = 0.0
        if model.wind_known:
            gain = true_wind_gain if model.wind_gain is None else model.wind_gain
        self.q = {d: min(model.p_spread * f * model.fuel, 0.999)
                  for d, f in spread_factors(wind_dir, gain).items()}

    @property
    def prob(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.L))

    def update(self, fp: tuple[slice, slice], z: np.ndarray) -> None:
        self.L[fp] += np.where(z, self.l_hit, self.l_miss)
        np.clip(self.L, -15, 15, out=self.L)

    def predict(self) -> None:
        b = self.prob
        src = np.clip(b - self.history[0], 0.0, 1.0)
        if self.gate is not None:
            src = src * (b >= self.gate)
        self.history = self.history[1:] + [b]
        log_not = np.zeros_like(b)
        for (dr, dc), q in self.q.items():
            log_not += np.log1p(-q * shift(src, dr, dc))
        b_new = b + (1 - b) * (1 - np.exp(log_not))
        b_new = np.clip(b_new, EPS, 1 - EPS)
        self.L = np.log(b_new / (1 - b_new))


__all__ = ["satellite_prior", "observe", "footprint", "binary_entropy",
           "mutual_information", "OccupancyFilter", "NEIGHBOURS"]
