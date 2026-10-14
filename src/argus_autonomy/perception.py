"""Satellite hotspot prior, thermal sensor model and Bayesian occupancy filter."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .config import (
    EMPTY,
    FIRE,
    HOT,
    K_CLASSES,
    WARM,
    DecoyConfig,
    ModelConfig,
    PriorConfig,
    SensorConfig,
)
from .world import NEIGHBOURS, shift, spread_factors

EPS = 1e-6
LOG_FLOOR = -15.0 - float(np.log1p(np.exp(-15.0)))  # log sigma(-15)


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
def spread_kernel(model: ModelConfig, wind_dir: float, true_wind_gain: float) -> dict:
    """Per-direction ignition probability the *estimator* assumes."""
    gain = 0.0
    if model.wind_known:
        gain = true_wind_gain if model.wind_gain is None else model.wind_gain
    return {d: min(model.p_spread * f * model.fuel, 0.999)
            for d, f in spread_factors(wind_dir, gain).items()}


def mean_field_ignition(src: np.ndarray, q: dict) -> np.ndarray:
    """P(cell ignites this fire step) = 1 - prod_j (1 - q_j s_j), neighbours independent."""
    log_not = np.zeros_like(src)
    for (dr, dc), qd in q.items():
        log_not += np.log1p(-qd * shift(src, dr, dc))
    return 1.0 - np.exp(log_not)


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
        self.q = spread_kernel(model, wind_dir, true_wind_gain)

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
        b_new = b + (1 - b) * mean_field_ignition(src, self.q)
        b_new = np.clip(b_new, EPS, 1 - EPS)
        self.L = np.log(b_new / (1 - b_new))





# --------------------------------------------------------------------------- v1.1
def class_theta(sensor: SensorConfig, decoys: DecoyConfig) -> np.ndarray:
    """P(z=1 | class) for the wide thermal look, indexed by class."""
    th = np.empty(K_CLASSES)
    th[EMPTY], th[FIRE] = sensor.p_false_alarm, sensor.p_detect
    th[WARM], th[HOT] = decoys.theta_warm, decoys.theta_hot
    return th


def observe_wide(classes: np.ndarray, pos: tuple[int, int], theta: np.ndarray, radius: int,
                 rng: np.random.Generator) -> tuple[tuple[slice, slice], np.ndarray]:
    """Class-conditional binary detections over the square footprint.

    Consumes the RNG exactly like v1.0 ``observe`` does, so with no decoys
    the measurement stream is bit-identical to v1.0.
    """
    fp = footprint(pos, radius, classes.shape[0])
    p = theta[classes[fp]]
    z = rng.random(p.shape) < p
    return fp, z


def observe_close(classes: np.ndarray, pos: tuple[int, int], close_lik: np.ndarray, radius: int,
                  rng: np.random.Generator) -> tuple[tuple[slice, slice], np.ndarray]:
    """Close look: one categorical reading per cell, y in {0 cold, 1 warm, 2 hot}."""
    fp = footprint(pos, radius, classes.shape[0])
    cdf = np.cumsum(close_lik[classes[fp]], axis=-1)
    y = (rng.random(cdf.shape[:-1])[..., None] > cdf).sum(axis=-1)
    return fp, np.minimum(y, close_lik.shape[1] - 1).astype(np.int8)


class CategoricalFilter:
    """Per-cell categorical Bayes filter over {empty, fire, warm, hot}.

    Classes with zero prior mass everywhere are *inactive* and stay at exactly
    zero (the H0 model has no fire class).  The state is the log-odds of each
    active class against `empty`, clipped like v1.0, so that restricted to
    {empty, fire} with no decoys it reproduces ``OccupancyFilter``.

    ``predict`` moves mass empty -> fire through the same gated mean-field
    spread model as v1.0 (when ``q`` is given); decoy mass never moves.
    """

    def __init__(self, prior: np.ndarray, theta: np.ndarray, q: dict | None = None,
                 burn_steps: int = 8, gate: float | None = 0.5):
        assert prior.shape[0] == K_CLASSES
        mass = prior.reshape(K_CLASSES, -1).max(axis=1)
        self.cls = [k for k in range(1, K_CLASSES) if mass[k] > 0]
        if q is not None and FIRE not in self.cls:
            raise ValueError("a spreading filter needs an active fire class")
        self.theta = np.asarray(theta, float)
        self.q, self.gate = q, gate
        p0 = np.maximum(prior[EMPTY], EPS)
        self.L = np.stack([np.log(np.maximum(prior[k], EPS)) - np.log(p0) for k in self.cls])
        self.history = [np.zeros(prior.shape[1:])] * burn_steps
        th0 = self.theta[EMPTY]
        self.l_hit = np.array([np.log(self.theta[k] / th0) for k in self.cls])
        self.l_miss = np.array([np.log((1 - self.theta[k]) / (1 - th0)) for k in self.cls])

    # ------------------------------------------------------------------ state
    @property
    def probs(self) -> np.ndarray:
        e = np.exp(self.L)
        denom = 1.0 + e.sum(axis=0)
        out = np.zeros((K_CLASSES,) + self.L.shape[1:])
        out[EMPTY] = 1.0 / denom
        for i, k in enumerate(self.cls):
            out[k] = e[i] / denom
        return out

    def prob(self, k: int) -> np.ndarray:
        return self.probs[k]

    def p_hit(self, fp: tuple[slice, slice] | None = None) -> np.ndarray:
        """Predictive P(z=1) per cell (optionally only over a footprint)."""
        b = self.probs if fp is None else self.probs[(slice(None),) + fp]
        return np.tensordot(self.theta, b, axes=1)

    # ------------------------------------------------------------------ Bayes
    def predictive_loglik(self, fp: tuple[slice, slice], z: np.ndarray) -> float:
        """log p(z_fp | model, past), cells independent within the model."""
        ph = np.clip(self.p_hit(fp), EPS, 1 - EPS)
        return float(np.sum(np.where(z, np.log(ph), np.log1p(-ph))))

    def update(self, fp: tuple[slice, slice], z: np.ndarray) -> None:
        """Exact Bayes, then floor every class probability at sigma(-15).

        For two classes this is v1.0's log-odds clip at +-15.  Clipping each
        class's log-odds against `empty` separately would be wrong here: fire,
        warm and hot all climb together on a hit, saturate at the same cap
        and lose the ratio between them.
        """
        idx = (slice(None),) + fp
        lg = self.L[idx] + np.where(z, self.l_hit[:, None, None], self.l_miss[:, None, None])
        full = np.concatenate([np.zeros((1,) + lg.shape[1:]), lg])
        logp = np.maximum(full - np.logaddexp.reduce(full, axis=0), LOG_FLOOR)
        logp -= np.logaddexp.reduce(logp, axis=0)
        self.L[idx] = logp[1:] - logp[0]

    # ------------------------------------------------------------------ close look
    def predictive_close(self, fp: tuple[slice, slice], close_lik: np.ndarray) -> np.ndarray:
        """Predictive P(y) per cell over the footprint, shape (n_out, h, w)."""
        b = self.probs[(slice(None),) + fp]
        return np.tensordot(close_lik.T, b, axes=1)

    def predictive_loglik_close(self, fp, y: np.ndarray, close_lik: np.ndarray) -> float:
        q = self.predictive_close(fp, close_lik)
        py = np.take_along_axis(q, y[None].astype(np.intp), axis=0)[0]
        return float(np.sum(np.log(np.clip(py, EPS, 1.0))))

    def update_close(self, fp: tuple[slice, slice], y: np.ndarray, close_lik: np.ndarray) -> None:
        """Exact Bayes for a categorical reading; same probability floor as ``update``."""
        idx = (slice(None),) + fp
        ll = np.log(close_lik)[:, y.astype(np.intp)]            # (K, h, w): log p(y | class)
        lg = self.L[idx] + (ll[self.cls] - ll[EMPTY][None])
        full = np.concatenate([np.zeros((1,) + lg.shape[1:]), lg])
        logp = np.maximum(full - np.logaddexp.reduce(full, axis=0), LOG_FLOOR)
        logp -= np.logaddexp.reduce(logp, axis=0)
        self.L[idx] = logp[1:] - logp[0]

    def predict(self) -> None:
        if self.q is None:
            return
        b = self.probs
        bf = b[FIRE].copy()     # pre-predict belief; history must not alias b
        src = np.clip(bf - self.history[0], 0.0, 1.0)
        if self.gate is not None:
            src = src * (bf >= self.gate)
        self.history = self.history[1:] + [bf]
        moved = b[EMPTY] * mean_field_ignition(src, self.q)
        b[FIRE] = bf + moved
        b[EMPTY] = b[EMPTY] - moved
        p0 = np.log(np.maximum(b[EMPTY], EPS))
        for i, k in enumerate(self.cls):
            self.L[i] = np.log(np.maximum(b[k], EPS)) - p0


__all__ = [
    "NEIGHBOURS",
    "CategoricalFilter",
    "OccupancyFilter",
    "binary_entropy",
    "class_theta",
    "footprint",
    "mean_field_ignition",
    "mutual_information",
    "observe",
    "observe_close",
    "observe_wide",
    "satellite_prior",
    "spread_kernel",
]
