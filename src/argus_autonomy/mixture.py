"""v1.3: object-level fire hypothesis as a mixture of location hypotheses.

The v1.1 fire hypothesis H1 puts independent fire mass on every cell of the
satellite cue. A cold look at any cue cell is then evidence against a fire
*anywhere*, so a cue seen cold is dismissed even when the fire lies just
outside it, in the ring (the coverage misses of v1.1-v1.2).

Here H1 is "H0 plus one fire object at an unknown location":

    p(z | H1, past) = sum_j  w_j  p(z | H1_j, past),

where component j is the H0 world (decoys explain the cue) with a fire blob
of radius r around centre c_j, and the centres tile the cue and its ring.
Prior weights follow the satellite map's excess over its background, so the
ring keeps the mass the satellite model gives it. Outside blob j, component
j predicts exactly what H0 predicts, so a cold look away from every blob is
evidence for neither hypothesis, and cold looks at the cue move weight to
ring components instead of lowering H1 as a whole: the log-odds cannot fall
below log(weight of the unseen ring components) from cold cue looks alone.

The class exposes the same interface as ``CategoricalFilter`` (``probs``,
``prob``, ``p_hit``, ``predictive_loglik``, ``update``, the close-look
variants and ``predict``), so planner, decision rule and safety monitor are
unchanged; only the fire hypothesis differs. Components are vectorised:
state L has shape (J, active classes, n, n).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.special import logsumexp

from .config import EMPTY, FIRE, K_CLASSES
from .perception import EPS, LOG_FLOOR

ACTIVE = [k for k in range(1, K_CLASSES)]          # fire, warm, hot: all active under H1_j
PRUNE_LOG_WEIGHT = np.log(1e-9)                    # components below this weight are dropped


def blob_centres(region: np.ndarray, step: int) -> list[tuple[int, int]]:
    """Lattice points (spacing ``step``, offset step // 2) that fall inside ``region``."""
    n, m = region.shape
    o = step // 2
    return [(r, c) for r in range(o, n, step) for c in range(o, m, step) if region[r, c]]


def disc(n: int, centre: tuple[int, int], radius: float) -> np.ndarray:
    rr, cc = np.indices((n, n))
    return (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2 <= radius ** 2


def object_priors(h0_prior: np.ndarray, sat: np.ndarray, background: float, region: np.ndarray,
                  radius: float, step: int, p_blob: float) -> tuple[np.ndarray, np.ndarray]:
    """Component priors (J, K_CLASSES, n, n) and normalised prior log-weights (J,).

    Component j = the H0 prior with, inside blob j, fire probability p_blob and
    the H0 classes scaled by (1 - p_blob). Weight_j proportional to the
    satellite excess over background summed over blob j.
    """
    n = sat.shape[0]
    excess = np.clip(sat - background, 0.0, None)
    centres = blob_centres(region, step)
    if not centres and region.any():
        # cue too small or too far off the lattice: widen it by one lattice step
        centres = blob_centres(binary_dilation(region, iterations=step), step)
    if not centres:
        # no cue at all: a fire anywhere, uniform location prior (equal excess everywhere)
        centres = blob_centres(np.ones_like(region, dtype=bool), step)
    priors, logw = [], []
    for c in centres:
        b = disc(n, c, radius)
        p = h0_prior.copy()
        p[:, b] *= (1.0 - p_blob)
        p[FIRE, b] = p_blob
        priors.append(p)
        logw.append(np.log(max(float(excess[b].sum()), 1e-12)))
    logw = np.array(logw)
    return np.stack(priors), logw - logsumexp(logw)


def _shift3(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``world.shift`` over the last two axes of a stack, zero fill."""
    out = np.zeros_like(a)
    n, m = a.shape[-2:]
    rs, cs = slice(max(dr, 0), n + min(dr, 0)), slice(max(dc, 0), m + min(dc, 0))
    rd, cd = slice(max(-dr, 0), n + min(-dr, 0)), slice(max(-dc, 0), m + min(-dc, 0))
    out[..., rs, cs] = a[..., rd, cd]
    return out


class MixtureFilter:
    """Mixture of categorical filters, one per fire-location hypothesis.

    Each component is updated exactly as ``CategoricalFilter`` (same floor),
    and the weights by Bayes on each component's predictive likelihood.
    """

    def __init__(self, priors: np.ndarray, logw: np.ndarray, theta: np.ndarray, q: dict,
                 burn_steps: int = 8, gate: float | None = 0.5):
        assert priors.ndim == 4 and priors.shape[1] == K_CLASSES
        self.theta = np.asarray(theta, float)
        self.q, self.gate = q, gate
        self.cls = ACTIVE
        p0 = np.maximum(priors[:, EMPTY], EPS)
        self.L = np.stack([np.log(np.maximum(priors[:, k], EPS)) - np.log(p0) for k in self.cls], axis=1)
        self.logw = np.asarray(logw, float).copy()
        self.history = [np.zeros((priors.shape[0],) + priors.shape[2:])] * burn_steps
        th0 = self.theta[EMPTY]
        self.l_hit = np.array([np.log(self.theta[k] / th0) for k in self.cls])
        self.l_miss = np.array([np.log((1 - self.theta[k]) / (1 - th0)) for k in self.cls])
        self._marg = None

    # ------------------------------------------------------------------ state
    @property
    def n_components(self) -> int:
        return self.L.shape[0]

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.logw)

    def _comp_probs(self, idx=(Ellipsis,)) -> np.ndarray:
        """Per-component class probabilities, shape (J, K_CLASSES, ...) over ``idx`` (cells)."""
        Lk = self.L[(slice(None), slice(None)) + idx]
        e = np.exp(Lk)
        denom = 1.0 + e.sum(axis=1)
        out = np.empty((Lk.shape[0], K_CLASSES) + Lk.shape[2:])
        out[:, EMPTY] = 1.0 / denom
        out[:, 1:] = e / denom[:, None]
        return out

    @property
    def probs(self) -> np.ndarray:
        """Marginal class probabilities, sum_j w_j b_j."""
        if self._marg is None:
            self._marg = np.tensordot(self.weights, self._comp_probs(), axes=1)
        return self._marg

    def prob(self, k: int) -> np.ndarray:
        return self.probs[k]

    def p_hit(self, fp: tuple[slice, slice] | None = None) -> np.ndarray:
        """Marginal predictive P(z=1) per cell (optionally over a footprint)."""
        b = self.probs if fp is None else self.probs[(slice(None),) + fp]
        return np.tensordot(self.theta, b, axes=1)

    # ------------------------------------------------------------------ Bayes
    def _comp_loglik(self, fp, z) -> np.ndarray:
        b = self._comp_probs(fp)                                   # (J, K, h, w)
        ph = np.clip(np.tensordot(b, self.theta, axes=([1], [0])), EPS, 1 - EPS)   # (J, h, w)
        return np.where(z[None], np.log(ph), np.log1p(-ph)).sum(axis=(1, 2))

    def predictive_loglik(self, fp: tuple[slice, slice], z: np.ndarray) -> float:
        return float(logsumexp(self.logw + self._comp_loglik(fp, z)))

    def _reweight(self, ll: np.ndarray) -> None:
        lw = self.logw + ll
        self.logw = lw - logsumexp(lw)
        keep = self.logw > PRUNE_LOG_WEIGHT
        if not keep.all() and keep.any():
            self.L, self.logw = self.L[keep], self.logw[keep]
            self.history = [h[keep] for h in self.history]
            self.logw -= logsumexp(self.logw)

    def _floor_update(self, idx, lg) -> None:
        full = np.concatenate([np.zeros((lg.shape[0], 1) + lg.shape[2:]), lg], axis=1)
        logp = np.maximum(full - logsumexp(full, axis=1, keepdims=True), LOG_FLOOR)
        logp -= logsumexp(logp, axis=1, keepdims=True)
        self.L[idx] = logp[:, 1:] - logp[:, :1]

    def update(self, fp: tuple[slice, slice], z: np.ndarray) -> None:
        ll = self._comp_loglik(fp, z)
        idx = (slice(None), slice(None)) + fp
        lg = self.L[idx] + np.where(z[None, None], self.l_hit[None, :, None, None], self.l_miss[None, :, None, None])
        self._floor_update(idx, lg)
        self._reweight(ll)
        self._marg = None

    # ------------------------------------------------------------------ close look
    def _comp_loglik_close(self, fp, y, close_lik) -> np.ndarray:
        b = self._comp_probs(fp)                                   # (J, K, h, w)
        q = np.tensordot(b, close_lik, axes=([1], [0]))            # (J, h, w, n_out)
        py = np.take_along_axis(q, y[None, ..., None].astype(np.intp), axis=-1)[..., 0]
        return np.log(np.clip(py, EPS, 1.0)).sum(axis=(1, 2))

    def predictive_loglik_close(self, fp, y: np.ndarray, close_lik: np.ndarray) -> float:
        return float(logsumexp(self.logw + self._comp_loglik_close(fp, y, close_lik)))

    def update_close(self, fp, y: np.ndarray, close_lik: np.ndarray) -> None:
        ll = self._comp_loglik_close(fp, y, close_lik)
        idx = (slice(None), slice(None)) + fp
        lcl = np.log(close_lik)[:, y.astype(np.intp)]              # (K, h, w)
        lg = self.L[idx] + (lcl[self.cls] - lcl[EMPTY][None])[None]
        self._floor_update(idx, lg)
        self._reweight(ll)
        self._marg = None

    # ------------------------------------------------------------------ dynamics
    def predict(self) -> None:
        """Gated mean-field spread of each component's fire mass, as CategoricalFilter."""
        b = self._comp_probs()
        bf = b[:, FIRE].copy()
        src = np.clip(bf - self.history[0], 0.0, 1.0)
        if self.gate is not None:
            src = src * (bf >= self.gate)
        self.history = self.history[1:] + [bf]
        log_not = np.zeros_like(src)
        for (dr, dc), qd in self.q.items():
            log_not += np.log1p(-qd * _shift3(src, dr, dc))
        moved = b[:, EMPTY] * (1.0 - np.exp(log_not))
        b[:, FIRE] = bf + moved
        b[:, EMPTY] = b[:, EMPTY] - moved
        p0 = np.log(np.maximum(b[:, EMPTY], EPS))
        for i, k in enumerate(self.cls):
            self.L[:, i] = np.log(np.maximum(b[:, k], EPS)) - p0
        self._marg = None
