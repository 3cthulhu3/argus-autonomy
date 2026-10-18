"""Two-model sequential test: is there a spreading fire, or only things that stay hot?

H1: the world contains a fire that spreads (plus background decoys).
H0: nothing spreads; whatever the satellite flagged is a warm or hot decoy.

Each hypothesis runs its own categorical filter.  Before every update, each
filter scores the incoming footprint with its predictive likelihood; the sum
of the differences is the posterior log-odds of H1 (a Bayesian SPRT
statistic, Wald 1945).  Persistence is explained by both models; *growth*
and the *absence of predicted growth* are what move the statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .config import (
    EMPTY,
    FIRE,
    HOT,
    K_CLASSES,
    WARM,
    DecoyConfig,
    MissionConfig,
    SPRTConfig,
)
from .mixture import MixtureFilter, object_priors
from .perception import CategoricalFilter, class_theta, spread_kernel

ALERT, CLEAR, UNDECIDED = "ALERT", "CLEAR", "UNDECIDED"


def hypothesis_priors(sat: np.ndarray, background: float, sprt: SPRTConfig,
                      decoys: DecoyConfig, known: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell class priors under H1 and H0 from the satellite map.

    Under H1 the satellite probability is fire mass.  Under H0 the *excess*
    over the background is decoy mass (split by omega); the background itself
    is assigned to the hot class, whose sensor response equals fire's, so
    flying over untouched ground is evidence for neither hypothesis.
    """
    omega = sprt.omega_for(decoys)
    excess = np.clip(sat - background, 0.0, None)

    phi = sprt.phi
    h1 = np.zeros((K_CLASSES,) + sat.shape)
    h1[FIRE] = np.minimum(sat, background) + phi * excess
    h1[WARM] = sprt.rho_warm + (1 - phi) * omega * excess
    h1[HOT] = sprt.rho_hot + (1 - phi) * (1 - omega) * excess
    h1[EMPTY] = 1.0 - h1[1:].sum(axis=0)

    h0 = np.zeros_like(h1)
    h0[WARM] = sprt.rho_warm + omega * excess
    h0[HOT] = sprt.rho_hot + (1 - omega) * excess + np.minimum(sat, background)
    h0[EMPTY] = 1.0 - h0[1:].sum(axis=0)
    if known is not None:
        h1, h0 = _oracle(h1, h0, sat, background, known)
    return h1, h0


def _oracle(h1, h0, sat, background, known, certainty: float = 0.999):
    """Oracle priors: every decoy's location and class is known to both models.

    Decoy cells are their class; elsewhere there are no decoys, so H0 has
    nothing but the background to explain the cue with.  A lower bound on
    decision delay, not a method.
    """
    h1, h0 = h1.copy(), h0.copy()
    h1[WARM] = h1[HOT] = 0.0
    h0[WARM] = 0.0
    h0[HOT] = np.minimum(sat, background)
    for h in (h1, h0):
        h[EMPTY] = 1.0 - h[1:].sum(axis=0)
    h0[EMPTY] = 1.0 - h0[HOT]      # the cue's excess is empty ground under H0
    for k in (WARM, HOT):
        m = known == k
        for h in (h1, h0):
            h[:, m] = 0.0
            h[k, m] = certainty
            h[EMPTY, m] = 1.0 - certainty
    return h1, h0


@dataclass
class MultiModelTest:
    h1: CategoricalFilter | MixtureFilter
    h0: CategoricalFilter
    llr: float              # log P(H1 | data) / P(H0 | data)
    alert: float
    clear: float

    @classmethod
    def build(cls, sat: np.ndarray, cfg: MissionConfig, decoys: DecoyConfig, sprt: SPRTConfig,
              wind_dir: float, known_decoys: np.ndarray | None = None) -> MultiModelTest:
        theta = class_theta(cfg.sensor, decoys)
        p1, p0 = hypothesis_priors(sat, cfg.prior.p_background, sprt, decoys, known_decoys)
        q = spread_kernel(cfg.model, wind_dir, cfg.fire.wind_gain)
        if sprt.h1_model == "object":
            if known_decoys is not None:
                raise ValueError("the object-level H1 has no oracle variant")
            region = sat >= cfg.prior.p_ring
            comps, logw = object_priors(p0, sat, cfg.prior.p_background, region,
                                        sprt.obj_radius, sprt.obj_step, sprt.obj_p_blob)
            h1 = MixtureFilter(comps, logw, theta, q, cfg.fire.burn_steps)
        elif sprt.h1_model == "cell":
            h1 = CategoricalFilter(p1, theta, q, cfg.fire.burn_steps)
        else:
            raise ValueError(f"unknown h1_model {sprt.h1_model!r}")
        return cls(
            h1=h1,
            h0=CategoricalFilter(p0, theta, None),
            llr=float(np.log(sprt.prior_h1 / (1 - sprt.prior_h1))),
            alert=sprt.alert, clear=sprt.clear,
        )

    @property
    def p_h1(self) -> float:
        return float(expit(self.llr))   # 1 / (1 + e^-llr), without overflow for very negative llr

    def predict(self) -> None:
        self.h1.predict()
        self.h0.predict()

    def update(self, fp: tuple[slice, slice], z: np.ndarray) -> float:
        """Score the observation under both models, then condition both on it."""
        self.llr += self.h1.predictive_loglik(fp, z) - self.h0.predictive_loglik(fp, z)
        self.h1.update(fp, z)
        self.h0.update(fp, z)
        return self.llr

    def update_close(self, fp: tuple[slice, slice], y: np.ndarray, close_lik: np.ndarray) -> float:
        self.llr += (self.h1.predictive_loglik_close(fp, y, close_lik)
                     - self.h0.predictive_loglik_close(fp, y, close_lik))
        self.h1.update_close(fp, y, close_lik)
        self.h0.update_close(fp, y, close_lik)
        return self.llr

    def decision(self) -> str:
        if self.llr >= self.alert:
            return ALERT
        if self.llr <= self.clear:
            return CLEAR
        return UNDECIDED
