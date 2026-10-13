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
from .perception import CategoricalFilter, class_theta, spread_kernel

ALERT, CLEAR, UNDECIDED = "ALERT", "CLEAR", "UNDECIDED"


def hypothesis_priors(sat: np.ndarray, background: float, sprt: SPRTConfig,
                      decoys: DecoyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell class priors under H1 and H0 from the satellite map.

    Under H1 the satellite probability is fire mass.  Under H0 the *excess*
    over the background is decoy mass (split by omega); the background itself
    is assigned to the hot class, whose sensor response equals fire's, so
    flying over untouched ground is evidence for neither hypothesis.
    """
    omega = sprt.omega_for(decoys)
    excess = np.clip(sat - background, 0.0, None)

    h1 = np.zeros((K_CLASSES,) + sat.shape)
    h1[FIRE], h1[WARM], h1[HOT] = sat, sprt.rho_warm, sprt.rho_hot
    h1[EMPTY] = 1.0 - h1[1:].sum(axis=0)

    h0 = np.zeros_like(h1)
    h0[WARM] = sprt.rho_warm + omega * excess
    h0[HOT] = sprt.rho_hot + (1 - omega) * excess + np.minimum(sat, background)
    h0[EMPTY] = 1.0 - h0[1:].sum(axis=0)
    return h1, h0


@dataclass
class MultiModelTest:
    h1: CategoricalFilter
    h0: CategoricalFilter
    llr: float              # log P(H1 | data) / P(H0 | data)
    alert: float
    clear: float

    @classmethod
    def build(cls, sat: np.ndarray, cfg: MissionConfig, decoys: DecoyConfig, sprt: SPRTConfig,
              wind_dir: float) -> MultiModelTest:
        theta = class_theta(cfg.sensor, decoys)
        p1, p0 = hypothesis_priors(sat, cfg.prior.p_background, sprt, decoys)
        q = spread_kernel(cfg.model, wind_dir, cfg.fire.wind_gain)
        return cls(
            h1=CategoricalFilter(p1, theta, q, cfg.fire.burn_steps),
            h0=CategoricalFilter(p0, theta, None),
            llr=float(np.log(sprt.prior_h1 / (1 - sprt.prior_h1))),
            alert=sprt.alert, clear=sprt.clear,
        )

    @property
    def p_h1(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.llr)))

    def predict(self) -> None:
        self.h1.predict()
        self.h0.predict()

    def update(self, fp: tuple[slice, slice], z: np.ndarray) -> float:
        """Score the observation under both models, then condition both on it."""
        self.llr += self.h1.predictive_loglik(fp, z) - self.h0.predictive_loglik(fp, z)
        self.h1.update(fp, z)
        self.h0.update(fp, z)
        return self.llr

    def decision(self) -> str:
        if self.llr >= self.alert:
            return ALERT
        if self.llr <= self.clear:
            return CLEAR
        return UNDECIDED
