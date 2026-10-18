"""v1.3: object-level fire hypothesis (mixture of location hypotheses)."""

import numpy as np
import pytest
from scipy.special import logsumexp

from argus_autonomy import NOMINAL
from argus_autonomy.config import EMPTY, FIRE, HOT, K_CLASSES, WARM, DecoyConfig, SPRTConfig
from argus_autonomy.decision import make_decoy_scenario, run_decision_mission
from argus_autonomy.hypothesis import MultiModelTest
from argus_autonomy.mixture import MixtureFilter, disc, object_priors
from argus_autonomy.perception import CategoricalFilter, class_theta, footprint, spread_kernel

N = 32
THETA = class_theta(NOMINAL.sensor, DecoyConfig())
Q = spread_kernel(NOMINAL.model, 0.7, NOMINAL.fire.wind_gain)
CLOSE = np.asarray(DecoyConfig().close_lik, float)


def _prior(rng):
    p = rng.dirichlet(np.ones(K_CLASSES), size=(N, N)).transpose(2, 0, 1)
    p[FIRE] *= 0.3
    p[EMPTY] = 1.0 - p[1:].sum(axis=0)
    return p


def test_single_component_mixture_equals_categorical_filter():
    """Regression invariant: J = 1 reproduces CategoricalFilter exactly, through
    wide updates, close looks and spread predictions."""
    rng = np.random.default_rng(0)
    p = _prior(rng)
    cat = CategoricalFilter(p, THETA, Q, 8)
    mix = MixtureFilter(p[None], np.zeros(1), THETA, Q, 8)
    for t in range(40):
        pos = tuple(int(x) for x in rng.integers(3, N - 3, 2))
        fp = footprint(pos, 3, N)
        if t % 7 == 3:
            fpc = footprint(pos, 1, N)
            y = rng.integers(0, 3, size=(3, 3))
            assert mix.predictive_loglik_close(fpc, y, CLOSE) == pytest.approx(cat.predictive_loglik_close(fpc, y, CLOSE), abs=1e-9)
            mix.update_close(fpc, y, CLOSE)
            cat.update_close(fpc, y, CLOSE)
        else:
            z = rng.random((7, 7)) < 0.3
            assert mix.predictive_loglik(fp, z) == pytest.approx(cat.predictive_loglik(fp, z), abs=1e-9)
            mix.update(fp, z)
            cat.update(fp, z)
        if t % 6 == 5:
            mix.predict()
            cat.predict()
        np.testing.assert_allclose(mix.probs, cat.probs, atol=1e-12)


def test_weights_stay_normalised_and_pruning_keeps_a_component():
    rng = np.random.default_rng(1)
    comps = np.stack([_prior(rng) for _ in range(5)])
    mix = MixtureFilter(comps, np.log(np.full(5, 0.2)), THETA, Q, 8)
    for _ in range(60):
        fp = footprint(tuple(int(x) for x in rng.integers(3, N - 3, 2)), 3, N)
        mix.update(fp, rng.random((7, 7)) < 0.5)
        assert logsumexp(mix.logw) == pytest.approx(0.0, abs=1e-9)
        assert mix.n_components >= 1
    np.testing.assert_allclose(mix.probs.sum(axis=0), 1.0, atol=1e-9)


def test_cold_looks_at_one_blob_cannot_push_h1_below_the_other_blobs_weight():
    """The property v1.3 exists for. Component B is H0 outside its blob, so cold looks
    away from blob B are scored identically by H0 and B; the log-odds of H1 against H0
    then cannot fall below log w_B, however many cold looks blob A receives."""
    base = np.zeros((K_CLASSES, N, N))
    base[WARM], base[HOT] = 0.01, 0.05
    base[EMPTY] = 1.0 - base[WARM] - base[HOT]
    sat = np.full((N, N), 0.002)
    sat[4:12, 4:12] = 0.25        # cue: blob A sits here
    sat[22:30, 22:30] = 0.02      # ring stand-in: blob B sits here
    region = sat >= 0.02
    comps, logw = object_priors(base, sat, 0.002, region, radius=3, step=8, p_blob=0.6)
    centres = [tuple(np.argwhere(c[FIRE] == 0.6).mean(axis=0).round().astype(int)) for c in comps]
    in_b = [i for i, c in enumerate(centres) if c[0] > 16]
    logw_b = float(logsumexp(logw[in_b]))
    mix = MixtureFilter(comps, logw, THETA, None, 8)
    h0 = CategoricalFilter(base, THETA, None)
    llr = 0.0
    for _ in range(30):
        for pos in ((6, 6), (6, 10), (10, 6), (10, 10)):
            fp = footprint(pos, 3, N)
            z = np.zeros((7, 7), bool)
            llr += mix.predictive_loglik(fp, z) - h0.predictive_loglik(fp, z)
            mix.update(fp, z)
            h0.update(fp, z)
    # Exact up to the probability floor: component B keeps fire mass of order EPS = 1e-6
    # outside its blob (H0 has none), worth at most ~1e-6 nats per cell look.
    cell_looks = 30 * 4 * 49
    assert llr >= logw_b - cell_looks * 1e-6
    assert llr < 0                       # but the cold looks did count against H1
    assert mix.weights[in_b].sum() > 0.99


def test_object_prior_weights_follow_the_satellite_excess():
    sc = make_decoy_scenario(NOMINAL, DecoyConfig(), "C", 5000)
    _, p0 = __import__("argus_autonomy.hypothesis", fromlist=["x"]).hypothesis_priors(
        sc.prior, NOMINAL.prior.p_background, SPRTConfig(), DecoyConfig())
    region = sc.prior >= NOMINAL.prior.p_ring
    comps, logw = object_priors(p0, sc.prior, NOMINAL.prior.p_background, region, 3, 4, 0.6)
    share = np.array([(sc.prior[c[FIRE] == 0.6] >= NOMINAL.prior.p_hotspot).mean() for c in comps])
    # A radius-3 disc rarely fits wholly inside an 8 x 8 pixel, so "in the cue" = at least 3/4 of it.
    in_cue, in_ring = share >= 0.75, share == 0.0
    assert in_cue.any() and in_ring.any()
    # Per cell, cue excess / ring excess = 0.248 / 0.018 = 13.8, so a blob 3/4 in the cue weighs
    # at least about 0.75 x 13.8 = 10 times a pure ring blob; 5 leaves room for edge effects.
    assert np.exp(logw[in_cue]).min() > 5 * np.exp(logw[in_ring]).max()
    assert logsumexp(logw) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("family", ["A", "B", "C"])
def test_object_h1_mission_runs_and_keeps_the_safety_invariants(family):
    r = run_decision_mission(NOMINAL, "M3", family, 5001,
                             sprt=SPRTConfig(steer_clear=-4.595, h1_model="object"))
    assert r.geofence_violations == 0 and r.returned_to_base
    t = MultiModelTest.build(make_decoy_scenario(NOMINAL, DecoyConfig(), family, 5001).prior, NOMINAL,
                             DecoyConfig(), SPRTConfig(h1_model="object"), 0.0)
    assert isinstance(t.h1, MixtureFilter) and t.h1.n_components > 1


def test_disc_is_a_disc():
    d = disc(9, (4, 4), 2)
    assert d.sum() == 13 and d[4, 4] and not d[0, 0]


def test_object_prior_falls_back_when_the_cue_misses_the_lattice_or_is_absent():
    base = np.zeros((K_CLASSES, N, N))
    base[EMPTY] = 1.0
    tiny = np.zeros((N, N), bool)
    tiny[5, 5] = True                        # off-lattice for step 4 (lattice at 2, 6, 10, ...)
    sat = np.where(tiny, 0.25, 0.002)
    comps, logw = object_priors(base, sat, 0.002, tiny, 3, 4, 0.6)
    assert len(comps) >= 1 and logsumexp(logw) == pytest.approx(0.0, abs=1e-12)
    none = np.zeros((N, N), bool)
    comps, logw = object_priors(base, np.full((N, N), 0.002), 0.002, none, 3, 4, 0.6)
    assert len(comps) == (N // 4) ** 2
    np.testing.assert_allclose(logw, logw[0])   # uniform
