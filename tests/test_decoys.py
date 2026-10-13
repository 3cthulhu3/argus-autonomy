"""v1.1: categorical filter, two-model test, decoy scenarios, decision mission."""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from argus_autonomy import NOMINAL, run_mission
from argus_autonomy.config import (
    EMPTY,
    FIRE,
    HOT,
    K_CLASSES,
    WARM,
    DecoyConfig,
    SPRTConfig,
)
from argus_autonomy.decision import make_decoy_scenario, run_decision_mission
from argus_autonomy.hypothesis import MultiModelTest
from argus_autonomy.mission import make_scenario
from argus_autonomy.perception import (
    CategoricalFilter,
    OccupancyFilter,
    class_theta,
    footprint,
    observe,
    spread_kernel,
)

CFG = NOMINAL


def _binary_pair(seed):
    sc = make_scenario(CFG, seed)
    occ = OccupancyFilter(sc.prior, CFG.sensor, CFG.model, sc.wind_dir, CFG.fire.wind_gain,
                          CFG.fire.burn_steps)
    prior = np.zeros((K_CLASSES,) + sc.prior.shape)
    prior[FIRE], prior[EMPTY] = sc.prior, 1 - sc.prior
    theta = class_theta(CFG.sensor, DecoyConfig())
    cat = CategoricalFilter(prior, theta, spread_kernel(CFG.model, sc.wind_dir, CFG.fire.wind_gain),
                            CFG.fire.burn_steps)
    return sc, occ, cat


# ------------------------------------------------------------------ test 1: regression invariant
@pytest.mark.parametrize("seed", [0, 2])
def test_categorical_filter_restricted_to_fire_equals_v10_filter(seed):
    """Over a whole v1.0 mission (well past burn_steps predicts, where the
    history matters), {empty, fire} with no decoys reproduces OccupancyFilter."""
    sc, occ, cat = _binary_pair(seed)
    path = run_mission(CFG, "info-rate", seed).path[1:]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t, pos in enumerate(path):
        if t and t % CFG.fire.fire_step_ticks == 0:
            sc.fire.step()
            occ.predict()
            cat.predict()
        fp, z = observe(sc.fire.affected, pos, CFG.sensor, rng)
        occ.update(fp, z)
        cat.update(fp, z)
        worst = max(worst, float(np.abs(occ.prob - cat.prob(FIRE)).max()))
    assert worst < 1e-7


def test_hits_preserve_the_ratio_between_classes_with_equal_response():
    """Fire and hot decoys answer a hit identically, so no number of hits may
    change their ratio (a per-class log-odds clip used to flatten it to 1:1)."""
    prior = np.zeros((K_CLASSES, 1, 1))
    prior[FIRE], prior[HOT], prior[WARM] = 0.02, 0.005, 0.005
    prior[EMPTY] = 1 - prior[1:].sum()
    f = CategoricalFilter(prior, class_theta(CFG.sensor, DecoyConfig()))
    fp = (slice(0, 1), slice(0, 1))
    for _ in range(60):
        f.update(fp, np.ones((1, 1), bool))
    b = f.probs[:, 0, 0]
    assert b[FIRE] / b[HOT] == pytest.approx(4.0, rel=1e-6)
    assert b[EMPTY] < 1e-6


# ------------------------------------------------------------------ test 2: normalisation
def test_beliefs_and_predictive_distributions_are_normalised():
    sc = make_decoy_scenario(CFG, DecoyConfig(), "A", 3)
    test = MultiModelTest.build(sc.prior, CFG, DecoyConfig(), SPRTConfig(), sc.wind_dir)
    rng = np.random.default_rng(1)
    for t in range(40):
        if t % 6 == 0:
            test.predict()
        pos = tuple(int(x) for x in rng.integers(5, 59, size=2))
        fp = footprint(pos, CFG.sensor.radius, CFG.n)
        test.update(fp, rng.random((7, 7)) < 0.3)   # footprint lies fully inside the grid
    for filt in (test.h1, test.h0):
        np.testing.assert_allclose(filt.probs.sum(axis=0), 1.0, atol=1e-12)
        assert (filt.probs >= 0).all()
        one = (slice(30, 31), slice(30, 31))
        p = sum(np.exp(filt.predictive_loglik(one, np.array([[v]]))) for v in (True, False))
        assert p == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ test 3: H0 is static
def test_h0_has_no_fire_class_and_never_moves_mass():
    sc = make_decoy_scenario(CFG, DecoyConfig(), "B", 4)
    test = MultiModelTest.build(sc.prior, CFG, DecoyConfig(), SPRTConfig(), sc.wind_dir)
    assert FIRE not in test.h0.cls
    before = test.h0.probs.copy()
    for _ in range(10):
        test.h0.predict()
    np.testing.assert_array_equal(test.h0.probs, before)
    assert test.h0.probs[FIRE].max() == 0.0


# ------------------------------------------------------------------ test 4: SPRT bookkeeping
def test_log_odds_is_prior_plus_sum_of_predictive_log_likelihood_ratios():
    sc = make_decoy_scenario(CFG, DecoyConfig(), "A", 5)
    sprt = SPRTConfig(prior_h1=0.3)
    test = MultiModelTest.build(sc.prior, CFG, DecoyConfig(), sprt, sc.wind_dir)
    twin = MultiModelTest.build(sc.prior, CFG, DecoyConfig(), sprt, sc.wind_dir)
    rng = np.random.default_rng(2)
    total = np.log(0.3 / 0.7)
    for t in range(30):
        if t % 6 == 0:
            test.predict()
            twin.predict()
        pos = tuple(int(x) for x in rng.integers(8, 56, size=2))
        fp = footprint(pos, CFG.sensor.radius, CFG.n)
        z = rng.random((7, 7)) < 0.2
        total += twin.h1.predictive_loglik(fp, z) - twin.h0.predictive_loglik(fp, z)
        twin.h1.update(fp, z)
        twin.h0.update(fp, z)
        test.update(fp, z)
    assert test.llr == pytest.approx(total, rel=1e-12)


def test_cold_untouched_ground_is_evidence_for_neither_hypothesis():
    """Background cells carry the same hot mass with the same sensor response
    under H0 and H1, so a cold look far from the cue must leave the log-odds alone."""
    sc = make_decoy_scenario(CFG, DecoyConfig(), "B", 6)
    test = MultiModelTest.build(sc.prior, CFG, DecoyConfig(), SPRTConfig(), sc.wind_dir)
    bg = np.isclose(sc.prior, CFG.prior.p_background)
    r, c = np.argwhere(bg)[len(np.argwhere(bg)) // 2]
    fp = footprint((int(r), int(c)), 0, CFG.n)
    llr0 = test.llr
    test.update(fp, np.zeros((1, 1), bool))
    assert test.llr == pytest.approx(llr0, abs=1e-12)


# ------------------------------------------------------------------ scenarios
def test_family_c_is_the_v10_scenario():
    a = make_scenario(CFG, 7)
    b = make_decoy_scenario(CFG, DecoyConfig(), "C", 7)
    np.testing.assert_array_equal(a.prior, b.prior)
    np.testing.assert_array_equal(a.fire.state, b.fire.state)
    assert not (b.decoys != EMPTY).any()


def test_family_b_has_a_cue_decoys_and_no_fire():
    for seed in range(5):
        sc = make_decoy_scenario(CFG, DecoyConfig(), "B", seed)
        assert not sc.fire.affected.any()
        assert (sc.decoys != EMPTY).any()
        assert (sc.prior >= CFG.prior.p_hotspot).any()      # something was flagged


def test_family_a_keeps_decoys_clear_of_the_fire():
    for seed in range(5):
        sc = make_decoy_scenario(CFG, DecoyConfig(), "A", seed)
        d = sc.decoys != EMPTY
        if d.any():
            assert distance_transform_edt(~sc.fire.affected)[d].min() > DecoyConfig().min_sep_fire


# ------------------------------------------------------------------ missions
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_m0_on_family_c_reproduces_v10(seed):
    v = run_mission(CFG, "info-rate", seed)
    m0 = run_decision_mission(CFG, "M0", "C", seed)
    assert m0.f1 == v.f1
    assert m0.t_decision_min == v.t_confirm


def test_decision_missions_are_deterministic_and_safe():
    for fam in "ABC":
        a = run_decision_mission(CFG, "M1", fam, 11)
        b = run_decision_mission(CFG, "M1", fam, 11)
        assert (a.decision, a.t_decision_min, a.ticks_flown) == (b.decision, b.t_decision_min, b.ticks_flown)
        assert a.llr == b.llr
        assert a.geofence_violations == 0
        assert a.returned_to_base
