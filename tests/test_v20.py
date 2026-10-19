"""v2.0: detection (patrol + hand-off) and response (brief, forecasts, suppression model)."""

import math

import numpy as np
import pytest

from argus_autonomy.config import NOMINAL, RECOMMENDED, DecoyConfig, PatrolConfig
from argus_autonomy.decision import make_decoy_scenario, run_decision_mission
from argus_autonomy.detection import (
    HazardFilter,
    LatentFire,
    cue_prior,
    hazard_per_step,
    make_patrol_scenario,
    run_patrol,
    scaled,
    slow_fire,
)
from argus_autonomy.response import (
    compass_from_rowcol,
    forecast_ensemble,
    forecast_meanfield,
    forecast_scores,
    responder_brief,
)
from argus_autonomy.suppression import (
    Airframe,
    FireCase,
    attack,
    critical_flow_per_m,
    ellipse_perimeter,
    flame_length,
    hb_ratio,
    head_distance,
    lb_ratio,
    perimeter,
)


# ------------------------------------------------------------------ hand-off hooks
def test_handoff_defaults_reproduce_v1():
    sprt = RECOMMENDED["NOMINAL"]
    a = run_decision_mission(NOMINAL, "M3", "A", 7, sprt=sprt).summary()
    sc = make_decoy_scenario(NOMINAL, DecoyConfig(), "A", 7)
    b = run_decision_mission(NOMINAL, "M3", "A", 7, sprt=sprt, scenario=sc, start_pos=NOMINAL.base,
                             start_remaining=NOMINAL.budget_ticks, t_offset=0).summary()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k] or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])), k


def test_return_on_decision_accounts_for_every_tick():
    for seed in range(3):
        r = run_decision_mission(NOMINAL, "M3", "A", seed, sprt=RECOMMENDED["NOMINAL"],
                                 start_remaining=150, return_on_decision=True)
        assert 150 - r.ticks_flown == r.reserve_left
        if r.decision != "UNDECIDED":
            assert r.final_pos is not None


def test_t_offset_keeps_fire_on_global_clock():
    sc = make_decoy_scenario(NOMINAL, DecoyConfig(), "C", 3)
    steps0 = sc.fire.steps
    r = run_decision_mission(NOMINAL, "M3", "C", 3, sprt=RECOMMENDED["NOMINAL"], scenario=sc,
                             t_offset=4, start_remaining=20, return_on_decision=True)
    fs = NOMINAL.fire.fire_step_ticks
    expected = sum(1 for t in range(r.ticks_flown) if (t + 4) > 0 and (t + 4) % fs == 0)
    assert sc.fire.steps - steps0 == expected


# ------------------------------------------------------------------ detection
def test_latent_fire_ignites_on_schedule():
    rng = np.random.default_rng(0)
    f = LatentFire(32, (16, 16), 0.0, NOMINAL.fire, rng, ignite_step=5)
    for _ in range(4):
        f.step()
        assert not f.affected.any()
    f.step()
    assert f.ignited and f.affected[16, 16]
    g = LatentFire(32, (16, 16), 0.0, NOMINAL.fire, np.random.default_rng(0), ignite_step=None)
    for _ in range(20):
        g.step()
    assert not g.affected.any()


def test_hazard_rates_sum_to_one_fire_per_window():
    risk = np.random.default_rng(1).random((16, 16))
    pc = PatrolConfig()
    a, b = pc.ign_window_min
    for uniform in (True, False):
        assert hazard_per_step(risk, pc, uniform).sum() == pytest.approx(1.0 / (b - a))
        assert hazard_per_step(risk, pc, uniform, 10.0).sum() == pytest.approx(10.0 / (b - a))


def test_hazard_filter_grows_unseen_belief_and_ignores_masked_cells():
    h = np.full((16, 16), 1e-3)
    f = HazardFilter(h, NOMINAL, 0.0)
    p0 = f.prob.copy()
    f.predict()
    assert (f.prob > p0).all()
    m = np.zeros((16, 16), bool)
    m[:4, :4] = True
    f.register_static(m)
    before = f.L.copy()
    f.update((slice(0, 8), slice(0, 8)), np.ones((8, 8), bool))
    assert np.array_equal(f.L[:4, :4], before[:4, :4])
    assert (f.L[4:8, 4:8] > before[4:8, 4:8]).all()


def test_cue_prior_block_and_ring():
    pc = PatrolConfig()
    p = cue_prior(64, (30, 30), NOMINAL, pc)
    assert (p >= NOMINAL.prior.p_hotspot).sum() == pc.cue_pixel ** 2
    assert p[30, 30] == NOMINAL.prior.p_hotspot
    assert p[30, 30 + pc.cue_pixel] == NOMINAL.prior.p_ring
    assert p[0, 0] == NOMINAL.prior.p_background


def test_patrol_is_deterministic_and_safe():
    pc = PatrolConfig(horizon_ticks=600)
    a = run_patrol(NOMINAL, pc, "P1", "F", 11).summary()
    b = run_patrol(NOMINAL, pc, "P1", "F", 11).summary()
    assert a == b or all(a[k] == b[k] or (isinstance(a[k], float) and math.isnan(a[k])) for k in a)
    for planner in ("P0", "P1u", "P1"):
        for fam in ("F", "N"):
            r = run_patrol(NOMINAL, pc, planner, fam, 12)
            assert r.geofence_violations == 0
            assert r.sorties - r.landed_safely in (0, 1)   # at most the sortie in progress at the end
            if fam == "N":
                assert math.isnan(r.t_alert_min)


def test_slow_fire_and_scaled_maps():
    s = slow_fire(NOMINAL)
    assert s.fire.fire_step_ticks == 10 * NOMINAL.fire.fire_step_ticks
    sc = make_patrol_scenario(s, PatrolConfig(), DecoyConfig(), "F", 4)
    assert sc.ign_minute % 10 == 0
    m = scaled(NOMINAL, 32)
    assert m.n == 32 and all(0 <= x <= 32 for x in m.nfz)


# ------------------------------------------------------------------ response
def test_compass():
    assert compass_from_rowcol(0.0) == pytest.approx(90.0)          # +col = east
    assert compass_from_rowcol(np.pi / 2) == pytest.approx(180.0)   # +row = south
    assert compass_from_rowcol(-np.pi / 2) == pytest.approx(0.0)


def test_forecasts_trivial_cases():
    z = np.full((24, 24), 1e-9)
    assert forecast_ensemble(z, NOMINAL.with_(n=24), 0.0, 10, k=20).sum() == 0
    b = np.zeros((24, 24))
    b[10:13, 10:13] = 1.0
    f0 = forecast_ensemble(b, NOMINAL.with_(n=24), 0.0, 0, k=20)
    assert np.array_equal(f0 > 0.5, b > 0.5)
    f30 = forecast_ensemble(b, NOMINAL.with_(n=24), 0.0, 30, k=50, rng=np.random.default_rng(2))
    assert f30.sum() > b.sum()                                       # fires grow
    assert (f30[b > 0.5] == 1).all()                                 # affected stays affected
    mf = forecast_meanfield(np.clip(b, 1e-6, 1 - 1e-6), NOMINAL.with_(n=24), 0.0, 0)
    assert np.allclose(mf, np.clip(b, 1e-6, 1 - 1e-6), atol=1e-6)


def test_forecast_scores_perfect_and_brief():
    t = np.zeros((20, 20), bool)
    t[5:8, 5:8] = True
    s = forecast_scores(t.astype(float), t)
    assert s["brier"] == 0 and s["cover10"] == 1 and s["area_ratio"] == 1
    belief = np.where(t, 0.99, 1e-4)
    br = responder_brief(belief, NOMINAL.with_(n=20), 0.0, 3.0, horizons_min=(5,), k=20)
    assert br.area_m2 == pytest.approx(9 * 625)
    assert br.forecast_area_m2[5] >= br.area_m2 * 0.9


# ------------------------------------------------------------------ suppression physics
def test_shape_relations():
    assert lb_ratio(0.0) == pytest.approx(1.0, abs=1e-12)
    assert hb_ratio(1.0) == pytest.approx(1.0)
    assert np.all(np.diff(lb_ratio(np.linspace(0, 10, 11))) > 0)
    assert ellipse_perimeter(3.0, 3.0) == pytest.approx(2 * np.pi * 3.0)
    # calm air: origin at the centre of a circle of radius = head distance
    assert perimeter(10.0, 1.0, 0.0, 0.0) == pytest.approx(2 * np.pi * 10.0)
    # acceleration: head distance below R t, approaching it for t >> tau
    assert head_distance(5.0, 2.0, 15.0) < 10.0
    assert head_distance(600.0, 2.0, 15.0) / (2.0 * 600.0) == pytest.approx(1 - 15 / 600, rel=1e-6)


def test_byram_and_penney():
    assert flame_length(186.0) == pytest.approx(0.86, abs=0.01)
    # Penney et al. 2019, 2 m depth: CF_10m = 2.72 RoS^0.42, RoS in km/h
    assert 10 * critical_flow_per_m(1000 / 60) == pytest.approx(2.72, rel=1e-3)
    assert critical_flow_per_m(0.0) == 0.0


def test_airframe_momentum_theory():
    ac = Airframe(p_avionics=0.0, k_cruise=1.0)
    assert ac.power(80.0) / ac.power(40.0) == pytest.approx(2 ** 1.5)
    loads = [Airframe().max_payload(d) for d in (500, 2000, 5000, 8000, 12000)]
    assert all(x >= y for x, y in zip(loads, loads[1:]))
    assert Airframe().max_payload(1e6) == 0.0


def test_attack_monotone_in_time_and_drones():
    case = FireCase(r_ss=1.5)
    early, late = attack(case, 5, 1500), attack(case, 40, 1500)
    assert early.need_l <= late.need_l
    assert not (late.knockdown_first and not early.knockdown_first)
    one, four = attack(case, 20, 1500, n_drones=1), attack(case, 20, 1500, n_drones=4)
    assert np.isfinite(four.contained_min) or not np.isfinite(one.contained_min)
    hot = attack(FireCase(r_ss=20.0, w=1.2), 10, 1000)
    assert hot.decision == "CUE" and hot.reason == "flame length"
