import numpy as np
import pytest

from argus_autonomy import NOMINAL, PLANNERS, run_mission
from argus_autonomy.config import ModelConfig, SensorConfig
from argus_autonomy.metrics import boundary_error, f1_iou, perimeter
from argus_autonomy.mission import make_scenario
from argus_autonomy.perception import OccupancyFilter, mutual_information, satellite_prior
from argus_autonomy.planners import InfoRate, serpentine
from argus_autonomy.safety import Geofence, RTBMonitor, advance
from argus_autonomy.world import Fire, shift, spread_factors
from argus_autonomy.config import FireConfig


# ------------------------------------------------------------------ world
def test_shift_has_no_wraparound():
    a = np.zeros((5, 5))
    a[0, 0] = 1
    assert shift(a, -1, 0).sum() == 0          # shifted off the grid, not wrapped
    assert shift(a, 1, 1)[1, 1] == 1


def test_wind_makes_downwind_spread_most_likely():
    f = spread_factors(wind_dir=0.0, wind_gain=1.6)   # blowing towards +col
    assert max(f, key=f.get) == (0, 1)
    assert min(f, key=f.get) == (0, -1)


def test_affected_area_is_monotone():
    rng = np.random.default_rng(0)
    fire = Fire(40, (20, 20), 0.0, FireConfig(), rng)
    prev = fire.affected.copy()
    for _ in range(25):
        fire.step()
        assert np.all(fire.affected >= prev)      # nothing un-burns
        prev = fire.affected.copy()
    assert (fire.state < 0).any()                 # some cells burnt out


# ------------------------------------------------------------------ perception
def test_bayes_update_matches_closed_form():
    s = SensorConfig(radius=0, p_detect=0.8, p_false_alarm=0.1)
    f = OccupancyFilter(np.full((3, 3), 0.3), s, ModelConfig(), 0.0, 1.0)
    fp = (slice(1, 2), slice(1, 2))
    f.update(fp, np.array([[True]]))
    expected = 0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.1)
    assert f.prob[1, 1] == pytest.approx(expected, rel=1e-9)
    f.update(fp, np.array([[False]]))
    expected = expected * 0.2 / (expected * 0.2 + (1 - expected) * 0.9)
    assert f.prob[1, 1] == pytest.approx(expected, rel=1e-9)
    assert f.prob[0, 0] == pytest.approx(0.3)


def test_mutual_information_is_zero_at_certainty_and_maximal_inside():
    s = SensorConfig()
    p = np.array([1e-9, 0.5, 1 - 1e-9])
    mi = mutual_information(p, s)
    assert mi[0] == pytest.approx(0, abs=1e-6) and mi[2] == pytest.approx(0, abs=1e-6)
    grid = np.linspace(0.01, 0.99, 99)
    assert 0 < mutual_information(grid, s).max() <= 1.0


def test_prediction_never_lowers_belief_and_gate_blocks_diffuse_mass():
    model = ModelConfig()
    b0 = np.full((30, 30), 0.01)
    b0[15, 15] = 0.99
    f = OccupancyFilter(b0, SensorConfig(), model, 0.0, 1.6, gate=0.5)
    for _ in range(5):
        before = f.prob.copy()
        f.predict()
        assert np.all(f.prob >= before - 1e-12)
    # far from the seed cell, gated prediction leaves the background untouched
    assert f.prob[0, 0] == pytest.approx(0.01, rel=1e-6)

    g = OccupancyFilter(b0, SensorConfig(), model, 0.0, 1.6, gate=None)
    for _ in range(5):
        g.predict()
    assert g.prob[0, 0] > 0.01            # ungated mean-field inflates diffuse mass


def test_satellite_prior_flags_the_fire_pixel():
    a = np.zeros((32, 32), bool)
    a[10:13, 10:13] = True
    from argus_autonomy.config import PriorConfig
    cfg = PriorConfig(geoloc_sigma=0.0)
    prior, hot = satellite_prior(a, cfg, np.random.default_rng(0))
    assert hot[11, 11] and prior[11, 11] == cfg.p_hotspot
    assert prior[31, 31] == cfg.p_background


# ------------------------------------------------------------------ safety
def test_geodesic_path_avoids_no_fly_zone():
    fence = Geofence(20, (5, 5, 15, 15))
    d, pred = fence.distances_from((10, 0))
    path = fence.path(pred, (10, 0), (10, 19))
    assert path and path[-1] == (10, 19)
    assert all(fence.contains(c) for c in path)
    assert d[10, 19] > 19                     # had to go around


def test_advance_respects_speed_and_carries_remainder():
    path = [(0, i) for i in range(1, 20)]
    pos, rest, carry = advance((0, 0), path, 4.0)
    assert pos == (0, 4) and carry == 0.0
    diag = [(i, i) for i in range(1, 20)]
    pos, rest, carry = advance((0, 0), diag, 4.0)
    assert pos == (2, 2) and 0 < carry < 1.5
    pos2, _, _ = advance(pos, rest, 4.0, carry)
    assert pos2 == (5, 5)                     # carry keeps average speed at ~4 cells/tick


def test_rtb_monitor_feasible_mask_keeps_return_possible():
    fence = Geofence(30, None)
    mon = RTBMonitor(fence, (0, 0), 2.0, reserve_ticks=2)
    d, _ = fence.distances_from((0, 0))
    m = mon.feasible_mask(d, remaining_ticks=10)
    rr, cc = np.nonzero(m)
    ticks = np.ceil(d[rr, cc] / 2) * 2
    assert ticks.max() <= 10 - 2 - 2


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mission_invariants(planner, seed):
    r = run_mission(NOMINAL, planner, seed)
    assert r.geofence_violations == 0
    assert r.returned_to_base
    assert r.reserve_left >= 0
    assert r.ticks_flown <= NOMINAL.budget_ticks
    assert 0.0 <= r.f1 <= 1.0
    assert all(NOMINAL.n > p[0] >= 0 and NOMINAL.n > p[1] >= 0 for p in r.path)


# ------------------------------------------------------------------ planners & metrics
def test_serpentine_covers_box():
    allowed = np.ones((20, 20), bool)
    wps = serpentine((0, 0, 20, 20), 3, allowed)
    rows = sorted({w[0] for w in wps})
    covered = set()
    for r in rows:
        covered |= set(range(r - 3, r + 4))
    assert set(range(20)) <= covered


def test_info_rate_prefers_uncertain_region():
    s = SensorConfig()
    belief = np.full((30, 30), 1e-4)
    belief[20:25, 20:25] = 0.5
    fence = Geofence(30, None)
    d, _ = fence.distances_from((0, 0))
    from argus_autonomy.planners import PlanContext
    ctx = PlanContext((0, 0), belief, d, np.ones((30, 30), bool), 4.0, 0)
    wp = InfoRate(s).plan(ctx)
    assert 18 <= wp[0] <= 26 and 18 <= wp[1] <= 26


def test_metrics_on_known_masks():
    t = np.zeros((20, 20), bool)
    t[5:15, 5:15] = True
    b = t.astype(float)
    assert f1_iou(b, t) == (1.0, 1.0)
    assert boundary_error(b, t, 25.0) == 0.0
    assert perimeter(t).sum() == 36


def test_same_seed_same_fire_for_every_planner():
    a = make_scenario(NOMINAL, 5)
    b = make_scenario(NOMINAL, 5)
    assert np.array_equal(a.fire.state, b.fire.state) and np.array_equal(a.prior, b.prior)
    assert run_mission(NOMINAL, "info-rate", 5).f1 == run_mission(NOMINAL, "info-rate", 5).f1
