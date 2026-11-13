"""v2.1: Rothermel, the level-set front world, the EnKF and the regime signal."""

import numpy as np
import pytest

from argus_autonomy import rothermel as rt
from argus_autonomy.assimilation import (DroneConfig, Ensemble, ExperimentConfig, FilterConfig, Obs,
                                         WorldConfig, analysis, gaspari_cohn, run_scenario)
from argus_autonomy.levelset import FrontConfig, FrontModel, Inputs, Landscape, ignite, reinit
from argus_autonomy.regime import convection_number, regime_class, regime_signal

MPH = 0.44704
# Anderson (1982, INT-GTR-122) table 1: flame length (ft) at 8 % dead, 100 % live, 5 mi/h midflame
ANDERSON_FL_FT = {1: 4.0, 2: 6.0, 8: 1.0, 9: 2.6, 10: 4.8}


# ------------------------------------------------------------------ Rothermel
@pytest.mark.parametrize("code", rt.FUEL_CODES)
def test_rothermel_matches_anderson_table(code):
    fm = rt.FUEL_MODELS[code]
    r = float(rt.spread_rate(code, 0.08, 5 * MPH / fm.waf))
    ref = fm.ref_ros_ch_h * rt.CHAIN_H_TO_M_MIN
    assert abs(r / ref - 1) < 0.12, (r, ref)
    fl = float(rt.flame_length_m(rt.byram_intensity(code, 0.08, r))) / rt.FT
    assert abs(fl / ANDERSON_FL_FT[code] - 1) < 0.15, (fl, ANDERSON_FL_FT[code])


def test_rothermel_moisture_and_wind_monotone():
    m = np.linspace(0.03, 0.2, 30)
    r0, _ = rt.no_wind_spread(1, m)
    assert np.all(np.diff(r0) <= 1e-12)
    assert r0[m >= rt.FUEL_MODELS[1].mx_dead].max() == 0.0          # extinguished above Mx
    u = np.linspace(0, 10, 30)
    r = rt.spread_rate(2, 0.06, u)
    assert np.all(np.diff(r) >= -1e-12)
    assert r[0] == pytest.approx(rt.no_wind_spread(2, 0.06)[0])


# ------------------------------------------------------------------ level set
def _uniform(code, n=120):
    cfg = FrontConfig(n=n)
    return cfg, FrontModel(Landscape(np.full((n, n), code), np.zeros((n, n)), cfg))


def _crossings(row, dx):
    burnt = np.where(row < 0)[0]
    i, j = burnt.max(), burnt.min()
    head = i * dx + dx * row[i] / (row[i] - row[i + 1])
    back = j * dx - dx * row[j] / (row[j] - row[j - 1])
    return head, back


@pytest.mark.parametrize("code,minutes", [(1, 20), (9, 60)])
def test_point_ignition_head_and_back_match_wavelet(code, minutes):
    """Exact solution from a point: the wavelet scaled by t (Hopf formula). Head = R_head t,
    back = R_back t, measured from the ignition circle."""
    cfg, model = _uniform(code)
    k = 2
    inp = Inputs(wind=np.tile([4.0, 0.0], (k, 1)), m1h=np.full(k, 0.06), alpha=np.ones(k))
    c0 = 40
    psi = model.advance(ignite(cfg, k, (60, c0)), inp, minutes)
    head, back = _crossings(psi[0, 60], cfg.cell_m)
    pc = model._prepare(inp, k)
    rh = float(pc["rh"][0, 60, c0])
    rb = float(2 * pc["a"][0, 60, c0] - rh)
    x0, r = c0 * cfg.cell_m, cfg.ign_radius_m
    assert (head - x0 - r) / minutes == pytest.approx(rh, rel=0.02)
    assert (x0 - back - r) / minutes == pytest.approx(rb, rel=0.05, abs=0.01)


def test_planar_front_exact_and_alpha_scales():
    cfg, model = _uniform(1)
    xs = np.arange(cfg.n) * cfg.cell_m
    psi = np.broadcast_to(xs[None, None, :] - 1000.0, (2, cfg.n, cfg.n)).copy()
    inp = Inputs(wind=np.array([[3.0, 0.0], [3.0, 0.0]]), m1h=np.full(2, 0.07), alpha=np.array([1.0, 0.5]))
    out = model.advance(psi, inp, 10)
    r = float(rt.spread_rate(1, 0.07, 3.0))
    h0, _ = _crossings(out[0, 60], cfg.cell_m)
    h1, _ = _crossings(out[1, 60], cfg.cell_m)
    assert (h0 - 1000) / 10 == pytest.approx(r, rel=0.01)
    assert (h1 - 1000) / 10 == pytest.approx(0.5 * r, rel=0.01)


def test_nonburnable_stops_the_front():
    n = 80
    fuel = np.full((n, n), 1)
    fuel[:, 50:53] = 0
    cfg = FrontConfig(n=n)
    model = FrontModel(Landscape(fuel, np.zeros((n, n)), cfg))
    inp = Inputs(wind=np.array([[5.0, 0.0]]), m1h=np.array([0.05]), alpha=np.ones(1))
    out = model.advance(ignite(cfg, 1, (40, 30)), inp, 30)
    assert (out[0, :, 53:] < 0).sum() == 0


def _zero(row, dx):
    i = int(np.where(np.diff(np.sign(row)) != 0)[0][0])
    return i * dx + dx * row[i] / (row[i] - row[i + 1])


def test_reinit_keeps_subcell_front():
    cfg = FrontConfig(n=60)
    xs = np.arange(cfg.n) * cfg.cell_m
    x0 = 512.3
    psi = np.broadcast_to(2.7 * (xs[None, None, :] - x0), (1, cfg.n, cfg.n)).copy()   # not a distance
    out = reinit(psi, cfg.cell_m)
    assert abs(_zero(out[0, 30], cfg.cell_m) - x0) < 0.5
    assert np.allclose(np.abs(np.diff(out[0, 30])), cfg.cell_m, atol=1e-6)            # unit slope


# ------------------------------------------------------------------ EnKF
def test_gaspari_cohn():
    r = np.array([0.0, 100.0, 199.999, 200.0, 300.0])
    g = gaspari_cohn(r, 100.0)
    assert g[0] == pytest.approx(1.0)
    assert g[1] == pytest.approx(gaspari_cohn(np.array([100.0001]), 100.0)[0], abs=1e-4)
    assert g[2] == pytest.approx(0.0, abs=1e-6) and g[3] == 0 and g[4] == 0


def test_analysis_moves_front_and_wind_towards_observations():
    """Members whose front is too far east and whose wind is too strong: one perimeter point at
    the true front and one wind reading must pull both towards the truth and shrink the spread."""
    n = 60
    cfg = FrontConfig(n=n)
    land = Landscape(np.full((n, n), 1), np.zeros((n, n)), cfg)
    rng = np.random.default_rng(0)
    k = 40
    xs = np.arange(n) * cfg.cell_m
    shift = rng.normal(100.0, 40.0, k)                     # members' fronts are 100 +- 40 m too far east
    psi = xs[None, None, :] - (700.0 + shift)[:, None, None]
    psi = np.broadcast_to(psi, (k, n, n)).copy()
    wind = np.column_stack([3.0 + 0.5 * (shift - 100) / 40 + rng.normal(0, 0.2, k), np.zeros(k)])
    ens = Ensemble(psi, wind, np.full(k, 0.06), np.zeros(k))
    pts = np.array([[700.0, -500.0], [700.0, -700.0]])
    obs = Obs(t=0, pts=pts, wind=np.array([2.0, 0.0]), frp_log=0.0, drone=pts[0],
              mask=np.zeros((n, n), bool))
    f = FilterConfig(k=k, r_frp_log=1e6)
    out = analysis(ens, obs, land, DroneConfig(), f, update_inputs=True, rng=rng)
    from argus_autonomy.levelset import sample
    before = np.abs(sample(ens.psi, pts, cfg).mean())
    after = np.abs(sample(out.psi, pts, cfg).mean())
    assert after < 0.5 * before
    assert abs(out.wind[:, 0].mean() - 2.0) < abs(ens.wind[:, 0].mean() - 2.0)
    assert out.wind[:, 0].std() < ens.wind[:, 0].std()
    so = analysis(ens, obs, land, DroneConfig(), f, update_inputs=False, rng=rng)
    assert np.array_equal(so.wind, ens.wind)               # state-only leaves the inputs alone


def test_small_scenario_is_deterministic():
    cfg = ExperimentConfig(world=WorldConfig(front=FrontConfig(n=80)),
                           drone=DroneConfig(t_last_obs_min=20.0),
                           filt=FilterConfig(k=8), issue_min=(20.0,), leads_min=(10.0,))
    a = run_scenario(cfg, 3)
    b = run_scenario(cfg, 3)
    assert a == b
    for m in ("OL", "SO", "JE"):
        assert 0.0 <= a[f"{m}@20+10:brier"] <= 1.0
        assert a[f"{m}@20:light"] in ("GREEN", "AMBER", "RED")


# ------------------------------------------------------------------ regime
def test_convection_number():
    # 1000 kW/m, 5 m/s wind, 30 m/min head: 2 g I / (rho cp T (U - R)^3)
    nc = float(convection_number(1000.0, 5.0, 30.0))
    assert nc == pytest.approx(2 * 9.81 * 1e6 / (1.17 * 1005 * 300 * 4.5 ** 3), rel=1e-9)
    assert np.isinf(convection_number(100.0, 0.4, 30.0))          # wind slower than the head
    assert list(regime_class([0.5, 2.0, 10.0, 10.5, np.inf])) == [0, 1, 1, 2, 2]


def test_regime_signal_lights():
    assert regime_signal([0.1] * 9 + [5.0]).light == "GREEN"       # P(wind-driven) = 0.9
    assert regime_signal([0.1] * 7 + [5.0] * 3).light == "AMBER"   # 0.7 wind-driven, none plume
    s = regime_signal([0.1] * 7 + [50.0, np.inf, 5.0])
    assert s.light == "RED" and s.p == pytest.approx((0.7, 0.1, 0.2))


def test_front_brief_reads_like_a_brief():
    from argus_autonomy.assimilation import front_brief, make_world, observe, prior_ensemble
    cfg = ExperimentConfig(world=WorldConfig(front=FrontConfig(n=80)), filt=FilterConfig(k=8))
    world = make_world(cfg.world, 5, 10.0)
    init = observe(world, 10.0, 0, cfg.drone, np.random.default_rng(0), full=True)
    ens = prior_ensemble(world, init, cfg.filt, np.random.default_rng(1))
    b = front_brief(ens, FrontModel(world.land_model), 10.0, leads=(15.0,))
    lines = b.lines()
    assert lines[0].startswith("t+10 min") and lines[-1].startswith("regime ")
    assert b.forecast_area_ha[15.0] >= b.area_ha * 0.9
    assert b.regime.light in ("GREEN", "AMBER", "RED")
