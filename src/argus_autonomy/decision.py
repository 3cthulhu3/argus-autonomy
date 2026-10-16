"""v1.1 decision mission: dispatched on a satellite cue, decide ALERT or CLEAR.

Scenario families (design doc §3.2):
  A  fire + decoys          -> detection and mislocalisation
  B  decoys only, cue exists -> false-alert rate (the cue is a decoy)
  C  fire only              -> regression against v1.0

Methods (design doc §6.1):
  M0  v1.0 occupancy filter; ALERT when any cell reaches 0.95; never CLEARs
  M1  two-model sequential test; info-rate planner on P(any heat)
  M2  two-model test; Chernoff planner (expected log-odds drift), wide camera only
  M3  as M2 plus the close-look action
  OR  oracle: as M3 with every decoy's location and class known (a delay bound, not a method)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from .config import EMPTY, FIRE, WARM, DecoyConfig, MissionConfig, SPRTConfig
from .hypothesis import ALERT, CLEAR, UNDECIDED, MultiModelTest
from .metrics import boundary_error, f1_iou
from .perception import (
    OccupancyFilter,
    class_theta,
    observe_close,
    observe_wide,
    satellite_prior,
)
from .planners import Chernoff, InfoRate, PlanContext
from .safety import Geofence, RTBMonitor, advance
from .world import Fire, class_map, place_decoys, rasterise_decoys

METHODS = ("M0", "M1", "M2", "M3", "OR")
CORRECT_RADIUS_CELLS = 4   # an alert is correct if it lies within 100 m of affected ground


@dataclass
class DecoyScenario:
    family: str
    fire: Fire
    has_fire: bool
    decoys: np.ndarray          # EMPTY / WARM / HOT per cell
    rects: list
    prior: np.ndarray
    hot: np.ndarray
    wind_dir: float
    sensor_rng: np.random.Generator


def _no_fire(fire: Fire) -> Fire:
    fire.state[:] = 0
    return fire


def make_decoy_scenario(cfg: MissionConfig, dcfg: DecoyConfig, family: str, seed: int) -> DecoyScenario:
    """Same RNG streams as v1.0 ``make_scenario`` plus a fifth for decoys, so
    family C reproduces the v1.0 scenario bit for bit."""
    if family not in ("A", "B", "C"):
        raise ValueError(f"unknown family {family!r}")
    ss = np.random.SeedSequence(seed)
    r_world, r_fire, r_prior, r_sensor, r_decoy = (np.random.default_rng(s) for s in ss.spawn(5))
    fence = Geofence(cfg.n, cfg.nfz)
    lo, hi = cfg.n // 4, cfg.n - cfg.n // 4
    while True:
        ign = tuple(int(x) for x in r_world.integers(lo, hi, size=2))
        if fence.allowed[ign] and np.hypot(*np.subtract(ign, cfg.base)) > 20:
            break
    wind_dir = float(r_world.uniform(0, 2 * np.pi))
    fire = Fire(cfg.n, ign, wind_dir, cfg.fire, r_fire)
    has_fire = family != "B"
    if not has_fire:
        _no_fire(fire)

    pre, lat = cfg.fire.preburn_steps, cfg.prior.latency_steps
    fire.advance(pre - lat)
    overpass_fire = fire.affected.copy()
    fire.advance(lat)

    base_keep = np.zeros((cfg.n, cfg.n), bool)
    base_keep[max(cfg.base[0] - 4, 0):cfg.base[0] + 5, max(cfg.base[1] - 4, 0):cfg.base[1] + 5] = True
    rects: list = []
    if family != "C":
        keep = base_keep | binary_dilation(fire.affected, iterations=dcfg.min_sep_fire)
        for _ in range(dcfg.max_tries):
            n_w, n_h = int(r_decoy.poisson(dcfg.mean_warm)), int(r_decoy.poisson(dcfg.mean_hot))
            rects = place_decoys(cfg.n, fence.allowed, keep, n_w, n_h, dcfg.max_side, r_decoy)
            sat_visible = [r for r in rects if r[4] != WARM or r_decoy.random() < dcfg.p_sat_warm]
            decoy_sat = rasterise_decoys(cfg.n, sat_visible) != EMPTY
            if family == "A":
                break
            # family B: a dispatch only happens if the satellite flagged something
            k = cfg.prior.coarse_factor
            m = cfg.n // k
            frac = decoy_sat[: m * k, : m * k].reshape(m, k, m, k).mean(axis=(1, 3))
            if (frac >= cfg.prior.detect_fraction).any():
                break
        else:  # pragma: no cover
            raise RuntimeError("no cued decoy configuration found")
    else:
        decoy_sat = np.zeros((cfg.n, cfg.n), bool)

    decoys = rasterise_decoys(cfg.n, rects)
    prior, hot = satellite_prior(overpass_fire | decoy_sat, cfg.prior, r_prior)
    return DecoyScenario(family, fire, has_fire, decoys, rects, prior, hot, wind_dir, r_sensor)


def alert_location(belief: np.ndarray) -> tuple[float, float] | None:
    """Belief-weighted centroid of the largest connected blob with belief >= 0.5."""
    lab, k = label(belief >= 0.5, structure=np.ones((3, 3), bool))
    if k == 0:
        r, c = np.unravel_index(int(np.argmax(belief)), belief.shape)
        return float(r), float(c)
    sizes = np.bincount(lab.ravel())[1:]
    m = lab == (1 + int(np.argmax(sizes)))
    w = belief * m
    rr, cc = np.indices(belief.shape)
    return float((rr * w).sum() / w.sum()), float((cc * w).sum() / w.sum())


@dataclass
class DecisionResult:
    method: str
    family: str
    seed: int
    has_fire: bool
    decision: str                 # ALERT / CLEAR / UNDECIDED
    t_decision_min: float         # nan if UNDECIDED
    alert_correct: bool | None    # only for ALERT
    alert_error_m: float          # distance from alert to nearest affected cell (nan if no fire / no alert)
    alert_on_decoy: int           # decoy class the alert sits on (0 if none)
    f1: float
    boundary_m: float
    geofence_violations: int
    returned_to_base: bool
    ticks_flown: int
    reserve_left: int
    n_close_looks: int = 0
    cue_seen_at_decision: float = float("nan")
    ring_seen_at_decision: float = float("nan")   # v1.2: fraction of ring cells seen (nan if no ring rule)
    llr: list[float] = field(repr=False, default_factory=list)
    # sweep mode only: (tick, log-odds, gate open) after every observation, and for
    # each alert level the first tick the log-odds reached it with where the alert lands
    trace: list = field(repr=False, default_factory=list)
    crossings: dict = field(repr=False, default_factory=dict)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("llr", "trace", "crossings")}


def run_decision_mission(cfg: MissionConfig, method: str, family: str, seed: int,
                         dcfg: DecoyConfig | None = None, sprt: SPRTConfig | None = None,
                         stop_on_clear: bool = True, plan_on: str = "heat",
                         min_drift: float | None = None, sweep: bool = False,
                         alert_levels: tuple[float, ...] = ()) -> DecisionResult:
    """``sweep=True`` never decides: the mission flies to endurance with the
    pre-decision policy and logs the log-odds trajectory, so the decision of any
    threshold pair is recovered afterwards as a first-passage time
    (``first_passage``). Needs ``sprt.steer_clear`` set, or the policy would depend
    on ``sprt.clear``."""
    dcfg = dcfg or DecoyConfig()
    sprt = sprt or SPRTConfig()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    sc = make_decoy_scenario(cfg, dcfg, family, seed)
    fence = Geofence(cfg.n, cfg.nfz)
    monitor = RTBMonitor(fence, cfg.base, cfg.speed, cfg.reserve_ticks)
    theta = class_theta(cfg.sensor, dcfg)
    close_lik = np.asarray(dcfg.close_lik, float)
    fire, rng = sc.fire, sc.sensor_rng
    min_per_tick = cfg.dt_s / 60.0

    info = InfoRate(cfg.sensor)
    chern = None
    if method in ("M2", "M3", "OR"):
        kw = {} if min_drift is None else {"min_drift": min_drift}
        chern = Chernoff(cfg.sensor, close_lik if method in ("M3", "OR") else None,
                         dcfg.close_radius, dcfg.t_close, **kw)

    if method == "M0":
        filt = OccupancyFilter(sc.prior, cfg.sensor, cfg.model, sc.wind_dir, cfg.fire.wind_gain,
                               cfg.fire.burn_steps)
        test = None
    else:
        known = sc.decoys if method == "OR" else None
        test = MultiModelTest.build(sc.prior, cfg, dcfg, sprt, sc.wind_dir, known)
        filt = None

    def belief() -> np.ndarray:
        return filt.prob if filt is not None else test.h1.prob(FIRE)

    def plan_belief() -> np.ndarray:
        # The wide camera resolves "hot or not", not "fire or hot decoy": planning
        # on P(fire) makes the planner chase uncertainty no look can reduce.
        if filt is not None or plan_on == "fire":
            return belief()
        return 1.0 - test.h1.prob(EMPTY)

    gate = None
    if sprt.clear_gate is not None and test is not None:
        lo = {"hotspot": cfg.prior.p_hotspot, "cue": cfg.prior.p_ring}[sprt.clear_gate]
        gate = (sc.prior >= lo) & fence.allowed
    seen = np.zeros((cfg.n, cfg.n), bool)
    cue_seen = [float("nan")]   # fraction of flagged allowed cells seen when the decision was made
    ring_seen = [float("nan")]

    ring = None
    if (sprt.ring_cover > 0 or sprt.dwell_ticks > 0) and gate is not None:
        ring = (sc.prior >= cfg.prior.p_ring) & (sc.prior < cfg.prior.p_hotspot) & fence.allowed
    dwell_start: list = [None]
    now_tick = [0]

    def flagged_seen() -> bool:
        return gate is None or bool(seen[gate].all())

    def ring_frac() -> float:
        return float(seen[ring].mean()) if ring is not None and ring.any() else 1.0

    def gate_open() -> bool:
        if not flagged_seen():
            return False
        if sprt.ring_cover > 0 and ring_frac() < sprt.ring_cover:
            return False
        if sprt.dwell_ticks > 0 and ring_frac() < 1.0:
            if dwell_start[0] is None or now_tick[0] - dwell_start[0] < sprt.dwell_ticks:
                return False
        return True

    pos, path, rtb, carry = cfg.base, [], False, 0.0
    remaining = cfg.budget_ticks
    decision, t_dec, where = UNDECIDED, float("nan"), None
    llr_log: list[float] = []
    violations = n_close = 0
    close_target, hover = None, 0
    t = 0

    steer = sprt.clear if sprt.steer_clear is None else sprt.steer_clear
    if sweep and (test is None or sprt.steer_clear is None):
        raise ValueError("sweep mode needs a two-model method and sprt.steer_clear")
    trace: list = []
    crossings: dict = {}
    pending = sorted(alert_levels)

    def log(now: int) -> None:
        now_tick[0] = now
        if sprt.dwell_ticks > 0:      # the dwell runs while the test would clear and the flagged pixels are seen
            if test.llr <= steer and flagged_seen():
                if dwell_start[0] is None:
                    dwell_start[0] = now
            else:
                dwell_start[0] = None
        llr_log.append(test.llr)
        if not sweep:
            return
        trace.append((now, test.llr, gate_open()))
        while pending and test.llr >= pending[0]:
            lvl = pending.pop(0)
            r_, c_ = (round(x) for x in alert_location(test.h1.prob(FIRE)))
            ok = None
            if sc.has_fire:
                ok = bool(distance_transform_edt(~fire.affected)[r_, c_] <= CORRECT_RADIUS_CELLS)
            crossings[lvl] = (now, ok, int(sc.decoys[r_, c_]))

    def decide(now: int) -> None:
        nonlocal decision, t_dec, where
        if decision != UNDECIDED or test is None or sweep:
            return
        d = test.decision()
        if d == CLEAR and not gate_open():
            d = UNDECIDED          # cue not fully seen yet: keep looking
        if d != UNDECIDED:
            decision, t_dec = d, (now + 1) * min_per_tick
            hs = (sc.prior >= cfg.prior.p_hotspot) & fence.allowed
            cue_seen[0] = float(seen[hs].mean()) if hs.any() else float("nan")
            if ring is not None:
                ring_seen[0] = ring_frac()
            if d == ALERT:
                where = alert_location(test.h1.prob(FIRE))

    for t in range(cfg.budget_ticks):
        if t > 0 and t % cfg.fire.fire_step_ticks == 0:
            fire.step()
            (filt or test).predict()

        if not rtb and (monitor.must_return(pos, remaining) or (decision == CLEAR and stop_on_clear)):
            rtb, close_target, hover = True, None, 0
            path = monitor.path_home(pos)

        # ---------------------------------------------------------- close look in progress
        if hover > 0:
            remaining -= 1
            hover -= 1
            if hover == 0:
                fp, y = observe_close(class_map(fire.affected, sc.decoys), pos, close_lik,
                                      dcfg.close_radius, rng)
                seen[fp] = True
                test.update_close(fp, y, close_lik)
                log(t)
                n_close += 1
                close_target = None
                decide(t)
            else:
                log(t)
            continue

        if not rtb and close_target is None and (t % cfg.replan_ticks == 0 or not path):
            dist, pred = fence.distances_from(pos)
            feas = monitor.feasible_mask(dist, remaining)
            pb = plan_belief()
            if test is not None and decision == UNDECIDED and test.llr <= steer and not gate_open():
                # the test would clear, but part of the cue is unseen: go and look there
                k = cfg.sensor.radius
                if flagged_seen() and ring is not None:   # v1.2: then search the ring
                    near = binary_dilation(ring & ~seen, iterations=k) & feas
                    if not near.any():
                        near = feas
                else:
                    near = binary_dilation(gate & ~seen, iterations=k) & feas
                wp = info.plan(PlanContext(pos, pb, dist, near, cfg.speed, t))
                mode = "wide"
            elif chern is not None and decision == UNDECIDED:
                feas_c = monitor.feasible_mask(dist, remaining - dcfg.t_close) if chern.close_lik is not None else None
                wp, mode = chern.plan(PlanContext(pos, pb, dist, feas, cfg.speed, t), test, feas_c)
            else:
                wp, mode = info.plan(PlanContext(pos, pb, dist, feas, cfg.speed, t)), "wide"
            path = fence.path(pred, pos, wp)
            if mode == "close":
                close_target = wp

        pos, path, carry = advance(pos, path, cfg.speed, carry)
        remaining -= 1
        violations += int(not fence.contains(pos))

        if close_target is not None and pos == close_target:
            hover = dcfg.t_close          # descend: the wide camera is off while hovering
            log(t)
            continue

        fp, z = observe_wide(class_map(fire.affected, sc.decoys), pos, theta, cfg.sensor.radius, rng)
        seen[fp] = True
        if filt is not None:
            filt.update(fp, z)
            if decision == UNDECIDED and (filt.prob >= cfg.confirm_threshold).any():
                decision, t_dec, where = ALERT, (t + 1) * min_per_tick, alert_location(filt.prob)
        else:
            test.update(fp, z)
            log(t)
            decide(t)
        if rtb and pos == cfg.base:
            break

    if method == "M0" and decision == UNDECIDED:   # M0 cannot clear; silence at landing counts as CLEAR
        decision, t_dec = CLEAR, (t + 1) * min_per_tick

    truth = fire.affected
    correct, err_m, on_decoy = None, float("nan"), 0
    if decision == ALERT and where is not None:
        ri, ci = round(where[0]), round(where[1])
        on_decoy = int(sc.decoys[ri, ci])
        if sc.has_fire:
            d = distance_transform_edt(~truth)[ri, ci]
            err_m = float(d * cfg.cell_m)
            correct = bool(d <= CORRECT_RADIUS_CELLS)
        else:
            correct = False
    b = belief()
    f1, _ = f1_iou(b, truth) if sc.has_fire else (float("nan"), None)
    return DecisionResult(
        method=method, family=family, seed=seed, has_fire=sc.has_fire, decision=decision,
        t_decision_min=t_dec, alert_correct=correct, alert_error_m=err_m, alert_on_decoy=on_decoy,
        f1=f1, boundary_m=boundary_error(b, truth, cfg.cell_m) if sc.has_fire else float("nan"),
        geofence_violations=violations, returned_to_base=(pos == cfg.base),
        ticks_flown=t + 1, reserve_left=remaining, n_close_looks=n_close,
        cue_seen_at_decision=cue_seen[0], ring_seen_at_decision=ring_seen[0], llr=llr_log, trace=trace, crossings=crossings,
    )


def first_passage(trace: list, a: float, b: float, min_per_tick: float) -> tuple[str, float, int | None]:
    """Decision of threshold pair (a, b) from a sweep-mode trace: ALERT at the first
    tick with log-odds >= a; CLEAR at the first tick with log-odds <= b and the cue
    gate open; else UNDECIDED. Returns (decision, minutes, tick)."""
    for now, llr, gate_ok in trace:
        if llr >= a:
            return ALERT, (now + 1) * min_per_tick, now
        if llr <= b and gate_ok:
            return CLEAR, (now + 1) * min_per_tick, now
    return UNDECIDED, float("nan"), None
