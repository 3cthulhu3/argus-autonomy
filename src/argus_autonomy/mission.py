"""Closed-loop mission: truth -> sensor -> filter -> planner -> safety monitor -> motion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MissionConfig
from .metrics import boundary_error, confirmed, f1_iou, mean_entropy
from .perception import OccupancyFilter, observe, satellite_prior
from .planners import InfoRate, PlanContext, full_map_lawnmower, prior_lawnmower
from .safety import Geofence, RTBMonitor, advance
from .world import Fire

PLANNERS = ("lawnmower", "prior-lawnmower", "info-rate")


@dataclass
class Scenario:
    fire: Fire
    prior: np.ndarray
    hot: np.ndarray
    ignition: tuple[int, int]
    wind_dir: float
    sensor_rng: np.random.Generator


def make_scenario(cfg: MissionConfig, seed: int) -> Scenario:
    """Fire, wind and satellite prior for one seed.  Independent RNG streams mean
    every planner faces exactly the same fire evolution for the same seed."""
    ss = np.random.SeedSequence(seed)
    r_world, r_fire, r_prior, r_sensor = (np.random.default_rng(s) for s in ss.spawn(4))
    fence = Geofence(cfg.n, cfg.nfz)
    lo, hi = cfg.n // 4, cfg.n - cfg.n // 4
    while True:
        ign = tuple(int(x) for x in r_world.integers(lo, hi, size=2))
        if fence.allowed[ign] and np.hypot(*np.subtract(ign, cfg.base)) > 20:
            break
    wind_dir = float(r_world.uniform(0, 2 * np.pi))
    fire = Fire(cfg.n, ign, wind_dir, cfg.fire, r_fire)
    fire.advance(cfg.fire.preburn_steps - cfg.prior.latency_steps)
    overpass = fire.affected.copy()
    fire.advance(cfg.prior.latency_steps)
    prior, hot = satellite_prior(overpass, cfg.prior, r_prior)
    return Scenario(fire, prior, hot, ign, wind_dir, r_sensor)


def make_planner(name: str, cfg: MissionConfig, sc: Scenario, fence: Geofence):
    r = cfg.sensor.radius
    if name == "lawnmower":
        return full_map_lawnmower(cfg.n, r, fence.allowed, cfg.base)
    if name == "prior-lawnmower":
        return prior_lawnmower(sc.prior, cfg.prior.p_ring, r, fence.allowed, cfg.base)
    if name == "info-rate":
        return InfoRate(cfg.sensor)
    raise ValueError(f"unknown planner {name!r}; choose from {PLANNERS}")


@dataclass
class MissionResult:
    planner: str
    seed: int
    t_confirm: float                 # minutes until first correct confirmation (nan if never)
    entropy: list[float]             # bits/cell per tick
    recall_confirmed: list[float]    # fraction of affected cells confirmed, per tick
    f1: float
    iou: float
    boundary_m: float
    false_confirmations: int
    geofence_violations: int
    returned_to_base: bool
    ticks_flown: int
    reserve_left: int
    path: list[tuple[int, int]] = field(repr=False, default_factory=list)
    frames: list[dict] = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in (
            "planner", "seed", "t_confirm", "f1", "iou", "boundary_m",
            "false_confirmations", "geofence_violations", "returned_to_base",
            "ticks_flown", "reserve_left")} | {
            "entropy_final": self.entropy[-1],
            "recall_final": self.recall_confirmed[-1],
        }


def run_mission(cfg: MissionConfig, planner_name: str, seed: int,
                record_every: int = 0) -> MissionResult:
    sc = make_scenario(cfg, seed)
    fence = Geofence(cfg.n, cfg.nfz)
    monitor = RTBMonitor(fence, cfg.base, cfg.speed, cfg.reserve_ticks)
    planner = make_planner(planner_name, cfg, sc, fence)
    filt = OccupancyFilter(sc.prior, cfg.sensor, cfg.model, sc.wind_dir, cfg.fire.wind_gain,
                           cfg.fire.burn_steps)
    fire, rng = sc.fire, sc.sensor_rng
    minutes_per_tick = cfg.dt_s / 60.0

    pos, path, rtb, carry = cfg.base, [], False, 0.0
    remaining = cfg.budget_ticks
    t_confirm = float("nan")
    ent, rec, trail, frames = [], [], [pos], []
    violations = 0

    for t in range(cfg.budget_ticks):
        if t > 0 and t % cfg.fire.fire_step_ticks == 0:
            fire.step()
            filt.predict()

        if not rtb and monitor.must_return(pos, remaining):
            rtb = True
            path = monitor.path_home(pos)
        if not rtb and (t % cfg.replan_ticks == 0 or not path):
            dist, pred = fence.distances_from(pos)
            ctx = PlanContext(pos, filt.prob, dist, monitor.feasible_mask(dist, remaining), cfg.speed, t)
            wp = planner.plan(ctx)
            path = fence.path(pred, pos, wp)

        pos, path, carry = advance(pos, path, cfg.speed, carry)
        remaining -= 1
        violations += int(not fence.contains(pos))
        trail.append(pos)

        fp, z = observe(fire.affected, pos, cfg.sensor, rng)
        filt.update(fp, z)

        b, truth = filt.prob, fire.affected
        tp, _ = confirmed(b, truth, cfg.confirm_threshold)
        if tp > 0 and np.isnan(t_confirm):
            t_confirm = (t + 1) * minutes_per_tick
        ent.append(mean_entropy(b))
        rec.append(tp / max(int(truth.sum()), 1))
        if record_every and t % record_every == 0:
            frames.append({"t": t, "belief": b.copy(), "truth": fire.state.copy(),
                           "pos": pos, "rtb": rtb})
        if rtb and pos == cfg.base:
            break

    b, truth = filt.prob, fire.affected
    f1, iou = f1_iou(b, truth)
    _, fc = confirmed(b, truth, cfg.confirm_threshold)
    if record_every:
        frames.append({"t": t, "belief": b.copy(), "truth": fire.state.copy(), "pos": pos, "rtb": rtb})
    return MissionResult(
        planner=planner_name, seed=seed, t_confirm=t_confirm, entropy=ent,
        recall_confirmed=rec, f1=f1, iou=iou,
        boundary_m=boundary_error(b, truth, cfg.cell_m), false_confirmations=fc,
        geofence_violations=violations, returned_to_base=(pos == cfg.base),
        ticks_flown=t + 1, reserve_left=remaining, path=trail, frames=frames,
    )
