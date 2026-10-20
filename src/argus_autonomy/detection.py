"""v2.0 detection layer: cue-free patrol from a dock, hand-off to the decision test.

The v1.x stack starts from a satellite cue. Here nothing tells the drone where or
when a fire starts. The drone patrols, lands, recharges and patrols again; a patrol
filter tracks P(heat) per cell with an ignition hazard in its prediction step, so
ground that has not been seen for a while slowly becomes worth looking at again.
When the filter is confident that something is hot, the drone hands off to the
v1.1-v1.2b decision test (M3, ``RECOMMENDED``), seeded with a cue map built around
the hot cell. ALERT ends the run; CLEAR adds the source to a static-source register
whose cells the patrol filter ignores from then on (the operational analogue of
masking FIRMS static sources before dispatch).

Planners (``PATROL_PLANNERS``):
  P0   lawnmower over the whole geofence, repeated
  P1u  information-rate on the patrol filter with a uniform hazard
  P1   information-rate with the operator's risk map as hazard

For small P, a cell's patrol belief is about lambda_i * (time since seen), so the
information-rate rule is a latency-weighted patrol rule in the sense of Smith,
Schwager & Rus (2012, IEEE T-RO 28(2)); nothing here claims their guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from .config import DecoyConfig, MissionConfig, PatrolConfig, SPRTConfig
from .decision import DecoyScenario, run_decision_mission
from .hypothesis import ALERT, CLEAR
from .perception import OccupancyFilter, class_theta, observe_wide
from .planners import InfoRate, PlanContext, full_map_lawnmower
from .safety import Geofence, RTBMonitor, advance
from .world import Fire, class_map, place_decoys, rasterise_decoys

PATROL_PLANNERS = ("P0", "P1u", "P1")

# Hand-off test: D5 (5-min ring search, CLEAR gate, steering at -log 99) at the
# a-priori thresholds +-log 99. RECOMMENDED's operating points are per satellite
# geometry and do not transfer (results/robustness_viirs_pixel.md); the patrol cue is
# a third geometry -- one exact block, no geolocation error -- and at clear = -15 the
# log-odds of a single-block cue saturates just above the threshold (screening seed 913).
PATROL_SPRT = SPRTConfig(steer_clear=-4.595, dwell_ticks=30, alert=4.595, clear=-4.595)
MAX_UNDECIDED = 2   # hand-offs on one source that end UNDECIDED before it is registered as unresolved


# --------------------------------------------------------------------------- world
class LatentFire(Fire):
    """A fire that ignites at fire step ``ignite_step`` (never if None).

    Before ignition every cell is unburnt and stepping is a no-op apart from the
    RNG draws ``Fire.step`` always makes.
    """

    def __init__(self, n, ignition, wind_dir, cfg, rng, ignite_step: int | None):
        super().__init__(n, ignition, wind_dir, cfg, rng)
        self.ignition = ignition
        self.ignite_step = ignite_step
        self.ignited = ignite_step is not None and ignite_step <= 0
        if not self.ignited:
            self.state[:] = 0

    def step(self) -> None:
        super().step()
        if not self.ignited and self.ignite_step is not None and self.steps >= self.ignite_step:
            self.state[self.ignition] = self.cfg.burn_steps
            self.ignited = True


def risk_map(n: int, allowed: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth risk field in [0.05, 1] on allowed cells, 0 elsewhere."""
    f = gaussian_filter(rng.standard_normal((n, n)), sigma=sigma)
    f = (f - f.min()) / (np.ptp(f) + 1e-12)
    return np.where(allowed, 0.05 + 0.95 * f, 0.0)


@dataclass
class PatrolScenario:
    family: str                 # "F": one latent ignition (+ decoys); "N": decoys only
    fire: LatentFire
    has_fire: bool
    ign_minute: float           # nan for family N
    ignition: tuple[int, int]
    risk: np.ndarray
    decoys: np.ndarray
    rects: list
    wind_dir: float
    sensor_rng: np.random.Generator


def make_patrol_scenario(cfg: MissionConfig, pcfg: PatrolConfig, dcfg: DecoyConfig,
                         family: str, seed: int) -> PatrolScenario:
    if family not in ("F", "N"):
        raise ValueError(f"unknown patrol family {family!r}")
    ss = np.random.SeedSequence([seed, 20])     # stream family disjoint from v1.x seeds
    r_world, r_fire, r_sensor, r_decoy = (np.random.default_rng(s) for s in ss.spawn(4))
    fence = Geofence(cfg.n, cfg.nfz)
    risk = risk_map(cfg.n, fence.allowed, pcfg.hazard_sigma, r_world)
    rr, cc = np.indices((cfg.n, cfg.n))
    far = np.hypot(rr - cfg.base[0], cc - cfg.base[1]) > pcfg.ign_min_dist
    w = np.where(far, risk ** pcfg.hazard_gamma, 0.0).ravel()
    i = int(r_world.choice(w.size, p=w / w.sum()))
    ign = (i // cfg.n, i % cfg.n)
    wind_dir = float(r_world.uniform(0, 2 * np.pi))
    a, b = pcfg.ign_window_min
    mps = cfg.fire.fire_step_ticks * cfg.dt_s / 60.0           # minutes per fire step
    step = int(np.ceil(r_world.integers(a, b) / mps)) if family == "F" else None
    ign_min = step * mps if family == "F" else float("nan")
    fire = LatentFire(cfg.n, ign, wind_dir, cfg.fire, r_fire, step)

    keep = np.zeros((cfg.n, cfg.n), bool)
    keep[max(cfg.base[0] - 4, 0):cfg.base[0] + 5, max(cfg.base[1] - 4, 0):cfg.base[1] + 5] = True
    keep |= _disc(cfg.n, ign, dcfg.min_sep_fire)       # decoys stay clear of the ignition point
    # decoys scale with area: DecoyConfig means are per 64 x 64 map
    scale = (cfg.n / 64) ** 2
    n_w = int(r_decoy.poisson(dcfg.mean_warm * scale))
    n_h = int(r_decoy.poisson(dcfg.mean_hot * scale))
    rects = place_decoys(cfg.n, fence.allowed, keep, n_w, n_h, dcfg.max_side, r_decoy)
    return PatrolScenario(family, fire, family == "F", ign_min, ign, risk,
                          rasterise_decoys(cfg.n, rects), rects, wind_dir, r_sensor)


def _disc(n: int, centre: tuple[int, int], radius: float) -> np.ndarray:
    rr, cc = np.indices((n, n))
    return np.hypot(rr - centre[0], cc - centre[1]) <= radius


# --------------------------------------------------------------------------- filter
class HazardFilter(OccupancyFilter):
    """P(something hot) per cell, with an ignition hazard and a static-source mask.

    Predict: b <- b + (1 - b) h per fire step, and nothing else. The v1.0 spread
    step is deliberately left out: applied to a heat source that has not yet been
    tested, it grows a phantom fire around every decoy, and the hand-off cue drifts
    off the source (seen on screening seeds 900-914). Whether heat spreads is the
    decision test's question, not the patrol filter's.
    Update: exact Bayes, except on masked cells (registered static sources),
    whose readings are explained and ignored.
    """

    def __init__(self, hazard: np.ndarray, cfg: MissionConfig, wind_dir: float):
        p0 = np.clip(hazard, 1e-6, None)
        super().__init__(p0, cfg.sensor, cfg.model, wind_dir, cfg.fire.wind_gain, cfg.fire.burn_steps)
        self.h = hazard
        self.mask = np.zeros(hazard.shape, bool)

    def predict(self) -> None:
        b = self.prob
        b = b + (1 - b) * self.h
        b = np.clip(b, 1e-6, 1 - 1e-6)
        self.L = np.log(b / (1 - b))

    def update(self, fp, z) -> None:
        keep = ~self.mask[fp]
        if keep.all():
            return super().update(fp, z)
        dl = np.where(z, self.l_hit, self.l_miss) * keep
        self.L[fp] += dl
        np.clip(self.L, -15, 15, out=self.L)

    def register_static(self, cells: np.ndarray) -> None:
        self.mask |= cells
        self.L[cells] = np.log(1e-6)


def hazard_per_step(risk: np.ndarray, pcfg: PatrolConfig, uniform: bool,
                    minutes_per_step: float = 1.0) -> np.ndarray:
    """Per-cell ignition probability per fire step, summing to one fire per window."""
    w = (risk > 0).astype(float) if uniform else risk ** pcfg.hazard_gamma
    a, b = pcfg.ign_window_min
    return w / w.sum() / (b - a) * minutes_per_step


def cue_prior(n: int, cell: tuple[int, int], cfg: MissionConfig, pcfg: PatrolConfig) -> np.ndarray:
    """Cue map for the decision test: a cue_pixel^2 'flagged' block centred on the
    hot cell and a ring of the same width, at the satellite prior's levels."""
    k, h = pcfg.cue_pixel, pcfg.cue_pixel // 2
    r, c = cell
    out = np.full((n, n), cfg.prior.p_background)
    out[max(r - h - k, 0):r + h + k, max(c - h - k, 0):c + h + k] = cfg.prior.p_ring
    out[max(r - h, 0):r + h, max(c - h, 0):c + h] = cfg.prior.p_hotspot
    return out


# --------------------------------------------------------------------------- mission
@dataclass
class PatrolEvent:
    tick: int
    kind: str            # "cue" -> decision "ALERT"/"CLEAR"/"UNDECIDED"
    decision: str
    correct: bool | None
    cell: tuple[int, int]
    minutes_spent: float


@dataclass
class PatrolResult:
    planner: str
    family: str
    seed: int
    has_fire: bool
    ign_minute: float
    ign_dist_m: float
    t_first_cue_min: float        # minutes after ignition of the first cue within 4 cells of the fire
    t_alert_min: float            # minutes after ignition of the first correct ALERT (nan if none)
    false_alerts: int
    n_cues: int
    n_clears: int
    n_unresolved: int             # sources registered after MAX_UNDECIDED undecided hand-offs
    fire_cleared: int             # CLEARs issued while the cue block held burning ground
    flight_hours: float
    geofence_violations: int
    landed_safely: int            # sorties that ended on the dock
    sorties: int
    alert_rc: tuple | None = field(repr=False, default=None)
    h1_fire: np.ndarray | None = field(repr=False, default=None)
    events: list = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("alert_rc", "h1_fire", "events")}


def run_patrol(cfg: MissionConfig, pcfg: PatrolConfig, planner: str, family: str, seed: int,
               dcfg: DecoyConfig | None = None, sprt: SPRTConfig | None = None,
               stop_on_alert: bool = True) -> PatrolResult:
    dcfg = dcfg or DecoyConfig()
    sprt = sprt or PATROL_SPRT
    if planner not in PATROL_PLANNERS:
        raise ValueError(f"unknown planner {planner!r}; choose from {PATROL_PLANNERS}")
    sc = make_patrol_scenario(cfg, pcfg, dcfg, family, seed)
    fence = Geofence(cfg.n, cfg.nfz)
    monitor = RTBMonitor(fence, cfg.base, cfg.speed, cfg.reserve_ticks)
    theta = class_theta(cfg.sensor, dcfg)
    mps = cfg.fire.fire_step_ticks * cfg.dt_s / 60.0
    filt = HazardFilter(hazard_per_step(sc.risk, pcfg, uniform=(planner == "P1u"), minutes_per_step=mps),
                        cfg, sc.wind_dir)
    plan = (full_map_lawnmower(cfg.n, cfg.sensor.radius, fence.allowed, cfg.base)
            if planner == "P0" else InfoRate(cfg.sensor))
    fire, rng = sc.fire, sc.sensor_rng
    fs = cfg.fire.fire_step_ticks
    mpt = cfg.dt_s / 60.0

    g = 0                                   # global tick
    pos, path, carry, rtb = cfg.base, [], 0.0, False
    airborne, remaining, ground_left = True, cfg.budget_ticks, 0
    sorties, landed_ok, flown, violations = 1, 0, 0, 0
    events: list[PatrolEvent] = []
    first_cue = t_alert = float("nan")
    false_alerts = n_clears = fire_cleared = n_unresolved = 0
    undecided: dict = {}
    alert_rc, h1_fire = None, None

    def near_fire(cell, radius=4) -> bool:
        aff = fire.affected
        if not aff.any():
            return False
        r, c = cell
        return bool(aff[max(r - radius, 0):r + radius + 1, max(c - radius, 0):c + radius + 1].any())

    def tick_clock() -> None:
        nonlocal g
        g += 1
        if g % fs == 0:
            fire.step()
            filt.predict()

    while g < pcfg.horizon_ticks:
        if not airborne:                     # on the dock: recharge
            tick_clock()
            ground_left -= 1
            if ground_left <= 0:
                airborne, remaining, rtb, path, carry = True, cfg.budget_ticks, False, [], 0.0
                sorties += 1
            continue

        if not rtb and monitor.must_return(pos, remaining):
            rtb, path = True, monitor.path_home(pos)
        if not rtb and (g % cfg.replan_ticks == 0 or not path):
            dist, pred = fence.distances_from(pos)
            feas = monitor.feasible_mask(dist, remaining)
            wp = plan.plan(PlanContext(pos, filt.prob, dist, feas, cfg.speed, g))
            path = fence.path(pred, pos, wp)
        pos, path, carry = advance(pos, path, cfg.speed, carry)
        remaining -= 1
        flown += 1
        violations += int(not fence.contains(pos))
        fp, z = observe_wide(class_map(fire.affected, sc.decoys), pos, theta, cfg.sensor.radius, rng)
        filt.update(fp, z)
        tick_clock()

        if rtb and pos == cfg.base:
            airborne, ground_left = False, pcfg.recharge_ticks
            landed_ok += 1
            continue

        # ------------------------------------------------ cue -> hand-off
        hot = (filt.prob >= pcfg.cue_threshold) & ~filt.mask & fence.allowed
        if rtb or not hot.any():
            continue
        cell = tuple(int(x) for x in np.unravel_index(int(np.argmax(np.where(hot, filt.prob, -1))), hot.shape))
        if sc.has_fire and fire.ignited and np.isnan(first_cue) and near_fire(cell):
            first_cue = g * mpt - sc.ign_minute
        prior = cue_prior(cfg.n, cell, cfg, pcfg)
        dsc = DecoyScenario("A" if sc.has_fire else "B", fire, sc.has_fire, sc.decoys, sc.rects,
                            prior, prior >= cfg.prior.p_hotspot, sc.wind_dir, rng)
        res = run_decision_mission(cfg, "M3", dsc.family, seed, dcfg, sprt, scenario=dsc,
                                   start_pos=pos, start_remaining=remaining, t_offset=g,
                                   return_on_decision=True)
        used = res.ticks_flown
        for _ in range(used):                # the decision loop stepped the fire on the global clock
            g += 1
            if g % fs == 0:
                filt.predict()
        flown += used
        remaining = res.reserve_left
        violations += res.geofence_violations
        pos = res.final_pos
        block = prior >= cfg.prior.p_hotspot
        events.append(PatrolEvent(g, "cue", res.decision, res.alert_correct, cell, used * mpt))
        if res.decision == ALERT:
            ok = bool(res.alert_correct)
            if ok and np.isnan(t_alert):
                t_alert = g * mpt - sc.ign_minute
                alert_rc, h1_fire = res.alert_rc, res.h1_fire
                if stop_on_alert:
                    break
            if not ok:
                false_alerts += 1
            # register what was alerted on so the patrol moves on
            filt.register_static(binary_dilation(block & (filt.prob >= 0.5), iterations=pcfg.mask_dilate)
                                 | _disc(cfg.n, cell, 1))
        elif res.decision == CLEAR:
            n_clears += 1
            if (fire.state[block] > 0).any():
                fire_cleared += 1
            src = block & (filt.prob >= 0.5)
            filt.register_static(binary_dilation(src | _disc(cfg.n, cell, 0), iterations=pcfg.mask_dilate))
        else:   # UNDECIDED: endurance ran out first
            key = (cell[0] // pcfg.cue_pixel, cell[1] // pcfg.cue_pixel)
            undecided[key] = undecided.get(key, 0) + 1
            if undecided[key] >= MAX_UNDECIDED:
                n_unresolved += 1
                filt.register_static(binary_dilation(block & (filt.prob >= 0.5) | _disc(cfg.n, cell, 0),
                                                     iterations=pcfg.mask_dilate))
        if res.returned_to_base and pos == cfg.base:
            airborne, ground_left = False, pcfg.recharge_ticks
            landed_ok += 1

    d_m = float(np.hypot(*np.subtract(sc.ignition, cfg.base)) * cfg.cell_m)
    return PatrolResult(
        planner=planner, family=family, seed=seed, has_fire=sc.has_fire, ign_minute=sc.ign_minute,
        ign_dist_m=d_m if sc.has_fire else float("nan"), t_first_cue_min=first_cue,
        t_alert_min=t_alert, false_alerts=false_alerts, n_cues=len(events), n_clears=n_clears,
        n_unresolved=n_unresolved,
        fire_cleared=fire_cleared, flight_hours=flown * cfg.dt_s / 3600, geofence_violations=violations,
        landed_safely=landed_ok, sorties=sorties, alert_rc=alert_rc, h1_fire=h1_fire, events=events,
    )


def slow_fire(cfg: MissionConfig, factor: int = 10) -> MissionConfig:
    """The same fire, ``factor`` times slower: one fire step every ``factor`` minutes.
    At the default factor the head moves about 2 m/min instead of about 20 m/min
    (25 m cells), the regime in which a small drone's payload can matter."""
    from dataclasses import replace
    return cfg.with_(fire=replace(cfg.fire, fire_step_ticks=cfg.fire.fire_step_ticks * factor))


def scaled(cfg: MissionConfig, n: int) -> MissionConfig:
    """Same scenario on an n x n map; the no-fly box scales with the map."""
    r0, c0, r1, c1 = cfg.nfz
    f = n / cfg.n
    nfz = tuple(int(round(x * f)) for x in (r0, c0, r1, c1))
    return cfg.with_(n=n, nfz=nfz)


__all__ = ["PATROL_PLANNERS", "HazardFilter", "LatentFire", "PatrolResult", "PatrolScenario",
           "cue_prior", "hazard_per_step", "make_patrol_scenario", "risk_map", "run_patrol", "scaled",
           "slow_fire"]
