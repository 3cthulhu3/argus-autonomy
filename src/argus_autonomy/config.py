"""Scenario and algorithm parameters.

All spatial quantities are in grid cells unless the name says otherwise;
all temporal quantities are in ticks (one tick = ``dt_s`` seconds).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class SensorConfig:
    radius: int = 3            # footprint half-width in cells -> (2r+1)^2 cells
    p_detect: float = 0.85     # P(z=1 | cell affected)
    p_false_alarm: float = 0.03  # P(z=1 | cell not affected)


@dataclass(frozen=True)
class FireConfig:
    p_spread: float = 0.10       # base ignition probability per burning neighbour per fire step
    wind_gain: float = 1.6       # exponent gain on cos(angle to wind)
    burn_steps: int = 8          # fire steps a cell burns before it is burnt out
    fire_step_ticks: int = 6     # ticks per fire-model step (6 x 10 s = 1 min)
    preburn_steps: int = 15      # fire steps between ignition and drone launch
    fuel_min: float = 0.3
    fuel_max: float = 1.3


@dataclass(frozen=True)
class PriorConfig:
    coarse_factor: int = 8        # satellite pixel = 8 x 8 cells (200 m at 25 m cells)
    detect_fraction: float = 0.05  # pixel flagged if >= 5 % of it is affected
    latency_steps: int = 6         # overpass happened this many fire steps before launch
    geoloc_sigma: float = 2.0      # std of geolocation error, in cells
    p_hotspot: float = 0.25
    p_ring: float = 0.02
    p_background: float = 0.002


@dataclass(frozen=True)
class ModelConfig:
    """What the onboard estimator *believes* about fire spread."""
    p_spread: float = 0.10
    wind_gain: float | None = None  # None -> use the true wind gain (matched model)
    wind_known: bool = True         # False -> estimator assumes calm air
    fuel: float = 0.8               # estimator assumes homogeneous mean fuel


@dataclass(frozen=True)
class MissionConfig:
    n: int = 64                  # grid is n x n cells
    cell_m: float = 25.0
    dt_s: float = 10.0
    speed: float = 4.0           # cells per tick (10 m/s)
    budget_ticks: int = 180      # 30 min endurance
    reserve_ticks: int = 6       # landing reserve the safety monitor keeps
    base: tuple[int, int] = (2, 2)
    nfz: tuple[int, int, int, int] = (40, 8, 52, 22)  # no-fly box (r0, c0, r1, c1), exclusive end
    replan_ticks: int = 2
    confirm_threshold: float = 0.95
    sensor: SensorConfig = field(default_factory=SensorConfig)
    fire: FireConfig = field(default_factory=FireConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def with_(self, **kw) -> MissionConfig:
        return replace(self, **kw)


NOMINAL = MissionConfig()


# ----------------------------------------------------------------- v1.1: decoys
# Latent cell classes. Index order is used everywhere (arrays of shape (K, n, n)).
EMPTY, FIRE, WARM, HOT = 0, 1, 2, 3
CLASS_NAMES = ("empty", "fire", "warm-decoy", "hot-decoy")
K_CLASSES = 4


@dataclass(frozen=True)
class DecoyConfig:
    """Persistent, non-spreading heat sources and how the scenario places them.

    `warm` = sunlit rock, vehicle, roof; `hot` = contained burn, campfire.
    The wide thermal camera's hit rate for empty and fire cells is taken from
    SensorConfig, so a scenario without decoys is exactly the v1.0 scenario.
    """
    theta_warm: float = 0.70       # P(z=1 | warm decoy), wide thermal look
    theta_hot: float = 0.85        # P(z=1 | hot decoy) -- identical to fire by design
    mean_warm: float = 1.5         # Poisson mean number of warm decoys per scenario
    mean_hot: float = 1.0          # Poisson mean number of hot decoys per scenario
    max_side: int = 2              # decoys are 1..max_side x 1..max_side cells
    min_sep_fire: int = 6          # cells from the fire at launch (family A)
    p_sat_warm: float = 0.5        # probability a warm decoy shows in the satellite overpass
    max_tries: int = 1000          # rejection sampling budget for family B (needs a cue)
    # Close look (hover/descend, 3x3 footprint): radiometric reading in {cold, warm, hot}.
    # Rows: empty, fire, warm decoy, hot decoy. Fire and hot decoy are identical by design
    # (decision D2): only growth separates them.
    close_lik: tuple = ((0.90, 0.08, 0.02),
                        (0.05, 0.15, 0.80),
                        (0.10, 0.80, 0.10),
                        (0.05, 0.15, 0.80))
    close_radius: int = 1
    t_close: int = 2               # ticks spent hovering for one close look


@dataclass(frozen=True)
class SPRTConfig:
    """Two-model sequential test: H1 = a spreading fire exists, H0 = nothing spreads.

    rho_* are the estimator's background densities of each decoy class per
    cell; the defaults match the scenario generator's expected decoy area
    (mean count x mean size 2.25 cells / 4096 cells).
    """
    prior_h1: float = 0.5           # P(H1) at dispatch
    alert: float = 4.595            # ALERT when log-odds >= alert   (log 99)
    clear: float = -4.595           # CLEAR when log-odds <= clear   (-log 99)
    rho_warm: float = 8.2e-4
    rho_hot: float = 5.5e-4
    # Share of the satellite cue H0 attributes to *warm* decoys. None (default) ->
    # matched to the scenario generator (~0.43). 0 puts the whole cue on hot decoys
    # (sensor response equal to fire's); a 60-seed smoke test favoured it, the full
    # benchmark did not: more false alerts, no fewer missed fires (results/decoys_m1.md).
    omega: float | None = None

    # Share of the satellite cue's excess that H1 attributes to fire (the rest to
    # decoys, split like H0's). 1 = v1.1 milestone (i): under H1 every cue cell is fire.
    phi: float = 1.0
    # CLEAR only once every cell of this cue region has been seen: None, "hotspot"
    # (flagged pixels) or "cue" (flagged pixels + ring). Not a test parameter but a
    # stopping rule: absence of evidence in unseen cue cells is not evidence of absence.
    clear_gate: str | None = "hotspot"   # adopted in milestone (ii); None = milestone (i)
    # Threshold that triggers "the test would clear but the cue is unseen: go look there"
    # steering. None -> same as `clear` (milestones i-ii). Milestone (iii) fixes it at
    # -log 99 so that the pre-decision policy does not depend on (alert, clear), which
    # makes every threshold pair's decision a first-passage time of one logged run.
    steer_clear: float | None = None
    # v1.2 ring-aware stopping rules (0 = off = v1.1). The ring is the band of pixels
    # around the flagged ones (prior p_ring), where the coverage misses of v1.1 lie.
    #   ring_cover   CLEAR needs at least this fraction of allowed ring cells seen.
    #   dwell_ticks  CLEAR needs the log-odds to have stayed <= steer_clear, with the
    #                flagged pixels seen, for this many ticks while the planner searches
    #                the ring; waived once every ring cell has been seen.
    ring_cover: float = 0.0
    dwell_ticks: int = 0
    # v1.3 fire hypothesis. "cell" (v1.1): independent fire mass on every cue cell.
    # "object": H0 plus ONE fire blob of radius obj_radius (cells) at an unknown centre on a
    # lattice of spacing obj_step over the cue and its ring, fire probability obj_p_blob
    # inside the blob; prior weights follow the satellite excess (mixture.py).
    h1_model: str = "cell"
    obj_radius: float = 3.0
    obj_step: int = 4
    obj_p_blob: float = 0.6

    def omega_for(self, decoys: DecoyConfig) -> float:
        if self.omega is not None:
            return self.omega
        w = decoys.p_sat_warm * self.rho_warm
        return w / (w + self.rho_hot)


FAMILIES = ("A", "B", "C")  # A: fire + decoys, B: decoys only (cue present), C: fire only

# Truth has wind, estimator assumes calm air: the idealised-model failure case.
WIND_MISMATCH = MissionConfig(model=ModelConfig(wind_known=False))

# Satellite geometry with VIIRS-sized pixels (400 m, same absolute detection area as NOMINAL).
VIIRS_PIXEL = replace(NOMINAL, prior=replace(NOMINAL.prior, coarse_factor=16, detect_fraction=3.2 / 256))

# Recommended decision configuration, adopted in v1.2b under a pre-registered criterion
# (docs/design/v1.2-coverage-misses.md, §8-9): method M3, CLEAR gate on the flagged
# pixels, steering at -log 99, and a 5-minute ring search (D5) before any CLEAR.
# Thresholds are per satellite geometry, because they do not transfer between them.
# SPRTConfig's own defaults are left at their v1.1 values so that every published
# v1.0/v1.1 result still reproduces from the scripts that created it.
RECOMMENDED = {
    "NOMINAL": SPRTConfig(steer_clear=-4.595, dwell_ticks=30, alert=6.0, clear=-15.0),
    "VIIRS_PIXEL": SPRTConfig(steer_clear=-4.595, dwell_ticks=30, alert=2.0, clear=-30.0),
}
