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

    def with_(self, **kw) -> "MissionConfig":
        return replace(self, **kw)


NOMINAL = MissionConfig()

# Truth has wind, estimator assumes calm air: the idealised-model failure case.
WIND_MISMATCH = MissionConfig(model=ModelConfig(wind_known=False))
