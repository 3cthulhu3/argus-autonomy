"""v2.1: which propagation regime is the fire in?  Byram's convection number N_c.

    N_c = 2 g I / (rho c_p T_a (U - R)^3)                     (Byram 1959; Nelson 1993, INT-415)

  I    Byram fireline intensity at the head, W/m
  U    open wind speed, m/s (here the 20-ft wind the forecaster uses; see below)
  R    head rate of spread, m/s
  rho, c_p, T_a   ambient air density, heat capacity, temperature

N_c compares the power of the fire's buoyant plume with the power of the wind
flowing through the front. Two regimes are reported in the literature, with a
transition between them: wind-driven fires for N_c < 2 and plume-dominated fires
for N_c > 10 (Morvan & Frangieh 2018, IJWF 27:636, as summarised by Accary et al.
2025/26). In a plume-dominated fire the fire makes its own wind; a kinematic
front model driven by the ambient wind is then outside the regime it was built
for. That is the decision-support content of the signal: GREEN means the
forecast is inside its regime of validity, RED means it is not.

What is and is not established (stated here, not discovered later):
  * The thresholds 2 and 10 are empirical, drawn from laboratory and field fires;
    the wind height in N_c is not standardised across studies. This module uses
    the 20-ft wind. A different convention shifts N_c by the cube of the wind ratio.
  * Inside this repository's front world nothing changes regime: the world itself
    is kinematic. What can be tested here is whether the drone's data make the
    *estimate* of N_c, and so the regime call, closer to the truth world's value.
    Whether the call predicts forecast failure can only be tested on real fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81
RHO_AIR = 1.17          # kg/m^3 at ~300 K, sea level (assumption)
CP_AIR = 1005.0         # J/kg/K
T_AMB = 300.0           # K (assumption)
NC_WIND, NC_PLUME = 2.0, 10.0
REGIMES = ("wind-driven", "transitional", "plume-dominated")


def convection_number(i_kwm, u_ms, r_mmin) -> np.ndarray:
    """Byram's N_c; +inf where the wind does not outrun the head (U <= R)."""
    i_w = np.asarray(i_kwm, float) * 1e3
    rel = np.asarray(u_ms, float) - np.asarray(r_mmin, float) / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        nc = 2 * G * i_w / (RHO_AIR * CP_AIR * T_AMB * np.power(np.maximum(rel, 0.0), 3))
    return np.where(rel > 0, nc, np.inf)


def regime_class(nc) -> np.ndarray:
    nc = np.asarray(nc, float)
    return np.where(nc < NC_WIND, 0, np.where(nc <= NC_PLUME, 1, 2))


@dataclass
class RegimeSignal:
    p: tuple[float, float, float]     # P(wind-driven), P(transitional), P(plume-dominated)
    nc_median: float
    nc_q10: float
    nc_q90: float
    light: str                        # GREEN / AMBER / RED

    def line(self) -> str:
        return (f"{self.light}: N_c {self.nc_median:.2g} (80 % {self.nc_q10:.2g}-{self.nc_q90:.2g}); "
                f"P(wind-driven) {self.p[0]:.0%}, P(plume-dominated) {self.p[2]:.0%}")


def regime_signal(nc_members, green_at: float = 0.8, red_at: float = 0.2) -> RegimeSignal:
    """Turn an ensemble of N_c values into a traffic light.

    GREEN  P(wind-driven) >= green_at: the kinematic forecast is inside its regime.
    RED    P(plume-dominated) >= red_at: the forecast should not be trusted for the head.
    AMBER  otherwise.
    The two cut-offs are fixed here, not tuned on results (design doc v2.1).
    """
    nc = np.asarray(nc_members, float)
    cls = regime_class(nc)
    p = tuple(float((cls == j).mean()) for j in range(3))
    finite = np.where(np.isfinite(nc), nc, 1e9)
    q10, q50, q90 = (float(np.quantile(finite, q)) for q in (0.1, 0.5, 0.9))
    light = "GREEN" if p[0] >= green_at else ("RED" if p[2] >= red_at else "AMBER")
    return RegimeSignal(p, q50, q10, q90, light)


def regime_brier(p, true_class: int) -> float:
    """Multi-class Brier score, sum over classes of (p_j - o_j)^2, in [0, 2]."""
    o = np.zeros(3)
    o[int(true_class)] = 1.0
    return float(((np.asarray(p, float) - o) ** 2).sum())


__all__ = ["NC_PLUME", "NC_WIND", "REGIMES", "RegimeSignal", "convection_number", "regime_brier",
           "regime_class", "regime_signal"]
