"""v2.1: Rothermel's surface fire spread model, for the front-scale fire world.

The 25 m cellular automaton of ``world.py`` has no physical inputs (its spread
probability is a free number), so there is nothing in it a drone could measure and
feed back. v2.1 therefore adds a second fire world whose spread rate is a function
of the inputs a drone *can* observe or constrain: wind, dead fuel moisture, slope.

Equations: Rothermel (1972) with Albini's (1976) revisions, as collected in
Andrews (2018), RMRS-GTR-371. Internal units are those of the original model
(lb, ft, min, Btu); the public functions take and return SI.

Definitions (fixed throughout):
  R0      no-wind, no-slope rate of spread, ft/min internally.
  phi_w   wind factor, C U^B (beta/beta_op)^-E, U = midflame wind (ft/min).
  phi_s   slope factor, 5.275 beta^-0.3 tan^2(theta).
  R       R0 (1 + phi_w + phi_s); capped at the effective midflame wind speed
          (Andrews, Cruz & Rothermel 2013 recommend this in place of the original
          wind limit), but never below R0 (our convention: the cap must not make a
          windy fire slower than a calm one).
  I_R     reaction intensity, Btu/ft^2/min.
  t_r     flame residence time, 384 / sigma min (Anderson 1969).
  I_B     Byram fireline intensity, I_R * R * t_r (Albini 1976), reported in kW/m.

Fuel models: five of Anderson's (1982, INT-GTR-122) thirteen, those that occur in
Central European landscapes: short grass (1), open timber with grass (2), closed
timber litter (8), hardwood / long-needle litter (9), timber litter with
understory (10). 10-h and 100-h moisture are taken as 1-h + 1 and + 2 percentage
points; live fuel moisture is fixed at 100 % (the condition of Anderson's table).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- constants
TONS_AC_TO_LB_FT2 = 2000.0 / 43_560.0          # 1 ton/acre in lb/ft^2
FT = 0.3048
FTMIN_TO_MMIN = FT
MS_TO_FTMIN = 60.0 / FT
BTU_FT_S_TO_KW_M = 1.055056 / FT               # 3.4613 kW/m per Btu/ft/s
CHAIN_H_TO_M_MIN = 20.1168 / 60.0              # 1 chain/h in m/min

H_BTU_LB = 8000.0          # heat content
RHO_P = 32.0               # oven-dry particle density, lb/ft^3
S_T = 0.0555               # total mineral content
S_E = 0.010                # effective (silica-free) mineral content
ETA_S = min(0.174 * S_E ** -0.19, 1.0)          # 0.4174
SAV_10H, SAV_100H = 109.0, 30.0
LIVE_MOISTURE = 1.0        # fraction (100 %)


@dataclass(frozen=True)
class FuelModel:
    """One standard fuel model (Anderson 1982, table 1). Loads in tons/acre."""
    code: int
    name: str
    load_1h: float
    load_10h: float
    load_100h: float
    load_live: float
    sav_1h: float
    sav_live: float
    depth_ft: float
    mx_dead: float           # dead fuel moisture of extinction, fraction
    waf: float               # wind adjustment factor, 20-ft wind -> midflame (assumption, see doc)
    ref_ros_ch_h: float      # Anderson (1982) table: 8 % dead, 100 % live, 5 mi/h midflame, flat

    def particles(self):
        """(load lb/ft^2, sav 1/ft, is_live, size index) for the non-empty classes."""
        out = []
        for load, sav, live, j in ((self.load_1h, self.sav_1h, False, 0),
                                   (self.load_10h, SAV_10H, False, 1),
                                   (self.load_100h, SAV_100H, False, 2),
                                   (self.load_live, self.sav_live, True, 3)):
            if load > 0:
                out.append((load * TONS_AC_TO_LB_FT2, sav, live, j))
        return out


# WAF: unsheltered grass beds 0.4; forest litter under a closed canopy 0.15. Both lie
# inside the ranges tabulated by Andrews (2012, RMRS-GTR-266); they are assumptions,
# not site measurements.
FUEL_MODELS: dict[int, FuelModel] = {
    1: FuelModel(1, "short grass", 0.74, 0.0, 0.0, 0.0, 3500, 1500, 1.0, 0.12, 0.40, 78.0),
    2: FuelModel(2, "timber, grass and understory", 2.0, 1.0, 0.5, 0.5, 3000, 1500, 1.0, 0.15, 0.40, 35.0),
    8: FuelModel(8, "closed timber litter", 1.5, 1.0, 2.5, 0.0, 2000, 1500, 0.2, 0.30, 0.15, 1.6),
    9: FuelModel(9, "hardwood litter", 2.92, 0.41, 0.15, 0.0, 2500, 1500, 0.2, 0.25, 0.15, 7.5),
    10: FuelModel(10, "timber, litter and understory", 3.01, 2.0, 5.01, 2.0, 2000, 1500, 1.0, 0.25, 0.15, 7.9),
}
FUEL_CODES = (1, 2, 8, 9, 10)


def _size_class(sav: float) -> int:
    """Albini (1976) SAV subclasses used for the net-load weights g_ij."""
    for k, lo in enumerate((1200.0, 192.0, 96.0, 48.0, 16.0)):
        if sav >= lo:
            return k
    return 5


@dataclass(frozen=True)
class FuelBed:
    """Moisture-independent quantities of a fuel bed, plus the pieces needed per moisture."""
    fm: FuelModel
    sigma: float             # characteristic SAV, 1/ft
    beta: float              # packing ratio
    beta_rel: float          # beta / beta_op
    rho_b: float             # bulk density, lb/ft^3
    gamma: float             # optimum reaction velocity Gamma', 1/min
    xi: float                # propagating flux ratio
    C: float
    B: float
    E: float
    phi_s_coef: float        # 5.275 beta^-0.3
    t_r: float               # residence time, min


def fuel_bed(fm: FuelModel) -> FuelBed:
    parts = fm.particles()
    a = {False: 0.0, True: 0.0}
    for w, s, live, _ in parts:
        a[live] += s * w / RHO_P
    a_t = a[False] + a[True]
    sigma = 0.0
    for w, s, live, _ in parts:
        f_ij = (s * w / RHO_P) / a[live]
        f_i = a[live] / a_t
        sigma += f_i * f_ij * s
    w_total = sum(p[0] for p in parts)
    rho_b = w_total / fm.depth_ft
    beta = rho_b / RHO_P
    beta_op = 3.348 * sigma ** -0.8189
    rel = beta / beta_op
    gmax = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    A = 133.0 * sigma ** -0.7913
    gamma = gmax * rel ** A * np.exp(A * (1.0 - rel))
    xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    C = 7.47 * np.exp(-0.133 * sigma ** 0.55)
    B = 0.02526 * sigma ** 0.54
    E = 0.715 * np.exp(-3.59e-4 * sigma)
    return FuelBed(fm, sigma, beta, rel, rho_b, float(gamma), float(xi), float(C), float(B), float(E),
                   float(5.275 * beta ** -0.3), 384.0 / sigma)


BEDS: dict[int, FuelBed] = {c: fuel_bed(m) for c, m in FUEL_MODELS.items()}


def _eta_m(m, mx):
    r = np.clip(np.asarray(m, float) / mx, 0.0, 1.0)
    eta = 1.0 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3
    return np.where(r >= 1.0, 0.0, np.maximum(eta, 0.0))     # the cubic is 0 at r = 1 up to rounding


def no_wind_spread(code: int, m1h) -> tuple[np.ndarray, np.ndarray]:
    """R0 (m/min) and reaction intensity I_R (kW/m^2) for 1-h dead moisture ``m1h`` (fraction).

    Vectorised over ``m1h``. Moisture of the 10-h and 100-h classes is m1h + 0.01 and
    + 0.02; live moisture is 100 %.
    """
    bed = BEDS[code]
    fm = bed.fm
    m1h = np.asarray(m1h, float)
    parts = fm.particles()
    moist = {0: m1h, 1: m1h + 0.01, 2: m1h + 0.02, 3: np.full_like(m1h, LIVE_MOISTURE)}
    a = {False: 0.0, True: 0.0}
    for w, s, live, _ in parts:
        a[live] += s * w / RHO_P
    a_t = a[False] + a[True]

    # live moisture of extinction (Albini 1976 / Andrews 2018 eq. 88)
    mx = {False: np.full_like(m1h, fm.mx_dead), True: None}
    if a[True] > 0:
        dead_fine = sum(w * np.exp(-138.0 / s) for w, s, live, _ in parts if not live)
        live_fine = sum(w * np.exp(-500.0 / s) for w, s, live, _ in parts if live)
        mf_dead = sum(w * np.exp(-138.0 / s) * moist[j] for w, s, live, j in parts if not live) / dead_fine
        mx_live = 2.9 * (dead_fine / live_fine) * (1.0 - mf_dead / fm.mx_dead) - 0.226
        mx[True] = np.maximum(mx_live, fm.mx_dead)

    reaction = np.zeros_like(m1h)
    heat_sink = np.zeros_like(m1h)
    for live in (False, True):
        if a[live] == 0:
            continue
        cls = [p for p in parts if p[2] == live]
        f_ij = [(s * w / RHO_P) / a[live] for w, s, _, _ in cls]
        f_i = a[live] / a_t
        # Albini's net-load weights: sum of f_ij over particles in the same SAV subclass
        sub = [_size_class(s) for _, s, _, _ in cls]
        g_ij = [sum(f for f, k2 in zip(f_ij, sub) if k2 == k) for k in sub]
        w_n = sum(g * w * (1.0 - S_T) for g, (w, _, _, _) in zip(g_ij, cls))
        m_f = sum(f * moist[j] for f, (_, _, _, j) in zip(f_ij, cls))
        reaction = reaction + w_n * H_BTU_LB * _eta_m(m_f, mx[live]) * ETA_S
        heat_sink = heat_sink + f_i * sum(
            f * np.exp(-138.0 / s) * (250.0 + 1116.0 * moist[j]) for f, (_, s, _, j) in zip(f_ij, cls))
    i_r = bed.gamma * reaction                                   # Btu/ft^2/min
    r0 = i_r * bed.xi / (bed.rho_b * heat_sink)                  # ft/min
    return r0 * FTMIN_TO_MMIN, i_r * 1.055056 / (FT * FT) / 60.0  # m/min, kW/m^2


def wind_factor(code: int, u_mid_ms) -> np.ndarray:
    """phi_w for a midflame wind (m/s), vectorised."""
    bed = BEDS[code]
    u = np.maximum(np.asarray(u_mid_ms, float), 0.0) * MS_TO_FTMIN
    return bed.C * u ** bed.B * bed.beta_rel ** -bed.E


def slope_factor(code: int, tan_slope) -> np.ndarray:
    t = np.maximum(np.asarray(tan_slope, float), 0.0)
    return BEDS[code].phi_s_coef * t * t


def effective_wind_ms(code: int, phi_e) -> np.ndarray:
    """Midflame wind (m/s) that alone would produce wind factor ``phi_e``."""
    bed = BEDS[code]
    phi = np.maximum(np.asarray(phi_e, float), 0.0)
    u = (phi * bed.beta_rel ** bed.E / bed.C) ** (1.0 / bed.B)
    return u / MS_TO_FTMIN


def spread_rate(code: int, m1h, u20_ms, tan_slope=0.0, cap: bool = True) -> np.ndarray:
    """Head rate of spread (m/min) for 20-ft wind ``u20_ms`` blowing upslope (or on the flat)."""
    r0, _ = no_wind_spread(code, m1h)
    fm = FUEL_MODELS[code]
    phi = wind_factor(code, fm.waf * np.asarray(u20_ms, float)) + slope_factor(code, tan_slope)
    r = r0 * (1.0 + phi)
    if cap:
        r = np.minimum(r, np.maximum(effective_wind_ms(code, phi) * 60.0, r0))
    return r


def byram_intensity(code: int, m1h, ros_mmin) -> np.ndarray:
    """Byram fireline intensity (kW/m) = I_R * R * t_r (Albini 1976)."""
    _, i_r = no_wind_spread(code, m1h)                         # kW/m^2
    # R * t_r is the flame depth in m, so kW/m^2 * m = kW/m
    return i_r * np.asarray(ros_mmin, float) * BEDS[code].t_r


def flame_length_m(i_kwm) -> np.ndarray:
    """Byram (1959): L = 0.0775 I^0.46 (m, kW/m), as in suppression.py."""
    return 0.0775 * np.power(np.maximum(np.asarray(i_kwm, float), 0.0), 0.46)


__all__ = ["BEDS", "CHAIN_H_TO_M_MIN", "FUEL_CODES", "FUEL_MODELS", "FuelModel", "byram_intensity",
           "effective_wind_ms", "flame_length_m", "no_wind_spread", "slope_factor", "spread_rate",
           "wind_factor"]
