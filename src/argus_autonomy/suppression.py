"""v2.0 response layer, part 2: can a drone from the dock knock the fire down?

This is a feasibility model, not a fire simulation. The 25 m cellular automaton
of the rest of the package is far too coarse for incipient-fire suppression (one
cell is already 625 m2), so this module works at front scale with closed-form
fire growth and a momentum-theory multirotor. Every relation is named with its
source; every number that is an assumption says so (``Assumptions``).

Definitions (fixed throughout):
  R(t)    head rate of spread, m/min.  R(t) = R_ss (1 - exp(-t / tau)): a fire
          accelerating from a point to its quasi-steady rate (the form used in the
          point-source acceleration literature, e.g. McAlpine & Wakimoto 1991; tau
          is NOT taken from it and is an explicit assumption). tau = 0 means
          no acceleration (worst case).
  shape   ellipse with the ignition at the rear focus. Length-to-breadth ratio
          LB(U) = 0.936 e^{0.2566 U} + 0.461 e^{-0.1548 U} - 0.397, U midflame wind
          in mph (Anderson 1983, as adjusted in FARSITE; LB(0) = 1 exactly).
          Head-to-back ratio HB = (LB + sqrt(LB^2 - 1)) / (LB - sqrt(LB^2 - 1))
          (Alexander 1985).
  I       Byram fireline intensity, I = H w R (kW/m), H = 18 600 kJ/kg.
  FL      flame length, FL = 0.0775 I^0.46 (Byram 1959), metres.
  cf      critical water flow per metre of front, L/s/m, from the 2 m flame-depth
          fit of Penney, Habibi & Cattani (2019, Fire 2(1):3):
          CF_10m = 2.72 RoS^0.42 (L/s per 10 m, RoS in km/h). Derived for head
          fires in Australian forest fuels; applying it to the whole perimeter of
          an incipient fire is conservative (flanks burn less intensely).
  q_k     knockdown water per metre of perimeter = cf * t_ext, where t_ext is the
          time the critical flow must be sustained to extinguish (an assumption).
  P(m)    electrical power of a multirotor of mass m: ideal induced power from
          momentum theory, (m g)^1.5 / sqrt(2 rho A), divided by figure of merit
          and drive efficiency, times a cruise factor. Forward-flight power of a
          multirotor at moderate speed is below hover power; k_cruise <= 1 makes
          the model optimistic, not pessimistic.

Direct-attack limit: flame length 1.2 m (4 ft) is the hand-crew direct-attack
limit of the fire-behaviour interpretation chart (Andrews & Rothermel 1982,
GTR INT-131); above it the policy only cues ground resources.

Containment race (multi-sortie): the untreated perimeter U grows like the free
ellipse's perimeter, in proportion to the untreated share; each sortie removes
V / q_k metres; the fire is contained when U <= 0. Treated perimeter is assumed
to stay out (no re-ignition): optimistic. Growth of the free ellipse is used for
the untreated share: roughly neutral. This is a simplified form of the
line-production-versus-perimeter-growth race (Fried & Fried 1996).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81
RHO = 1.225
H_COMB = 18_600.0            # kJ/kg (Penney et al. 2019 use the same value)
MPH_PER_KMH = 1 / 1.609344
FL_DIRECT_ATTACK = 1.2       # m, Andrews & Rothermel (1982)
LB_MAX = 8.0                 # cap on the length-to-breadth ratio (assumption; not reached for U <= 10 km/h)


# --------------------------------------------------------------------------- fire
def lb_ratio(u_kmh: float | np.ndarray) -> np.ndarray:
    u = np.asarray(u_kmh, float) * MPH_PER_KMH
    lb = 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397
    return np.clip(lb, 1.0, LB_MAX)


def hb_ratio(lb: float | np.ndarray) -> np.ndarray:
    lb = np.asarray(lb, float)
    s = np.sqrt(np.maximum(lb * lb - 1.0, 0.0))
    return (lb + s) / (lb - s)


def head_distance(t_min, r_ss: float, tau_min: float) -> np.ndarray:
    """Distance (m) the head has travelled t minutes after a point ignition."""
    t = np.asarray(t_min, float)
    if tau_min <= 0:
        return r_ss * t
    return r_ss * (t - tau_min * (1.0 - np.exp(-t / tau_min)))


def ros(t_min, r_ss: float, tau_min: float) -> np.ndarray:
    t = np.asarray(t_min, float)
    return r_ss * np.ones_like(t) if tau_min <= 0 else r_ss * (1.0 - np.exp(-t / tau_min))


def ellipse_perimeter(a, b) -> np.ndarray:
    """Ramanujan's second approximation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = ((a - b) / np.maximum(a + b, 1e-12)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def perimeter(t_min, r_ss: float, tau_min: float, u_kmh: float) -> np.ndarray:
    lb = lb_ratio(u_kmh)
    head = head_distance(t_min, r_ss, tau_min)
    a = 0.5 * head * (1 + 1 / hb_ratio(lb))
    return ellipse_perimeter(a, a / lb)


def intensity(r_mpm, w_kgm2: float) -> np.ndarray:
    return H_COMB * w_kgm2 * np.asarray(r_mpm, float) / 60.0


def flame_length(i_kwm) -> np.ndarray:
    return 0.0775 * np.power(np.maximum(np.asarray(i_kwm, float), 0.0), 0.46)


def critical_flow_per_m(r_mpm) -> np.ndarray:
    """L/s per metre of front (Penney et al. 2019, 2 m flame-depth fit)."""
    kmh = np.maximum(np.asarray(r_mpm, float), 0.0) * 0.06
    return 0.272 * np.power(kmh, 0.42)


# --------------------------------------------------------------------------- airframe
@dataclass(frozen=True)
class Airframe:
    """An illustrative heavy-lift multirotor, not a specific product.

    The defaults give about 12 min of hover at the full 40 L and 41 min empty.
    Dealer figures for 40 L agricultural spray drones are about 7-10 min fully
    loaded (unofficial; the manufacturer publishes none), so the model is
    deliberately optimistic: a negative result is robust to it, a positive one is
    an upper bound.
    """
    m_frame: float = 18.0         # kg, everything except battery and water
    m_batt: float = 12.0          # kg
    e_pack: float = 180.0         # Wh/kg at pack level (~250 Wh/kg cells)
    dod: float = 0.8              # usable fraction of pack energy
    n_rotors: int = 6
    d_rotor: float = 1.2          # m
    fm: float = 0.60              # rotor figure of merit
    eta_drive: float = 0.85       # motor + ESC
    k_cruise: float = 0.9         # cruise power / hover power
    p_avionics: float = 100.0     # W
    v_cruise: float = 10.0        # m/s
    payload_max: float = 40.0     # L of water (1 kg/L)
    t_launch: float = 30.0        # s from alarm to airborne
    t_drop: float = 30.0          # s over the fire per sortie
    t_turn: float = 120.0         # s on the dock between sorties: refill + battery swap (assumed automated)
    reserve: float = 0.2          # energy fraction kept in reserve

    @property
    def disk_area(self) -> float:
        return self.n_rotors * np.pi * (self.d_rotor / 2) ** 2

    @property
    def m_empty(self) -> float:
        return self.m_frame + self.m_batt

    @property
    def energy_j(self) -> float:
        return self.m_batt * self.e_pack * 3600.0 * self.dod

    def power(self, mass: float, cruise: bool = True) -> float:
        p_ind = (mass * G) ** 1.5 / np.sqrt(2 * RHO * self.disk_area)
        p = p_ind / (self.fm * self.eta_drive)
        return (self.k_cruise if cruise else 1.0) * p + self.p_avionics

    def sortie_energy(self, payload_l: float, dist_m: float) -> float:
        t_leg = dist_m / self.v_cruise
        full = self.m_empty + payload_l
        return (self.power(full) * t_leg + self.power(full, cruise=False) * self.t_drop
                + self.power(self.m_empty) * t_leg)

    def max_payload(self, dist_m: float) -> float:
        """Largest water load (L) that can be carried to dist_m, dropped and flown back."""
        budget = self.energy_j * (1 - self.reserve)
        if self.sortie_energy(0.0, dist_m) > budget:
            return 0.0
        if self.sortie_energy(self.payload_max, dist_m) <= budget:
            return self.payload_max
        lo, hi = 0.0, self.payload_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.sortie_energy(mid, dist_m) <= budget else (lo, mid)
        return lo

    def endurance_min(self, payload_l: float) -> float:
        return self.energy_j / self.power(self.m_empty + payload_l) / 60.0


# --------------------------------------------------------------------------- attack
@dataclass(frozen=True)
class FireCase:
    r_ss: float                  # m/min, quasi-steady head rate of spread
    w: float = 0.6               # kg/m2 of fuel consumed in flaming
    u_kmh: float = 5.0           # midflame wind
    tau_min: float = 15.0        # acceleration time scale (assumption)
    t_ext_s: float = 30.0        # knockdown time at critical flow (assumption)


@dataclass
class AttackOutcome:
    t_arrive_min: float          # minutes after ignition, first drop
    perimeter_m: float           # at first drop
    flame_m: float               # at first drop
    payload_l: float
    need_l: float                # water to knock the whole perimeter down at first drop
    knockdown_first: bool        # one sortie suffices (and flame length allows direct attack)
    contained_min: float         # minutes after ignition, nan if not within the sortie limit
    sorties: int
    decision: str                # "ATTACK" or "CUE"
    reason: str = ""


def attack(fire: FireCase, t_alert_min: float, dist_m: float, ac: Airframe = Airframe(),
           n_drones: int = 1, max_sorties: int = 6, fl_max: float = FL_DIRECT_ATTACK) -> AttackOutcome:
    """Dock-launched direct attack on a fire alerted ``t_alert_min`` after ignition,
    ``dist_m`` from the dock. Drones fly in rotation; each sortie lands ``V`` litres."""
    v = ac.max_payload(dist_m)
    leg = dist_m / ac.v_cruise
    t1 = t_alert_min + (ac.t_launch + leg) / 60.0
    cycle = (2 * leg + ac.t_drop + ac.t_turn) / 60.0       # one drone, minutes between its drops
    spacing = cycle / n_drones

    r1 = float(ros(t1, fire.r_ss, fire.tau_min))
    p1 = float(perimeter(t1, fire.r_ss, fire.tau_min, fire.u_kmh))
    fl1 = float(flame_length(intensity(r1, fire.w)))
    qk1 = float(critical_flow_per_m(r1)) * fire.t_ext_s
    need = p1 * qk1
    if v <= 0:
        return AttackOutcome(t1, p1, fl1, v, need, False, float("nan"), 0, "CUE", "out of range")

    u_left, t, prev_p, contained = p1, t1, p1, float("nan")
    k = 0
    for k in range(1, max_sorties + 1):
        t = t1 + (k - 1) * spacing
        p = float(perimeter(t, fire.r_ss, fire.tau_min, fire.u_kmh))
        if k > 1:
            u_left += (p - prev_p) * (u_left / max(prev_p, 1e-9))
        prev_p = p
        r = float(ros(t, fire.r_ss, fire.tau_min))
        if float(flame_length(intensity(r, fire.w))) > fl_max:
            break                                             # direct attack no longer possible
        u_left -= v / max(float(critical_flow_per_m(r)) * fire.t_ext_s, 1e-9)
        if u_left <= 0:
            contained = t
            break
    first = bool(fl1 <= fl_max and v >= need)
    if fl1 > fl_max:
        return AttackOutcome(t1, p1, fl1, v, need, False, float("nan"), 0, "CUE", "flame length")
    dec = "ATTACK" if np.isfinite(contained) else "CUE"
    return AttackOutcome(t1, p1, fl1, v, need, first, contained, k, dec,
                         "" if dec == "ATTACK" else "not contained within sortie limit")


# --------------------------------------------------------------------------- Monte Carlo
@dataclass(frozen=True)
class Assumptions:
    """Ranges for the quantities the literature above does not pin down."""
    w: tuple = (0.3, 1.2)            # kg/m2 consumed
    u_kmh: tuple = (0.0, 10.0)      # midflame, i.e. well below the 10 m wind
    tau_min: tuple = (5.0, 30.0)
    t_ext_s: tuple = (15.0, 60.0)


def sample_cases(r_ss: float, n: int, rng: np.random.Generator, a: Assumptions = Assumptions(),
                 tau_override: float | None = None) -> list[FireCase]:
    u = lambda lo_hi: rng.uniform(*lo_hi, size=n)   # noqa: E731
    w, wind, tau, te = u(a.w), u(a.u_kmh), u(a.tau_min), u(a.t_ext_s)
    if tau_override is not None:
        tau = np.full(n, tau_override)
    return [FireCase(r_ss, float(w[i]), float(wind[i]), float(tau[i]), float(te[i])) for i in range(n)]


def success_rates(cases: list[FireCase], t_alert_min: float, dist_m: float, ac: Airframe = Airframe(),
                  n_drones: int = 1, max_sorties: int = 6) -> dict:
    outs = [attack(c, t_alert_min, dist_m, ac, n_drones, max_sorties) for c in cases]
    return {"p_knockdown_first": float(np.mean([o.knockdown_first for o in outs])),
            "p_contained": float(np.mean([np.isfinite(o.contained_min) for o in outs])),
            "p_flame_ok": float(np.mean([o.flame_m <= FL_DIRECT_ATTACK for o in outs])),
            "payload_l": outs[0].payload_l}


__all__ = ["Airframe", "AttackOutcome", "Assumptions", "FL_DIRECT_ATTACK", "FireCase", "attack",
           "critical_flow_per_m", "ellipse_perimeter", "flame_length", "hb_ratio", "head_distance",
           "intensity", "lb_ratio", "perimeter", "ros", "sample_cases", "success_rates"]
