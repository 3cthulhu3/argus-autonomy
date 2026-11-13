"""v2.1: what a drone's observations do to a fire-spread forecast -- an ensemble Kalman filter.

The world (``make_world``) is a level-set fire driven by Rothermel spread rates on
a 4 km landscape with five Anderson fuel models, terrain, a spatially varying
wind with a possible wind shift, a spatially varying dead fuel moisture and an
unresolved rate-of-spread noise field. The forecaster knows the terrain, a fuel
map with patchy misclassification, and prior inputs of NWP-like quality.

The drone, on station from the alert, observes every ``obs_every_min`` minutes:
  perimeter   points of the true front inside its footprint, with position noise;
  wind        the local wind at its position, with noise;
  FRP         fire radiative power per metre of flaming front around it, chi_r * mean I_B.

Three forecasters see the same ensemble draws (paired):
  OL  open loop: initialised from the first mapped perimeter, run with prior inputs.
  SO  state only: EnKF update of the front (psi) from perimeter points; inputs stay at the prior.
  JE  joint: EnKF update of the front *and* the inputs (wind, moisture, ROS factor) from all three
      observation types (augmented-state EnKF; Evensen 2009; for fire, Mandel et al. 2009,
      IEEE Control Systems 29(3):47).

Analysis: stochastic EnKF with perturbed observations (Burgers, van Leeuwen & Evensen 1998)
and Gaspari-Cohn (1999) localisation of front cells against perimeter points. The inputs
are global per member, so their updates are not localised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from . import regime as rg
from .levelset import (FrontConfig, FrontModel, Inputs, Landscape, cell_xy, front_points, ignite,
                       reinit, sample)
from .response import forecast_scores

FUELS = (1, 2, 8, 9, 10)
# physical bounds on the members' inputs (stated assumptions, applied after every draw and update)
WIND_MAX_MS = 12.0        # 20-ft wind; the world draws mean winds of 1-7 m/s
M_BOUNDS = (0.02, 0.30)
LA_BOUNDS = (-1.0, 1.0)   # ROS factor between 0.37 and 2.7


def _bound(ens: "Ensemble") -> None:
    sp = np.hypot(ens.wind[:, 0], ens.wind[:, 1])
    ens.wind *= (np.minimum(sp, WIND_MAX_MS) / np.maximum(sp, 1e-12))[:, None]
    ens.m = np.clip(ens.m, *M_BOUNDS)
    ens.la = np.clip(ens.la, *LA_BOUNDS)


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class WorldConfig:
    front: FrontConfig = field(default_factory=FrontConfig)
    fuel_weights: tuple[float, ...] = (0.25, 0.20, 0.20, 0.15, 0.20)   # for FUELS
    nonburnable_frac: float = 0.03
    relief_m: float = 40.0              # std of terrain height
    wind_range: tuple[float, float] = (1.0, 7.0)        # mean 20-ft wind speed, m/s
    m1h_range: tuple[float, float] = (0.04, 0.12)       # mean dead 1-h moisture
    m_field_sd: float = 0.01
    wind_dir_field_sd_deg: float = 10.0
    wind_speed_field_rel_sd: float = 0.15
    ros_noise_sd: float = 0.20          # log-normal, spatially correlated
    p_wind_shift: float = 0.4
    shift_deg: tuple[float, float] = (30.0, 70.0)
    shift_time_min: tuple[float, float] = (20.0, 40.0)
    misclass_frac: float = 0.10
    chi_r_range: tuple[float, float] = (0.10, 0.20)     # true radiative fraction (FRP / HRR)


@dataclass(frozen=True)
class DroneConfig:
    t_alert_min: float = 10.0           # first full perimeter map
    obs_every_min: float = 5.0
    t_last_obs_min: float = 40.0
    footprint_m: float = 300.0
    max_points: int = 30
    pos_sd_m: float = 10.0              # geolocation error of a perimeter point
    wind_sd_ms: float = 0.7
    frp_log_sd: float = 0.2
    chi_r_model: float = 0.15


@dataclass(frozen=True)
class FilterConfig:
    k: int = 40
    # prior input uncertainty (what an NWP + fuel-moisture model would give)
    prior_speed_logsd: float = 0.30
    prior_dir_sd_deg: float = 25.0
    prior_m_sd: float = 0.02
    prior_la_sd: float = 0.30
    init_pos_sd_m: float = 10.0
    # observation error used by the filter
    r_perim_m: float = 15.0
    r_wind_ms: float = 1.0
    r_frp_log: float = 0.40
    loc_m: float = 150.0                # Gaspari-Cohn half-width; support is 2x
    # input random walk per 5 min (JE only)
    rw_wind_ms: float = 0.4
    rw_m: float = 0.003
    rw_la: float = 0.03
    infl: float = 1.0                   # multiplicative inflation of input anomalies after analysis
    member_ros_sd: float = 0.0          # per-member log-normal ROS field (model error), same statistics family as the truth's


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    issue_min: tuple[float, ...] = (20.0, 30.0, 40.0)
    leads_min: tuple[float, ...] = (15.0, 30.0)

    def with_filter(self, **kw) -> ExperimentConfig:
        return replace(self, filt=replace(self.filt, **kw))


# --------------------------------------------------------------------------- world
def _smooth(rng, n, sigma):
    f = gaussian_filter(rng.standard_normal((n, n)), sigma=sigma, mode="reflect")
    return f / (f.std() + 1e-12)


def _quantile_classes(z: np.ndarray, weights, codes) -> np.ndarray:
    cum = np.cumsum(weights) / np.sum(weights)
    cuts = np.quantile(z, cum[:-1])
    return np.asarray(codes)[np.searchsorted(cuts, z)]


@dataclass
class World:
    cfg: WorldConfig
    land_true: Landscape
    land_model: Landscape
    ign_rc: tuple[float, float]
    wind_mean: np.ndarray            # (2,) before any shift
    shift_t: float | None
    shift_rad: float
    m_mean: float
    m_field: np.ndarray              # (n, n)
    wind_field_rel: np.ndarray       # (n, n, 2) multiplicative speed and additive direction offsets
    alpha_field: np.ndarray          # (n, n)
    chi_r: float
    truth: dict = field(default_factory=dict)    # minute -> psi (n, n)

    def wind_true(self, t: float) -> np.ndarray:
        """True 20-ft wind field (n, n, 2) at minute t."""
        sp = np.hypot(*self.wind_mean)
        th = np.arctan2(self.wind_mean[1], self.wind_mean[0])
        if self.shift_t is not None and t >= self.shift_t:
            th = th + self.shift_rad
        s = sp * self.wind_field_rel[..., 0]
        a = th + self.wind_field_rel[..., 1]
        return np.stack([s * np.cos(a), s * np.sin(a)], axis=-1)

    def wind_mean_at(self, t: float) -> np.ndarray:
        sp = np.hypot(*self.wind_mean)
        th = np.arctan2(self.wind_mean[1], self.wind_mean[0])
        if self.shift_t is not None and t >= self.shift_t:
            th += self.shift_rad
        return np.array([sp * np.cos(th), sp * np.sin(th)])

    def inputs_true(self, t: float) -> Inputs:
        return Inputs(wind=self.wind_true(t)[None], m1h=self.m_field[None], alpha=self.alpha_field[None])


def make_world(cfg: WorldConfig, seed: int, t_end: float) -> World:
    rng = np.random.default_rng(seed)
    fc = cfg.front
    n = fc.n
    z = _smooth(rng, n, 6)
    fuel = _quantile_classes(z, cfg.fuel_weights, FUELS)
    if cfg.nonburnable_frac > 0:          # roads, water, rock: small patches
        nb = _smooth(rng, n, 2)
        fuel = np.where(nb > np.quantile(nb, 1 - cfg.nonburnable_frac), 0, fuel)
    elev = gaussian_filter(rng.standard_normal((n, n)), 12, mode="reflect")
    elev = cfg.relief_m * elev / (elev.std() + 1e-12)
    # model fuel map: patchy misclassification into another burnable model
    patch = _smooth(rng, n, 3) > np.quantile(_smooth(rng, n, 3), 1 - cfg.misclass_frac)
    other = rng.choice(FUELS, size=(n, n))
    fuel_model = np.where(patch & (fuel > 0) & (other != fuel), other, fuel)
    land_true, land_model = Landscape(fuel, elev, fc), Landscape(fuel_model, elev, fc)

    c = n // 2
    while True:
        ign = (float(rng.uniform(c - 20, c + 20)), float(rng.uniform(c - 20, c + 20)))
        if fuel[int(ign[0]), int(ign[1])] in (1, 2, 9, 10):   # starts in a fuel that carries a fire
            break
    speed = rng.uniform(*cfg.wind_range)
    th = rng.uniform(0, 2 * np.pi)
    shift = rng.random() < cfg.p_wind_shift
    world = World(
        cfg=cfg, land_true=land_true, land_model=land_model, ign_rc=ign,
        wind_mean=np.array([speed * np.cos(th), speed * np.sin(th)]),
        shift_t=float(rng.uniform(*cfg.shift_time_min)) if shift else None,
        shift_rad=float(np.radians(rng.uniform(*cfg.shift_deg)) * rng.choice([-1, 1])),
        m_mean=float(rng.uniform(*cfg.m1h_range)),
        m_field=np.zeros((n, n)),
        wind_field_rel=np.stack([np.exp(cfg.wind_speed_field_rel_sd * _smooth(rng, n, 15)),
                                 np.radians(cfg.wind_dir_field_sd_deg) * _smooth(rng, n, 15)], axis=-1),
        alpha_field=np.exp(cfg.ros_noise_sd * _smooth(rng, n, 4) - 0.5 * cfg.ros_noise_sd ** 2),
        chi_r=float(rng.uniform(*cfg.chi_r_range)),
    )
    world.m_field = np.clip(world.m_mean + cfg.m_field_sd * _smooth(rng, n, 10), 0.02, 0.30)
    # truth, stored every minute on which anything is observed or scored (multiples of 5 min)
    model = FrontModel(land_true)
    psi = ignite(fc, 1, ign)
    t = 0.0
    world.truth[0.0] = psi[0].copy()
    marks = sorted(set(np.arange(5.0, t_end + 1e-9, 5.0)) | ({world.shift_t} if shift else set()))
    for m in marks:
        if m <= t:
            continue
        psi = model.advance(psi, world.inputs_true(t), m - t)
        t = m
        world.truth[round(t, 6)] = psi[0].copy()
    return world


# --------------------------------------------------------------------------- observations
@dataclass
class Obs:
    t: float
    pts: np.ndarray            # (P, 2) east, north
    wind: np.ndarray | None    # (2,)
    frp_log: float | None
    drone: np.ndarray | None   # (2,) position
    mask: np.ndarray | None    # (n, n) footprint


def frp_density_log(psi: np.ndarray, i_b: np.ndarray, drone: np.ndarray, fc: FrontConfig,
                    radius_m: float, chi_r: float) -> np.ndarray:
    """log of FRP per metre of flaming front seen around the drone, log(chi_r * mean I_B) (kW/m).

    The drone's thermal image gives both the fire radiative power and the length of the
    flaming line in its footprint; their ratio is chi_r times the mean Byram intensity
    along that line. The mean is weighted by a Gaussian of the footprint radius around the
    drone, so a member whose front lies outside the footprint still has a defined value.
    Returns (K,).
    """
    x, y = cell_xy(fc)
    w = np.exp(-((x - drone[0]) ** 2 + (y - drone[1]) ** 2) / (2 * radius_m ** 2))
    eps = 1.5 * fc.cell_m
    delta = np.where(np.abs(psi) < eps, (1.0 + np.cos(np.pi * psi / eps)) / (2 * eps), 0.0)
    num = (i_b * delta * w).sum(axis=(1, 2))
    den = (delta * w).sum(axis=(1, 2))
    return np.log(chi_r * np.maximum(num / np.maximum(den, 1e-12), 1e-3))


def observe(world: World, t: float, k_obs: int, d: DroneConfig, rng, full: bool = False) -> Obs:
    fc = world.cfg.front
    psi = world.truth[round(t, 6)]
    pts = front_points(psi, fc)
    x, y = cell_xy(fc)
    if full:
        sel = pts
        drone, mask = None, None
    else:
        inside = psi < 0
        cx, cy = x[inside].mean(), y[inside].mean()
        bearing = np.radians(37.0) + k_obs * np.radians(100.0)
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        j = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - bearing))))))
        drone = pts[j]
        mask = np.hypot(x - drone[0], y - drone[1]) <= d.footprint_m
        sel = pts[np.hypot(pts[:, 0] - drone[0], pts[:, 1] - drone[1]) <= d.footprint_m]
    if len(sel) > d.max_points:
        sel = sel[np.linspace(0, len(sel) - 1, d.max_points).round().astype(int)]
    sel = sel + rng.normal(0, d.pos_sd_m, sel.shape)
    if full:
        return Obs(t, sel, None, None, None, None)
    # local wind at the drone
    r = int(np.clip(round(-drone[1] / fc.cell_m), 0, fc.n - 1))
    c = int(np.clip(round(drone[0] / fc.cell_m), 0, fc.n - 1))
    wind = world.wind_true(t)[r, c] + rng.normal(0, d.wind_sd_ms, 2)
    # FRP per metre of flaming front around the drone
    model = FrontModel(world.land_true)
    i_b = model.intensity(psi[None], world.inputs_true(t))
    frp_log = float(frp_density_log(psi[None], i_b, drone, fc, d.footprint_m, world.chi_r)[0]
                    + rng.normal(0, d.frp_log_sd))
    return Obs(t, sel, wind, frp_log, drone, mask)


# --------------------------------------------------------------------------- ensemble
@dataclass
class Ensemble:
    psi: np.ndarray            # (K, n, n)
    wind: np.ndarray           # (K, 2)
    m: np.ndarray              # (K,)
    la: np.ndarray             # (K,) log ROS factor
    ros_field: np.ndarray | None = None   # (K, n, n) fixed per-member log ROS perturbation

    def inputs(self) -> Inputs:
        if self.ros_field is None:
            return Inputs(wind=self.wind, m1h=self.m, alpha=np.exp(self.la))
        return Inputs(wind=self.wind, m1h=self.m, alpha=np.exp(self.la[:, None, None] + self.ros_field))

    def copy(self) -> Ensemble:
        return Ensemble(self.psi.copy(), self.wind.copy(), self.m.copy(), self.la.copy(), self.ros_field)


def prior_ensemble(world: World, init: Obs, f: FilterConfig, rng) -> Ensemble:
    """Inputs drawn around an NWP-like forecast whose own error is drawn once per scenario."""
    fc = world.cfg.front
    k = f.k
    wm = world.wind_mean_at(0.0)
    sp, th = np.hypot(*wm), np.arctan2(wm[1], wm[0])
    # the forecast itself is wrong by one draw of the prior error...
    sp_fc = sp * np.exp(rng.normal(0, f.prior_speed_logsd))
    th_fc = th + np.radians(rng.normal(0, f.prior_dir_sd_deg))
    m_fc = float(np.clip(world.m_mean + rng.normal(0, f.prior_m_sd), 0.02, 0.30))
    la_fc = 0.0
    # ...and the ensemble spreads around the forecast with the same error statistics
    sp_k = sp_fc * np.exp(rng.normal(0, f.prior_speed_logsd, k))
    th_k = th_fc + np.radians(rng.normal(0, f.prior_dir_sd_deg, k))
    wind = np.stack([sp_k * np.cos(th_k), sp_k * np.sin(th_k)], axis=1)
    m = np.clip(m_fc + rng.normal(0, f.prior_m_sd, k), 0.02, 0.30)
    la = la_fc + rng.normal(0, f.prior_la_sd, k)
    # initial front: the mapped perimeter, as a signed distance, with smooth per-member error
    psi0 = perimeter_sdf(init.pts, fc)
    psi = np.empty((k, fc.n, fc.n))
    for i in range(k):
        psi[i] = psi0 + f.init_pos_sd_m * _smooth(rng, fc.n, 4)
    ros = None
    if f.member_ros_sd > 0:   # unresolved fuel structure: each member its own field, ~100 m correlation
        ros = np.stack([f.member_ros_sd * _smooth(rng, fc.n, 4) - 0.5 * f.member_ros_sd ** 2 for _ in range(k)])
    ens = Ensemble(reinit(psi, fc.cell_m), wind, m, la, ros)
    _bound(ens)
    return ens


def perimeter_sdf(pts: np.ndarray, fc: FrontConfig) -> np.ndarray:
    """Signed distance to a mapped perimeter given as points.

    Inside/outside: the mapped region is taken as star-shaped about the points' centroid
    (true for the small, near-elliptical fire of the first orbit): a cell is inside if it is
    closer to the centroid than the perimeter in its direction.
    """
    x, y = cell_xy(fc)
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    ang_c = np.arctan2(y - cy, x - cx)
    r_edge = np.interp(ang_c, np.r_[ang[-1] - 2 * np.pi, ang, ang[0] + 2 * np.pi],
                       np.r_[rad[-1], rad, rad[0]])
    inside = np.hypot(x - cx, y - cy) < r_edge
    d = np.min(np.hypot(x[..., None] - pts[:, 0], y[..., None] - pts[:, 1]), axis=-1)
    return np.where(inside, -d, d)


def gaspari_cohn(r: np.ndarray, c: float) -> np.ndarray:
    """Gaspari & Cohn (1999) eq. 4.10, compact support 2c."""
    z = np.abs(r) / c
    out = np.zeros_like(z)
    m1 = z <= 1
    m2 = (z > 1) & (z < 2)
    z1, z2 = z[m1], z[m2]
    out[m1] = -0.25 * z1 ** 5 + 0.5 * z1 ** 4 + 0.625 * z1 ** 3 - 5 / 3 * z1 ** 2 + 1
    out[m2] = (z2 ** 5 / 12 - 0.5 * z2 ** 4 + 0.625 * z2 ** 3 + 5 / 3 * z2 ** 2 - 5 * z2 + 4
               - 2 / 3 / z2)
    return out


def predicted_obs(ens: Ensemble, obs: Obs, land: Landscape, d: DroneConfig) -> np.ndarray:
    """H(x) per member: [psi at perimeter points, wind east, wind north, log FRP]."""
    fc = land.cfg
    parts = [sample(ens.psi, obs.pts, fc)]
    if obs.wind is not None:
        parts.append(ens.wind)
        i_b = FrontModel(land).intensity(ens.psi, ens.inputs())
        parts.append(frp_density_log(ens.psi, i_b, obs.drone, fc, d.footprint_m, d.chi_r_model)[:, None])
    return np.concatenate(parts, axis=1)


def analysis(ens: Ensemble, obs: Obs, land: Landscape, d: DroneConfig, f: FilterConfig,
             update_inputs: bool, rng) -> Ensemble:
    """One stochastic-EnKF analysis. Front cells are localised against perimeter points."""
    fc = land.cfg
    k, n = f.k, fc.n
    hx = predicted_obs(ens, obs, land, d)
    p = len(obs.pts)
    y = np.zeros(hx.shape[1])
    r_diag = np.full(hx.shape[1], f.r_perim_m ** 2)
    if obs.wind is not None:
        y[p:p + 2] = obs.wind
        y[p + 2] = obs.frp_log
        r_diag[p:p + 2] = f.r_wind_ms ** 2
        r_diag[p + 2] = f.r_frp_log ** 2
    if not update_inputs:          # SO: only perimeter points, only the front
        hx, y, r_diag = hx[:, :p], y[:p], r_diag[:p]
    ny = len(y)
    ay = hx - hx.mean(axis=0)
    cyy = (ay.T @ ay) / (k - 1)
    # Front cells: localised gain. P_yy is localised with the same Gaspari-Cohn function
    # (perimeter-perimeter by distance), as in Houtekamer & Mitchell (2001).
    rho_yy = np.ones((ny, ny))
    dpp = np.hypot(obs.pts[:, None, 0] - obs.pts[None, :, 0], obs.pts[:, None, 1] - obs.pts[None, :, 1])
    rho_yy[:p, :p] = gaspari_cohn(dpp, f.loc_m)
    innov = y[None, :] + rng.normal(0, np.sqrt(r_diag), (k, ny)) - hx          # (K, ny)
    w = np.linalg.solve(rho_yy * cyy + np.diag(r_diag), innov.T)                 # (ny, K)
    # Inputs are global per member: their gain must use the *unlocalised* P_yy. Localising
    # P_yy while leaving P_theta,y whole counts distant perimeter points as independent
    # evidence about the same wind (measured on screening seed 17: the wind went from
    # (3.8, 4.0) to (-1.5, 6.6) m/s against a direct reading of (0.8, 1.8)).
    w_in = np.linalg.solve(cyy + np.diag(r_diag), innov.T)

    out = ens.copy()
    # front cells within the support of some perimeter point
    x, yy = cell_xy(fc)
    dist = np.hypot(x.ravel()[:, None] - obs.pts[None, :, 0], yy.ravel()[:, None] - obs.pts[None, :, 1])
    near = dist.min(axis=1) < 2 * f.loc_m
    if near.any():
        idx = np.where(near)[0]
        xs = ens.psi.reshape(k, -1)[:, idx]
        ax = xs - xs.mean(axis=0)
        rho = np.zeros((len(idx), ny))
        rho[:, :p] = gaspari_cohn(dist[idx], f.loc_m)
        pxy = rho * (ax.T @ ay) / (k - 1)
        flat = out.psi.reshape(k, -1)
        flat[:, idx] = xs + (pxy @ w).T
        out.psi = reinit(flat.reshape(k, n, n), fc.cell_m)
    if update_inputs:
        th = np.column_stack([ens.wind, ens.m, ens.la])
        at = th - th.mean(axis=0)
        th_a = th + ((at.T @ ay) / (k - 1) @ w_in).T
        mean = th_a.mean(axis=0)
        th_a = mean + f.infl * (th_a - mean)
        out.wind, out.m, out.la = th_a[:, :2].copy(), th_a[:, 2].copy(), th_a[:, 3].copy()
        _bound(out)
    return out


def random_walk(ens: Ensemble, f: FilterConfig, minutes: float, rng) -> None:
    s = np.sqrt(minutes / 5.0)
    k = len(ens.m)
    ens.wind += rng.normal(0, f.rw_wind_ms * s, (k, 2))
    ens.m = ens.m + rng.normal(0, f.rw_m * s, k)
    ens.la = ens.la + rng.normal(0, f.rw_la * s, k)
    _bound(ens)


# --------------------------------------------------------------------------- regime
def head_nc(model: FrontModel, psi: np.ndarray, inp: Inputs, wind_speed: np.ndarray) -> np.ndarray:
    """N_c at each member's head: the front cell of maximum Byram intensity, its local front
    speed, and the wind speed there. ``wind_speed``: (K,) or (K, n, n), m/s."""
    fc = model.cfg
    k = len(psi)
    i_b = model.intensity(psi, inp)
    front = np.abs(psi) < 0.75 * fc.cell_m
    flat = np.where(front, i_b, -1.0).reshape(k, -1)
    j = flat.argmax(axis=1)
    rows = np.arange(k)
    i_head = np.maximum(flat[rows, j], 0.0)
    r_head = model.normal_speed(psi, inp).reshape(k, -1)[rows, j]
    ws = np.asarray(wind_speed, float)
    u = ws if ws.ndim == 1 else ws.reshape(k, -1)[rows, j]
    return rg.convection_number(i_head, u, r_head)


def truth_nc(world: World, t: float) -> float:
    psi = world.truth[round(t, 6)][None]
    model = FrontModel(world.land_true)
    w = world.wind_true(t)
    return float(head_nc(model, psi, world.inputs_true(t), np.hypot(w[..., 0], w[..., 1])[None])[0])


# --------------------------------------------------------------------------- responder brief
@dataclass
class FrontBrief:
    """What the front-world forecaster hands to the people who respond (v2.1)."""
    minute: float
    area_ha: float                   # expected burnt area now
    head_ros_mmin: tuple             # median and 80 % range of the head rate of spread
    head_intensity_kwm: float        # median Byram intensity at the head
    flame_length_m: float            # Byram (1959) from that intensity
    wind_ms: tuple                   # posterior mean (east, north) and speed s.d.
    forecast_area_ha: dict           # lead (min) -> expected burnt area
    envelope: dict                   # lead (min) -> boolean map, P(burnt) >= 0.1
    regime: rg.RegimeSignal

    def lines(self) -> list[str]:
        r = self.head_ros_mmin
        out = [f"t+{self.minute:.0f} min: {self.area_ha:.1f} ha burning or burnt",
               f"head: {r[0]:.1f} m/min (80 % {r[1]:.1f}-{r[2]:.1f}), {self.head_intensity_kwm:.0f} kW/m, "
               f"flames ~{self.flame_length_m:.1f} m",
               f"wind: {np.hypot(*self.wind_ms[0]):.1f} m/s (s.d. {self.wind_ms[1]:.1f})"]
        out += [f"+{int(k)} min: {v:.1f} ha expected" for k, v in self.forecast_area_ha.items()]
        out.append("regime " + self.regime.line())
        return out


def front_brief(ens: Ensemble, model: FrontModel, minute: float, leads=(15.0, 30.0)) -> FrontBrief:
    fc = model.cfg
    k = len(ens.m)
    inp = ens.inputs()
    i_b = model.intensity(ens.psi, inp)
    speed = model.normal_speed(ens.psi, inp)
    front = np.abs(ens.psi) < 0.75 * fc.cell_m
    flat = np.where(front, i_b, -1.0).reshape(k, -1)
    j = flat.argmax(axis=1)
    rows = np.arange(k)
    i_head = np.maximum(flat[rows, j], 0.0)
    r_head = speed.reshape(k, -1)[rows, j]
    nc = rg.convection_number(i_head, np.hypot(ens.wind[:, 0], ens.wind[:, 1]), r_head)
    fa, env = {}, {}
    psi, t_prev = ens.psi, 0.0
    for lead in sorted(leads):
        psi = model.advance(psi, inp, lead - t_prev)
        t_prev = lead
        prob = (psi < 0).mean(axis=0)
        fa[lead] = float(prob.sum() * fc.cell_m ** 2 / 1e4)
        env[lead] = prob >= 0.1
    from .rothermel import flame_length_m  # noqa: PLC0415
    i_med = float(np.median(i_head))
    return FrontBrief(
        minute=minute, area_ha=float((ens.psi < 0).mean(axis=0).sum() * fc.cell_m ** 2 / 1e4),
        head_ros_mmin=tuple(float(np.quantile(r_head, q)) for q in (0.5, 0.1, 0.9)),
        head_intensity_kwm=i_med, flame_length_m=float(flame_length_m(i_med)),
        wind_ms=(ens.wind.mean(axis=0), float(np.hypot(*ens.wind.std(axis=0)))),
        forecast_area_ha=fa, envelope=env, regime=rg.regime_signal(nc))


# --------------------------------------------------------------------------- one scenario
METHODS = ("OL", "SO", "JE")


def run_scenario(cfg: ExperimentConfig, seed: int, methods=METHODS) -> dict:
    """Run the truth, the drone and the three forecasters on one seed. Returns a flat record."""
    t_end = max(cfg.issue_min) + max(cfg.leads_min)
    world = make_world(cfg.world, seed, t_end)
    d, f = cfg.drone, cfg.filt
    rng_obs = np.random.default_rng((seed, 1))
    times = np.arange(d.t_alert_min + d.obs_every_min, d.t_last_obs_min + 1e-9, d.obs_every_min)
    init = observe(world, d.t_alert_min, 0, d, rng_obs, full=True)
    obs = {float(t): observe(world, float(t), i + 1, d, rng_obs) for i, t in enumerate(times)}
    land = world.land_model
    model = FrontModel(land)
    fc = cfg.world.front
    rec = {"seed": seed, "wind_ms": float(np.hypot(*world.wind_mean)), "m1h": world.m_mean,
           "shift": world.shift_t is not None, "chi_r": world.chi_r,
           "edge_hit": bool((world.truth[round(t_end, 6)] < 0)[[0, -1], :].any()
                            or (world.truth[round(t_end, 6)] < 0)[:, [0, -1]].any())}
    for meth in methods:
        rng = np.random.default_rng((seed, 2))          # same prior draws for every method
        ens = prior_ensemble(world, init, f, rng)
        rng_f = np.random.default_rng((seed, 3, METHODS.index(meth)))
        t = d.t_alert_min
        checkpoints = sorted(set(times.tolist()) | set(cfg.issue_min))
        for tc in checkpoints:
            ens.psi = model.advance(ens.psi, ens.inputs(), tc - t)
            if meth == "JE":
                random_walk(ens, f, tc - t, rng_f)
            t = tc
            if tc in obs and meth != "OL":
                ens = analysis(ens, obs[tc], land, d, f, update_inputs=(meth == "JE"), rng=rng_f)
            if tc in cfg.issue_min:
                _score_issue(rec, meth, tc, ens, model, world, cfg)
    return rec


def _score_issue(rec, meth, tc, ens, model, world, cfg):
    fc = cfg.world.front
    tag = f"{meth}@{int(tc)}"
    psi = ens.psi
    fpsi, t_prev = psi, 0.0
    inp = ens.inputs()
    for lead in sorted(cfg.leads_min):
        fpsi = model.advance(fpsi, inp, lead - t_prev)
        t_prev = lead
        prob = (fpsi < 0).mean(axis=0)
        truth = world.truth[round(tc + lead, 6)] < 0
        sc = forecast_scores(prob, truth)
        # domain-summed squared error, m^2: proper and comparable between forecasts of different
        # support (the v2.0 Brier is a mean over each forecast's own relevant set)
        sc["sse_m2"] = float(((prob - truth) ** 2).sum() * fc.cell_m ** 2)
        sc["true_m2"] = float(truth.sum() * fc.cell_m ** 2)
        p50 = prob >= 0.5
        sc["iou50"] = float((p50 & truth).sum() / max((p50 | truth).sum(), 1))
        for kk, v in sc.items():
            rec[f"{tag}+{int(lead)}:{kk}"] = v
    # inputs against the truth's means at this minute
    wm = world.wind_mean_at(tc)
    rec[f"{tag}:wind_err"] = float(np.hypot(*(ens.wind.mean(axis=0) - wm)))
    rec[f"{tag}:m_err"] = float(abs(ens.m.mean() - world.m_mean))
    rec[f"{tag}:la_mean"] = float(ens.la.mean())
    # regime
    nc = head_nc(model, psi, ens.inputs(), np.hypot(ens.wind[:, 0], ens.wind[:, 1]))
    sig = rg.regime_signal(nc)
    nc_t = truth_nc(world, tc)
    cls_t = int(rg.regime_class(nc_t))
    rec[f"{tag}:nc_true"] = nc_t if np.isfinite(nc_t) else 1e9
    rec[f"{tag}:nc_med"] = sig.nc_median
    rec[f"{tag}:regime_true"] = cls_t
    rec[f"{tag}:regime_brier"] = rg.regime_brier(sig.p, cls_t)
    rec[f"{tag}:regime_hit"] = int(int(np.argmax(sig.p)) == cls_t)
    rec[f"{tag}:light"] = sig.light
    rec[f"{tag}:area_m2"] = float((psi < 0).mean(axis=0).sum() * fc.cell_m ** 2)


__all__ = ["DroneConfig", "Ensemble", "ExperimentConfig", "FilterConfig", "FrontBrief", "METHODS", "Obs",
           "World", "WorldConfig", "analysis", "front_brief", "gaspari_cohn", "head_nc", "make_world", "observe",
           "perimeter_sdf", "prior_ensemble", "run_scenario", "truth_nc"]
