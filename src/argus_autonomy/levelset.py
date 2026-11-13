"""v2.1: front-scale fire world -- a level set driven by Rothermel spread rates.

The fire is the set {psi < 0} of a signed distance function psi (metres, negative
inside). It evolves by the Hamilton-Jacobi equation

    d psi / dt + H(x, grad psi) = 0,     H(x, p) = h_W(x, p)                    (1)

where h_W is the support function of the local *fire-spread wavelet* W(x): the
ellipse that a point ignition at x would burn in one minute. This is Richards'
(1990) differential form of the Huygens model used by FARSITE (Finney 1998),
written as a level set. Because a support function is convex and positively
homogeneous, the viscosity solution of (1) from a point is exactly W scaled by t
(Hopf formula), so the head of a point-ignition fire runs at Rothermel's head rate.

    (A note on a formulation *not* used. Driving the front with the wind component
    normal to it, R(n) = R0 (1 + phi_w(U . n)), as WRF-SFIRE does, gives a
    non-convex Hamiltonian. For Anderson model 1 at 4 m/s its exact point-ignition
    head speed is min_n R(n)/cos(n) = 8.8 m/min, about half of Rothermel's
    16.5 m/min; a naive isotropic upwind scheme instead returns 14.8 m/min, i.e. it
    is not even consistent with that equation. Measured during development; see the
    v2.1 design document.)

The wavelet at x (Finney 1998, Anderson 1983, Alexander 1985):
  phi_E   = | phi_w d_wind + phi_s d_upslope |   (vector sum of wind and slope factors)
  d       = direction of that vector (direction of maximum spread)
  R_head  = alpha * min(R0 (1 + phi_E), max(U_eff, R0))   (Andrews, Cruz & Rothermel 2013 cap)
  LB      = Anderson (1983) length-to-breadth ratio of the effective midflame wind U_eff
  R_back  = R_head / HB(LB);  ellipse with the ignition at its rear focus:
            a = (R_head + R_back)/2,  b = a / LB,  c = (R_head - R_back)/2
  h_W(n)  = c (d . n) + sqrt(a^2 (d . n)^2 + b^2 (d_perp . n)^2)
alpha is a rate-of-spread adjustment factor (1 in the truth model); it is the knob
through which unresolved model error enters the assimilation.

Numerics: first-order upwind discretisation of the control form max_{v in W} v . grad psi
(monotone; see ``FrontModel.advance``), explicit Euler with a CFL-limited step, and periodic
re-initialisation to a signed distance by a sub-cell-preserving rule (``reinit``).
Every array carries a leading ensemble axis K.

Coordinates: arrays are [row, col]; east = +col, north = -row. Wind vectors are
(east, north) in m/s and denote the 20-ft (6.1 m) open wind the fire sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import rothermel as rt
from .suppression import hb_ratio, lb_ratio

NONBURNABLE = 0


@dataclass(frozen=True)
class FrontConfig:
    n: int = 160               # grid is n x n cells (4 km at 25 m)
    cell_m: float = 25.0       # same cell size as the 25 m automaton of world.py
    cfl: float = 0.8           # fraction of the stability limit dt <= dx / (sqrt(2) R_head)
    max_dt_min: float = 1.0
    reinit_every: int = 20
    ign_radius_m: float = 60.0


class Landscape:
    """Fuel map (Anderson codes, 0 = non-burnable) and terrain, with per-cell Rothermel constants."""

    def __init__(self, fuel: np.ndarray, elev_m: np.ndarray, cfg: FrontConfig):
        self.cfg = cfg
        self.fuel = fuel.astype(np.int16)
        self.elev = elev_m.astype(float)
        dx = cfg.cell_m
        dzdr, dzdc = np.gradient(self.elev, dx)
        self.dz_east, self.dz_north = dzdc, -dzdr
        self.burnable = self.fuel != NONBURNABLE
        n = fuel.shape[0]
        self.C = np.zeros((n, n)); self.B = np.ones((n, n)); self.E = np.zeros((n, n))
        self.rel = np.ones((n, n)); self.phis = np.zeros((n, n)); self.waf = np.zeros((n, n))
        self.t_r = np.zeros((n, n))
        for c in rt.FUEL_CODES:
            m = self.fuel == c
            b = rt.BEDS[c]
            self.C[m], self.B[m], self.E[m] = b.C, b.B, b.E
            self.rel[m], self.phis[m], self.t_r[m] = b.beta_rel, b.phi_s_coef, b.t_r
            self.waf[m] = rt.FUEL_MODELS[c].waf
        self.rel_negE = np.where(self.burnable, self.rel ** -self.E, 0.0)
        self.rel_E = np.where(self.burnable, self.rel ** self.E, 1.0)

    def r0_ir(self, m1h) -> tuple[np.ndarray, np.ndarray]:
        """No-wind spread rate (m/min) and reaction intensity (kW/m^2) on the grid.

        ``m1h``: dead 1-h moisture, shape (K,) (one value per member) or (K, n, n).
        """
        n = self.fuel.shape[0]
        m = np.asarray(m1h, float)
        m = np.broadcast_to(m.reshape(m.shape + (1, 1)) if m.ndim == 1 else m, (m.shape[0], n, n))
        r0 = np.zeros(m.shape); ir = np.zeros(m.shape)
        for c in rt.FUEL_CODES:
            mask = self.fuel == c
            if mask.any():
                r0[:, mask], ir[:, mask] = rt.no_wind_spread(c, m[:, mask])
        return r0, ir


@dataclass
class Inputs:
    """Spread inputs for K members, constant over one integration segment."""
    wind: np.ndarray          # (K, 2) or (K, n, n, 2): (east, north) 20-ft wind, m/s
    m1h: np.ndarray           # (K,) or (K, n, n): dead 1-h moisture, fraction
    alpha: np.ndarray         # (K,) or (K, n, n): ROS adjustment factor
    _cache: dict | None = None


def _grad_central(psi: np.ndarray, dx: float):
    p = np.pad(psi, ((0, 0), (1, 1), (1, 1)), mode="edge")
    g_c = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2 * dx)
    g_r = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / (2 * dx)
    return g_r, g_c


def _upwind_norm(psi: np.ndarray, dx: float) -> np.ndarray:
    """Godunov |grad psi| for an outward-moving front (speed >= 0)."""
    p = np.pad(psi, ((0, 0), (1, 1), (1, 1)), mode="edge")
    c = p[:, 1:-1, 1:-1]
    dxm = (c - p[:, 1:-1, :-2]) / dx
    dxp = (p[:, 1:-1, 2:] - c) / dx
    dym = (c - p[:, :-2, 1:-1]) / dx
    dyp = (p[:, 2:, 1:-1] - c) / dx
    return np.sqrt(np.maximum(dxm, 0) ** 2 + np.minimum(dxp, 0) ** 2
                   + np.maximum(dym, 0) ** 2 + np.minimum(dyp, 0) ** 2)


def _expand(a: np.ndarray, k: int, n: int) -> np.ndarray:
    a = np.asarray(a, float)
    if a.ndim == 1:
        return a.reshape(k, 1, 1)
    return a


class FrontModel:
    """Level-set propagation of K fires on one landscape."""

    def __init__(self, land: Landscape):
        self.land = land
        self.cfg = land.cfg

    # ----------------------------------------------------------------- wavelet
    def _prepare(self, inp: Inputs, k: int):
        """Per-cell wavelet (a, b, c, d) for every member; cached on the Inputs."""
        if inp._cache is not None:
            return inp._cache
        land = self.land
        r0, ir = land.r0_ir(inp.m1h)
        alpha = _expand(inp.alpha, k, self.cfg.n)
        w = np.asarray(inp.wind, float)
        if w.ndim == 2:
            we, wn = w[:, 0].reshape(k, 1, 1), w[:, 1].reshape(k, 1, 1)
        else:
            we, wn = w[..., 0], w[..., 1]
        speed = np.hypot(we, wn)
        u_mid = land.waf * speed * rt.MS_TO_FTMIN                               # ft/min
        phi_w = land.C * np.power(u_mid, land.B) * land.rel_negE
        tan_s = np.hypot(land.dz_east, land.dz_north)
        phi_s = land.phis * tan_s * tan_s
        with np.errstate(invalid="ignore", divide="ignore"):
            wx = np.where(speed > 0, we / np.maximum(speed, 1e-12), 0.0)
            wy = np.where(speed > 0, wn / np.maximum(speed, 1e-12), 0.0)
            sx = np.where(tan_s > 0, land.dz_east / np.maximum(tan_s, 1e-12), 0.0)
            sy = np.where(tan_s > 0, land.dz_north / np.maximum(tan_s, 1e-12), 0.0)
        vx, vy = phi_w * wx + phi_s * sx, phi_w * wy + phi_s * sy
        phi_e = np.hypot(vx, vy)
        dx_, dy_ = np.where(phi_e > 0, vx / np.maximum(phi_e, 1e-12), 1.0), \
            np.where(phi_e > 0, vy / np.maximum(phi_e, 1e-12), 0.0)
        u_eff = np.power(phi_e * land.rel_E / np.where(land.C > 0, land.C, 1.0), 1.0 / land.B) \
            / rt.MS_TO_FTMIN                                                    # m/s, midflame
        r_head = np.minimum(r0 * (1.0 + phi_e), np.maximum(u_eff * 60.0, r0)) * alpha
        r_head = np.where(land.burnable, r_head, 0.0)
        lb = lb_ratio(u_eff * 3.6)
        r_back = r_head / hb_ratio(lb)
        a_ = 0.5 * (r_head + r_back)
        inp._cache = {"a": a_, "b": a_ / lb, "c": 0.5 * (r_head - r_back),
                      "dx": dx_, "dy": dy_, "rh": r_head, "ir": ir}
        return inp._cache

    def hamiltonian(self, pc: dict, p_e: np.ndarray, p_n: np.ndarray) -> np.ndarray:
        """h_W(p) for gradient components (east, north); homogeneous of degree 1."""
        along = pc["dx"] * p_e + pc["dy"] * p_n
        across = -pc["dy"] * p_e + pc["dx"] * p_n
        return pc["c"] * along + np.sqrt((pc["a"] * along) ** 2 + (pc["b"] * across) ** 2)

    def normal_speed(self, psi: np.ndarray, inp: Inputs) -> np.ndarray:
        """Local front speed h_W(n) in m/min, n = grad psi / |grad psi|."""
        pc = self._prepare(inp, psi.shape[0])
        g_r, g_c = _grad_central(psi, self.cfg.cell_m)
        nrm = np.sqrt(g_r ** 2 + g_c ** 2) + 1e-12
        return self.hamiltonian(pc, g_c / nrm, -g_r / nrm)

    def intensity(self, psi: np.ndarray, inp: Inputs) -> np.ndarray:
        """Byram intensity (kW/m) of the local front, I_R * h_W(n) * t_r, on the grid."""
        pc = self._prepare(inp, psi.shape[0])
        return pc["ir"] * self.normal_speed(psi, inp) * self.land.t_r

    def head_rate(self, inp: Inputs, k: int) -> np.ndarray:
        return self._prepare(inp, k)["rh"]

    # ----------------------------------------------------------------- integration
    def _axis_roots(self, pc: dict):
        """Where the wavelet boundary crosses the east and north axes through the ignition
        point (the origin, which lies inside the wavelet: it is the rear focus).
        Returns (x_pos, x_neg, y_pos, y_neg), each (K, n, n)."""
        a, b, c = pc["a"], pc["b"], pc["c"]
        dx_, dy_ = pc["dx"], pc["dy"]
        ox, oy = c * dx_, c * dy_
        ia2 = 1.0 / np.maximum(a * a, 1e-18)
        ib2 = 1.0 / np.maximum(b * b, 1e-18)

        def roots(o_par, o_perp, e_par_d, e_par_dp):
            # point = o + t e, e unit axis; ((t e_d + o_d)^2 / a^2 + (t e_dp + o_dp)^2 / b^2 = 1) in t
            A = e_par_d ** 2 * ia2 + e_par_dp ** 2 * ib2
            B = 2 * (e_par_d * o_par * ia2 + e_par_dp * o_perp * ib2)
            C = o_par ** 2 * ia2 + o_perp ** 2 * ib2 - 1.0
            disc = np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))
            return (-B + disc) / (2 * A), (-B - disc) / (2 * A)

        # coordinates of -o in the (d, d_perp) frame: the origin relative to the centre
        o_d = -(ox * dx_ + oy * dy_)
        o_dp = -(-ox * dy_ + oy * dx_)
        # east axis e = (1, 0): components along d and d_perp are dx_, -dy_
        xp, xn = roots(o_d, o_dp, dx_, -dy_)
        # north axis e = (0, 1): components dy_, dx_
        yp, yn = roots(o_d, o_dp, dy_, dx_)
        ok = (a > 0) & (b > 0)
        z = np.zeros_like(a)
        return (np.where(ok, xp, z), np.where(ok, xn, z), np.where(ok, yp, z), np.where(ok, yn, z))

    def advance(self, psi: np.ndarray, inp: Inputs, minutes: float, group: int = 8) -> np.ndarray:
        """Advance every member ``minutes`` with constant inputs. Returns a new array.

        Members are integrated in groups of similar head rate, each group with its own
        CFL step and bounding box, so one fast member does not set the step for all.
        """
        k = psi.shape[0]
        if k <= group:
            return self._advance_group(psi, self._prepare(inp, k), minutes)
        full = self._prepare(inp, k)
        n = self.cfg.n
        rmax_k = np.broadcast_to(full["rh"], (k, n, n)).reshape(k, -1).max(axis=1)
        order = np.argsort(rmax_k)
        out = np.empty_like(psi, dtype=float)
        for start in range(0, k, group):
            idx = order[start:start + group]
            sub = {key: np.broadcast_to(v, (k, n, n))[idx] for key, v in full.items()}
            out[idx] = self._advance_group(psi[idx], sub, minutes)
        return out

    def _advance_group(self, psi: np.ndarray, full: dict, minutes: float) -> np.ndarray:
        """Integrate one group of members (``full``: its wavelet arrays, (k, n, n)).

        Scheme: psi_t + max_{v in W} [v_e^+ D^-_e + v_e^- D^+_e + v_n^+ D^-_n + v_n^- D^+_n] psi = 0,
        the upwind discretisation of the control form psi_t + max_{v in W} v . grad psi = 0
        (monotone and consistent for any wavelet). On each quadrant of v the bracket is
        linear in v, so its maximum over W is attained at the support point of W, at a
        crossing of W with an axis, or at the origin; all are closed-form.
        """
        psi = psi.astype(float, copy=True)
        k = psi.shape[0]
        dx, n = self.cfg.cell_m, self.cfg.n
        # work on the bounding box the fire can reach in this segment (plus a margin)
        reach = float(full["rh"].max()) * minutes + 4 * dx
        inside = (psi < reach).any(axis=0)
        if not inside.any():
            return psi
        rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
        r0_, r1_ = max(rows[0] - 2, 0), min(rows[-1] + 3, n)
        c0_, c1_ = max(cols[0] - 2, 0), min(cols[-1] + 3, n)
        box = (slice(None), slice(r0_, r1_), slice(c0_, c1_))
        pc = {key: np.broadcast_to(v, (k, n, n))[box] for key, v in full.items()}
        out = psi
        psi = psi[box].copy()
        f32 = np.float32
        c = pc["c"].astype(f32)
        d_e, d_n = pc["dx"].astype(f32), pc["dy"].astype(f32)
        o_e, o_n = c * d_e, c * d_n
        a2, b2 = (pc["a"] ** 2).astype(f32), (pc["b"] ** 2).astype(f32)
        xp, xn, yp, yn = (v.astype(f32) for v in self._axis_roots(pc))
        rmax = float(pc["rh"].max())
        psi = psi.astype(f32)
        t, it = 0.0, 0
        while t < minutes - 1e-9:
            dt = min(self.cfg.max_dt_min, minutes - t,
                     self.cfg.cfl * dx / (np.sqrt(2.0) * rmax) if rmax > 0 else self.cfg.max_dt_min)
            p = np.pad(psi, ((0, 0), (1, 1), (1, 1)), mode="edge")
            cen = p[:, 1:-1, 1:-1]
            de_m = (cen - p[:, 1:-1, :-2]) / dx            # backward in east
            de_p = (p[:, 1:-1, 2:] - cen) / dx             # forward in east
            dn_m = (cen - p[:, 2:, 1:-1]) / dx             # backward in north (south neighbour)
            dn_p = (p[:, :-2, 1:-1] - cen) / dx            # forward in north
            # origin and axis-crossing candidates (independent of the quadrant pairing)
            h = np.maximum(np.maximum(xp * de_m, xn * de_p), np.maximum(yp * dn_m, yn * dn_p))
            np.maximum(h, 0, out=h)
            ge_terms = ((1.0, de_m * d_e, de_m * d_n), (-1.0, de_p * d_e, de_p * d_n))
            gn_terms = ((1.0, dn_m * d_n, dn_m * d_e), (-1.0, dn_p * d_n, dn_p * d_e))
            for se, ee, en in ge_terms:
                for sn, nn, ne in gn_terms:
                    gd = ee + nn                           # g . d
                    gp = ne - en                           # g . d_perp
                    q = np.sqrt(a2 * gd * gd + b2 * gp * gp)
                    # support point v = o + M g / q; test its quadrant without dividing by q
                    ve_q = o_e * q + a2 * gd * d_e - b2 * gp * d_n
                    vn_q = o_n * q + a2 * gd * d_n + b2 * gp * d_e
                    inq = (se * ve_q >= 0) & (sn * vn_q >= 0)
                    np.maximum(h, np.where(inq, c * gd + q, 0), out=h)   # h_W(g) = c g.d + q
            psi -= f32(dt) * h
            t += dt
            it += 1
            if it % self.cfg.reinit_every == 0:
                psi = reinit(psi, dx).astype(f32)
        out[box] = psi
        return reinit(out, dx)


# --------------------------------------------------------------------------- geometry
def reinit(psi: np.ndarray, dx: float) -> np.ndarray:
    """Signed distance from the zero level set, keeping sub-cell information.

    Cells adjacent to the front (a sign change to a 4-neighbour) get psi / |grad psi|
    (the first-order sub-cell estimate of Russo & Smereka 2000); every other cell gets
    its distance to the nearest such cell plus that cell's own distance.
    """
    out = np.empty_like(psi)
    g_r, g_c = _grad_central(psi, dx)
    gnorm = np.maximum(np.sqrt(g_r ** 2 + g_c ** 2), 1e-6)
    s = psi < 0
    for i in range(psi.shape[0]):
        si = s[i]
        band = np.zeros_like(si)
        band[:-1, :] |= si[:-1, :] != si[1:, :]
        band[1:, :] |= si[1:, :] != si[:-1, :]
        band[:, :-1] |= si[:, :-1] != si[:, 1:]
        band[:, 1:] |= si[:, 1:] != si[:, :-1]
        if not band.any():
            out[i] = np.where(si, -1e6, 1e6)
            continue
        d_band = np.abs(psi[i] / gnorm[i])
        d_band = np.minimum(d_band, dx)            # a band cell is within one cell of the front
        dist, (ir, ic) = distance_transform_edt(~band, sampling=dx, return_indices=True)
        d = dist + d_band[ir, ic]
        d[band] = d_band[band]
        out[i] = np.where(si, -d, d)
    return out


def ignite(cfg: FrontConfig, k: int, rc: tuple[float, float]) -> np.ndarray:
    """psi for a circular ignition of radius ``ign_radius_m`` at (row, col)."""
    n = cfg.n
    rr, cc = np.indices((n, n), dtype=float)
    d = np.hypot(rr - rc[0], cc - rc[1]) * cfg.cell_m - cfg.ign_radius_m
    return np.broadcast_to(d, (k, n, n)).copy()


def cell_xy(cfg: FrontConfig) -> tuple[np.ndarray, np.ndarray]:
    """East and north coordinates (m) of cell centres."""
    rr, cc = np.indices((cfg.n, cfg.n), dtype=float)
    return cc * cfg.cell_m, -rr * cfg.cell_m


def front_points(psi2d: np.ndarray, cfg: FrontConfig) -> np.ndarray:
    """Points (east, north) on the zero level set: band cells projected onto the front."""
    dx = cfg.cell_m
    g_r, g_c = _grad_central(psi2d[None], dx)
    g_r, g_c = g_r[0], g_c[0]
    gn = np.sqrt(g_r ** 2 + g_c ** 2) + 1e-12
    band = (np.abs(psi2d) < 0.75 * dx) & (psi2d < 0)
    x, y = cell_xy(cfg)
    ne, nn = g_c / gn, -g_r / gn
    px = x[band] - psi2d[band] * ne[band]
    py = y[band] - psi2d[band] * nn[band]
    return np.stack([px, py], axis=1)


def sample(psi: np.ndarray, pts: np.ndarray, cfg: FrontConfig) -> np.ndarray:
    """Bilinear interpolation of psi (K, n, n) at points (east, north); returns (K, P)."""
    dx, n = cfg.cell_m, cfg.n
    c = np.clip(pts[:, 0] / dx, 0, n - 1 - 1e-9)
    r = np.clip(-pts[:, 1] / dx, 0, n - 1 - 1e-9)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    r1, c1 = np.minimum(r0 + 1, n - 1), np.minimum(c0 + 1, n - 1)
    return ((1 - fr) * (1 - fc) * psi[:, r0, c0] + (1 - fr) * fc * psi[:, r0, c1]
            + fr * (1 - fc) * psi[:, r1, c0] + fr * fc * psi[:, r1, c1])


def front_integral(psi: np.ndarray, field: np.ndarray, mask: np.ndarray, cfg: FrontConfig) -> np.ndarray:
    """Integral of ``field`` along the front inside ``mask``: sum field * delta_eps(psi) dA.

    psi is a signed distance (|grad psi| = 1), so delta_eps(psi) dA is a line element.
    The smoothed delta has half-width eps = 1.5 cells. Returns (K,) in field-units * m.
    """
    eps = 1.5 * cfg.cell_m
    a = np.abs(psi)
    delta = np.where(a < eps, (1.0 + np.cos(np.pi * psi / eps)) / (2 * eps), 0.0)
    return (field * delta * mask).sum(axis=(1, 2)) * cfg.cell_m ** 2


__all__ = ["FrontConfig", "FrontModel", "Inputs", "Landscape", "NONBURNABLE", "cell_xy", "front_integral",
           "front_points", "ignite", "reinit", "sample"]
