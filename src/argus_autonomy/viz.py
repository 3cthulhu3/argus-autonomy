"""Rendering helpers (optional dependency: matplotlib, imageio)."""

from __future__ import annotations

import numpy as np

from .config import MissionConfig
from .metrics import perimeter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BURNING = "#eb6834"
BURNT = "#52514e"
UNBURNT = "#e9e8e2"
DRONE = "#0b0b0b"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    return plt, LinearSegmentedColormap, ListedColormap


def belief_cmap():
    _, LSC, _ = _mpl()
    return LSC.from_list("belief", ["#fcfcfb", "#b7d3f2", "#2a78d6", "#0d3d7a"])


def truth_rgb(state: np.ndarray) -> np.ndarray:
    from matplotlib.colors import to_rgb
    img = np.empty(state.shape + (3,))
    img[:] = to_rgb(UNBURNT)
    img[state > 0] = to_rgb(BURNING)
    img[state < 0] = to_rgb(BURNT)
    return img


def _decorate(ax, cfg: MissionConfig):
    from matplotlib.patches import Rectangle
    r0, c0, r1, c1 = cfg.nfz
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, hatch="////",
                           edgecolor=INK_2, lw=0.8))
    ax.text((c0 + c1) / 2, (r0 + r1) / 2, "no-fly", ha="center", va="center", fontsize=7, color=INK_2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.plot(cfg.base[1], cfg.base[0], marker="s", ms=6, color=INK, mec=SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7")


def render_frame(cfg: MissionConfig, frame: dict, trail: list, title: str = ""):
    """Two panels: hidden truth | onboard belief with trajectory and sensor footprint."""
    plt, _, _ = _mpl()
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), facecolor=SURFACE)
    ax0, ax1 = axes
    ax0.imshow(truth_rgb(frame["truth"]), interpolation="nearest")
    ax0.set_title("ground truth (hidden from the drone)", fontsize=8, color=INK)
    ax1.imshow(frame["belief"], cmap=belief_cmap(), vmin=0, vmax=1, interpolation="nearest")
    per = perimeter(frame["truth"] != 0)
    rr, cc = np.nonzero(per)
    ax1.scatter(cc, rr, s=1.2, color=BURNING, lw=0, alpha=0.9)
    ax1.set_title("onboard belief P(fire-affected)  ·  true perimeter (orange)", fontsize=8, color=INK)
    for ax in axes:
        _decorate(ax, cfg)
        if trail:
            tr = np.array(trail)
            ax.plot(tr[:, 1], tr[:, 0], color=DRONE, lw=0.8, alpha=0.55)
        r, c = frame["pos"]
        k = cfg.sensor.radius
        ax.add_patch(Rectangle((c - k - 0.5, r - k - 0.5), 2 * k + 1, 2 * k + 1, fill=False,
                               edgecolor=DRONE, lw=1.0))
        ax.plot(c, r, marker="o", ms=3.5, color=DRONE)
    minutes = (frame["t"] + 1) * cfg.dt_s / 60
    mode = "  ·  returning to base" if frame["rtb"] else ""
    fig.suptitle(f"{title}t = {minutes:4.1f} min{mode}", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_gif(cfg: MissionConfig, result, path: str, fps: int = 10, title: str = "") -> None:
    import imageio.v2 as imageio
    plt, _, _ = _mpl()
    images = []
    for fr in result.frames:
        fig = render_frame(cfg, fr, result.path[: fr["t"] + 2], title)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(img[::1, ::1])
        plt.close(fig)
    images += [images[-1]] * (2 * fps)   # hold the last frame
    imageio.mimsave(path, images, duration=1000 / fps, loop=0)
