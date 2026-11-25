"""Compare the released matched predictions with HST and the solved source plane."""

import matplotlib.pyplot as plt
import numpy as np
from autolens.point.plot.source_plane_panel import source_plane_data_from
from inputs import FIELD
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

dark = "#172b36"
amber = "#ffbd49"
cyan = "#5ee8fb"
STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titleweight": "semibold",
    "axes.labelcolor": dark,
    "text.color": dark,
    "axes.titlecolor": dark,
}


def sky_axes(ax):
    ax.set_xlabel("x  ·  west (″)")
    ax.set_ylabel("y  ·  north (″)")
    ax.tick_params(labelsize=9)


def ellipse(ax, pos, c, color):
    vals, vec = np.linalg.eigh(c[::-1, ::-1])
    direction = vec[:, -1]
    ax.add_patch(
        Ellipse(
            pos[::-1],
            width=2 * np.sqrt(vals[-1]),
            height=2 * np.sqrt(vals[0]),
            angle=np.rad2deg(np.arctan2(direction[1], direction[0])),
            facecolor="none",
            edgecolor=color,
            lw=1,
            alpha=0.8,
        )
    )


def centre(ax, c):
    ax.scatter(c[1], c[0], marker="+", s=100, color="#965f15", lw=1.8, zorder=4)


def bounds_at(pos, r):
    y, x = pos
    return x - r, x + r, y - r, y + r


def published_offsets(fit, field_positions, residuals, mosaic, output):
    """Use the released image matches and back-trace the observed positions with the fixed model."""
    name = fit.dataset.name
    ids = np.asarray(fit.dataset.image_ids)
    theta = np.asarray(fit.dataset.positions.array)
    offset = np.asarray([residuals[ident] for ident in ids])
    dist = np.linalg.norm(offset, axis=1)
    rms = np.sqrt(np.mean(dist**2))
    source_plane = source_plane_data_from(fit.positions)
    beta, cov, sigma = source_plane.traced, source_plane.covariances, source_plane.sigma
    c = source_plane.centre
    base = mosaic.raster(FIELD, 0.09)
    good = base[np.isfinite(base) & (base != 0)]
    lo, hi = np.percentile(good, [2, 99.8])

    def image(ax, bounds=FIELD):
        pixels = base if bounds == FIELD else mosaic.raster(bounds, 0.06)
        stretched = np.arcsinh(np.clip((pixels - lo) / (hi - lo), 0, 1) * 15) / np.arcsinh(15)
        ax.imshow(
            stretched,
            extent=bounds,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal")

    markers = ["o", "s", "D", "^", "v", "P"]
    count = len(ids)
    cols = min(3, count)
    rows = (count + cols - 1) // cols
    side = 2.70
    tile = {2: 2.00, 3: 1.57}[cols]
    gap = 0.31
    left = 0.69
    right = left + side + 0.63
    top_h = max(side, rows * tile + (rows - 1) * 0.42)
    bot = 0.54
    summary_y = bot
    top_y = bot + side + (0.94 if rows == 2 else 0.69)
    top_end = top_y + top_h
    width = right + max(cols * tile + (cols - 1) * gap, side + 1.43) + 0.22
    height = top_end + 0.82
    fig = plt.figure(figsize=(width, height))

    def panel(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    def words(x, y, content, **kwargs):
        return fig.text(x / width, y / height, content, transform=fig.transFigure, **kwargs)

    words(
        left - 0.08,
        height - 0.08,
        f"MACS J0416   |   Family {name}   ·   z = {fit.dataset.redshift:.3f}",
        fontsize=17,
        weight="semibold",
        va="top",
    )
    # Use the space beside the field for the image key, zooms and their true sky coordinates.
    field = panel(left, top_y + top_h - side, side, side)
    image(field)
    sky_axes(field)
    field.set_title("Cluster field", loc="left", pad=6)
    field.scatter(
        field_positions[:, 1],
        field_positions[:, 0],
        s=5,
        facecolors="none",
        edgecolors="0.77",
        lw=0.48,
    )
    field.scatter(theta[:, 1], theta[:, 0], s=59, facecolors="none", edgecolors=amber, lw=1.3)
    used = []
    for j in range(count):
        y, x = theta[j]
        # Try nearby label positions before lengthening a leader in crowded families.
        candidates = [
            (x + sx * dx, y + sy * dy)
            for dx, dy in ((5, 4), (8, 7), (7, 12), (12, 7), (18, 12), (23, 16))
            for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))
        ]
        tx, ty = next(
            (
                (u, v)
                for u, v in candidates
                if FIELD[0] + 8 < u < FIELD[1] - 8
                and FIELD[2] + 4 < v < FIELD[3] - 4
                and all(abs(u - a) > 17 or abs(v - b) > 7 for a, b in used)
                and all(np.hypot(u - theta[k, 1], v - theta[k, 0]) > 2.4 for k in range(count))
            ),
            candidates[-1],
        )
        used.append((tx, ty))
        field.annotate(
            str(ids[j]),
            (x, y),
            (tx, ty),
            ha="center",
            va="center",
            color=amber,
            fontsize=9,
            bbox=dict(facecolor="#111", alpha=0.75, edgecolor="none", pad=0.5),
            arrowprops=dict(arrowstyle="-", color=amber, lw=0.65),
        )
    img_half = max(1.45, 1.40 * np.max(np.abs(offset)), 1.5 * sigma.max())
    stack = rows * tile + (rows - 1) * 0.42
    # One row leaves room for a marker key directly above it.
    ztop = top_end - (top_h - stack) / 2 - (0.10 if rows == 1 else 0)
    ztitle = ztop + 0.28 if rows == 1 else top_end + 0.23
    words(right, ztitle, "Image-plane zooms", fontsize=12, weight="semibold", va="bottom")
    for j in range(count):
        col = j % cols
        row = j // cols
        ax = panel(right + col * (tile + gap), ztop - (row + 1) * tile - row * 0.42, tile, tile)
        image(ax, bounds_at(theta[j], img_half))
        ax.add_patch(Circle(theta[j, ::-1], sigma[j], fill=False, edgecolor=amber, lw=1.05, ls=":"))
        ax.scatter(
            theta[j, 1], theta[j, 0], marker="o", s=60, facecolors="none", edgecolors=amber, lw=1.65
        )
        pp = theta[j] + offset[j]
        ax.plot(
            pp[1],
            pp[0],
            marker="x",
            ls="none",
            ms=7,
            color=cyan,
            mew=1.65,
            path_effects=[pe.withStroke(linewidth=3.3, foreground="#20333a")],
        )
        ax.set_title(ids[j], fontsize=10, pad=4)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=8, pad=1)
    # A real legend matches the marker colours on the photographic background.
    handles = [
        Line2D([], [], marker="o", ls="none", mec=amber, mfc="none", mew=1.5, label="Observed"),
        Line2D([], [], marker="x", ls="none", color=cyan, mew=1.6, label="Published prediction"),
        Line2D([], [], ls=":", color=amber, label="1σ"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(
            (right + (cols * tile + (cols - 1) * gap) / 2) / width,
            (ztop - stack - 0.25) / height,
        ),
        ncol=3,
        fontsize=8,
        labelcolor="white",
        handletextpad=0.4,
        columnspacing=1.1,
        facecolor="#24343a",
        framealpha=1,
        edgecolor="none",
    )
    ax = panel(left, summary_y, side, side)
    values = dist
    order = np.argsort(-dist)
    ax.barh(np.arange(count), values[order], height=0.53, color="#377e90")
    ax.set_yticks(np.arange(count), ids[order])
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, values[order[0]] * 1.24)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
    for k, j in enumerate(order):
        ax.text(
            values[j] + 0.018 * values[order[0]], k, f"{values[j]:.2f}", va="center", fontsize=9
        )
    ax.set_xlabel("Observed–predicted separation (″)")
    ax.set_title(f"Per-image offsets   ·   RMS {rms:.2f}″", loc="left", fontsize=11, pad=6)
    ax = panel(right, summary_y, side, side)
    ax.set_facecolor("#f1f5f5")
    err = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    lows = np.minimum(np.min(beta - err, axis=0), c)
    highs = np.maximum(np.max(beta + err, axis=0), c)
    b = bounds_at((lows + highs) / 2, max(0.055, np.max(highs - lows) * 0.64))
    ax.set(xlim=b[:2], ylim=b[2:], aspect="equal")
    keys = []
    for j in range(count):
        ellipse(ax, beta[j], cov[j], "#438698")
        (point,) = ax.plot(
            beta[j, 1],
            beta[j, 0],
            marker=markers[j],
            ls="none",
            ms=6,
            mew=0.6,
            mec="white",
            color="#14667b",
            label=ids[j],
        )
        keys.append(point)
    centre(ax, c)
    sky_axes(ax)
    ax.set_title("All back-traced images", fontsize=13, loc="left", pad=7)
    legend = panel(right + side + 0.08, summary_y + 0.04, width - right - side - 0.14, side - 0.08)
    legend.axis("off")
    keys.extend(
        [
            Line2D(
                [],
                [],
                marker="+",
                ls="none",
                color="#965f15",
                ms=10,
                mew=1.6,
                label="Solved centre",
            ),
            Line2D([], [], color="#438698", lw=1, label="1σ ellipse"),
        ]
    )
    legend.legend(
        handles=keys,
        title="Source plane",
        loc="center left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0,
        handlelength=1.3,
        handletextpad=0.45,
        labelspacing=0.56,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
