"""Shared Matplotlib style for the course figures.

The categorical palette was checked with a colour-vision-deficiency validator
(worst adjacent CVD Delta E 9.1 for the first five slots). Several colours
have a contrast below 3:1 against the white background, so figures always
add a legend or direct labels: colour is never the only way to read them.

Rules used in the lessons:
* categorical colours are assigned in this fixed order, never cycled;
* magnitude uses one hue from light to dark (SEQUENTIAL_CMAP);
* signed values (correlations) use blue <-> red with a neutral grey middle;
* one y-axis per chart (no dual axes).
"""

from __future__ import annotations

from pathlib import Path

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
NEUTRAL = "#8a8a85"
TEXT = "#2b2b29"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("lab_blue", BLUE_RAMP)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("lab_diverging", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a32a2a"])


def apply_style() -> None:
    """Thin marks, recessive grid and axes, readable text."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c9c9c4",
            "axes.labelcolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e1",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.frameon": False,
            "font.size": 9.5,
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, name: str, folder: str = "outputs/module04") -> Path:
    """Save a figure as PNG under outputs/ and close it. Returns the path."""
    import matplotlib.pyplot as plt

    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def label_line_ends(ax, x_last, values: dict, min_gap: float | None = None, fontsize: float = 8) -> None:
    """Write each series name at the end of its line, nudging labels apart.

    ``values`` maps a label to the last y value of its line. Labels closer than
    ``min_gap`` (default: 4 % of the y range) are pushed apart vertically.
    """
    if min_gap is None:
        low, high = ax.get_ylim()
        min_gap = 0.04 * (high - low)
    placed: list[tuple[str, float]] = []
    for label, y in sorted(values.items(), key=lambda item: item[1]):
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((label, y))
    for label, y in placed:
        ax.annotate(label, (x_last, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=fontsize, color=TEXT)
