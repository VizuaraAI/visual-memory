"""One visual system for every figure in the paper.

The palette is the website's, validated with the dataviz skill's checker
(lightness band, chroma floor, CVD separation, normal-vision separation,
contrast against the surface). Colour means the same thing in every figure:

  * slate blue      the untouched condition, and the primary measured series
  * magenta (hot)   the intervention, and the fast stratum
  * orange (shelf)  the second comparison series (2.7B scale, mid tercile)
  * green (good)    the third comparison series (1.2B scale)

Conventions applied everywhere:

  * Eyebrow titles: letter-spaced muted uppercase, never bold black headlines.
  * Direct labels at line ends with no legend boxes; identity never colour
    alone.
  * 2.2pt lines with round caps, >=7pt markers ringed in the surface colour.
  * The chance rule is dashed, muted, labelled clear of data.
  * No top/right spines; hairline left/bottom; warm-white surface.
  * The finding is annotated on the figure itself, the way a good chalkboard
    would, so a reader who only looks at figures still gets the argument.
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects as _pe


def punch(size: float = 2.6):
    """Stroke a text label in the surface colour so it reads over rules."""
    return [_pe.withStroke(linewidth=size, foreground="#ffffff")]

# Website palette (validated).
BLUE = "#155e8c"    # slate: untouched / primary
RED = "#b3006b"     # hot: intervention / fast stratum
GREEN = "#1c7a55"   # good: third series
SHELF = "#c0641a"   # orange: second series
PURPLE = SHELF      # legacy alias; the old purple slot now carries shelf
SERIES = (BLUE, RED, SHELF, GREEN)

INK = "#16130d"
INK2 = "#3a352b"
MUTED = "#6d665a"
FAINT = "#a49c8c"
RULE = "#e7e2d5"
PAPERBG = "#ffffff"

CHANCE = 12.5

# Text width of the paper body at 1.1in margins on US letter.
TEXT_WIDTH = 6.3


def _register_fonts() -> None:
    for path in glob.glob(os.path.expanduser("~/.fonts/SourceSans3*.ttf")):
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass


def apply() -> None:
    _register_fonts()
    plt.rcParams.update({
        "font.family": ["Source Sans 3", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.edgecolor": RULE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK2,
        "axes.labelsize": 10.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 5,
        "figure.facecolor": PAPERBG,
        "axes.facecolor": PAPERBG,
        "savefig.facecolor": PAPERBG,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.linewidth": 2.2,
    })


def frame(ax) -> None:
    """Drop the top and right spines and soften the rest."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)


def grid_y(ax) -> None:
    """Hairline horizontal gridlines behind the data, where they help."""
    ax.grid(axis="y", color=RULE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _track(text: str) -> str:
    """Letter-space a short label with hair spaces (matplotlib has no tracking)."""
    return " ".join(text.upper())


def title(ax, text: str, size: float = 9.0) -> None:
    """Eyebrow title: letter-spaced muted caps, left aligned."""
    ax.set_title(_track(text), color=FAINT, fontsize=size, loc="left", pad=10,
                 fontweight="semibold")


def line(ax, x, y, colour, lw: float = 2.2, marker: bool = True, ms: float = 7.0,
         zorder: int = 3, ls: str = "-", alpha: float = 1.0):
    """A series in the house style: round-capped line, ringed markers."""
    (ln,) = ax.plot(x, y, ls, color=colour, lw=lw, zorder=zorder, alpha=alpha)
    if marker:
        ax.plot(x, y, "o", color=colour, ms=ms, markeredgecolor=PAPERBG,
                markeredgewidth=1.6, zorder=zorder + 1, alpha=alpha)
    return ln


def chance_rule(ax, label: str = "chance, 12.5%", side: str = "left") -> None:
    """Dashed chance line, labelled clear of the data and the end labels."""
    ax.axhline(CHANCE, color=MUTED, lw=1.2, ls=(0, (6, 5)), zorder=1)
    lo, hi = ax.get_xlim()
    x = lo + 0.015 * (hi - lo) if side == "left" else hi - 0.015 * (hi - lo)
    ax.text(x, CHANCE - 1.8, label, color=MUTED, fontsize=9,
            ha="left" if side == "left" else "right", va="top")


def chance_rule_above(ax, label: str = "chance, 12.5%") -> None:
    """Chance rule labelled above the line, for panels whose bars sit low."""
    ax.axhline(CHANCE, color=MUTED, lw=1.2, ls=(0, (6, 5)), zorder=1)
    lo, hi = ax.get_xlim()
    ax.text(lo + 0.015 * (hi - lo), CHANCE + 1.8, label, color=MUTED,
            fontsize=9, ha="left", va="bottom")


def end_label(ax, x, y, text: str, colour: str, dy: float = 0.0) -> None:
    """Label a series at its last point: a colour key dot plus ink text."""
    ax.annotate("●", (x, y), xytext=(7, dy), textcoords="offset points",
                color=colour, fontsize=7, va="center")
    ax.annotate(text, (x, y), xytext=(16, dy), textcoords="offset points",
                color=INK2, fontsize=9.5, va="center", fontweight="medium")


def percent_axis(ax, top: float = 100.0) -> None:
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75, 100] if top > 90 else [0, 20, 40, 60, 80])
    ax.set_yticklabels([f"{int(t)}%" for t in ax.get_yticks()])


def note(ax, x, y, text: str, colour: str = MUTED, size: float = 9.0,
         ha: str = "left", va: str = "center", weight: str = "normal",
         stroke: bool = False, box: bool = False) -> None:
    """A short teaching annotation inside the panel."""
    ax.text(x, y, text, color=colour, fontsize=size, ha=ha, va=va,
            fontweight=weight, linespacing=1.35,
            path_effects=punch() if stroke else None,
            bbox=dict(facecolor=PAPERBG, edgecolor="none", pad=1.2) if box else None,
            zorder=6)


def gap_bracket(ax, x, y_hi, y_lo, label_hi: str, label_lo: str,
                label_mid: str, colour: str = RED) -> None:
    """A two-headed vertical bracket marking the distance between two values."""
    ax.annotate("", xy=(x, y_hi), xytext=(x, y_lo),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4,
                                shrinkA=2, shrinkB=2), zorder=6)
    ax.annotate(label_hi, (x, y_hi), xytext=(-8, 5), textcoords="offset points",
                color=INK2, fontsize=9, ha="right", va="bottom",
                fontweight="semibold", zorder=6)
    ax.annotate(label_lo, (x, y_lo), xytext=(-8, -5), textcoords="offset points",
                color=MUTED, fontsize=9, ha="right", va="top", zorder=6)
    ax.annotate(label_mid, (x, (y_hi + y_lo) / 2), xytext=(-8, 0),
                textcoords="offset points", color=colour, fontsize=9,
                ha="right", va="center", fontweight="semibold", zorder=6)


def stack_end_labels(ax, x, items, gap=9.0, pad_pt=8, floor=None):
    """Direct labels at a shared x, nudged apart when series converge.

    `items` is a sequence of (y, text, colour), `gap` is the minimum vertical
    separation in data units, and `floor` keeps the lowest label clear of
    something already occupying the bottom of the panel, normally the chance
    rule and its own label.
    """
    ordered = sorted(items, key=lambda item: item[0])
    placed = []
    for y, text, colour in ordered:
        if floor is not None:
            y = max(y, floor)
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, text, colour))

    top = ax.get_ylim()[1]
    overshoot = placed[-1][0] - (top - gap * 0.5)
    if overshoot > 0:                      # whole stack slides down together
        placed = [(y - overshoot, t, c) for y, t, c in placed]

    for (y, text, colour), (y0, _, _) in zip(placed, ordered):
        if abs(y - y0) > 0.5:
            ax.annotate("", xy=(x, y0), xytext=(x, y), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=colour, lw=0.8,
                                        alpha=0.55, shrinkA=1, shrinkB=1),
                        zorder=2)
        ax.annotate("●", (x, y), xytext=(pad_pt - 1, 0),
                    textcoords="offset points", color=colour, fontsize=7,
                    va="center")
        ax.annotate(text, (x, y), xytext=(pad_pt + 8, 0),
                    textcoords="offset points", color=INK2, fontsize=9.5,
                    va="center", fontweight="medium")
