"""Rebuilt figures.

The first version drew all three property families as lines on one axis. All
three sit near the 12.5% chance rate for most of their length, so the plot read
as noise and buried the only dramatic movement in the data. These replace it:

  fig_cliff  one story per panel: the recurrent state empties, behaviour does not
  fig_gist   the coarse-to-fine gradient, as bars at the moment of writing
  fig_splice the causal result, as a slope between two conditions
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#16130d"
INK2 = "#3a352b"
MUTED = "#6d665a"
FAINT = "#a49c8c"
LINE = "#e7e2d5"
SLATE = "#155e8c"
SHELF = "#c0641a"
HOT = "#b3006b"
CHANCE = 12.5


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def tidy(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)


def load(path: str) -> dict:
    with open(path) as fh:
        report = json.load(fh)
    out = {}
    for fam in report["families"]:
        rows = [r for r in fam["per_distance"] if not r["is_control"]]
        rows.sort(key=lambda r: r["distance"])
        out[fam["family"]] = {
            "d": [r["distance"] for r in rows],
            "probe": [r["probe_accuracy"] * 100 for r in rows],
            "behaviour": [r.get("behaviour_accuracy", float("nan")) * 100 for r in rows],
            "control": next(
                (r["probe_accuracy"] * 100 for r in fam["per_distance"] if r["is_control"]),
                None,
            ),
        }
    return out


def fig_cliff(data: dict, out_dir: str) -> None:
    """The single most important panel: the state empties, behaviour does not."""
    fam = data["identity"]
    xs = np.arange(len(fam["d"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    tidy(ax)

    ax.plot(xs, fam["behaviour"], color=MUTED, linewidth=2.4, linestyle=(0, (6, 3)),
            marker="s", markersize=6, zorder=3)
    ax.plot(xs, fam["probe"], color=SLATE, linewidth=3.0, marker="o", markersize=8, zorder=4)

    ax.axhline(CHANCE, color=FAINT, linestyle=(0, (3, 4)), linewidth=1.6, zorder=1)
    ax.text(xs[-1], CHANCE + 3.2, "chance", va="bottom", ha="right", color=FAINT, fontsize=12)

    # Name the collapse without drawing over the curve that already shows it.
    ax.text(xs[0] + 0.42, 55, "collapse within\n32 tokens", color=HOT, fontsize=13.5,
            ha="left", va="center", linespacing=1.35)

    ax.text(xs[0] - 0.02, fam["probe"][0] + 5.5, f"{fam['probe'][0]:.1f}%",
            color=SLATE, fontsize=15, ha="left", va="bottom")
    ax.text(xs[-1] - 0.1, 103, "the model's own answer",
            color=MUTED, fontsize=13, ha="right", va="top")
    ax.text(xs[4], fam["probe"][4] - 6.5, "decoded from the recurrent state",
            color=SLATE, fontsize=13, ha="center", va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in fam["d"]])
    ax.set_xlim(-0.35, len(xs) - 0.4)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("tokens between the image and the readout")
    ax.set_title("The recurrent state empties. The answer does not change.",
                 loc="left", pad=14)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_cliff.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_gist(data: dict, out_dir: str) -> None:
    """The coarse-to-fine gradient at the moment of writing."""
    labels = ["which object\nis present", "how many\nthere are", "where it is\nin the frame"]
    keys = ["identity", "count", "position"]
    values = [data[k]["probe"][0] for k in keys]
    colours = [SLATE, SHELF, SHELF]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    tidy(ax, grid_axis="x")
    ys = np.arange(len(keys))[::-1]
    ax.set_ylim(-0.9, len(keys) - 0.35)

    ax.barh(ys, values, height=0.52, color=colours, zorder=3)
    ax.axvline(CHANCE, color=INK2, linestyle=(0, (3, 4)), linewidth=1.8, zorder=4)
    ax.text(CHANCE + 1.6, ys[-1] - 0.46, "chance, 12.5%", color=INK2, fontsize=12.5,
            va="center", ha="left")

    for y, v in zip(ys, values):
        ax.text(v + 1.6, y, f"{v:.1f}%", va="center", ha="left",
                color=INK, fontsize=15)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=13.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("recovered from the recurrent state alone, at the moment of writing")
    ax.set_title("Gist survives. Particulars do not.", loc="left", pad=14)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_gist.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_splice(summary: dict, out_dir: str) -> None:
    """The causal result as a slope: what the answer follows, per condition."""
    conditions = ["none", "recurrent", "attention", "both"]
    labels = ["no swap", "recurrent state\nexchanged (81 layers)",
              "attention cache\nexchanged (13 layers)", "both\nexchanged"]
    block = summary["identity"]
    n = max(block["none"]["n"], 1)
    own = [block[c]["follows_host"] / n * 100 for c in conditions]
    other = [block[c]["follows_donor"] / n * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    tidy(ax)
    xs = np.arange(len(conditions))

    ax.plot(xs, own, color=SLATE, linewidth=2.8, marker="o", markersize=10, zorder=4)
    ax.plot(xs, other, color=SHELF, linewidth=2.8, marker="s", markersize=9, zorder=4)

    for x, v in zip(xs, own):
        ax.text(x, v + 3.4, f"{v:.0f}%", ha="center", color=SLATE, fontsize=13.5)
    for x, v in zip(xs, other):
        ax.text(x, v - 7.6, f"{v:.0f}%", ha="center", color=SHELF, fontsize=13.5)

    ax.text(xs[0] - 0.28, own[0] + 11, "answers with its own image",
            color=SLATE, fontsize=13.5, ha="left")
    ax.text(xs[-1] + 0.12, other[-1] + 6, "answers with the\nother image",
            color=SHELF, fontsize=13.5, ha="right", va="bottom", linespacing=1.3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(-0.45, len(xs) - 0.35)
    ax.set_ylim(-8, 80)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20", "40", "60%"])
    ax.set_ylabel("of 30 paired trials")
    ax.set_title("Destroying 81 recurrent states changes nothing.\n"
                 "Replacing 13 attention caches flips the answer.",
                 loc="left", pad=14)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_splice.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay-json", required=True)
    parser.add_argument("--splice-json", required=True)
    parser.add_argument("--out-dir", default="figures/pub")
    args = parser.parse_args()

    configure()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load(args.decay_json)
    with open(args.splice_json) as fh:
        summary = json.load(fh)

    fig_cliff(data, args.out_dir)
    fig_gist(data, args.out_dir)
    fig_splice(summary, args.out_dir)
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
