"""Publication figures for the visual memory study.

Consumes the JSON written by analysis/decode_decay.py and the splice summary,
and writes PDF and PNG into figures/pub/. Bright ground throughout, serif
type, no chartjunk, one idea per panel.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANCE = 0.125

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
CHANCE_LINE = "#9a9a9a"

# One hue per property family, ordered from coarse to fine.
FAMILY_COLOUR = {
    "identity": "#1f5fa9",
    "count": "#c2571a",
    "position": "#2f7d5f",
}
FAMILY_LABEL = {
    "identity": "which object",
    "count": "how many",
    "position": "where",
}

CHANNEL_COLOUR = {
    "none": "#8a8a8a",
    "recurrent": "#1f5fa9",
    "attention": "#c2571a",
}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _series(family: dict) -> tuple[list[int], list[float], list[float]]:
    rows = [r for r in family["per_distance"] if not r["is_control"]]
    rows.sort(key=lambda r: r["distance"])
    distances = [r["distance"] for r in rows]
    probe = [r["probe_accuracy"] for r in rows]
    behaviour = [r.get("behaviour_accuracy", float("nan")) for r in rows]
    return distances, probe, behaviour


def _xpos(distances: list[int]) -> list[float]:
    """Log-like axis that still shows distance zero."""
    return [np.log10(d + 32) for d in distances]


def fig_decay(report: dict, out_dir: str) -> None:
    """What survives in the state, as a function of distance from the image."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    tidy(ax)

    for family in report["families"]:
        name = family["family"]
        distances, probe, _ = _series(family)
        xs = _xpos(distances)
        ax.plot(
            xs, probe, marker="o", markersize=5, linewidth=1.8,
            color=FAMILY_COLOUR[name], label=FAMILY_LABEL[name],
        )
        control = [r for r in family["per_distance"] if r["is_control"]]
        if control:
            ax.scatter(
                [xs[0] - 0.22], [control[0]["probe_accuracy"]], marker="x",
                s=42, color=FAMILY_COLOUR[name], linewidth=1.6,
            )

    ax.axhline(CHANCE, color=CHANCE_LINE, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(
        _xpos([2048])[0], CHANCE + 0.012, "chance", ha="right", va="bottom",
        color=MUTED, fontsize=9.5,
    )

    distances, _, _ = _series(report["families"][0])
    ax.set_xticks(_xpos(distances))
    ax.set_xticklabels([str(d) for d in distances])
    ax.set_xlabel("tokens between the picture and the readout")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "What is still readable from the recurrent memory", loc="left", pad=10
    )
    ax.legend(loc="upper right")
    fig.text(
        0.125, -0.02,
        "Crosses at left: the same probe with the picture never shown.",
        fontsize=9, color=MUTED,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_decay.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_gap(report: dict, out_dir: str) -> None:
    """What the state holds against what the model can say."""
    families = report["families"]
    fig, axes = plt.subplots(1, len(families), figsize=(3.3 * len(families), 3.5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        tidy(ax)
        name = family["family"]
        distances, probe, behaviour = _series(family)
        xs = _xpos(distances)
        ax.plot(xs, probe, marker="o", markersize=4.5, linewidth=1.8,
                color=FAMILY_COLOUR[name], label="read from state")
        ax.plot(xs, behaviour, marker="s", markersize=4.5, linewidth=1.8,
                color=MUTED, linestyle=(0, (5, 2)), label="model's answer")
        ax.axhline(CHANCE, color=CHANCE_LINE, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_xticks(_xpos([0, 128, 2048]))
        ax.set_xticklabels(["0", "128", "2048"])
        ax.set_title(FAMILY_LABEL[name], loc="left")
        ax.set_xlabel("tokens after the picture")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_gap.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_splice(summary: dict, out_dir: str) -> None:
    """Which memory the answer follows when the two are put in conflict."""
    families = [k for k in ("identity", "count", "position") if k in summary]
    conditions = ("none", "recurrent", "attention")
    labels = {
        "none": "no swap",
        "recurrent": "recurrent memory\nreplaced",
        "attention": "attention cache\nreplaced",
    }

    fig, axes = plt.subplots(1, len(families), figsize=(3.4 * len(families), 3.8), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        tidy(ax)
        block = summary[family]
        xs = np.arange(len(conditions))
        host = [block[c]["follows_host"] / max(block[c]["n"], 1) for c in conditions]
        donor = [block[c]["follows_donor"] / max(block[c]["n"], 1) for c in conditions]
        width = 0.36
        ax.bar(xs - width / 2, host, width, color="#3d6fa8", label="answers with its own picture")
        ax.bar(xs + width / 2, donor, width, color="#c2571a", label="answers with the other picture")
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[c] for c in conditions], fontsize=9)
        ax.set_title(FAMILY_LABEL[family], loc="left")
    axes[0].set_ylabel("fraction of trials")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.suptitle(
        "Where the picture is read from", x=0.02, ha="left", fontsize=12, y=1.0
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_splice.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_depth(report: dict, out_dir: str, attention_layers: tuple[int, ...]) -> None:
    """Where in depth the picture is decodable, with attention layers marked."""
    target = None
    for family in report["families"]:
        rows = [r for r in family["per_distance"] if r["distance"] == 0]
        if rows and "layerwise" in rows[0]:
            target = (family["family"], rows[0]["layerwise"])
            break
    if target is None:
        return

    name, layerwise = target
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    tidy(ax)
    layers = [row["layer"] for row in layerwise]
    accuracy = [row["accuracy"] for row in layerwise]
    ax.plot(layers, accuracy, linewidth=1.6, color=FAMILY_COLOUR[name])
    for layer in attention_layers:
        ax.axvline(layer, color="#c2571a", linewidth=0.8, alpha=0.35)
    ax.axhline(CHANCE, color=CHANCE_LINE, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy")
    ax.set_title(
        f"Depth profile, {FAMILY_LABEL[name]} (orange lines: the 13 attention layers)",
        loc="left", pad=10,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_depth.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


ATTENTION_LAYERS = (6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay-json", required=True)
    parser.add_argument("--splice-json", default=None)
    parser.add_argument("--out-dir", default="figures/pub")
    args = parser.parse_args()

    configure()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.decay_json) as fh:
        report = json.load(fh)
    fig_decay(report, args.out_dir)
    fig_gap(report, args.out_dir)
    fig_depth(report, args.out_dir, ATTENTION_LAYERS)

    if args.splice_json and os.path.exists(args.splice_json):
        with open(args.splice_json) as fh:
            summary = json.load(fh)
        fig_splice(summary, args.out_dir)

    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
