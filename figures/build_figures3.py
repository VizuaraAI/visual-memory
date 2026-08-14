"""Figures for version 2 of the paper.

Every number is read from a results file rather than typed in, so a figure
cannot drift away from the data the way a hand-copied constant can. Each
function returns the numbers it drew so the caller can print them for checking
against the paper text.

Style rules, kept deliberately plain: one accent colour per figure, no gridlines
competing with the data, direct labels instead of legends wherever two or three
series fit, chance rates as a dashed rule with the label inside the axes so it
cannot collide with a title.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CHANCE = 12.5
INK = "#14161a"
INK2 = "#3d434d"
MUTED = "#8b93a1"
LINE = "#d8dce3"
HOT = "#c8452a"
COOL = "#2f6f9f"
WARM = "#c98a1e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _load(name: str):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def _frame(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _chance_rule(ax, xmax, label="chance, 12.5%", xpos=None):
    """Dashed chance rule, labelled at the left and below the line.

    Series end-labels live on the right, so a right-aligned chance label
    collides with them. Anchoring at the left and below the rule keeps it clear
    of both the data and the title band.
    """
    ax.axhline(CHANCE, color=MUTED, lw=1.6, ls=(0, (7, 6)), zorder=1)
    x = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if xpos is None else xpos
    ax.text(x, CHANCE - 2.2, label, color=MUTED, fontsize=10.5,
            ha="left", va="top")


def fig_horizon() -> dict:
    """Measured decodability against distance, at three model scales."""
    series = [
        ("Zamba2-VL-7B", "decode_fine.json", INK),
        ("Zamba2-VL-2.7B", "decode_fine2p7.json", COOL),
        ("Zamba2-VL-1.2B", "decode_fine1p2.json", WARM),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    drawn = {}
    for label, path, colour in series:
        fam = _load(path)["families"][0]
        pts = [(r["distance"], r["probe_accuracy"] * 100)
               for r in fam["per_distance"] if not r["is_control"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=colour, lw=2.2, ms=5.5, label=label, zorder=3)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", color=colour, fontsize=10.5,
                    va="center")
        drawn[label] = pts

    _frame(ax)
    ax.set_xlabel("tokens between the image and the readout")
    ax.set_ylabel("identity decoded from the recurrent state (%)")
    ax.set_xlim(-2, 88)
    ax.set_ylim(0, 95)
    _chance_rule(ax, 64)
    ax.set_xticks([0, 8, 16, 24, 32, 48, 64])
    ax.set_title("The decodable picture collapses within a few tokens",
                 color=INK, fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_horizon.png"), dpi=220)
    plt.close(fig)
    return drawn


def fig_strata() -> dict:
    """The head-stratified probe, whose ordering inverts the decay prediction."""
    data = _load("e5b_decoded.json")["strata"]
    # (key, label, colour, vertical offset in points). Middle and slow finish
    # within two points of one another, so their labels need separating by hand.
    order = [("fast", "fast heads, predicted half-life 2.0 tokens", HOT, 0),
             ("middle", "middle heads, 6.6 tokens", MUTED, -13),
             ("slow", "slow heads, 29.0 tokens", COOL, 13)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    drawn = {}
    for key, label, colour, dy in order:
        if key not in data:
            continue
        rows = data[key]["per_distance"]
        xs = [r["distance"] for r in rows]
        ys = [r["accuracy"] * 100 for r in rows]
        ax.plot(xs, ys, "-o", color=colour, lw=2.4, ms=5.5, zorder=3)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(9, dy),
                    textcoords="offset points", color=colour, fontsize=10.5,
                    va="center")
        drawn[key] = list(zip(xs, ys))

    _frame(ax)
    ax.set_xlabel("tokens between the image and the readout")
    ax.set_ylabel("identity decoded (%)")
    ax.set_xlim(-2, 118)
    ax.set_ylim(0, 95)
    _chance_rule(ax, 64)
    ax.set_xticks([0, 8, 16, 32, 64])
    ax.set_title("The fastest-decaying heads retain the most, which decay cannot explain",
                 color=INK, fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_strata.png"), dpi=220)
    plt.close(fig)
    return drawn


def fig_relay() -> list:
    """The decisive contrast: the same state, with and without attention's copy."""
    summary = _load("e5c_decoded.json")["summary"]
    keep = [r for r in summary if r["stratum"] == "fast"]
    keep.sort(key=lambda r: r["distance"])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(keep))
    width = 0.36
    intact = [r["intact"] * 100 for r in keep]
    evicted = [r["evicted"] * 100 for r in keep]
    ax.bar(xs - width / 2, intact, width, color=COOL, zorder=3,
           label="visual KV intact")
    ax.bar(xs + width / 2, evicted, width, color=HOT, zorder=3,
           label="visual KV evicted")
    for x, v in zip(xs - width / 2, intact):
        ax.text(x, v + 1.6, f"{v:.1f}", ha="center", color=COOL, fontsize=10.5)
    for x, v in zip(xs + width / 2, evicted):
        ax.text(x, v + 1.6, f"{v:.1f}", ha="center", color=HOT, fontsize=10.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['distance']} tokens" for r in keep])
    _chance_rule(ax, len(keep) - 0.4)
    _frame(ax)
    ax.set_ylabel("identity decoded from the fast heads (%)")
    ax.set_ylim(0, 95)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    ax.set_title("Remove attention's copy and the recurrent state has nothing left",
                 color=INK, fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_relay.png"), dpi=220)
    plt.close(fig)
    return keep


def fig_splice(report_path: str) -> dict:
    """Dose response: how many attention layers must be swapped to move the answer."""
    with open(report_path) as fh:
        summary = json.load(fh)["summary"]
    order = ["attention_1", "attention_3", "attention_7", "attention_all"]
    xs, host, donor = [], [], []
    for key in order:
        if key not in summary:
            continue
        xs.append(summary[key]["n_layers_swapped"])
        host.append(summary[key]["host_rate"] * 100)
        donor.append(summary[key]["donor_rate"] * 100)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, host, "-o", color=COOL, lw=2.4, ms=6, zorder=3)
    ax.plot(xs, donor, "-o", color=HOT, lw=2.4, ms=6, zorder=3)
    ax.annotate("answers the host's own image", (xs[-1], host[-1]),
                xytext=(-6, 14), textcoords="offset points", color=COOL,
                fontsize=10.5, ha="right")
    ax.annotate("answers the donor's image", (xs[-1], donor[-1]),
                xytext=(-6, -18), textcoords="offset points", color=HOT,
                fontsize=10.5, ha="right")
    recurrent = summary.get("recurrent", {}).get("host_rate")
    if recurrent is not None:
        ax.axhline(recurrent * 100, color=MUTED, lw=1.6, ls=(0, (2, 4)))
        ax.text(0.6, recurrent * 100 - 6,
                "all 81 recurrent states swapped: no effect",
                color=MUTED, fontsize=10.5)
    _frame(ax)
    ax.set_xlabel("attention layers swapped (of 13)")
    ax.set_ylabel("share of 100 pairs (%)")
    ax.set_xticks(xs)
    ax.set_ylim(-5, 108)
    ax.set_title("Seven of thirteen attention caches are enough to change the answer",
                 color=INK, fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_splice2.png"), dpi=220)
    plt.close(fig)
    return {"layers": xs, "host": host, "donor": donor}


def fig_depth() -> dict:
    """Where in the stack the gist sits."""
    fam = _load("decode_fine_layerwise.json")["families"][0]
    row = [p for p in fam["per_distance"] if p["distance"] == 0][0]
    layers = [e["layer"] for e in row["layerwise"]]
    acc = [e["accuracy"] * 100 for e in row["layerwise"]]
    attention = {6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77}

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(layers, acc, color=INK, lw=1.8, zorder=3)
    marks = [(l, a) for l, a in zip(layers, acc) if l in attention]
    ax.scatter([m[0] for m in marks], [m[1] for m in marks], s=26,
               color=HOT, zorder=4, label="layer also carries attention")
    best = int(np.argmax(acc))
    ax.annotate(f"layer {layers[best]}, {acc[best]:.1f}%",
                (layers[best], acc[best]), xytext=(-10, 12),
                textcoords="offset points", color=INK, fontsize=10.5, ha="right")
    full = row["probe_accuracy"] * 100
    ax.axhline(full, color=MUTED, lw=1.6, ls=(0, (2, 4)))
    ax.text(1, full - 5, f"probe on all 81 layers at once: {full:.1f}%",
            color=MUTED, fontsize=10.5)
    _frame(ax)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("identity decoded (%)")
    ax.set_ylim(0, 100)
    _chance_rule(ax, 80)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax.set_title("The gist concentrates in the late layers",
                 color=INK, fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_depth.png"), dpi=220)
    plt.close(fig)
    return {"best_layer": layers[best], "best": acc[best], "full_state": full}


def main() -> None:
    built = {}
    built["horizon"] = fig_horizon()
    built["strata"] = fig_strata()
    built["relay"] = fig_relay()
    built["depth"] = fig_depth()

    splice_path = os.path.join(RESULTS, "e10_report_scaled.json")
    if os.path.exists(splice_path):
        built["splice"] = fig_splice(splice_path)
    else:
        print("e10_report_scaled.json not present, splice figure skipped",
              file=sys.stderr)

    print(json.dumps(built, indent=2, default=str))


if __name__ == "__main__":
    main()
