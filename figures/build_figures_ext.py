"""Extra figures for the extended (arXiv) version of the paper.

The conference version has to earn every inch, so it carries only the figures
that are evidence. This module adds the ones that are explanation: what the
decay parameters look like across nine thousand heads, how the predicted
retention curve is built, where in the stack the fast heads live, and what the
experiments physically do to the token stream.

Same rule as build_figures4.py: every number is read from a results file, and
each function returns what it drew so the caller can check it against the text.
The two schematics are the exception and are marked as such; they encode the
architecture and the protocol, both of which are stated in the paper's prose.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
style.apply()

S = style

SCALES = (
    ("Zamba2-VL-7B", "e5_dynamics.npz", S.BLUE),
    ("Zamba2-VL-2.7B", "e5_dynamics_2p7b.npz", S.SHELF),
    ("Zamba2-VL-1.2B", "e5_dynamics_1p2b.npz", S.GREEN),
)
ATTENTION_LAYERS_7B = (6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77)


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def dynamics(name):
    return np.load(os.path.join(RESULTS, name))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def decay_rate(data):
    """Per-head decay per token: |A| times the mean gate over the filler."""
    return (data["abs_a"] * data["mean_dt_first32"]).ravel()


# --------------------------------------------------------------------------
def fig_halflife_distribution():
    """What the horizon looks like across every head, and at every scale.

    The paper reports a median and an interquartile range. Those three numbers
    hide the shape, which is the interesting part: a mass of heads with
    single-digit half-lives and a thin tail of slow ones that carries whatever
    long-distance signal survives.
    """
    seven = dynamics("e5_dynamics.npz")
    half = seven["half_life_first32"].ravel()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.85), gridspec_kw={"wspace": 0.34}
    )

    bins = np.logspace(np.log10(max(half.min(), 0.05)), np.log10(half.max()), 46)
    counts, _, _ = left.hist(half, bins=bins, color=S.BLUE, alpha=0.85, lw=0)
    q25, q50, q75 = np.percentile(half, [25, 50, 75])
    left.set_ylim(0, counts.max() * 1.34)     # headroom for the two callouts
    top = left.get_ylim()[1]
    left.axvspan(q25, q75, color=S.BLUE, alpha=0.12, lw=0, zorder=0)
    left.axvline(q50, color=S.INK, lw=1.3, zorder=4)
    left.annotate(f"median\n{q50:.1f} tokens", (q50, top * 0.97),
                  xytext=(-9, 0), textcoords="offset points", color=S.INK,
                  fontsize=9.5, ha="right", va="top", linespacing=1.25)
    left.axvline(32, color=S.RED, lw=1.2, ls=(0, (5, 4)), zorder=4)
    left.annotate(f"32 tokens\n{(half < 32).mean() * 100:.0f}% of heads below",
                  (32, top * 0.97), xytext=(9, 0), textcoords="offset points",
                  color=S.RED, fontsize=9.5, ha="left", va="top",
                  linespacing=1.25)
    S.frame(left)
    left.set_xscale("log")
    left.set_xticks([0.1, 1, 10, 100, 1000])
    left.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    left.set_xlabel("predicted half-life (tokens)")
    left.set_ylabel("heads")
    S.title(left, "Half-lives of 9,072 heads")

    # All three curves finish at 100%, so end labels would pile up; the key
    # goes in the empty upper left, which no curve reaches.
    for offset, (name, path, colour) in enumerate(SCALES):
        hl = np.sort(dynamics(path)["half_life_first32"].ravel())
        share = np.arange(1, len(hl) + 1) / len(hl) * 100
        right.plot(hl, share, color=colour, lw=1.9, zorder=3)
        right.text(0.075, 97 - 9 * offset, "\u25cf", color=colour,
                   fontsize=7, va="center", ha="left")
        right.text(0.135, 97 - 9 * offset, name.replace("Zamba2-VL-", ""),
                   color=S.INK2, fontsize=9.5, va="center", ha="left",
                   fontweight="medium")
    right.axvline(32, color=S.MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
    right.text(29, 8, "32 tokens", color=S.MUTED, fontsize=9.5, ha="right")
    S.frame(right)
    right.set_xscale("log")
    right.set_xlim(0.05, 6000)
    right.set_ylim(0, 100)
    right.set_yticks([0, 25, 50, 75, 100])
    right.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    right.set_xticks([0.1, 1, 10, 100, 1000])
    right.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    right.set_xlabel("predicted half-life (tokens)")
    right.set_ylabel("share of heads below")
    S.title(right, "All three scales")

    return save(fig, "e1_halflife_distribution.png"), {
        "median": float(q50), "q25": float(q25), "q75": float(q75),
        "fraction_below_32": float((half < 32).mean()),
        "n_heads": int(half.size),
    }


# --------------------------------------------------------------------------
def fig_halflife_by_layer():
    """Where the fast heads sit, which is the confound the paper has to face.

    The stratified probe splits heads globally rather than within a layer, so
    the strata draw unevenly from the stack. Showing the per-layer spread makes
    that visible instead of asking the reader to take the summary statistics on
    trust: the fast third really is concentrated early, in the layers that
    decode worst, and it wins anyway.
    """
    data = dynamics("e5_dynamics.npz")
    half = data["half_life_first32"]
    layers = np.arange(half.shape[0])
    med = np.median(half, axis=1)
    lo = np.percentile(half, 25, axis=1)
    hi = np.percentile(half, 75, axis=1)

    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH, 2.6))
    ax.fill_between(layers, lo, hi, color=S.BLUE, alpha=0.16, lw=0, zorder=2)
    ax.plot(layers, med, color=S.BLUE, lw=1.8, zorder=3)
    overall = np.median(half)
    ax.axhline(overall, color=S.MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
    ax.text(83.5, overall, f"median over\nall heads: {overall:.1f}",
            color=S.MUTED, fontsize=9.5, va="center", ha="left",
            linespacing=1.25)
    attention = np.array(ATTENTION_LAYERS_7B)
    ax.scatter(attention, med[attention], s=18, color=S.RED, zorder=4,
               edgecolor="white", linewidth=0.5)
    # Keyed by colour rather than by a leader: any leader long enough to reach
    # a red dot from clear space has to cross the curve on the way.
    ax.text(0.5, 104, "\u25cf layers that also carry attention", color=S.RED,
            fontsize=9.0, ha="left", va="center")
    S.frame(ax)
    ax.set_yscale("log")
    ax.set_xlim(-2, 106)
    ax.set_ylim(0.62, 190)
    ax.set_yticks([1, 4, 16, 64])
    ax.set_yticklabels(["1", "4", "16", "64"])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("half-life (tokens)")
    S.title(ax, "Half-life against depth: median, with interquartile band")
    return save(fig, "e2_halflife_by_layer.png"), {
        "layer_median_min": float(med.min()), "layer_median_max": float(med.max()),
        "overall_median": float(overall),
    }


# --------------------------------------------------------------------------
def fig_retention_curves():
    """The prediction, drawn, and then held against what was measured.

    Left is the mechanism with nothing fitted: multiply by exp(-|A| dt) once per
    token and this is what is left. Right is that prediction against the probe,
    which agrees on the 7B at r = 0.973 and, as Section 8 reports, does not
    replicate across scales. Drawing both keeps the agreement and its failure in
    the same eye-line.
    """
    data = dynamics("e5_dynamics.npz")
    rate = decay_rate(data)
    ks = np.arange(0, 65)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.85), gridspec_kw={"wspace": 0.46}
    )

    # Strata by predicted half-life, the same thirds the probe used.
    half = np.log(2) / rate
    order = np.argsort(half)
    thirds = np.array_split(order, 3)
    ends = []
    for idx, (name, colour) in zip(thirds, (("fast", S.RED),
                                            ("middle", S.MUTED),
                                            ("slow", S.BLUE))):
        curve = np.exp(-np.median(rate[idx]) * ks)
        left.plot(ks, curve * 100, color=colour, lw=1.9, zorder=3)
        ends.append((curve[-1] * 100, f"{name}, {np.median(half[idx]):.1f}",
                     colour))
    # Two of the three curves have reached the floor by 64 tokens, which is the
    # point of the figure and also puts their labels on top of each other.
    S.stack_end_labels(left, ks[-1], ends, gap=10.0, floor=2.0)
    left.axhline(50, color=S.MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    left.text(103, 53, "half of what was written", color=S.MUTED,
              fontsize=9.5, ha="right", va="bottom")
    S.frame(left)
    left.set_xlim(0, 104)
    left.set_ylim(0, 100)
    left.set_xticks([0, 16, 32, 48, 64])
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    left.set_xlabel("tokens after writing")
    left.set_ylabel("retained")
    S.title(left, "What the weights alone predict")

    fine = load("decode_fine.json")["families"][0]
    rows = sorted((r["distance"], r["probe_accuracy"] * 100)
                  for r in fine["per_distance"] if not r["is_control"])
    dist = np.array([r[0] for r in rows])
    acc = np.array([r[1] for r in rows])
    excess = acc - S.CHANCE
    measured = excess / excess[0]
    predicted = np.array([np.exp(-rate * k).mean() for k in dist])
    r_value = float(np.corrcoef(measured, predicted)[0, 1])

    right.plot([0, 1], [0, 1], color=S.RULE, lw=1.2, zorder=1)
    right.scatter(predicted, measured, s=34, color=S.BLUE, zorder=3,
                  edgecolor="white", linewidth=0.7)
    # Labelled sparsely and outward, because the short distances crowd into the
    # top right and the long ones into the bottom left.
    placement = {0: (7, -3, "left"), 4: (7, -3, "left"),
                 16: (0, 9, "center"), 64: (-7, -3, "right")}
    for k, x, y in zip(dist, predicted, measured):
        if k in placement:
            dx, dy, ha = placement[k]
            right.annotate(f"{k} tok" if k == 0 else f"{k}", (x, y),
                           xytext=(dx, dy), textcoords="offset points",
                           color=S.INK2, fontsize=9.5, ha=ha)
    right.text(0.04, 0.99, f"$r = {r_value:.3f}$\nnothing fitted", color=S.INK,
               fontsize=10, va="top", linespacing=1.3)
    S.frame(right)
    right.set_xlim(-0.04, 1.20)
    right.set_ylim(-0.04, 1.08)
    right.set_xticks([0, 0.5, 1.0])
    right.set_yticks([0, 0.5, 1.0])
    right.set_xlabel("predicted retention")
    right.set_ylabel("measured, normalised")
    right.set_ylim(-0.06, 1.12)
    S.title(right, "Prediction against probe")

    return save(fig, "e3_retention_curves.png"), {
        "r": r_value,
        "predicted": [round(float(p), 4) for p in predicted],
        "measured": [round(float(m), 4) for m in measured],
    }


# --------------------------------------------------------------------------
def fig_two_channels():
    """The setup, drawn: what each channel holds when the question arrives.

    Schematic rather than measured. Every quantity shown is stated in the
    paper's Section 3: 81 layers all carrying a Mamba2 mixer, 13 of them also
    carrying attention, a recurrent state of 458,752 numbers per layer that is
    the same size at every position, and an attention cache that grows by one
    entry per token and therefore still holds the image when the question is
    asked.
    """
    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, x1 = 0.155, 0.995
    img_end = 0.47
    fill_end = 0.845

    def band(y, height, xa, xb, colour, alpha, label=None, text_colour=None):
        ax.add_patch(FancyBboxPatch(
            (xa, y), xb - xa, height,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=colour, edgecolor="none", alpha=alpha, zorder=2))
        if label:
            ax.text((xa + xb) / 2, y + height / 2, label, ha="center",
                    va="center", fontsize=9, color=text_colour or S.INK,
                    zorder=3)

    # ---- the token stream -------------------------------------------------
    ax.text(x0 - 0.014, 0.885, "tokens", ha="right", va="center", fontsize=10,
            color=S.INK, fontweight="semibold")
    band(0.83, 0.11, x0, img_end, S.BLUE, 0.85, "image, 345 to 415", "white")
    band(0.83, 0.11, img_end + 0.005, fill_end, S.MUTED, 0.28,
         "filler text, $d$ tokens")
    band(0.83, 0.11, fill_end + 0.005, x1, S.MUTED, 0.55, "question", "white")

    # ---- attention channel ------------------------------------------------
    ax.text(x0 - 0.014, 0.585, "attention\n13 layers", ha="right", va="center",
            fontsize=10, color=S.INK, fontweight="semibold", linespacing=1.3)
    n = 34
    edges = np.linspace(x0, x1, n + 1)
    for i in range(n):
        mid = (edges[i] + edges[i + 1]) / 2
        colour = S.BLUE if mid < img_end else S.MUTED
        alpha = 0.85 if mid < img_end else 0.28
        ax.add_patch(Rectangle((edges[i] + 0.0018, 0.53),
                               edges[i + 1] - edges[i] - 0.0036, 0.11,
                               facecolor=colour, edgecolor="none", alpha=alpha,
                               zorder=2))
    ax.text(x0, 0.455, "one entry per token, all kept: the image is still "
            "there when the question is asked",
            fontsize=9.5, color=S.INK2, ha="left", va="center")

    # ---- recurrent channel ------------------------------------------------
    ax.text(x0 - 0.014, 0.235, "recurrent\nall 81 layers", ha="right",
            va="center", fontsize=10, color=S.INK, fontweight="semibold",
            linespacing=1.3)
    width = 0.050
    for left_edge in np.linspace(x0, x1 - width, 9):
        centre = left_edge + width / 2
        # Same box everywhere; what fades is how much of the picture is in it.
        alpha = 0.85 if centre < img_end else \
            0.85 * float(np.exp(-(centre - img_end) / 0.105))
        ax.add_patch(Rectangle((left_edge, 0.18), width, 0.11,
                               facecolor=S.RULE, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((left_edge, 0.18), width, 0.11,
                               facecolor=S.BLUE, edgecolor="none",
                               alpha=max(alpha, 0.02), zorder=2))
    ax.text(x0, 0.10, "the same 458,752 numbers per layer at every position, "
            "rewritten at each token", fontsize=9.5, color=S.INK2, ha="left",
            va="center")
    ax.annotate("", xy=(x1 - 0.02, 0.035), xytext=(img_end, 0.035),
                arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.0))
    ax.text(img_end - 0.01, 0.035, "what the image wrote fades", fontsize=9.5,
            color=S.MUTED, ha="right", va="center")
    return save(fig, "e4_two_channels.png"), {"schematic": True}


# --------------------------------------------------------------------------
def fig_protocols():
    """What each of the three experiments does to the run.

    Schematic. The probe reads the recurrent state at the position just before
    the question; the eviction deletes the image's key-value entries at the
    moment the image finishes and before any filler token is processed; the
    splice exchanges one channel between two runs that saw different images and
    then asks the host's question.

    Panels this narrow hold about thirty characters a line, so the labels stay
    to a few words each and the sentences live in the LaTeX caption.
    """
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_WIDTH, 1.95),
                             gridspec_kw={"wspace": 0.30})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def strip(ax, y, height=0.12, image_colour=S.BLUE, x0=0.02, x1=0.98,
              img_end=0.44, fill_end=0.80, labels=True):
        ax.add_patch(Rectangle((x0, y), img_end - x0, height,
                               facecolor=image_colour, alpha=0.85,
                               edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((img_end + 0.006, y), fill_end - img_end - 0.006,
                               height, facecolor=S.MUTED, alpha=0.26,
                               edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((fill_end + 0.006, y), x1 - fill_end - 0.006,
                               height, facecolor=S.MUTED, alpha=0.55,
                               edgecolor="none", zorder=2))
        if labels:
            ax.text((x0 + img_end) / 2, y + height / 2, "image", ha="center",
                    va="center", fontsize=9, color="white", zorder=3)
            ax.text((img_end + fill_end) / 2, y + height / 2, "$d$ tok",
                    ha="center", va="center", fontsize=9, color=S.INK, zorder=3)
        return img_end, fill_end

    # ---- 1. probe ---------------------------------------------------------
    ax = axes[0]
    S.title(ax, "1. Read the state", size=10)
    img_end, fill_end = strip(ax, 0.68)
    ax.annotate("", xy=(fill_end, 0.46), xytext=(fill_end, 0.66),
                arrowprops=dict(arrowstyle="->", color=S.INK2, lw=1.2))
    ax.add_patch(FancyBboxPatch((0.24, 0.22), 0.62, 0.21,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=S.BLUE, alpha=0.14, edgecolor="none"))
    ax.text(0.55, 0.325, "linear probe", ha="center", va="center", fontsize=9.5,
            color=S.INK)
    ax.text(0.55, 0.09, "8 choices, chance 12.5%", ha="center", fontsize=9,
            color=S.INK2, va="center")

    # ---- 2. evict ---------------------------------------------------------
    ax = axes[1]
    S.title(ax, "2. Evict, then read", size=10)
    img_end, fill_end = strip(ax, 0.68)
    ax.plot([0.02, img_end], [0.785, 0.785], color=S.RED, lw=1.4, zorder=4)
    ax.annotate("", xy=(img_end, 0.66), xytext=(img_end, 0.50),
                arrowprops=dict(arrowstyle="->", color=S.RED, lw=1.3))
    ax.text(img_end, 0.44, "drop the image's\nattention entries", fontsize=9,
            color=S.RED, ha="center", va="top", linespacing=1.35)
    ax.text(0.50, 0.09, "state left untouched", ha="center", fontsize=9,
            color=S.INK2, va="center")

    # ---- 3. splice --------------------------------------------------------
    ax = axes[2]
    S.title(ax, "3. Swap a channel", size=10)
    strip(ax, 0.72, height=0.10, image_colour=S.BLUE, labels=False)
    strip(ax, 0.44, height=0.10, image_colour=S.GREEN, labels=False)
    ax.text(0.23, 0.77, "run A", ha="center", va="center", fontsize=9,
            color="white", zorder=3)
    ax.text(0.23, 0.49, "run B", ha="center", va="center", fontsize=9,
            color="white", zorder=3)
    ax.add_patch(FancyArrowPatch((0.60, 0.545), (0.60, 0.715),
                                 arrowstyle="<->", color=S.INK2, lw=1.2,
                                 connectionstyle="arc3,rad=0.6",
                                 mutation_scale=11, zorder=4))
    ax.text(0.68, 0.63, "swap one\nchannel", fontsize=9, color=S.INK2,
            ha="left", va="center", linespacing=1.3)
    ax.text(0.50, 0.24, "then ask A's question", ha="center", fontsize=9,
            color=S.INK2, va="center")
    ax.text(0.50, 0.09, "whose picture answers?", ha="center", fontsize=9,
            color=S.INK2, va="center")
    return save(fig, "e5_protocols.png"), {"schematic": True}


def main() -> None:
    built = {}
    for func in (fig_halflife_distribution, fig_halflife_by_layer,
                 fig_retention_curves, fig_two_channels,
                 fig_protocols):
        path, values = func()
        built[os.path.basename(path)] = values
        print(f"wrote {path}")
    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()
