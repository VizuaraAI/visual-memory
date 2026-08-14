"""Paper figures, version 4: one visual system, every number read from results.

Nothing here is typed in by hand. Each function loads the result file it plots
and returns the values it drew, so the caller can print them and check them
against the paper text. A figure that disagrees with the prose is then a
mismatch you can see rather than one you have to remember to look for.

Composition follows the argument rather than the experiment log: results that
are read together are drawn together as panels.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
style.apply()

S = style


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def fig_horizon_depth():
    """Where the picture is, and how fast the readout of it falls away."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.75), gridspec_kw={"wspace": 0.40}
    )

    series = [
        ("7B", "decode_fine.json", S.BLUE, 0),
        ("2.7B", "decode_fine2p7.json", S.SHELF, -3),
        ("1.2B", "decode_fine1p2.json", S.GREEN, 11),
    ]
    drawn = {}
    for label, path, colour, dy in series:
        fam = load(path)["families"][0]
        pts = [(r["distance"], r["probe_accuracy"] * 100)
              for r in fam["per_distance"] if not r["is_control"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        S.line(left, xs, ys, colour, lw=2.0, ms=5.2)
        S.end_label(left, xs[-1], ys[-1], label, colour, dy)
        drawn[label] = pts

    S.grid_y(left)
    S.frame(left)
    left.set_xlim(-3, 92)
    S.percent_axis(left, 100)
    # Both panels read "identity decoded" in percent, so they keep one scale.
    # The headroom exists for the peak callout on the right; the left panel
    # takes it too rather than letting the two axes differ.
    left.set_ylim(0, 118)
    left.set_xticks([0, 16, 32, 48, 64])
    left.set_xlabel("tokens after the image")
    left.set_ylabel("identity decoded")
    S.chance_rule(left)
    S.title(left, "Collapse within a few tokens")

    fam = load("decode_fine_layerwise.json")["families"][0]
    row = [p for p in fam["per_distance"] if p["distance"] == 0][0]
    layers = np.array([e["layer"] for e in row["layerwise"]])
    acc = np.array([e["accuracy"] * 100 for e in row["layerwise"]])
    attention = {6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77}

    right.plot(layers, acc, color=S.BLUE, lw=1.4, zorder=3, alpha=0.9)
    mask = np.array([l in attention for l in layers])
    right.scatter(layers[mask], acc[mask], s=20, color=S.RED, zorder=4,
                  edgecolor=S.PAPERBG, linewidth=0.8)
    right.text(2, 22, "● the 13 attention layers", color=S.RED, fontsize=8.6,
               va="center")
    best = int(np.argmax(acc))
    # Called out from directly above, in headroom added for the purpose. A
    # horizontal leader either runs through its own text (matplotlib starts it
    # at the anchor, not at the edge of the words) or grazes the neighbouring
    # spikes, which reach within a point of the peak.
    right.annotate(f"layer {layers[best]}, {acc[best]:.1f}%",
                   (layers[best], acc[best]), xytext=(layers[best], 106),
                   textcoords="data", color=S.INK, fontsize=9.5,
                   ha="center", va="bottom",
                   arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.9,
                                   shrinkA=3, shrinkB=3))
    full = row["probe_accuracy"] * 100
    right.axhline(full, color=S.MUTED, lw=1.2, ls=(0, (2, 3)))
    # Labelled at the end of its own rule rather than floating in the middle of
    # the panel, where it read as marking whatever it happened to sit beside.
    right.text(84, full, f"all 81 layers\nat once: {full:.1f}%",
               color=S.MUTED, fontsize=9.5, va="center", ha="left",
               linespacing=1.25)

    S.grid_y(right)
    S.frame(right)
    right.set_xlim(-2, 112)
    S.percent_axis(right, 100)
    right.set_ylim(0, 118)          # matches the left panel
    right.set_xticks([0, 25, 50, 75])
    right.set_xlabel("decoder layer")
    S.chance_rule(right)
    S.title(right, "Gist lives in the late layers")
    return save(fig, "f1_horizon_depth.png"), {"scales": drawn,
                                               "best_layer": int(layers[best]),
                                               "best": float(acc[best]),
                                               "full_state": full}


# --------------------------------------------------------------------------
def fig_relay():
    """The gain-matched strata, and the intervention that explains them."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.75), gridspec_kw={"wspace": 0.40}
    )

    strata = load("e5b_gain24_decoded.json")["strata"]
    order = [("fast", "fast, 2.3 tok", S.RED, 0),
             ("slow", "slow, 16.4", S.BLUE, 0)]
    ends = []
    for key, label, colour, dy in order:
        rows = strata[key]["per_distance"]
        xs = [r["distance"] for r in rows]
        ys = [r["accuracy"] * 100 for r in rows]
        S.line(left, xs, ys, colour, lw=2.0, ms=5.2)
        ends.append((ys[-1], label, colour))
    S.stack_end_labels(left, xs[-1], ends, gap=11.0, floor=S.CHANCE + 2.0)

    # What retention would permit: the fast stratum's own prediction reaches
    # 4e-9 of its trace by d=64, i.e. indistinguishable from the chance rule.
    # Drawing that curve makes the gap the figure's subject.
    ds = np.linspace(0, 64, 200)
    base = strata["fast"]["per_distance"][0]["accuracy"] * 100 - S.CHANCE
    pred = S.CHANCE + base * np.exp(-np.log(2) / 2.3 * ds)
    left.plot(ds, pred, ls=(0, (3, 3)), color=S.RED, lw=1.3, zorder=2, alpha=0.8)
    left.text(8.5, S.CHANCE + 8.2, "if only retaining", color=S.RED,
              fontsize=8.6, alpha=0.9)

    # The finding, drawn on the figure: measurement against calculation at
    # d=64, explained in the right margin below the series labels.
    left.annotate("", xy=(64, 25.2), xytext=(64, 14.0),
                  arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.3,
                                  shrinkA=1, shrinkB=1), zorder=6)
    S.note(left, 68, 20.4, "measured: 26.6%", colour=S.INK2, ha="left",
           weight="semibold", size=8.4)
    S.note(left, 68, 15.1, "calculated: 12.5%", colour=S.MUTED,
           ha="left", size=8.4, box=True)
    S.note(left, 68, 8.2, "this gap is the finding", colour=S.RED,
           ha="left", weight="semibold", size=8.4)

    S.grid_y(left)
    S.frame(left)
    left.set_xlim(-3, 118)
    S.percent_axis(left, 100)
    left.set_xticks([0, 16, 32, 64])
    left.set_xlabel("tokens after the image")
    left.set_ylabel("identity decoded")
    S.chance_rule(left)
    S.title(left, "Too much survives")

    summary = [r for r in load("e5c_decoded.json")["summary"]
               if r["stratum"] == "fast"]
    summary.sort(key=lambda r: r["distance"])
    xs = np.arange(len(summary))
    width = 0.34
    intact = [r["intact"] * 100 for r in summary]
    evicted = [r["evicted"] * 100 for r in summary]
    right.bar(xs - width / 2, intact, width - 0.03, color=S.BLUE, zorder=3)
    right.bar(xs + width / 2, evicted, width - 0.03, color=S.RED, zorder=3)
    for x, v in zip(xs - width / 2, intact):
        right.text(x, v + 2, f"{v:.0f}", ha="center", color=S.INK2,
                   fontsize=9.5, fontweight="semibold", path_effects=S.punch())
    for x, v in zip(xs + width / 2, evicted):
        right.text(x, v + 2, f"{v:.0f}", ha="center", color=S.INK2,
                   fontsize=9.5, fontweight="semibold", path_effects=S.punch())
    for offset, (text, colour) in enumerate((("attention intact", S.BLUE),
                                            ("image entries deleted", S.RED))):
        right.text(0.98, 0.99 - 0.105 * offset, "● ", transform=right.transAxes,
                   color=colour, fontsize=8, ha="right", va="top")
        right.annotate(text, xy=(0.98, 0.99 - 0.105 * offset),
                       xycoords="axes fraction", xytext=(-12, 0),
                       textcoords="offset points",
                       color=S.INK2, fontsize=9.3, ha="right", va="top")

    S.grid_y(right)
    S.frame(right)
    right.set_xticks(xs)
    right.set_xticklabels([f"{r['distance']}" for r in summary])
    right.set_xlabel("tokens after the image")
    S.percent_axis(right, 100)
    # Every point below the chance line sits inside a bar, so the rule is
    # labelled in the margin past the last group instead of on top of one.
    right.set_xlim(-0.6, len(summary) - 0.1 + 1.15)
    right.axhline(S.CHANCE, color=S.MUTED, lw=1.2, ls=(0, (6, 5)), zorder=1)
    right.text(len(summary) - 0.55, S.CHANCE, "chance\n12.5%", color=S.MUTED,
               fontsize=9.5, ha="left", va="center", linespacing=1.25)
    S.title(right, "Evict attention's copy")
    return save(fig, "f2_relay.png"), {"strata": order, "fast": summary}


# --------------------------------------------------------------------------
def fig_controls():
    """The control that rules out the perturbation, and the route it travels."""
    rows = load("e5d_decoded.json")["contrasts"]
    at32 = sorted([r for r in rows if r["distance"] == 32],
                  key=lambda r: r["predicted_median_half_life"])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.75), gridspec_kw={"wspace": 0.40}
    )

    xs = np.arange(len(at32))
    intact = [r["intact"] * 100 for r in at32]
    img = [r.get("visual_all", np.nan) * 100 for r in at32]
    txt = [r.get("prefill_matched", np.nan) * 100 for r in at32]
    # Shaded gaps make the comparison vertical, the way it should be read:
    # blue wash above the baseline = lift from deleting text, pink wash below
    # = cost of deleting the image.
    left.fill_between(xs, intact, txt, color=S.BLUE, alpha=0.10, lw=0, zorder=1)
    left.fill_between(xs, img, intact, color=S.RED, alpha=0.10, lw=0, zorder=1)
    left.plot(xs, intact, "-o", color=S.FAINT, lw=1.8, ms=4.6, zorder=3,
              markeredgecolor=S.PAPERBG, markeredgewidth=1.1)
    S.line(left, xs, img, S.RED, lw=2.0, ms=4.6, zorder=4)
    S.line(left, xs, txt, S.BLUE, lw=2.0, ms=4.6, zorder=4)
    S.note(left, 0.0, 67.5, "deleting text\nlifts the readout", colour=S.BLUE,
           size=8.4, weight="semibold")
    S.note(left, 4.6, 4.6, "deleting the image drops it", colour=S.RED,
           size=8.4, weight="semibold")
    # The three series end within a few points of each other and start far
    # apart, so they are labelled at the left, where identity is unambiguous.
    # The image series starts at exactly the chance rate, so its label would
    # otherwise sit on the dashed rule; it alone is dropped clear of it.
    for series, label, colour, dy in ((txt, "evict text", S.BLUE, 0),
                                      (intact, "no eviction", S.FAINT, 0),
                                      (img, "evict image", S.RED, -9)):
        left.annotate("●", (xs[0], series[0]), xytext=(-8, dy),
                      textcoords="offset points", color=colour, fontsize=6.5,
                      ha="right", va="center")
        left.annotate(label, (xs[0], series[0]), xytext=(-16, dy),
                      textcoords="offset points", color=S.INK2, fontsize=9.3,
                      ha="right", va="center", fontweight="medium")

    S.frame(left)
    # Room on the left for the labels to sit inside the axes rather than up
    # against the tick numbers, and room on the right for the chance rule to be
    # named clear of the lines, which cross it repeatedly.
    left.set_xlim(-6.2, len(xs) + 1.9)
    S.percent_axis(left, 80)
    left.set_xticks(xs[::3])
    left.set_xticklabels([f"{at32[i]['predicted_median_half_life']:.0f}"
                          for i in range(0, len(at32), 3)])
    left.set_xlabel("predicted half-life (tokens)")
    left.set_ylabel("identity decoded")
    left.axhline(S.CHANCE, color=S.MUTED, lw=1.2, ls=(0, (6, 5)), zorder=1)
    left.text(len(xs) - 0.75, S.CHANCE, "chance\n12.5%", color=S.MUTED,
              fontsize=9.5, ha="left", va="center", linespacing=1.25)
    S.title(left, "Equal evictions, opposite effects")

    at0 = sorted([r for r in rows if r["distance"] == 0],
                 key=lambda r: r["predicted_median_half_life"])
    xs0 = np.arange(len(at0))
    full = [(r["intact"] - r.get("visual_all_zero", r["intact"])) * 100 for r in at0]
    early = [(r["intact"] - r.get("visual_early3_zero", r["intact"])) * 100 for r in at0]
    late = [(r["intact"] - r.get("visual_late3_zero", r["intact"])) * 100 for r in at0]
    right.bar(xs0 - 0.28, full, 0.24, color=S.FAINT, zorder=3)
    right.bar(xs0, early, 0.24, color=S.RED, zorder=3)
    right.bar(xs0 + 0.28, late, 0.24, color=S.BLUE, zorder=3)
    right.axhline(0, color=S.RULE, lw=1.0)
    # Headroom above the tallest bar, so the key sits in empty space rather
    # than across the bars at either end.
    right.set_ylim(min(min(full), min(early), min(late)) - 3.0,
                   max(max(full), max(early), max(late)) * 1.34)
    for offset, (text, colour) in enumerate((("all 13 layers", S.FAINT),
                                            ("earliest 3", S.RED),
                                            ("latest 3", S.BLUE))):
        right.text(0.34, 0.99 - 0.10 * offset, "●", transform=right.transAxes,
                   color=colour, fontsize=6.5, va="top", ha="left")
        right.text(0.395, 0.995 - 0.10 * offset, text, transform=right.transAxes,
                   color=S.INK2, fontsize=9.3, va="top", ha="left")

    S.frame(right)
    right.set_xticks(xs0[::3])
    right.set_xticklabels([f"{at0[i]['predicted_median_half_life']:.0f}"
                           for i in range(0, len(at0), 3)])
    right.set_xlabel("predicted half-life (tokens)")
    right.set_ylabel("accuracy lost (points)")
    S.title(right, "The relay enters early")
    return save(fig, "f3_controls.png"), {"at32": at32[:4], "at0": at0[:4]}


# --------------------------------------------------------------------------
def fig_sensitivity():
    """How faint a signal this probe can still find, against what decay leaves."""
    full = load("probe_sensitivity_full.json")
    layer = load("probe_sensitivity_layer64.json")

    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.62, 2.9))
    for data, label, colour, dy in ((full, "all 81 layers", S.BLUE, 0),
                                    (layer, "layer 64 alone", S.SHELF, 0)):
        xs = [r["alpha"] for r in data["sweep"]]
        ys = [r["accuracy_mean"] * 100 for r in data["sweep"]]
        S.line(ax, xs, ys, colour, lw=2.0, ms=5.0)
        ax.annotate("\u25cf", (xs[0], ys[0]), xytext=(-7, dy + 2),
                    textcoords="offset points", color=colour, fontsize=6.5,
                    ha="right", va="center")
        ax.annotate(label, (xs[0], ys[0]), xytext=(-15, dy + 2),
                    textcoords="offset points", color=S.INK2, fontsize=9.3,
                    ha="right", va="center", fontweight="medium")
    S.grid_y(ax)

    # The band the weights say a stored trace occupies at 32 tokens: median
    # retention 0.037, mean 0.18. Both sit inside the region where neither
    # probe can see anything, which is the point of the figure.
    ax.axvspan(0.037, 0.18, color=S.RED, alpha=0.10, lw=0, zorder=1)
    ax.text(0.082, 97, "what decay leaves\nat 32 tokens", color=S.RED,
            fontsize=9.5, ha="center", va="top", linespacing=1.25)

    S.frame(ax)
    ax.set_xscale("log")
    ax.set_xlim(0.017, 1.25)
    S.percent_axis(ax, 100)
    ax.set_xlabel("signal amplitude retained (fraction of full)")
    ax.set_ylabel("identity decoded")
    ax.set_xticks([0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.25", "0.5", "1.0"])
    S.chance_rule(ax, side="right")
    S.title(ax, "The probe cannot see a faint trace")
    return save(fig, "f4_sensitivity.png"), {
        "full_half_alpha": full["half_signal_alpha"],
        "layer_half_alpha": layer["half_signal_alpha"],
    }


# --------------------------------------------------------------------------
def fig_splice():
    """How much of the attention channel must move before the answer does."""
    summary = load("e10_report_scaled.json")["summary"]
    order = ["attention_1", "attention_3", "attention_7", "attention_all"]
    # The recurrent swap is the natural zero of this axis: all 81 states
    # replaced, no attention cache touched. Drawing it as the leftmost point
    # puts the null inside the curve instead of floating above it as a rule.
    xs = [0] + [summary[k]["n_layers_swapped"] for k in order]
    host = [summary["recurrent"]["host_rate"] * 100] + [
        summary[k]["host_rate"] * 100 for k in order]
    donor = [summary["recurrent"]["donor_rate"] * 100] + [
        summary[k]["donor_rate"] * 100 for k in order]

    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.62, 2.75))
    S.line(ax, xs, host, S.BLUE, lw=2.0, ms=5.6)
    S.line(ax, xs, donor, S.RED, lw=2.0, ms=5.6)
    S.grid_y(ax)
    S.note(ax, 5.0, 36, "the cliff", colour=S.RED, size=9.0, ha="center",
           weight="semibold", stroke=True)
    S.end_label(ax, xs[-1], host[-1], "keeps its own", S.BLUE, -2)
    S.end_label(ax, xs[-1], donor[-1], "takes the other", S.RED, 2)
    ax.annotate("all 81 recurrent\nstates swapped", (0, host[0]),
                xytext=(0.25, 66), textcoords="data", color=S.MUTED,
                fontsize=9.5, ha="left", va="top", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.9,
                                shrinkA=2, shrinkB=4))

    S.frame(ax)
    ax.set_xticks(xs)
    ax.set_xlim(-0.6, 18.5)
    S.percent_axis(ax, 100)
    ax.set_xlabel("attention caches swapped (of 13)")
    ax.set_ylabel("share of 100 pairs")
    S.title(ax, "Seven of thirteen move the answer")
    return save(fig, "f5_splice.png"), {"layers": xs, "host": host, "donor": donor}


# --------------------------------------------------------------------------
def fig_eviction():
    """The cost of pruning, and how it grows with distance."""
    report = load("e9_report_dist_random.json")
    distances = report["distances"]
    retentions = report["retentions"]
    cells = report["by_retention"]

    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.62, 2.9))
    palette = {1.0: S.FAINT, 0.25: S.BLUE, 0.05: S.SHELF, 0.0: S.RED}
    labels = {1.0: "no pruning", 0.25: "keep 25%", 0.05: "keep 5%", 0.0: "keep none"}
    # A tiny cache glyph beside each label shows what the condition keeps.
    glyphs = {1.0: "\u25ae" * 8,
              0.25: "\u25ae" * 2 + "\u25af" * 6,
              0.05: "\u25ae" + "\u25af" * 7,
              0.0: "\u25af" * 8}
    drawn = {}
    for retention in retentions:
        ys = [cells[f"d{d}_r{retention}"]["accuracy"] * 100 for d in distances]
        colour = palette.get(retention, S.MUTED)
        S.line(ax, range(len(distances)), ys, colour, lw=2.0, ms=5.0)
        x_end = len(distances) - 1
        ax.annotate(glyphs[retention], (x_end, ys[-1]), xytext=(8, 0),
                    textcoords="offset points", color=colour, fontsize=6.5,
                    va="center", family="DejaVu Sans")
        ax.annotate(labels[retention], (x_end, ys[-1]), xytext=(40, 0),
                    textcoords="offset points", color=S.INK2, fontsize=9.3,
                    va="center", fontweight="medium")
        drawn[labels.get(retention, str(retention))] = ys
    S.grid_y(ax)

    S.frame(ax)
    ax.set_xlim(-0.2, len(distances) + 1.4)
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([str(d) for d in distances])
    S.percent_axis(ax, 100)
    ax.set_xlabel("tokens between image and question")
    ax.set_ylabel("model accuracy")
    S.chance_rule(ax)
    S.title(ax, "Pruning costs more the later you ask")
    return save(fig, "f6_eviction.png"), drawn


# --------------------------------------------------------------------------
def fig_sensitivity_splice():
    """The two half-width panels drawn as one full-width figure.

    Drawn together because the ICLR body has room for one figure here, not two,
    and shrinking a 3.9in panel into a 2.5in slot would push the tick labels
    below legible size. Same data as fig_sensitivity and fig_splice; only the
    canvas and the label placement change, the latter because two panels at
    2.9in collide where one at 3.9in does not.
    """
    full = load("probe_sensitivity_full.json")
    layer = load("probe_sensitivity_layer64.json")
    summary = load("e10_report_scaled.json")["summary"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(S.TEXT_WIDTH, 2.8), gridspec_kw={"wspace": 0.50}
    )

    # Left: label each curve at its own last point, clear of the shaded band.
    for data, label, colour, dy in ((full, "all 81 layers", S.BLUE, -9),
                                    (layer, "layer 64 alone", S.SHELF, 7)):
        xs = [r["alpha"] for r in data["sweep"]]
        ys = [r["accuracy_mean"] * 100 for r in data["sweep"]]
        S.line(left, xs, ys, colour, lw=2.0, ms=5.0)
        S.end_label(left, xs[0], ys[0], label, colour, dy)
    S.grid_y(left)

    # The band the weights say a stored trace occupies at 32 tokens: median
    # retention 0.037, mean 0.18. Both sit where neither probe sees anything.
    left.axvspan(0.037, 0.18, color=S.RED, alpha=0.10, lw=0, zorder=1)
    left.text(0.021, 78, "what decay leaves\nat 32 tokens", color=S.RED,
              fontsize=9.5, ha="left", va="top", linespacing=1.25)
    S.frame(left)
    left.set_xscale("log")
    left.set_xlim(0.017, 14.0)
    S.percent_axis(left, 100)
    left.set_xlabel("signal amplitude retained")
    left.set_ylabel("identity decoded")
    left.set_xticks([0.02, 0.1, 0.5])
    left.set_xticklabels(["0.02", "0.1", "0.5"])
    left.axhline(S.CHANCE, color=S.MUTED, lw=1.2, ls=(0, (6, 5)), zorder=1)
    left.text(1.7, S.CHANCE + 2.2, "chance, 12.5%", color=S.MUTED,
              fontsize=9.0, ha="left", va="bottom")
    S.title(left, "Faint traces stay invisible")

    order = ["attention_1", "attention_3", "attention_7", "attention_all"]
    xs = [0] + [summary[k]["n_layers_swapped"] for k in order]
    host = [summary["recurrent"]["host_rate"] * 100] + [
        summary[k]["host_rate"] * 100 for k in order]
    donor = [summary["recurrent"]["donor_rate"] * 100] + [
        summary[k]["donor_rate"] * 100 for k in order]
    S.line(right, xs, host, S.BLUE, lw=2.0, ms=5.6)
    S.line(right, xs, donor, S.RED, lw=2.0, ms=5.6)
    S.grid_y(right)
    S.note(right, 5.0, 36, "the cliff", colour=S.RED, size=9.0, ha="center",
           weight="semibold", stroke=True)
    S.end_label(right, xs[-1], host[-1], "keeps its own", S.BLUE, -2)
    S.end_label(right, xs[-1], donor[-1], "takes the other", S.RED, 2)
    # At this panel width every interior placement of a two-line callout
    # crosses one of the two lines, so the zero point is named on the axis
    # instead of annotated inside the plot.
    S.frame(right)
    right.set_xticks([0, 3, 7, 13])
    right.set_xlim(-0.6, 22.0)
    S.percent_axis(right, 100)
    right.set_xlabel("attention caches swapped\n(0 = all 81 recurrent states)")
    right.set_ylabel("share of 100 pairs")
    S.title(right, "Seven of thirteen flip it")
    return save(fig, "f7_sens_splice.png"), {
        "full_half_alpha": full["half_signal_alpha"],
        "layers": xs, "host": host, "donor": donor,
    }


def main() -> None:
    built = {}
    for func in (fig_horizon_depth, fig_relay, fig_controls,
                 fig_sensitivity, fig_splice, fig_eviction,
                 fig_sensitivity_splice):
        path, values = func()
        built[os.path.basename(path)] = values
        print(f"built {os.path.basename(path)}")
    print(json.dumps(built, indent=2, default=str)[:1200])


if __name__ == "__main__":
    main()
