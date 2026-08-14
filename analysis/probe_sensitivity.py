"""How weak a signal can this probe still find?

The relay result rests on a collapse: the probe reads the recurrent state at
distance only while the attention channel still holds the image. The standing
alternative is that the probe is simply insensitive to weak signals. Freshly
re-encoded content plausibly sits at higher amplitude than content written many
tokens earlier and decayed, so a probe that finds strong signals and misses weak
ones would produce everything we observe without the state having stopped
storing anything.

That alternative makes a quantitative prediction and can therefore be tested.
Take the distance-zero features, where the image is present at full strength,
and attenuate each trial's own contribution while substituting an equal amount
of unrelated variation drawn from another trial:

    x_i(alpha) = mu + alpha * (x_i - mu) + sqrt(1 - alpha^2) * (x_j - mu)

Total variance is held roughly constant while the discriminative component
shrinks, which is what decay followed by overwriting does to a stored trace.
Sweeping alpha gives the amplitude at which this probe, at this sample size,
loses the signal.

Compare that threshold to the retention the weights predict at the distance in
question. If the probe still finds the signal well below the predicted
retention, then it would have found retained content at that distance had any
been there, and the sensitivity account fails. If the probe dies first, the
account survives and the paper must say so.

Runs on CPU, on features already collected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab import probe  # noqa: E402

N_CLASSES = 8
CHANCE = 1.0 / N_CLASSES
ALPHAS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.037, 0.02)


def attenuate(x: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Shrink each row's own deviation, replacing it with another row's."""
    mu = x.mean(axis=0, keepdims=True)
    centred = x - mu
    partner = rng.permutation(len(x))
    substitute = centred[partner]
    return mu + alpha * centred + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * substitute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--base", required=True,
                        help="collection stem, e.g. decay_fine_identity")
    parser.add_argument("--distance", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None,
                        help="probe one layer only; default is the whole state")
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stem = os.path.join(args.results_dir, args.base)
    with open(f"{stem}.jsonl") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    features = np.load(f"{stem}_features.npz")["features"]

    idx = [i for i, r in enumerate(records) if r.get("distance") == args.distance]
    if not idx:
        raise SystemExit(f"no records at distance {args.distance}")
    x = features[idx].astype(np.float32)
    labels = torch.tensor([records[i]["answer_index"] for i in idx])
    groups = [records[i]["block_id"] for i in idx]

    if args.layer is not None:
        n_layers = x.shape[1] // args.n_features
        x = x.reshape(len(idx), n_layers, args.n_features)[:, args.layer, :]

    rows = []
    for alpha in ALPHAS:
        accuracies = []
        for repeat in range(args.repeats):
            rng = np.random.default_rng(1000 + repeat)
            result = probe.cross_validated_accuracy(
                torch.tensor(attenuate(x, alpha, rng)), labels, groups,
                n_classes=N_CLASSES,
            )
            accuracies.append(result["accuracy"])
        rows.append({
            "alpha": alpha,
            "accuracy_mean": round(float(np.mean(accuracies)), 4),
            "accuracy_min": round(float(np.min(accuracies)), 4),
            "accuracy_max": round(float(np.max(accuracies)), 4),
            "n": len(idx),
        })
        print(f"alpha={alpha:<6} accuracy {np.mean(accuracies)*100:5.1f}% "
              f"[{np.min(accuracies)*100:.1f}, {np.max(accuracies)*100:.1f}]")

    # Where does it cross halfway between the full-signal value and chance?
    full = rows[0]["accuracy_mean"]
    target = CHANCE + (full - CHANCE) / 2
    threshold = None
    for a, b in zip(rows, rows[1:]):
        if a["accuracy_mean"] >= target >= b["accuracy_mean"]:
            span = a["accuracy_mean"] - b["accuracy_mean"]
            frac = (a["accuracy_mean"] - target) / span if span > 0 else 0.0
            threshold = a["alpha"] + frac * (b["alpha"] - a["alpha"])
            break

    out = {
        "base": args.base,
        "distance": args.distance,
        "layer": args.layer,
        "chance": CHANCE,
        "full_signal_accuracy": full,
        "half_signal_alpha": None if threshold is None else round(threshold, 4),
        "sweep": rows,
    }
    print("\n" + json.dumps(
        {k: v for k, v in out.items() if k != "sweep"}, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
