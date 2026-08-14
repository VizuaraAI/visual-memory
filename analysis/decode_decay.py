"""Offline decoding of the collected state features.

Reads the JSONL records and NPZ feature matrices written by collect_decay and
produces, per family and per distance: probe accuracy with its exact binomial
p-value, the behavioural accuracy measured on the same trials, the two
controls, and a fitted exponential half-life.

Runs on CPU. No GPU is needed to redo any of the analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab import probe  # noqa: E402

N_CLASSES = 8
CHANCE = 1.0 / N_CLASSES
# Width of the per-layer random projection used by the collectors.
N_PROJECT_FEATURES = 512


def load_family(base: str) -> tuple[list[dict], np.ndarray]:
    with open(f"{base}.jsonl") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    features = np.load(f"{base}_features.npz")["features"]
    if len(records) != features.shape[0]:
        raise ValueError(
            f"{base}: {len(records)} records but {features.shape[0]} feature rows"
        )
    return records, features


def half_life(distances: list[int], accuracies: list[float]) -> dict:
    """Fit accuracy(d) = chance + A * exp(-d / tau) by least squares on the
    excess-over-chance, using only the strictly positive points."""
    points = [
        (d, a - CHANCE) for d, a in zip(distances, accuracies) if a - CHANCE > 1e-6 and d >= 0
    ]
    if len(points) < 3:
        return {"half_life_tokens": None, "note": "too few points above chance"}
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.log(np.array([p[1] for p in points], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope >= 0:
        return {"half_life_tokens": None, "note": "no decay detected"}
    tau = -1.0 / slope
    return {
        "half_life_tokens": round(tau * math.log(2), 1),
        "tau_tokens": round(tau, 1),
        "amplitude_at_zero": round(float(np.exp(intercept)), 4),
    }


def analyse(base: str, family: str, layerwise: bool) -> dict:
    records, features = load_family(base)
    labels = torch.tensor([r["answer_index"] for r in records])
    groups = [r["block_id"] for r in records]
    distances = sorted({r["distance"] for r in records})

    out: dict = {"family": family, "n_records": len(records), "per_distance": []}
    accuracies: list[float] = []
    positive_distances: list[int] = []

    for distance in distances:
        idx = [i for i, r in enumerate(records) if r["distance"] == distance]
        x = torch.tensor(features[idx].astype(np.float32))
        y = labels[idx]
        g = [groups[i] for i in idx]

        result = probe.cross_validated_accuracy(x, y, g, n_classes=N_CLASSES)
        row = {
            "distance": distance,
            "is_control": distance < 0,
            "probe_accuracy": round(result["accuracy"], 4),
            "p_value": result["p_value"],
            "n": result["total"],
        }

        behaviour = [
            r["behaviour"]["correct"] for i, r in enumerate(records)
            if r["distance"] == distance and "behaviour" in r
        ]
        if behaviour:
            row["behaviour_accuracy"] = round(sum(behaviour) / len(behaviour), 4)

        zero_shot = [
            int(r.get("zero_shot_index") == r["answer_index"])
            for i, r in enumerate(records) if r["distance"] == distance
        ]
        if zero_shot:
            row["zero_shot_accuracy"] = round(sum(zero_shot) / len(zero_shot), 4)

        if distance == 0:
            perm = probe.permuted_accuracy(x, y, g, n_classes=N_CLASSES, n_repeats=5)
            row["permutation_mean"] = round(perm["mean"], 4)
            row["permutation_max"] = round(perm["max"], 4)
            if layerwise:
                # The projection width is a property of the collection, not a
                # constant. An earlier hardcoded 128 silently reported 324
                # "layers" for a 512-wide collection of an 81-layer model,
                # which is 41472/128 rather than anything anatomical.
                n_layers = features.shape[1] // N_PROJECT_FEATURES
                row["layerwise"] = probe.layerwise_accuracy(
                    x, y, g, N_CLASSES, n_layers=n_layers
                )

        out["per_distance"].append(row)
        if distance >= 0:
            positive_distances.append(distance)
            accuracies.append(result["accuracy"])

    out["decay"] = half_life(positive_distances, accuracies)
    out["chance"] = CHANCE
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument(
        "--families", default="identity,count,position",
    )
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = {"tag": args.tag, "families": []}
    for family in args.families.split(","):
        base = os.path.join(args.results_dir, f"decay_{args.tag}_{family}")
        if not os.path.exists(f"{base}.jsonl"):
            print(f"skipping {family}: {base}.jsonl not found", file=sys.stderr)
            continue
        result = analyse(base, family, args.layerwise)
        report["families"].append(result)
        print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
