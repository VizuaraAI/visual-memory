"""Decode the head-stratified collections from E5b.

The question is whether a head's predicted half-life governs how long its slice
of the recurrent state still carries the picture. Decay says yes: the slow
third should stay decodable to larger distances than the fast third. If all
three strata behave alike, decay is not what drives the collapse.

Runs on CPU.
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
STRATA = ("fast", "middle", "slow")


def decode_one(base: str) -> list[dict]:
    with open(f"{base}.jsonl") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    features = np.load(f"{base}_features.npz")["features"]
    if len(records) != features.shape[0]:
        raise ValueError(f"{base}: {len(records)} records, {features.shape[0]} rows")

    labels = torch.tensor([r["answer_index"] for r in records])
    groups = [r["block_id"] for r in records]
    rows = []
    for distance in sorted({r["distance"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["distance"] == distance]
        result = probe.cross_validated_accuracy(
            torch.tensor(features[idx].astype(np.float32)),
            labels[idx],
            [groups[i] for i in idx],
            n_classes=N_CLASSES,
        )
        rows.append(
            {
                "distance": distance,
                "accuracy": round(result["accuracy"], 4),
                "p_value": result["p_value"],
                "n": result["total"],
            }
        )
    return rows


def half_life_of(rows: list[dict]) -> float | None:
    """Distance at which the above-chance signal falls to half its d=0 value.

    Linear interpolation between the bracketing measured points, which is
    honest about the grid rather than fitting a curve through a cliff.
    """
    excess = [(r["distance"], r["accuracy"] - CHANCE) for r in rows]
    if not excess or excess[0][1] <= 0:
        return None
    target = excess[0][1] / 2.0
    for (d0, e0), (d1, e1) in zip(excess, excess[1:]):
        if e0 >= target >= e1 and e0 > e1:
            return round(d0 + (e0 - target) / (e0 - e1) * (d1 - d0), 2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tag", default="7b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report_path = os.path.join(args.results_dir, f"e5b_report_{args.tag}.json")
    predicted = {}
    if os.path.exists(report_path):
        predicted = json.load(open(report_path)).get("strata", {})

    out: dict = {"tag": args.tag, "chance": CHANCE, "strata": {}}
    for stratum in STRATA:
        base = os.path.join(args.results_dir, f"e5b_{args.tag}_{stratum}")
        if not os.path.exists(f"{base}.jsonl"):
            print(f"skipping {stratum}: not collected", file=sys.stderr)
            continue
        rows = decode_one(base)
        out["strata"][stratum] = {
            "predicted_median_half_life": predicted.get(stratum, {}).get(
                "median_half_life"
            ),
            "measured_half_life": half_life_of(rows),
            "per_distance": rows,
        }

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
