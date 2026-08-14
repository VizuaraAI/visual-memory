"""Decode E5c: is the recurrent state remembering the image, or re-reading it?

Each collection holds the projected recurrent state under two conditions that
differ only in whether the visual key-value entries were evicted from the
attention layers immediately after the image was read, before any filler token
was processed. Everything else, including the recurrent state at the moment of
eviction, is identical.

If the state is remembering, evicting the attention entries should not change
what a probe can read out of it, because the recurrent channel was written
during the image and is not touched by the eviction.

If the state is a relay, continuously rewritten from a residual stream that the
attention channel keeps supplying, then eviction should collapse decodability at
distance while leaving distance zero comparatively intact, and it should do so
most severely for the fast-decaying heads, whose state is dominated by the most
recent writes.

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
STRATA = ("all", "fast", "middle", "slow")


def decode(base: str) -> list[dict]:
    with open(f"{base}.jsonl") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    features = np.load(f"{base}_features.npz")["features"]
    if len(records) != features.shape[0]:
        raise ValueError(f"{base}: {len(records)} records, {features.shape[0]} rows")

    labels = torch.tensor([r["answer_index"] for r in records])
    groups = [r["block_id"] for r in records]
    rows = []
    distances = sorted({r["distance"] for r in records})
    for distance in distances:
        for evicted in (False, True):
            idx = [
                i
                for i, r in enumerate(records)
                if r["distance"] == distance and r["evicted"] == evicted
            ]
            if not idx:
                continue
            result = probe.cross_validated_accuracy(
                torch.tensor(features[idx].astype(np.float32)),
                labels[idx],
                [groups[i] for i in idx],
                n_classes=N_CLASSES,
            )
            rows.append(
                {
                    "distance": distance,
                    "evicted": evicted,
                    "accuracy": round(result["accuracy"], 4),
                    "p_value": result["p_value"],
                    "n": result["total"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tag", default="7b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out: dict = {"tag": args.tag, "chance": CHANCE, "strata": {}}
    for stratum in STRATA:
        base = os.path.join(args.results_dir, f"e5c_{args.tag}_{stratum}")
        if not os.path.exists(f"{base}.jsonl"):
            print(f"skipping {stratum}: not collected", file=sys.stderr)
            continue
        out["strata"][stratum] = decode(base)

    # The headline contrast: how much of the decodable signal at each distance
    # survives removing the attention channel's copy of the image.
    summary = []
    for stratum, rows in out["strata"].items():
        intact = {r["distance"]: r["accuracy"] for r in rows if not r["evicted"]}
        evicted = {r["distance"]: r["accuracy"] for r in rows if r["evicted"]}
        for distance in sorted(intact):
            if distance not in evicted:
                continue
            excess_intact = intact[distance] - CHANCE
            excess_evicted = evicted[distance] - CHANCE
            summary.append(
                {
                    "stratum": stratum,
                    "distance": distance,
                    "intact": intact[distance],
                    "evicted": evicted[distance],
                    "drop_pp": round((intact[distance] - evicted[distance]) * 100, 1),
                    "fraction_surviving_eviction": (
                        round(excess_evicted / excess_intact, 3)
                        if excess_intact > 1e-9
                        else None
                    ),
                }
            )
    out["summary"] = summary

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
