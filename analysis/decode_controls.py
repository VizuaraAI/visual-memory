"""Decode E5d: the controls that decide whether the relay result survives.

Three questions, one collection.

1. Is the collapse caused by the missing image, or by the perturbation of
   shortening every attention cache? Compare `visual_all` against
   `prefill_matched`, which removes the same number of positions from the same
   caches but takes them from the text preceding the image. If only the visual
   eviction collapses the readout, the perturbation account is dead.

2. Which attention layers supply the relayed content? Compare
   `visual_early3` against `visual_late3`.

3. Does a head's dependence on the attention channel scale with its own decay
   rate? With ten bins rather than three, this becomes a measured relationship.

Runs on CPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab import probe  # noqa: E402

N_CLASSES = 8
CHANCE = 1.0 / N_CLASSES


def decode(base: str) -> list[dict]:
    with open(f"{base}.jsonl") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    features = np.load(f"{base}_features.npz")["features"]
    if len(records) != features.shape[0]:
        raise ValueError(f"{base}: {len(records)} records, {features.shape[0]} rows")

    labels = torch.tensor([r["answer_index"] for r in records])
    groups = [r["block_id"] for r in records]
    rows = []
    conditions = sorted({r["condition"] for r in records})
    for distance in sorted({r["distance"] for r in records}):
        for condition in conditions:
            idx = [
                i for i, r in enumerate(records)
                if r["distance"] == distance and r["condition"] == condition
            ]
            if not idx:
                continue
            result = probe.cross_validated_accuracy(
                torch.tensor(features[idx].astype(np.float32)),
                labels[idx],
                [groups[i] for i in idx],
                n_classes=N_CLASSES,
            )
            rows.append({
                "distance": distance,
                "condition": condition,
                "accuracy": round(result["accuracy"], 4),
                "p_value": result["p_value"],
                "n": result["total"],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tag", default="ctrl")
    parser.add_argument("--prefix", default="e5d")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, f"{args.prefix}_{args.tag}_*.jsonl")
    bases = sorted(p[: -len(".jsonl")] for p in glob.glob(pattern))
    if not bases:
        print(f"no collections matching {pattern}", file=sys.stderr)
        sys.exit(1)

    report_path = os.path.join(args.results_dir, f"{args.prefix}_report_{args.tag}.json")
    predicted = {}
    if os.path.exists(report_path):
        predicted = json.load(open(report_path)).get("strata", {})

    out: dict = {"tag": args.tag, "chance": CHANCE, "strata": {}}
    for base in bases:
        name = re.sub(rf"^.*{re.escape(args.prefix)}_{re.escape(args.tag)}_", "", base)
        out["strata"][name] = {
            "predicted_median_half_life": predicted.get(
                name.split("_s")[0], {}
            ).get("median_half_life"),
            "per_condition": decode(base),
        }

    # Headline contrasts, per stratum and distance.
    contrasts = []
    for name, entry in out["strata"].items():
        by = {(r["distance"], r["condition"]): r["accuracy"]
              for r in entry["per_condition"]}
        for distance in sorted({d for d, _ in by}):
            intact = by.get((distance, "intact"))
            if intact is None:
                continue
            row = {
                "stratum": name,
                "distance": distance,
                "predicted_median_half_life": entry["predicted_median_half_life"],
                "intact": intact,
            }
            present = sorted({c for _, c in by} - {"intact"})
            for condition in present:
                value = by.get((distance, condition))
                if value is not None:
                    row[condition] = value
                    row[f"drop_{condition}"] = round((intact - value) * 100, 1)
            contrasts.append(row)
    out["contrasts"] = contrasts

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
