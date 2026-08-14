"""Trial materials built from COCO val2017 instance annotations.

Three property families, all eight-way forced choice so chance is exactly
12.5%, chosen to span the gist-to-detail range we expect the recurrent state
to compress along:

  identity  which object is in the picture
  count     how many of a named object are in the picture
  position  where in the picture a named object is

Every family is built as a BLOCK of eight trials that share one identical
candidate list, with each candidate correct exactly once inside the block.
That construction is what makes the pre-image control interpretable: with the
candidate set held fixed and the answer balanced, a decoder that has not read
the image cannot do better than chance by exploiting a prior over candidates.
An earlier project of ours reported a spurious 31.2% against a 12.5% chance
rate because trial identity leaked the label; this design forecloses that.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
COCO_ANNOTATION_ZIP = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

COUNT_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight")

# Eight cells of a 3x3 grid, centre excluded so the set is exactly eight.
GRID_CELLS = (
    "top left", "top centre", "top right",
    "middle left", "middle right",
    "bottom left", "bottom centre", "bottom right",
)

FAMILY_QUESTIONS = {
    "identity": "Which one of these is in the picture: {options}? Answer with one option.",
    "count": "How many {subject} are in the picture: {options}? Answer with one option.",
    "position": "Where in the picture is the {subject}: {options}? Answer with one option.",
}

# A neutral suffix appended before the state is read, identical in every trial
# so that whatever it contributes to the state is a constant across conditions.
PROBE_SUFFIX = "\nRecalling the picture:"


@dataclass(frozen=True)
class Trial:
    """One image, one property family, one correct answer among eight."""

    trial_id: str
    family: str
    block_id: str
    image_id: int
    file_name: str
    subject: str
    candidates: tuple[str, ...]
    answer: str
    answer_index: int
    metadata: dict = field(default_factory=dict)

    def question(self) -> str:
        options = ", ".join(self.candidates)
        return FAMILY_QUESTIONS[self.family].format(
            options=options, subject=self.subject
        )


def load_coco(annotation_path: str) -> dict:
    """Index COCO instance annotations by image and category."""
    with open(annotation_path) as fh:
        raw = json.load(fh)

    categories = {c["id"]: c["name"] for c in raw["categories"]}
    images = {img["id"]: img for img in raw["images"]}

    by_image: dict[int, dict[int, list]] = {}
    for ann in raw["annotations"]:
        if ann.get("iscrowd"):
            continue
        by_image.setdefault(ann["image_id"], {}).setdefault(ann["category_id"], []).append(ann)

    return {"categories": categories, "images": images, "by_image": by_image}


def _grid_cell(bbox: list, width: int, height: int) -> str | None:
    """Which of the eight grid cells holds a box centroid, or None if centre."""
    cx = (bbox[0] + bbox[2] / 2) / max(width, 1)
    cy = (bbox[1] + bbox[3] / 2) / max(height, 1)
    col = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    if row == 1 and col == 1:
        return None
    names = {
        (0, 0): "top left", (0, 1): "top centre", (0, 2): "top right",
        (1, 0): "middle left", (1, 2): "middle right",
        (2, 0): "bottom left", (2, 1): "bottom centre", (2, 2): "bottom right",
    }
    return names[(row, col)]


# One fixed inventory of eight visually distinct categories, used in every
# identity block. Holding the candidate list global (rather than resampling it
# per block, as count and position already do) makes all three families the
# same kind of eight-way problem, so one probe architecture and one chance
# rate apply throughout and the families are directly comparable.
IDENTITY_CATEGORIES = (
    "person", "car", "dog", "cat", "chair", "bird", "pizza", "elephant",
)


def build_identity_blocks(coco: dict, n_blocks: int, rng: random.Random) -> list[Trial]:
    """Blocks over a fixed eight-category inventory.

    Each trial's image contains its answer category and none of the other
    seven, so the correct answer is unambiguous, and within a block each of
    the eight categories is correct exactly once.
    """
    categories = coco["categories"]
    present: dict[str, set[int]] = {name: set() for name in IDENTITY_CATEGORIES}
    for image_id, cats in coco["by_image"].items():
        names = {categories[cid] for cid in cats}
        for name in IDENTITY_CATEGORIES:
            if name in names:
                present[name].add(image_id)

    # An image qualifies for category c only if c is the sole inventory
    # member it contains.
    exclusive: dict[str, list[int]] = {}
    for name in IDENTITY_CATEGORIES:
        others = set().union(*[present[o] for o in IDENTITY_CATEGORIES if o != name])
        exclusive[name] = sorted(present[name] - others)

    shortest = min(len(v) for v in exclusive.values())
    if shortest < n_blocks:
        n_blocks = shortest

    pools = {name: rng.sample(v, len(v)) for name, v in exclusive.items()}
    trials: list[Trial] = []
    for block in range(n_blocks):
        block_id = f"identity-{block:04d}"
        for name in IDENTITY_CATEGORIES:
            image_id = pools[name][block]
            trials.append(
                Trial(
                    trial_id=f"{block_id}-{name.replace(' ', '_')}",
                    family="identity",
                    block_id=block_id,
                    image_id=image_id,
                    file_name=coco["images"][image_id]["file_name"],
                    subject="object",
                    candidates=IDENTITY_CATEGORIES,
                    answer=name,
                    answer_index=IDENTITY_CATEGORIES.index(name),
                    metadata={"category": name},
                )
            )
    return trials


def build_count_blocks(coco: dict, n_blocks: int, rng: random.Random) -> list[Trial]:
    """Blocks where one category appears with each count from one to eight."""
    categories = coco["categories"]
    by_image = coco["by_image"]

    counts: dict[str, dict[int, list[int]]] = {}
    for image_id, cats in by_image.items():
        for cid, anns in cats.items():
            n = len(anns)
            if 1 <= n <= 8:
                counts.setdefault(categories[cid], {}).setdefault(n, []).append(image_id)

    usable = [
        name for name, per_count in counts.items()
        if all(per_count.get(n) for n in range(1, 9))
    ]
    trials: list[Trial] = []
    if not usable:
        return trials

    for block in range(n_blocks):
        name = usable[block % len(usable)]
        block_id = f"count-{block:04d}-{name.replace(' ', '_')}"
        for n in range(1, 9):
            image_id = rng.choice(counts[name][n])
            trials.append(
                Trial(
                    trial_id=f"{block_id}-{n}",
                    family="count",
                    block_id=block_id,
                    image_id=image_id,
                    file_name=coco["images"][image_id]["file_name"],
                    subject=name if n == 1 else f"{name}s",
                    candidates=COUNT_WORDS,
                    answer=COUNT_WORDS[n - 1],
                    answer_index=n - 1,
                    metadata={"true_count": n, "category": name},
                )
            )
    return trials


def build_position_blocks(coco: dict, n_blocks: int, rng: random.Random) -> list[Trial]:
    """Blocks where a single instance of one category sits in each grid cell."""
    categories = coco["categories"]
    images = coco["images"]

    by_cell: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for image_id, cats in coco["by_image"].items():
        img = images[image_id]
        for cid, anns in cats.items():
            if len(anns) != 1:
                continue  # unambiguous subject only
            cell = _grid_cell(anns[0]["bbox"], img["width"], img["height"])
            if cell is None:
                continue
            name = categories[cid]
            by_cell.setdefault(name, {}).setdefault(cell, []).append((image_id, cell))

    usable = [
        name for name, cells in by_cell.items()
        if all(cells.get(cell) for cell in GRID_CELLS)
    ]
    trials: list[Trial] = []
    if not usable:
        return trials

    for block in range(n_blocks):
        name = usable[block % len(usable)]
        block_id = f"position-{block:04d}-{name.replace(' ', '_')}"
        for cell in GRID_CELLS:
            image_id, _ = rng.choice(by_cell[name][cell])
            trials.append(
                Trial(
                    trial_id=f"{block_id}-{cell.replace(' ', '_')}",
                    family="position",
                    block_id=block_id,
                    image_id=image_id,
                    file_name=images[image_id]["file_name"],
                    subject=name,
                    candidates=GRID_CELLS,
                    answer=cell,
                    answer_index=GRID_CELLS.index(cell),
                    metadata={"category": name},
                )
            )
    return trials


BUILDERS = {
    "identity": build_identity_blocks,
    "count": build_count_blocks,
    "position": build_position_blocks,
}


def build_trials(coco: dict, families: list[str], n_blocks: int, seed: int) -> list[Trial]:
    rng = random.Random(seed)
    out: list[Trial] = []
    for family in families:
        builder = BUILDERS.get(family)
        if builder is None:
            raise ValueError(f"unknown family {family!r}")
        built = builder(coco, n_blocks, rng)
        if not built:
            raise RuntimeError(f"family {family!r} produced no trials")
        out.extend(built)
    return out


def filler_text(words: list[str], n_tokens_approx: int, rng: random.Random) -> str:
    """Filler drawn from a public-domain corpus, sized in words.

    Sized deliberately loosely: the exact token count is measured after
    tokenisation and recorded per trial, rather than assumed here.
    """
    if n_tokens_approx <= 0:
        return ""
    n_words = max(1, int(n_tokens_approx * 0.75))
    start = rng.randrange(0, max(1, len(words) - n_words))
    return " ".join(words[start:start + n_words])
