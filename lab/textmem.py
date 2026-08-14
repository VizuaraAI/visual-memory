"""Trial materials for the text analogue of the visual memory experiment.

The second model family is a language model, so the item to be remembered is a
code word rather than a picture. Everything else mirrors the visual design so
the two are comparable:

  * eight-way forced choice, so chance is exactly 12.5%
  * blocks of eight trials sharing one candidate list, each candidate correct
    exactly once, which is what makes the pre-item control interpretable
  * a fixed neutral readout suffix, identical across conditions
  * filler measured in tokens between the item and the readout

The visual experiment's central intervention was evicting the image's key-value
entries. Its analogue here is evicting the code word's entries, which is a much
smaller intervention in absolute terms, a handful of positions rather than
several hundred. That is the point: if the relay account is architectural, the
size of the item should not matter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Chosen to be single common words that tokenise short and are unlikely to be
# predictable from the surrounding filler.
CODE_WORDS = (
    "CRIMSON",
    "HARBOUR",
    "LANTERN",
    "MEADOW",
    "PELICAN",
    "QUARTZ",
    "THUNDER",
    "WALNUT",
)

CARRIER = "Remember this for later. The code word is {word}."
QUESTION = (
    "Which one of these was the code word: {options}? Answer with one option."
)
PROBE_SUFFIX = "\nRecalling the code word:"


@dataclass(frozen=True)
class TextTrial:
    """One code word, one block, one correct answer among eight."""

    trial_id: str
    block_id: str
    candidates: tuple[str, ...]
    answer: str
    answer_index: int

    def carrier(self) -> str:
        return CARRIER.format(word=self.answer)

    def question(self) -> str:
        return QUESTION.format(options=", ".join(self.candidates))


def build_blocks(n_blocks: int, seed: int = 17) -> list[TextTrial]:
    """Blocks of eight, each candidate correct exactly once inside a block.

    The candidate list is shuffled per block but identical across the eight
    trials of that block, so a decoder that never saw the code word cannot beat
    chance by exploiting a prior over candidates.
    """
    rng = random.Random(seed)
    trials: list[TextTrial] = []
    for block in range(n_blocks):
        candidates = list(CODE_WORDS)
        rng.shuffle(candidates)
        candidates = tuple(candidates)
        block_id = f"text-{block:04d}"
        for index, answer in enumerate(candidates):
            trials.append(
                TextTrial(
                    trial_id=f"{block_id}-{answer.lower()}",
                    block_id=block_id,
                    candidates=candidates,
                    answer=answer,
                    answer_index=index,
                )
            )
    return trials


def score_answer(text: str, trial: TextTrial) -> dict:
    """Which candidate the generated text names, if exactly one."""
    lowered = text.lower()
    hits = [c for c in trial.candidates if c.lower() in lowered]
    chosen = hits[0] if len(hits) == 1 else None
    return {
        "chosen": chosen,
        "correct": bool(chosen is not None and chosen == trial.answer),
        "ambiguous": len(hits) > 1,
        "unparsed": not hits,
    }
