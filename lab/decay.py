"""Runner for the visual decay experiment.

For one trial at one distance it produces two things:

  readout   an [L, H, C] tensor of similarities between the recurrent state
            and each of the eight candidate answers, which says what the
            state still contains
  behaviour the answer the model actually gives, which says what it can use

The gap between those two is the point of the study.
"""

from __future__ import annotations

import io
import random

import requests
import torch

from . import vtelemetry as vt
from .materials import COCO_IMAGE_URL, PROBE_SUFFIX, Trial, filler_text

USER_AGENT = "visual-memory-lab/0.1"


def fetch_image(file_name: str, cache_dir: str, size: int | None = None):
    """Fetch a COCO image, caching it on the Modal volume.

    `size` forces a square resize. COCO images vary in dimensions, and the
    vision encoder emits a token count that follows the image geometry, so
    two runs on different images otherwise produce different-length attention
    caches. Any experiment that exchanges those caches between runs must fix
    the geometry first, or the intervention corrupts the sequence instead of
    transplanting a memory.
    """
    import os

    from PIL import Image as PILImage

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, file_name)
    if not os.path.exists(path):
        resp = requests.get(
            COCO_IMAGE_URL.format(file_name=file_name),
            timeout=120,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    image = PILImage.open(path).convert("RGB")
    if size is not None:
        image = letterbox(image, size)
    return image


def letterbox(image, size: int):
    """Fit an image onto a square canvas without distorting it.

    Stretching every image to a square equalises the vision-token count but
    deforms the content, and the model pays for it: identity accuracy fell
    from 17/30 to 8/30 under a plain square resize. Letterboxing preserves the
    aspect ratio and pads to the same canvas, so token counts still match
    across runs while the picture itself is left alone.
    """
    from PIL import Image as PILImage

    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, PILImage.BICUBIC)
    canvas = PILImage.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(
        resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    )
    return canvas


def build_codebook(
    model, processor, words: tuple[str, ...], geom: vt.Geometry
) -> torch.Tensor:
    """Value-space vectors for each candidate answer.

    Each word is run in a fixed neutral carrier and its written value x is
    captured at the word's final token, giving [n_words, L, H, P]. The carrier
    is constant so differences between entries come from the word alone.
    """
    carrier = "The answer is {word}"
    vectors = []
    for word in words:
        text = carrier.format(word=word)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        sink: dict = {}
        with vt.capture_projections(model, sink, geom):
            with torch.no_grad():
                model(**inputs, use_cache=False)
        vectors.append(vt.split_projection(sink, geom).x)
    return torch.stack(vectors)


def _readout_prompt(processor, filler: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": filler}],
        }
    ]
    base = processor.apply_chat_template(messages, add_generation_prompt=True)
    return base + PROBE_SUFFIX


def _question_prompt(processor, filler: str, question: str) -> str:
    text = f"{filler}\n\n{question}" if filler else question
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def run_readout(
    model,
    processor,
    image,
    filler: str,
    codebook: torch.Tensor,
    matrix: torch.Tensor,
    geom: vt.Geometry,
) -> dict:
    """Prefill image + filler + neutral suffix, then read the state two ways.

    `scores` is the zero-shot codebook similarity, kept as a secondary and
    assumption-light measure. `features` is the projected state, which is what
    the trained probe consumes and which makes no assumption that image-written
    content lands in the same place as word-written content.
    """
    prompt = _readout_prompt(processor, filler)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

    sink: dict = {}
    with vt.capture_projections(model, sink, geom):
        with torch.no_grad():
            out = model(**inputs, use_cache=True)

    cache = out.past_key_values
    capture = vt.split_projection(sink, geom)
    readout = vt.read_state(cache, capture.c, geom)
    scores = vt.codebook_scores(readout, codebook)
    features = vt.project_state(cache, matrix, geom)
    return {
        "scores": scores.to(torch.float16),
        "features": features.to(torch.float16),
        "n_tokens": int(inputs["input_ids"].shape[1]),
    }


def run_behaviour(model, processor, image, filler: str, trial: Trial, max_new_tokens: int = 12) -> dict:
    """Ask the actual question and score the generated answer."""
    prompt = _question_prompt(processor, filler, trial.question())
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

    return {
        "raw": text,
        **score_answer(text, trial),
        "n_tokens": int(inputs["input_ids"].shape[1]),
    }


# Count questions offer word candidates, but the model frequently replies with
# a digit. Accepting both is a scoring fix, not a relaxation: the mapping is
# exact and applies identically to correct and incorrect answers.
DIGIT_FOR_WORD = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
}


def _mentions(candidate: str, lowered: str) -> bool:
    if candidate in lowered:
        return True
    digit = DIGIT_FOR_WORD.get(candidate)
    if digit is None:
        return False
    # Digit match must not be part of a longer number.
    for pos in range(len(lowered)):
        if lowered[pos] != digit:
            continue
        before = lowered[pos - 1] if pos > 0 else " "
        after = lowered[pos + 1] if pos + 1 < len(lowered) else " "
        if not before.isdigit() and not after.isdigit():
            return True
    return False


def score_answer(text: str, trial: Trial) -> dict:
    """Which candidate the generated text names, if exactly one."""
    lowered = text.lower()
    hits = [c for c in trial.candidates if _mentions(c.lower(), lowered)]
    chosen = hits[0] if len(hits) == 1 else None
    return {
        "chosen": chosen,
        "correct": bool(chosen is not None and chosen.lower() == trial.answer.lower()),
        "ambiguous": len(hits) > 1,
        "unparsed": not hits,
    }


def run_trial(
    model,
    processor,
    trial: Trial,
    distance: int,
    filler_words: list[str],
    codebooks: dict,
    matrix,
    image_cache_dir: str,
    seed: int,
    geom: vt.Geometry,
    do_behaviour: bool = True,
) -> dict:
    """One trial at one distance, both instruments."""
    rng = random.Random(f"{trial.trial_id}-{distance}-{seed}")
    filler = filler_text(filler_words, distance, rng)
    image = fetch_image(trial.file_name, image_cache_dir)

    readout = run_readout(
        model, processor, image, filler, codebooks[trial.family], matrix, geom
    )
    record = {
        "trial_id": trial.trial_id,
        "family": trial.family,
        "block_id": trial.block_id,
        "image_id": trial.image_id,
        "distance": distance,
        "answer_index": trial.answer_index,
        "candidates": list(trial.candidates),
        "readout_tokens": readout["n_tokens"],
    }
    if do_behaviour:
        behaviour = run_behaviour(model, processor, image, filler, trial)
        record["behaviour"] = behaviour
    return record, readout["scores"], readout["features"]


def run_pre_image_control(
    model,
    processor,
    trial: Trial,
    filler_words: list[str],
    codebooks: dict,
    matrix,
    seed: int,
    geom: vt.Geometry,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Identical prompt with the image never shown.

    If the decoder can beat 12.5% here, the signal is coming from the trial
    construction rather than from the picture, and everything downstream is
    void. This is the control that caught our confound last time.
    """
    rng = random.Random(f"preimage-{trial.trial_id}-{seed}")
    filler = filler_text(filler_words, 0, rng)
    messages = [{"role": "user", "content": [{"type": "text", "text": filler}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True) + PROBE_SUFFIX
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    sink: dict = {}
    with vt.capture_projections(model, sink, geom):
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    capture = vt.split_projection(sink, geom)
    readout = vt.read_state(cache, capture.c, geom)
    scores = vt.codebook_scores(readout, codebooks[trial.family])
    features = vt.project_state(cache, matrix, geom)
    record = {
        "trial_id": trial.trial_id,
        "family": trial.family,
        "block_id": trial.block_id,
        "image_id": trial.image_id,
        "distance": -1,
        "answer_index": trial.answer_index,
        "candidates": list(trial.candidates),
        "control": "pre_image",
    }
    return record, scores.to(torch.float16), features.to(torch.float16)
