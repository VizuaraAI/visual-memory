"""Shared machinery for cache-splice experiments.

Factored out of the original in-line implementation so the scaled-up splice
(E10) and the eviction experiment (E9) use one definition of prefill and of
feeding a question through an existing cache, rather than three copies that can
drift apart.
"""

from __future__ import annotations

import torch


def prefill_context(model, processor, picture, filler: str):
    """Run image plus filler and return (cache, prompt_length).

    `add_generation_prompt=False` because the question is fed through the cache
    afterwards rather than being part of the prompt.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": filler}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(text=[prompt], images=[picture], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    return out.past_key_values, int(inputs["input_ids"].shape[1])


def ask_through(model, processor, cache, question: str, max_new_tokens: int = 10) -> str:
    """Feed a question through an existing cache and greedily decode an answer.

    Zamba2's naive path accepts only one token at a time once the cache holds a
    previous state: `torch_forward` does `input_states.squeeze(1)` under
    `has_previous_state`, so a multi-token feed makes the gate four-dimensional
    and raises. Every token therefore goes through individually.
    """
    ids = processor.tokenizer(
        question, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    language_model = model.language_model
    with torch.no_grad():
        out = None
        for position in range(ids.shape[1]):
            out = language_model(
                input_ids=ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
        produced = []
        for _ in range(max_new_tokens):
            token = int(torch.argmax(out.logits[0, -1]))
            produced.append(token)
            if token == processor.tokenizer.eos_token_id:
                break
            out = language_model(
                input_ids=torch.tensor([[token]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
    return processor.tokenizer.decode(produced, skip_special_tokens=True)


def full_prompt_inputs(model, processor, picture, question: str):
    """The prompt the model normally sees, image and question together.

    Asking a bare question through an image-only cache is not a format this
    model performs under. It tends to continue the prompt rather than answer,
    which names every candidate and scores as ambiguous. Building the real
    prompt and splitting it at the image boundary keeps the intervention in the
    right place while leaving the format intact.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=[prompt], images=[picture], return_tensors="pt").to(
        model.device
    )


def prefill_to(model, inputs, split: int):
    """Run the prompt up to `split`, which should be the token after the image."""
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"][:, :split],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=True,
        )
    return out.past_key_values


def finish_prompt(model, processor, cache, inputs, split: int, max_new_tokens: int = 10) -> str:
    """Feed the remaining prompt tokens through the cache, then decode greedily."""
    ids = inputs["input_ids"]
    language_model = model.language_model
    with torch.no_grad():
        out = None
        for position in range(split, ids.shape[1]):
            out = language_model(
                input_ids=ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
        produced = []
        for _ in range(max_new_tokens):
            token = int(torch.argmax(out.logits[0, -1]))
            produced.append(token)
            if token == processor.tokenizer.eos_token_id:
                break
            out = language_model(
                input_ids=torch.tensor([[token]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
    return processor.tokenizer.decode(produced, skip_special_tokens=True)


def state_fingerprint(cache, layer: int = 40) -> float:
    """A scalar summary of one recurrent state, for verifying a swap took."""
    return float(cache.ssm_states[layer].float().abs().sum())


def attention_subsets(attn_ids: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    """Named subsets of the attention layers for the dose-response splice.

    The evenly spaced subsets answer "how many attention layers are needed to
    move the answer", and the early/late pair answers "which ones", which the
    all-or-nothing splice could not distinguish.
    """
    n = len(attn_ids)
    return {
        "attention_1": (attn_ids[n // 2],),
        "attention_3": (attn_ids[0], attn_ids[n // 2], attn_ids[-1]),
        "attention_7": tuple(attn_ids[i] for i in range(0, n, max(1, n // 7))[:7]),
        "attention_all": tuple(attn_ids),
        "attention_early3": tuple(attn_ids[:3]),
        "attention_late3": tuple(attn_ids[-3:]),
    }
