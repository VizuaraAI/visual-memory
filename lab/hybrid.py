"""Family-agnostic access to a hybrid Mamba2 model's two memory channels.

Zamba2 and the newer upstream hybrids (Granite 4.0, Bamba) implement the same
recurrence but expose it through different objects. Zamba2's cache carries flat
`ssm_states` and `key_cache` dictionaries; recent transformers replaced those
with a per-layer list, where a Mamba2 layer holds `recurrent_states[0]` and an
attention layer holds `keys` and `values`.

The physics is identical in both. Verified against each model's own source:

  * Zamba2: `dA = exp(dt * A)`, `A = -exp(A_log)` per head.
  * GraniteMoeHybrid: `ssm_states = state * dA + dBx`, with the same per-head
    `A_log` and the same `C` group expansion.

So the instruments differ only in where they reach for the tensors, which is
what this module isolates. Everything above it stays unchanged.
"""

from __future__ import annotations

import torch

from .vtelemetry import Geometry


def geometry_for(model) -> Geometry:
    """Read a Geometry from whichever config schema the model happens to use.

    Zamba2 names its fields `n_mamba_heads`, `mamba_headdim`, `mamba_ngroups`
    and lists attention layers in `hybrid_layer_ids`. Granite and Bamba use
    `mamba_n_heads`, `mamba_d_head`, `mamba_n_groups`, and identify attention
    layers through `layer_types` or `attn_layer_indices`. Both are read here so
    no experiment has to know which family it is running on.
    """
    cfg = getattr(model.config, "text_config", model.config)

    def pick(*names, required=True):
        for name in names:
            value = getattr(cfg, name, None)
            if value is not None:
                return value
        if required:
            raise RuntimeError(f"config exposes none of {names}")
        return None

    n_layers = int(pick("num_hidden_layers"))
    attention_layers = _attention_layers(cfg, n_layers)
    return Geometry(
        n_layers=n_layers,
        n_heads=int(pick("n_mamba_heads", "mamba_n_heads", "mamba_num_heads")),
        head_dim=int(pick("mamba_headdim", "mamba_d_head", "mamba_head_dim")),
        state_dim=int(pick("mamba_d_state")),
        n_groups=int(pick("mamba_ngroups", "mamba_n_groups", "n_groups")),
        intermediate=int(pick("mamba_expand", "expand")) * int(pick("hidden_size")),
        attention_layers=attention_layers,
    )


def _attention_layers(cfg, n_layers: int) -> tuple[int, ...]:
    explicit = getattr(cfg, "hybrid_layer_ids", None)
    if explicit:
        return tuple(int(i) for i in explicit)
    explicit = getattr(cfg, "attn_layer_indices", None)
    if explicit:
        return tuple(int(i) for i in explicit)
    types = getattr(cfg, "layer_types", None)
    if types:
        return tuple(
            i for i, t in enumerate(types) if "full" in str(t) or t == "attention"
        )
    pattern = getattr(cfg, "hybrid_override_pattern", None)
    if pattern:
        return tuple(i for i, ch in enumerate(pattern) if ch == "*")
    raise RuntimeError("could not determine which layers carry attention")


def _entry(cache, layer: int):
    layers = getattr(cache, "layers", None)
    return None if layers is None else layers[layer]


def recurrent_state(cache, layer: int) -> torch.Tensor:
    """The Mamba2 state for one layer, whichever way the cache stores it."""
    flat = getattr(cache, "ssm_states", None)
    if flat is not None:
        return flat[layer]
    entry = _entry(cache, layer)
    states = getattr(entry, "recurrent_states", None)
    if states is None:
        raise RuntimeError(f"layer {layer} cache exposes no recurrent state")
    return states[0]


def set_recurrent_state(cache, layer: int, value: torch.Tensor) -> None:
    flat = getattr(cache, "ssm_states", None)
    if flat is not None:
        flat[layer] = value
        return
    _entry(cache, layer).recurrent_states[0] = value


def attention_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = getattr(cache, "key_cache", None)
    if keys is not None:
        return keys[layer], cache.value_cache[layer]
    entry = _entry(cache, layer)
    return entry.keys, entry.values


def set_attention_kv(cache, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    existing = getattr(cache, "key_cache", None)
    if existing is not None:
        existing[layer] = keys
        cache.value_cache[layer] = values
        return
    entry = _entry(cache, layer)
    entry.keys = keys
    entry.values = values


def mamba_mixer(layer):
    """The Mamba2 mixer for a decoder layer, across naming conventions."""
    for name in ("mamba", "mixer", "self_mamba"):
        found = getattr(layer, name, None)
        if found is not None:
            return found
    inner = getattr(layer, "mamba_decoder", None)
    if inner is not None:
        return inner.mamba
    raise RuntimeError(f"no mamba mixer on layer of type {type(layer).__name__}")


def mamba_layer_ids(model, geom: Geometry) -> tuple[int, ...]:
    """Indices of layers that actually carry a Mamba2 mixer.

    Zamba2 puts a mixer on every layer and attention on some of them. Granite
    alternates, so its attention layers have no mixer at all and the recurrent
    channel spans only the remainder. Any code that loops over "the recurrent
    layers" has to ask rather than assume.
    """
    inner = getattr(getattr(model, "model", model), "layers", None)
    if inner is None:
        raise RuntimeError("could not locate decoder layers")
    ids = []
    for index, layer in enumerate(inner):
        try:
            mamba_mixer(layer)
        except RuntimeError:
            continue
        ids.append(index)
    return tuple(ids)


def project_state_masked(
    cache,
    matrix: torch.Tensor,
    head_masks: torch.Tensor,
    layer_ids: tuple[int, ...],
    geom: Geometry,
) -> torch.Tensor:
    """Project the recurrent state of the given layers, other heads zeroed.

    Returns [len(layer_ids), n_features]. Zeroing unselected heads and using the
    full matrix is exactly equivalent to slicing the matrix rows for those
    heads, and needs one matrix instead of one per layer per stratum.
    """
    rows = []
    for position, layer in enumerate(layer_ids):
        state = recurrent_state(cache, layer)[0].float()
        mask = head_masks[position].to(state.device).to(state.dtype)
        masked = (state * mask[:, None, None]).reshape(-1)
        rows.append((masked @ matrix.to(state.device).float()).cpu())
    return torch.stack(rows)
