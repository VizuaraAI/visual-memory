"""Instruments for reading and splicing the memory of a hybrid Mamba2 VLM.

Every shape-dependent instrument takes a `Geometry`, read from the loaded
model's own config by `geometry(model)`. Nothing here assumes a particular
layer count, head count or state width, so the same instruments run on
Zamba2-VL at 7B, 2.7B and 1.2B, and on any other Mamba2 hybrid whose mixer
exposes the same `in_proj` layout.

Architecture facts these instruments depend on, verified in the e0/e1 gates
rather than assumed. For Zamba2-VL-7B specifically:

  * 81 decoder layers, every one carrying a Mamba2 mixer. Thirteen of them
    (indices 6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77) additionally
    carry a shared attention block, so the recurrent channel spans all 81
    layers while the attention channel spans 13.
  * The recurrent state is `cache.ssm_states[layer] : [1, 112, 64, 64]`, read
    as [batch, head, head_dim, state_dim].
  * `mamba.in_proj` emits one tensor of width 14704, laid out as
    z(7168) | x(7168) | B(128) | C(128) | dt(112), with n_groups = 2 so heads
    0..55 share B/C group 0 and heads 56..111 share group 1.
  * Mamba2 recurrence: S_t = S_{t-1} exp(dt A) + dt (x_t B_t^T), and
    y_t = S_t C_t. Writing uses (x, B), reading uses C, and the retention of
    anything already in the state is governed by A and dt alone.

The layout above is what `Geometry.offsets()` reconstructs from config values,
so a model with different widths is handled without editing this file.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

MODEL_ID = "Zyphra/Zamba2-VL-7B"


@dataclass(frozen=True)
class Geometry:
    """Everything the instruments need to know about a model's shape."""

    n_layers: int
    n_heads: int
    head_dim: int
    state_dim: int
    n_groups: int
    intermediate: int
    attention_layers: tuple[int, ...]

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_groups

    @property
    def state_numel(self) -> int:
        return self.n_heads * self.head_dim * self.state_dim

    @property
    def projection_width(self) -> int:
        """Expected width of the in_proj output: z | x | B | C | dt."""
        return 2 * self.intermediate + 2 * self.n_groups * self.state_dim + self.n_heads

    def offsets(self) -> tuple[int, int, int, int]:
        """(z_end, x_end, b_end, c_end) into the in_proj output."""
        z_end = self.intermediate
        x_end = z_end + self.intermediate
        b_end = x_end + self.n_groups * self.state_dim
        c_end = b_end + self.n_groups * self.state_dim
        return z_end, x_end, b_end, c_end

    def head_group_index(self) -> torch.Tensor:
        """Map each head to its B/C group. Shape [n_heads]."""
        return torch.arange(self.n_heads) // self.heads_per_group


def geometry(model) -> Geometry:
    """Read the model's actual shape rather than assuming any one checkpoint's."""
    cfg = getattr(model.config, "text_config", model.config)
    return Geometry(
        n_layers=int(cfg.num_hidden_layers),
        n_heads=int(cfg.n_mamba_heads),
        head_dim=int(cfg.mamba_headdim),
        state_dim=int(cfg.mamba_d_state),
        n_groups=int(cfg.mamba_ngroups),
        intermediate=int(cfg.mamba_expand) * int(cfg.hidden_size),
        attention_layers=tuple(cfg.hybrid_layer_ids),
    )


def load_model(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load a Zamba2-VL checkpoint and its processor. Returns (model, processor)."""
    from transformers import AutoProcessor
    from transformers.models.zamba2_vl.modeling_zamba2_vl import (
        Zamba2_VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Zamba2_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, processor


def decoder_layers(model, geom: Geometry | None = None):
    """The decoder layers, whatever the wrapper nesting happens to be.

    Passing a geometry turns a silent shape mismatch into an immediate error,
    which is the failure we want when a new checkpoint is wired up wrongly.
    """
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate decoder layers on the model")
    if geom is not None and len(layers) != geom.n_layers:
        raise RuntimeError(
            f"config declares {geom.n_layers} layers, model exposes {len(layers)}"
        )
    return layers


def mamba_mixer(layer):
    """The Mamba2 mixer for a layer, for both plain and hybrid layer types."""
    if hasattr(layer, "mamba"):
        return layer.mamba
    if hasattr(layer, "mamba_decoder"):
        return layer.mamba_decoder.mamba
    raise RuntimeError(f"no mamba mixer on layer of type {type(layer).__name__}")


def attention_layer_ids(model) -> tuple[int, ...]:
    cfg = getattr(model.config, "text_config", model.config)
    return tuple(cfg.hybrid_layer_ids)


@dataclass(frozen=True)
class ProjectionCapture:
    """Per-layer in_proj components for one chosen token position.

    Each field is [n_layers, ...] on CPU in float32. `x` is the value written
    into the state, `b` the write key, `c` the read vector, `dt` the raw gate
    before softplus and bias.
    """

    x: torch.Tensor  # [L, n_heads, head_dim]
    b: torch.Tensor  # [L, n_groups, state_dim]
    c: torch.Tensor  # [L, n_groups, state_dim]
    dt: torch.Tensor  # [L, n_heads]


@contextlib.contextmanager
def capture_projections(model, sink: dict, geom: Geometry | None = None):
    """Hook every mamba in_proj and record its output for the last position.

    The hook stores only the final token of the sequence, which is what the
    probe designs need, keeping memory flat regardless of prompt length.
    """
    handles = []
    layers = decoder_layers(model, geom)

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # output: [batch, seq, projection_width]
            sink[layer_idx] = output[0, -1, :].detach().float().cpu()

        return hook

    try:
        for idx, layer in enumerate(layers):
            handles.append(
                mamba_mixer(layer).in_proj.register_forward_hook(make_hook(idx))
            )
        yield sink
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def capture_dt(model, sink: dict, geom: Geometry):
    """Hook every mamba in_proj and record the dt slice at every position.

    E5 needs the gate at each token of the filler, not just the last, because
    retention over k tokens is exp(A * sum of dt over those k tokens). Only the
    final `n_heads` channels are kept, so the memory cost is a small fraction
    of storing the whole projection, and one trial at 2048 tokens fits
    comfortably in host memory.

    Each sink entry is a list of [seq, n_heads] tensors, one per forward call,
    in call order. Feeding tokens one at a time therefore yields many entries
    of length one, which `stacked_dt` concatenates back into a sequence.
    """
    handles = []
    layers = decoder_layers(model, geom)
    _, _, _, c_end = geom.offsets()

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            sink.setdefault(layer_idx, []).append(
                output[0, :, c_end:].detach().float().cpu()
            )

        return hook

    try:
        for idx, layer in enumerate(layers):
            handles.append(
                mamba_mixer(layer).in_proj.register_forward_hook(make_hook(idx))
            )
        yield sink
    finally:
        for handle in handles:
            handle.remove()


def stacked_dt(sink: dict, geom: Geometry) -> torch.Tensor:
    """Concatenate a `capture_dt` sink into [n_layers, seq, n_heads]."""
    if len(sink) != geom.n_layers:
        raise ValueError(f"expected {geom.n_layers} captured layers, got {len(sink)}")
    rows = [torch.cat(sink[idx], dim=0) for idx in range(geom.n_layers)]
    lengths = {row.shape[0] for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"layers disagree on sequence length: {sorted(lengths)}")
    return torch.stack(rows)


def split_projection(sink: dict, geom: Geometry) -> ProjectionCapture:
    """Turn raw in_proj vectors into (x, B, C, dt), stacked over layers."""
    if len(sink) != geom.n_layers:
        raise ValueError(f"expected {geom.n_layers} captured layers, got {len(sink)}")
    z_end, x_end, b_end, c_end = geom.offsets()
    xs, bs, cs, dts = [], [], [], []
    for idx in range(geom.n_layers):
        vec = sink[idx]
        if vec.shape[-1] != geom.projection_width:
            raise ValueError(
                f"in_proj width {vec.shape[-1]} at layer {idx}, "
                f"expected {geom.projection_width}"
            )
        xs.append(vec[z_end:x_end].reshape(geom.n_heads, geom.head_dim))
        bs.append(vec[x_end:b_end].reshape(geom.n_groups, geom.state_dim))
        cs.append(vec[b_end:c_end].reshape(geom.n_groups, geom.state_dim))
        dts.append(vec[c_end:])
    return ProjectionCapture(
        x=torch.stack(xs), b=torch.stack(bs), c=torch.stack(cs), dt=torch.stack(dts)
    )


def read_state(cache, read_vectors: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Read every layer's recurrent state with a per-layer read vector C.

    read_vectors: [L, n_groups, state_dim]
    returns:      [L, n_heads, head_dim] in float32 on CPU

    Implements y[h] = S[h] @ C[group(h)], the Mamba2 output map.
    """
    groups = geom.head_group_index()
    outputs = []
    for idx in range(geom.n_layers):
        state = cache.ssm_states[idx][0].float()  # [H, P, N]
        per_head_c = read_vectors[idx].to(state.device)[groups].float()  # [H, N]
        outputs.append(torch.einsum("hpn,hn->hp", state, per_head_c).cpu())
    return torch.stack(outputs)


def codebook_scores(readout: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between a state readout and each candidate value.

    readout:  [L, H, P]
    codebook: [C, L, H, P]
    returns:  [L, H, C]
    """
    r = torch.nn.functional.normalize(readout, dim=-1)
    k = torch.nn.functional.normalize(codebook, dim=-1)
    return torch.einsum("lhp,clhp->lhc", r, k)


def projection_matrix(
    n_features: int,
    geom: Geometry,
    seed: int = 20260808,
    device: str = "cuda",
) -> torch.Tensor:
    """A fixed random projection from one layer's flattened state.

    In the 7B model the state is 112*64*64 = 458,752 numbers per layer, far too
    wide to probe directly. A fixed Gaussian projection preserves linear
    separability up to a controlled distortion (Johnson-Lindenstrauss), so a
    linear probe on the projected features is, to that tolerance, a linear probe
    on the state. The matrix is regenerated deterministically from `seed` rather
    than stored, so every run and every layer shares the same basis.

    Because the columns are drawn independently, the first k columns are
    themselves a valid k-feature projection. That is what makes the probe
    capacity ladder cost one collection instead of four.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(geom.state_numel, n_features, generator=generator)
    return (mat / (n_features ** 0.5)).to(device)


def project_state(cache, matrix: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Project every layer's recurrent state. Returns [L, n_features] float32."""
    rows = []
    for idx in range(geom.n_layers):
        state = cache.ssm_states[idx][0].float().reshape(-1)
        rows.append((state @ matrix.to(state.device).float()).cpu())
    return torch.stack(rows)


def project_state_masked(
    cache,
    matrix: torch.Tensor,
    head_masks: torch.Tensor,
    geom: Geometry,
) -> torch.Tensor:
    """Project the state with all but the selected heads zeroed. [L, n_features].

    This is what makes the head-stratified test possible: the full projection
    mixes all heads irreversibly, so a stratum must be selected before the
    projection rather than after it.

    Zeroing the unselected heads and projecting with the full matrix is exactly
    equivalent to projecting the selected sub-vector with the matrix rows that
    belong to those heads, because head h occupies rows [h*P*N, (h+1)*P*N) of
    the flattened state. The equivalent-but-obvious implementation, one matrix
    per layer per stratum, needs tens of gigabytes and exhausts an H100; this
    one reuses the single matrix the rest of the project already uses, which
    also makes the strata directly comparable to the unstratified probe.

    head_masks: [n_layers, n_heads] boolean.
    """
    rows = []
    for idx in range(geom.n_layers):
        state = cache.ssm_states[idx][0].float()  # [H, P, N]
        mask = head_masks[idx].to(state.device).to(state.dtype)
        masked = (state * mask[:, None, None]).reshape(-1)
        rows.append((masked @ matrix.to(state.device).float()).cpu())
    return torch.stack(rows)


def snapshot_recurrent(cache, geom: Geometry) -> dict:
    """Clone the recurrent channel so it can be restored exactly."""
    return {
        "ssm": {i: cache.ssm_states[i].clone() for i in range(geom.n_layers)},
        "conv": {i: cache.conv_states[i].clone() for i in range(geom.n_layers)},
    }


def restore_recurrent(cache, snap: dict) -> None:
    for i, state in snap["ssm"].items():
        cache.ssm_states[i] = state
    for i, state in snap["conv"].items():
        cache.conv_states[i] = state


@contextlib.contextmanager
def channel_swapped(
    host,
    donor,
    channel: str,
    attn_ids: tuple[int, ...],
    geom: Geometry,
):
    """Temporarily give `host` one memory channel taken from `donor`.

    channel="recurrent" replaces every Mamba state and conv buffer, leaving the
    attention caches alone. channel="attention" does the opposite. The host
    cache is restored exactly on exit, so one prefill supports many splices.
    Nothing outside the swapped channel is touched.
    """
    if channel not in ("recurrent", "attention"):
        raise ValueError(f"unknown channel {channel!r}")

    if channel == "recurrent":
        saved = snapshot_recurrent(host, geom)
        try:
            for i in range(geom.n_layers):
                host.ssm_states[i] = donor.ssm_states[i].clone()
                host.conv_states[i] = donor.conv_states[i].clone()
            yield host
        finally:
            restore_recurrent(host, saved)
    else:
        saved_k = {i: host.key_cache[i].clone() for i in attn_ids}
        saved_v = {i: host.value_cache[i].clone() for i in attn_ids}
        try:
            for i in attn_ids:
                host.key_cache[i] = donor.key_cache[i].clone()
                host.value_cache[i] = donor.value_cache[i].clone()
            yield host
        finally:
            for i in attn_ids:
                host.key_cache[i] = saved_k[i]
                host.value_cache[i] = saved_v[i]
