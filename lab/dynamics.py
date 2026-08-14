"""Decay dynamics of the recurrent channel, read from the model's parameters.

E5 asks whether the measured forgetting horizon can be predicted from
quantities that were never fitted to it. In Mamba2 the state update is

    S_t = S_{t-1} exp(dt_t A) + dt_t (x_t B_t^T)

so anything already in the state at time zero is multiplied, after k further
tokens, by

    R(k) = exp(A * sum_{t=1..k} dt_t)

with A strictly negative. Both ingredients are available without any fitting.

Verified against the installed Zamba2 source rather than assumed:

  * `A = -torch.exp(self.A_log.float())`, shape [num_heads]. So the decay rate
    is |A_h| = exp(A_log_h) and is a per-head scalar, shared across head_dim
    and state_dim (the forward pass expands it over both).
  * `dt = softplus(dt_raw + dt_bias)` followed by `clamp(dt, time_step_min)`.
    The upper clamp is commented out in the source, so dt has a floor and no
    ceiling. `dt_bias` is shape [num_heads].
  * Zamba2-VL-7B config: time_step_min = 0.001, time_step_max = 0.1 (unused),
    time_step_limit = None.

`dt_raw` is the final `n_heads` channels of the in_proj output, which
`vtelemetry.capture_dt` records at every position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .vtelemetry import Geometry, decoder_layers, mamba_mixer

LN2 = 0.6931471805599453


@dataclass(frozen=True)
class DecayParameters:
    """Per-layer, per-head decay constants taken straight from the weights.

    abs_a:   [n_layers, n_heads]  the magnitude |A_h| = exp(A_log_h)
    dt_bias: [n_layers, n_heads]  the additive bias inside the softplus
    """

    abs_a: torch.Tensor
    dt_bias: torch.Tensor
    time_step_min: float

    @property
    def n_layers(self) -> int:
        return int(self.abs_a.shape[0])


def decay_parameters(model, geom: Geometry) -> DecayParameters:
    """Extract |A| and dt_bias from every mixer. No forward pass required."""
    layers = decoder_layers(model, geom)
    cfg = getattr(model.config, "text_config", model.config)
    time_step_min = float(getattr(cfg, "time_step_min", 0.001))

    abs_a, biases = [], []
    for idx, layer in enumerate(layers):
        mixer = mamba_mixer(layer)
        if not hasattr(mixer, "A_log") or not hasattr(mixer, "dt_bias"):
            raise RuntimeError(f"layer {idx} mixer exposes no A_log/dt_bias")
        a_log = mixer.A_log.detach().float().cpu()
        bias = mixer.dt_bias.detach().float().cpu()
        if a_log.shape != (geom.n_heads,) or bias.shape != (geom.n_heads,):
            raise RuntimeError(
                f"layer {idx}: expected [{geom.n_heads}] for A_log and dt_bias, "
                f"got {tuple(a_log.shape)} and {tuple(bias.shape)}"
            )
        abs_a.append(torch.exp(a_log))
        biases.append(bias)

    return DecayParameters(
        abs_a=torch.stack(abs_a),
        dt_bias=torch.stack(biases),
        time_step_min=time_step_min,
    )


def effective_dt(dt_raw: torch.Tensor, params: DecayParameters) -> torch.Tensor:
    """Apply the model's own gate transform to raw in_proj dt channels.

    dt_raw:  [n_layers, seq, n_heads]
    returns: [n_layers, seq, n_heads], the dt actually used by the recurrence.
    """
    if dt_raw.dim() != 3:
        raise ValueError(f"expected [layers, seq, heads], got {tuple(dt_raw.shape)}")
    biased = dt_raw.float() + params.dt_bias.unsqueeze(1)
    return torch.clamp(torch.nn.functional.softplus(biased), min=params.time_step_min)


def half_life_tokens(params: DecayParameters, mean_dt: torch.Tensor) -> torch.Tensor:
    """Predicted half-life in tokens, per layer and head.

    mean_dt: [n_layers, n_heads], the mean effective gate over the region of
             interest. Half-life is ln2 / (|A| * mean_dt), the token count at
             which the retention factor reaches one half.
    """
    if mean_dt.shape != params.abs_a.shape:
        raise ValueError(
            f"mean_dt {tuple(mean_dt.shape)} does not match "
            f"abs_a {tuple(params.abs_a.shape)}"
        )
    return LN2 / (params.abs_a * mean_dt)


def retention_curve(
    params: DecayParameters,
    dt_sequence: torch.Tensor,
    distances: tuple[int, ...],
) -> torch.Tensor:
    """Predicted retention at each distance, per layer and head.

    dt_sequence: [n_layers, seq, n_heads] effective dt over the filler region.
    returns:     [len(distances), n_layers, n_heads] in [0, 1].

    Uses the true cumulative sum of dt rather than a mean times k, so a filler
    whose gate varies along its length is handled exactly.
    """
    cumulative = torch.cumsum(dt_sequence.float(), dim=1)
    seq_len = cumulative.shape[1]
    rows = []
    for k in distances:
        if k == 0:
            rows.append(torch.ones_like(params.abs_a))
            continue
        if k > seq_len:
            raise ValueError(f"distance {k} exceeds captured length {seq_len}")
        rows.append(torch.exp(-params.abs_a * cumulative[:, k - 1, :]))
    return torch.stack(rows)


def head_tercile_masks(half_lives: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split (layer, head) pairs into fast, middle and slow thirds by half-life.

    half_lives: [n_layers, n_heads]
    returns:    {"fast"|"middle"|"slow": [n_layers, n_heads] boolean}

    The split is global rather than within-layer, because the physical claim is
    about absolute forgetting horizon in tokens, not about a head being fast
    relative to its neighbours. A consequence is that layers contribute unequal
    numbers of heads to each stratum, which the caller should report.
    """
    flat = half_lives.reshape(-1)
    lo, hi = torch.quantile(flat, torch.tensor([1 / 3, 2 / 3]))
    return {
        "fast": (flat <= lo).reshape(half_lives.shape),
        "middle": ((flat > lo) & (flat < hi)).reshape(half_lives.shape),
        "slow": (flat >= hi).reshape(half_lives.shape),
    }


def head_quantile_masks(half_lives: torch.Tensor, n_bins: int) -> dict[str, torch.Tensor]:
    """Split (layer, head) pairs into `n_bins` equal groups by half-life.

    Generalises the tercile split. Ten bins turn the dependence of eviction
    sensitivity on decay rate from three points into ten, which is the
    difference between a suggestive ordering and a measured relationship.
    Bins are named by index, slowest last, so "bin00" is the fastest.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    flat = half_lives.reshape(-1)
    edges = torch.quantile(
        flat, torch.linspace(0, 1, n_bins + 1)[1:-1]
    )
    masks = {}
    for index in range(n_bins):
        lower = edges[index - 1] if index > 0 else None
        upper = edges[index] if index < n_bins - 1 else None
        selected = torch.ones_like(flat, dtype=torch.bool)
        if lower is not None:
            selected &= flat > lower
        if upper is not None:
            selected &= flat <= upper
        masks[f"bin{index:02d}"] = selected.reshape(half_lives.shape)
    return masks


def stratum_summary(half_lives: torch.Tensor, mask: torch.Tensor) -> dict:
    """Predicted half-life statistics for one stratum, and its layer coverage."""
    values = half_lives[mask]
    per_layer = mask.sum(dim=1)
    return {
        "n_heads": int(mask.sum()),
        "median_half_life": float(values.median()),
        "min_half_life": float(values.min()),
        "max_half_life": float(values.max()),
        "layers_with_no_heads": int((per_layer == 0).sum()),
        "heads_per_layer_min": int(per_layer.min()),
        "heads_per_layer_max": int(per_layer.max()),
    }


def summarise(half_lives: torch.Tensor, boundaries: tuple[int, ...] = (32, 512)) -> dict:
    """Distribution summary of predicted half-lives, for the paper's table."""
    flat = half_lives.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    lo, hi = boundaries
    return {
        "n_heads_total": int(flat.numel()),
        "n_finite": int(finite.numel()),
        "median": float(finite.median()),
        "quantiles": {
            f"q{int(q * 100):02d}": float(torch.quantile(finite, q)) for q in quantiles
        },
        f"fraction_below_{lo}": float((finite < lo).float().mean()),
        f"fraction_{lo}_to_{hi}": float(
            ((finite >= lo) & (finite < hi)).float().mean()
        ),
        f"fraction_above_{hi}": float((finite >= hi).float().mean()),
    }


def gain_matched_masks(
    abs_a,
    mean_dt,
    band: tuple[float, float] = (30.0, 70.0),
    split: float = 25.0,
):
    """Head masks that vary the decay weight while holding the write gain fixed.

    The stratification used elsewhere splits heads by half-life, which is
    ln2/(|A| dt). Because dt scales the write term as well as the decay term,
    that split is also a split by write amplitude: the fast third writes about
    seven times harder than the slow third. Any result that follows could then
    be explained by signal amplitude rather than by retention, which is the most
    serious objection this work faces.

    This isolates the two. Keep only heads whose mean gate falls inside a narrow
    percentile band of dt, so write amplitude is matched by construction, and
    split what remains by |A| alone. On Zamba2-VL-7B the middle 20% band holds
    1,814 heads whose |A| still spans three orders of magnitude, giving strata
    with median half-lives near 2.5 and 15.6 tokens at equal write gain.

    Returns (fast_mask, slow_mask, report), each mask shaped like abs_a.
    """
    import numpy as np

    a = np.asarray(abs_a, dtype=np.float64)
    dt = np.asarray(mean_dt, dtype=np.float64)
    lo, hi = np.percentile(dt, band)
    in_band = (dt >= lo) & (dt <= hi)

    a_in = a[in_band]
    q_low, q_high = np.percentile(a_in, [split, 100.0 - split])
    fast = in_band & (a >= q_high)   # large |A| decays fastest
    slow = in_band & (a <= q_low)

    half = np.log(2.0) / (a * dt)
    report = {
        "n_in_band": int(in_band.sum()),
        "n_fast": int(fast.sum()),
        "n_slow": int(slow.sum()),
        "dt_band": [float(lo), float(hi)],
        "median_dt_fast": float(np.median(dt[fast])),
        "median_dt_slow": float(np.median(dt[slow])),
        "median_half_life_fast": float(np.median(half[fast])),
        "median_half_life_slow": float(np.median(half[slow])),
    }
    return fast, slow, report
