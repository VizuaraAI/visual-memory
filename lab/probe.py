"""Linear decoding of the recurrent state, with the controls that matter.

The probe is ridge regression onto one-hot targets, solved in dual form
because the projected state has far more dimensions (81 layers x 128) than we
will ever have trials. Cross-validation is grouped by block so that the eight
trials sharing a candidate list never straddle a fold, which would let the
probe learn the block rather than the picture.

Two controls accompany every number:
  pre-image     the same probe on states captured before any image was shown
  permutation   the same probe on shuffled labels
Both must land at chance. If either does not, the reported accuracy is an
artifact of the design and not a fact about the model.
"""

from __future__ import annotations

import math

import torch


def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    out = torch.zeros(y.shape[0], n_classes, dtype=torch.float64)
    out[torch.arange(y.shape[0]), y] = 1.0
    return out


def ridge_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    n_classes: int,
    alpha: float,
) -> torch.Tensor:
    """Dual-form ridge. Returns class scores for the test rows."""
    xtr = x_train.double()
    xte = x_test.double()
    mean = xtr.mean(dim=0, keepdim=True)
    xtr = xtr - mean
    xte = xte - mean

    targets = _one_hot(y_train, n_classes)
    targets = targets - targets.mean(dim=0, keepdim=True)

    gram = xtr @ xtr.T
    gram = gram + alpha * torch.eye(gram.shape[0], dtype=torch.float64)
    weights = torch.linalg.solve(gram, targets)
    return (xte @ xtr.T) @ weights


def grouped_folds(groups: list[str], n_folds: int, seed: int = 0) -> list[torch.Tensor]:
    """Assign whole groups to folds so no group is split across folds."""
    unique = sorted(set(groups))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique), generator=generator).tolist()
    assignment = {unique[g]: i % n_folds for i, g in enumerate(order)}
    index = torch.tensor([assignment[g] for g in groups])
    return [(index == k).nonzero(as_tuple=True)[0] for k in range(n_folds)]


def cross_validated_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
    n_classes: int,
    alphas: tuple[float, ...] = (1e1, 1e2, 1e3, 1e4, 1e5),
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Grouped k-fold accuracy, with alpha chosen on the training folds only."""
    folds = grouped_folds(groups, n_folds, seed=seed)
    correct = 0
    total = 0
    chosen_alphas: list[float] = []

    for held_out in folds:
        mask = torch.ones(features.shape[0], dtype=torch.bool)
        mask[held_out] = False
        x_train, y_train = features[mask], labels[mask]
        x_test, y_test = features[held_out], labels[held_out]
        if x_test.numel() == 0 or x_train.numel() == 0:
            continue

        inner_groups = [g for g, keep in zip(groups, mask.tolist()) if keep]
        inner = grouped_folds(inner_groups, min(4, n_folds), seed=seed + 1)
        best_alpha, best_score = alphas[0], -1.0
        for alpha in alphas:
            hits, seen = 0, 0
            for inner_test in inner:
                inner_mask = torch.ones(x_train.shape[0], dtype=torch.bool)
                inner_mask[inner_test] = False
                if inner_test.numel() == 0 or inner_mask.sum() == 0:
                    continue
                scores = ridge_predict(
                    x_train[inner_mask], y_train[inner_mask],
                    x_train[inner_test], n_classes, alpha,
                )
                hits += int((scores.argmax(dim=1) == y_train[inner_test]).sum())
                seen += int(inner_test.numel())
            score = hits / max(seen, 1)
            if score > best_score:
                best_alpha, best_score = alpha, score
        chosen_alphas.append(best_alpha)

        scores = ridge_predict(x_train, y_train, x_test, n_classes, best_alpha)
        correct += int((scores.argmax(dim=1) == y_test).sum())
        total += int(y_test.numel())

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "chance": 1.0 / n_classes,
        "alphas": chosen_alphas,
        "p_value": binomial_tail(correct, total, 1.0 / n_classes),
    }


def permuted_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
    n_classes: int,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """The same probe with labels shuffled, as an overfitting check."""
    accuracies = []
    for r in range(n_repeats):
        generator = torch.Generator().manual_seed(seed + r)
        shuffled = labels[torch.randperm(labels.shape[0], generator=generator)]
        result = cross_validated_accuracy(
            features, shuffled, groups, n_classes, seed=seed + r
        )
        accuracies.append(result["accuracy"])
    tensor = torch.tensor(accuracies)
    return {
        "mean": float(tensor.mean()),
        "max": float(tensor.max()),
        "runs": accuracies,
    }


def binomial_tail(successes: int, trials: int, probability: float) -> float:
    """One-sided P(X >= successes) under the null, exact."""
    if trials == 0:
        return 1.0
    total = 0.0
    for k in range(successes, trials + 1):
        total += math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * math.log(probability)
            + (trials - k) * math.log1p(-probability)
        )
    return min(1.0, total)


def layerwise_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
    n_classes: int,
    n_layers: int,
    seed: int = 0,
) -> list[dict]:
    """Probe each layer separately, to see where in depth the picture sits."""
    per_layer = features.reshape(features.shape[0], n_layers, -1)
    out = []
    for layer in range(n_layers):
        result = cross_validated_accuracy(
            per_layer[:, layer, :], labels, groups, n_classes, seed=seed
        )
        out.append({"layer": layer, "accuracy": result["accuracy"], "p_value": result["p_value"]})
    return out
