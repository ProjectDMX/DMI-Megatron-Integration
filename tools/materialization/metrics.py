"""Pure metric computations used by training payload materialization."""

from __future__ import annotations

from typing import Sequence

import torch

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
except ImportError:
    RapidLevenshtein = None


def coefficient_of_variation(values: torch.Tensor) -> float:
    values_f = values.to(torch.float64)
    mean = float(values_f.mean().item()) if values_f.numel() else 0.0
    if mean == 0.0:
        return 0.0
    return float(values_f.std(unbiased=False).item() / mean)


def pathway_tokens(probs: torch.Tensor, threshold: float) -> tuple[int, ...]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("pathway threshold must be in (0, 1]")
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    cumulative = torch.cumsum(sorted_probs, dim=0)
    count = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(threshold, dtype=cumulative.dtype),
        ).item()
    ) + 1
    count = max(1, min(count, int(probs.numel())))
    return tuple(int(value) for value in order[:count].tolist())


def paper_pathway_sequence(layer_tokens: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Encode expert choices while keeping layer separators distinct."""

    sequence: list[int] = []
    separator_base = 1_000_000
    for layer_index, tokens in enumerate(layer_tokens):
        if layer_index > 0:
            sequence.append(separator_base + layer_index)
        sequence.extend(tokens)
    return tuple(sequence)


def levenshtein(a: str | tuple[int, ...], b: str | tuple[int, ...]) -> int:
    if RapidLevenshtein is not None:
        return int(RapidLevenshtein.distance(a, b))
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for left_index, left_value in enumerate(a, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(b, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1]
                    + (0 if left_value == right_value else 1),
                )
            )
        previous = current
    return previous[-1]


def pathway_consistency(
    data: torch.Tensor,
    layers: list[int],
    router_weights: dict[int, torch.Tensor],
    *,
    eps: float,
) -> torch.Tensor:
    """Compute the paper's mapped adjacent-layer routing-transition metric."""

    missing = [layer for layer in layers if layer not in router_weights]
    if missing:
        raise ValueError(f"Missing router weights for layers: {missing}")
    if len(layers) < 2:
        raise ValueError("Pathway consistency requires at least two layers")
    mapped_routing_vectors = []
    for layer_index, layer_no in enumerate(layers):
        weights = router_weights[layer_no].float()
        if weights.shape[0] != data.shape[-1]:
            raise ValueError(
                f"Router weight expert dim mismatch for layer {layer_no}: "
                f"weights={tuple(weights.shape)}, data={tuple(data.shape)}"
            )
        mapped_routing_vectors.append(data[:, layer_index, :].float() @ weights)
    stacked = torch.stack(mapped_routing_vectors, dim=1)
    cosine = torch.nn.functional.cosine_similarity(
        stacked[:, :-1, :],
        stacked[:, 1:, :],
        dim=-1,
        eps=eps,
    )
    denominator = cosine.max(dim=1, keepdim=True).values + eps
    return 1.0 - (cosine / denominator).mean(dim=1)


def pairwise_distance_sum_blockwise(
    pathways: Sequence[tuple[int, ...]],
    *,
    block_size: int = 64,
) -> tuple[float, int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total = 0.0
    pair_count = 0
    size = len(pathways)
    for left_start in range(0, size, block_size):
        left_end = min(size, left_start + block_size)
        for right_start in range(left_start, size, block_size):
            right_end = min(size, right_start + block_size)
            for left_index in range(left_start, left_end):
                first_right = max(right_start, left_index + 1)
                for right_index in range(first_right, right_end):
                    total += float(levenshtein(pathways[left_index], pathways[right_index]))
                    pair_count += 1
    expected_pairs = size * (size - 1) // 2
    if pair_count != expected_pairs:
        raise RuntimeError(
            f"Blockwise pair traversal produced {pair_count} pairs, "
            f"expected {expected_pairs}"
        )
    return total, pair_count


def materialize_pathway_windows(
    *,
    model_id: str,
    run_name: str,
    phase: str,
    completed_iteration_ids: Sequence[int],
    consistency_eps: float,
    pathway_threshold: float,
    consistency_by_dataset_iteration: dict[tuple[int, int], list[float]],
    pathways_by_dataset_iteration: dict[tuple[int, int], list[tuple[int, ...]]],
    window_sizes: Sequence[int],
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    iteration_ids = [int(value) for value in completed_iteration_ids]
    if not iteration_ids:
        raise ValueError("completed_iteration_ids must not be empty")
    if iteration_ids != sorted(set(iteration_ids)):
        raise ValueError("completed_iteration_ids must be sorted and unique")
    if set(consistency_by_dataset_iteration) != set(pathways_by_dataset_iteration):
        raise RuntimeError("Pathway consistency and sequence groups do not match")
    iteration_set = set(iteration_ids)
    for dataset_id, iteration_id in consistency_by_dataset_iteration:
        if dataset_id < -1:
            raise RuntimeError(f"Invalid pathway dataset_id: {dataset_id}")
        if iteration_id not in iteration_set:
            raise RuntimeError(
                f"Pathway data references incomplete iteration {iteration_id}"
            )
    represented_iterations = {
        iteration_id
        for _dataset_id, iteration_id in consistency_by_dataset_iteration
    }
    if represented_iterations != iteration_set:
        raise RuntimeError("Pathway sample coverage is incomplete")

    consistency_rows = []
    distance_rows = []
    for window_size in window_sizes:
        if window_size <= 0:
            raise ValueError(f"Invalid pathway window size {window_size}")
        for end_index, window_end in enumerate(iteration_ids):
            start_index = max(0, end_index - window_size + 1)
            window_iterations = iteration_ids[start_index : end_index + 1]
            window_iteration_set = set(window_iterations)
            dataset_ids = sorted(
                {
                    dataset_id
                    for dataset_id, iteration_id in consistency_by_dataset_iteration
                    if iteration_id in window_iteration_set
                }
            )
            for dataset_id in dataset_ids:
                sample_scores: list[float] = []
                pathways: list[tuple[int, ...]] = []
                for iteration_id in window_iterations:
                    key = (dataset_id, iteration_id)
                    sample_scores.extend(
                        consistency_by_dataset_iteration.get(key, ())
                    )
                    pathways.extend(pathways_by_dataset_iteration.get(key, ()))
                if len(sample_scores) != len(pathways) or not sample_scores:
                    raise RuntimeError(
                        f"Pathway window {window_iterations[0]}..{window_end} "
                        f"for dataset_id={dataset_id} has inconsistent samples"
                    )
                consistency_sum = float(sum(sample_scores))
                distance_sum, pair_count = pairwise_distance_sum_blockwise(pathways)
                consistency_rows.append(
                    (
                        model_id,
                        run_name,
                        phase,
                        window_end,
                        dataset_id,
                        window_size,
                        len(window_iterations),
                        window_iterations[0],
                        window_end,
                        len(sample_scores),
                        consistency_sum,
                        consistency_sum / len(sample_scores),
                        float(consistency_eps),
                    )
                )
                distance_rows.append(
                    (
                        model_id,
                        run_name,
                        phase,
                        window_end,
                        dataset_id,
                        window_size,
                        len(window_iterations),
                        window_iterations[0],
                        window_end,
                        len(pathways),
                        pair_count,
                        distance_sum,
                        None if pair_count == 0 else distance_sum / pair_count,
                        float(pathway_threshold),
                    )
                )
    return consistency_rows, distance_rows
