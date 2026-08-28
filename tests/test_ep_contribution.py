from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from tools.materialization.ep_contribution import (
    CSV_COLUMNS,
    ExpertContributionValidationError,
    analyze_expert_contributions,
    read_primary_forward_rows,
    write_expert_contribution_outputs,
)
from dmi_megatron_integration.materialization.ep_reconstruction import (
    MoEExecutionKey,
    ReconstructedMoEInvocation,
    ReconstructedSourceDomain,
    SourceTokenCoordinate,
)


def _domain(
    *,
    dense_dp_rank: int,
    weights: list[list[float]],
    outputs: list[list[list[float]]],
    sample_offset: int = 0,
) -> ReconstructedSourceDomain:
    selected_weights = torch.tensor(weights, dtype=torch.float32)
    weighted_outputs = torch.tensor(outputs, dtype=torch.float32)
    token_count, top_k = selected_weights.shape
    return ReconstructedSourceDomain(
        dense_dp_rank=dense_dp_rank,
        token_coordinates=tuple(
            SourceTokenCoordinate(
                dataset_id=10 + dense_dp_rank,
                sample_index=sample_offset + token_index,
                token_index=token_index,
            )
            for token_index in range(token_count)
        ),
        selected_expert_ids=torch.arange(top_k, dtype=torch.int64)
        .repeat(token_count, 1),
        selected_weights=selected_weights,
        weighted_outputs=weighted_outputs,
        combined_output=weighted_outputs.sum(dim=1),
    )


def _invocation(
    *,
    global_batch_id: int,
    microbatch_id: int,
    layer_no: int,
    domains: tuple[ReconstructedSourceDomain, ...],
    attempt_id: int = 2,
    direction: str = "fwd",
) -> ReconstructedMoEInvocation:
    return ReconstructedMoEInvocation(
        key=MoEExecutionKey(
            model_id="olmoe-real",
            phase="train",
            global_batch_id=global_batch_id,
            attempt_id=attempt_id,
            microbatch_id=microbatch_id,
            layer_no=layer_no,
            direction=direction,
        ),
        source_domains=domains,
    )


def test_metrics_group_domains_and_exclude_ambiguous_ties() -> None:
    first = _invocation(
        global_batch_id=7,
        microbatch_id=0,
        layer_no=5,
        domains=(
            _domain(
                dense_dp_rank=0,
                weights=[[0.9, 0.1], [0.5, 0.5]],
                outputs=[
                    [[1.0, 0.0], [0.0, 2.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
            ),
            _domain(
                dense_dp_rank=1,
                weights=[[0.2, 0.8]],
                outputs=[[[0.0, 0.0], [3.0, 0.0]]],
                sample_offset=20,
            ),
        ),
    )
    second = _invocation(
        global_batch_id=7,
        microbatch_id=1,
        layer_no=5,
        domains=(
            _domain(
                dense_dp_rank=0,
                weights=[[0.5, 0.5]],
                outputs=[[[2.0, 0.0], [2.0, 0.0]]],
                sample_offset=30,
            ),
        ),
    )

    rows, validation = analyze_expert_contributions(
        (first, second),
        expected_model_id="olmoe-real",
        expected_step_count=1,
        expected_layer_ids=(5,),
        expected_microbatches_per_step=2,
        source_row_count_by_act={
            "router_topk_expert_ids": 8,
            "router_topk_weights": 8,
            "moe_inverse_map": 4,
            "moe_packed_weighted_output": 4,
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.global_moe_layer == 5
    assert row.valid_token_count == 4
    # Two tokens have tied maximum output norms and two have tied maximum
    # router weights.  Neither metric assigns those tokens an arbitrary slot.
    assert row.magnitude_tie_token_count == 2
    assert row.router_rank_dominance_counts == (1, 1)
    assert row.top_weight_tie_token_count == 2
    assert row.top_route_alignment_q1 == pytest.approx(
        0.75 / math.sqrt(5) + 0.25
    )
    assert row.top_route_alignment_median == pytest.approx(
        (1 / math.sqrt(5) + 1.0) / 2.0
    )
    assert row.top_route_alignment_q3 == pytest.approx(
        0.25 / math.sqrt(5) + 0.75
    )
    assert validation.status == "passed"
    assert validation.primary_forward_invocation_count == 2
    assert validation.source_domain_count == 3
    assert validation.output_row_count == 1
    assert validation.valid_token_count == 4
    assert validation.router_rank_1_dominance_count == 1
    assert validation.magnitude_tie_token_count == 2
    assert validation.top_weight_tie_token_count == 2
    assert validation.expected_step_count == 1
    assert validation.observed_step_count == 1
    assert validation.expected_global_batch_ids == (7,)
    assert validation.observed_global_batch_ids == (7,)
    assert validation.expected_global_moe_layer_ids == (5,)
    assert validation.observed_global_moe_layer_ids == (5,)
    assert validation.expected_layer_count == 1
    assert validation.observed_layer_count == 1
    assert validation.expected_microbatches_per_step == 2
    assert validation.observed_microbatch_invocation_counts == ((7, 5, 2),)
    assert validation.source_row_count_by_act["router_topk_expert_ids"] == 8


def test_router_rank_is_derived_from_weights_not_reconstructed_slot_order() -> None:
    invocation = _invocation(
        global_batch_id=1,
        microbatch_id=0,
        layer_no=0,
        domains=(
            _domain(
                dense_dp_rank=0,
                weights=[[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]],
                outputs=[
                    [[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]],
                    [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]],
                ],
            ),
        ),
    )

    rows, _ = analyze_expert_contributions(
        (invocation,),
        expected_model_id="olmoe-real",
        expected_step_count=1,
        expected_layer_ids=(0,),
        expected_microbatches_per_step=1,
    )

    # The magnitude winners are reconstructed slots 0 and 1, but those slots
    # have router-weight ranks 2 and 3 for their respective tokens.
    assert rows[0].magnitude_tie_token_count == 0
    assert rows[0].router_rank_dominance_counts == (0, 1, 1)
    assert rows[0].top_weight_tie_token_count == 0


@pytest.mark.parametrize(
    ("outputs", "match"),
    (
        ([[[0.0, 0.0], [1.0, 0.0]]], "zero-norm"),
        ([[[1.0, 0.0], [-1.0, 0.0]]], "zero-norm"),
        ([[[float("nan"), 0.0], [1.0, 0.0]]], "non-finite"),
    ),
)
def test_metrics_reject_invalid_cosine_operands(
    outputs: list[list[list[float]]], match: str
) -> None:
    invocation = _invocation(
        global_batch_id=1,
        microbatch_id=0,
        layer_no=0,
        domains=(
            _domain(
                dense_dp_rank=0,
                weights=[[0.9, 0.1]],
                outputs=outputs,
            ),
        ),
    )
    with pytest.raises(ValueError, match=match):
        analyze_expert_contributions(
            (invocation,),
            expected_model_id="olmoe-real",
            expected_step_count=1,
            expected_layer_ids=(0,),
            expected_microbatches_per_step=1,
        )


def test_metrics_reject_nonforward_and_multiple_attempts() -> None:
    domain = _domain(
        dense_dp_rank=0,
        weights=[[0.8, 0.2]],
        outputs=[[[1.0, 0.0], [0.0, 1.0]]],
    )
    backward = _invocation(
        global_batch_id=1,
        microbatch_id=0,
        layer_no=0,
        domains=(domain,),
        direction="bwd",
    )
    with pytest.raises(ValueError, match="Only forward"):
        analyze_expert_contributions(
            (backward,),
            expected_model_id="olmoe-real",
            expected_step_count=1,
            expected_layer_ids=(0,),
            expected_microbatches_per_step=1,
        )

    invocations = (
        _invocation(
            global_batch_id=2,
            microbatch_id=0,
            layer_no=0,
            domains=(domain,),
            attempt_id=0,
        ),
        _invocation(
            global_batch_id=2,
            microbatch_id=1,
            layer_no=0,
            domains=(domain,),
            attempt_id=1,
        ),
    )
    with pytest.raises(ValueError, match="More than one attempt"):
        analyze_expert_contributions(
            invocations,
            expected_model_id="olmoe-real",
            expected_step_count=1,
            expected_layer_ids=(0,),
            expected_microbatches_per_step=2,
        )


def test_fixed_run_requires_complete_10_step_by_16_layer_grid() -> None:
    domain = _domain(
        dense_dp_rank=0,
        weights=[[0.8, 0.2]],
        outputs=[[[2.0, 0.0], [0.0, 1.0]]],
    )
    layer_ids = tuple(range(1, 32, 2))
    invocations = tuple(
        _invocation(
            global_batch_id=batch_id,
            microbatch_id=0,
            layer_no=layer_id,
            domains=(domain,),
        )
        for batch_id in range(101, 111)
        for layer_id in layer_ids
    )

    rows, validation = analyze_expert_contributions(
        invocations,
        expected_model_id="olmoe-real",
        expected_step_count=10,
        expected_layer_ids=layer_ids,
        expected_microbatches_per_step=1,
    )

    assert len(rows) == 16
    assert all(row.valid_token_count == 10 for row in rows)
    assert all(row.magnitude_tie_token_count == 0 for row in rows)
    assert all(row.router_rank_dominance_counts == (10, 0) for row in rows)
    assert all(row.top_weight_tie_token_count == 0 for row in rows)
    assert validation.status == "passed"
    assert validation.expected_global_batch_ids == tuple(range(101, 111))
    assert validation.observed_global_batch_ids == tuple(range(101, 111))
    assert validation.expected_global_moe_layer_ids == layer_ids
    assert validation.observed_global_moe_layer_ids == layer_ids
    assert validation.expected_layer_count == 16
    assert validation.observed_layer_count == 16
    assert validation.primary_forward_invocation_count == 10 * 16
    assert validation.output_row_count == 16
    assert validation.valid_token_count == 10 * 16
    assert validation.router_rank_1_dominance_count == 10 * 16
    assert validation.magnitude_tie_token_count == 0
    assert validation.top_weight_tie_token_count == 0
    assert len(validation.observed_microbatch_invocation_counts) == 10 * 16
    assert all(
        count == 1
        for _, _, count in validation.observed_microbatch_invocation_counts
    )


def test_grid_validation_rejects_missing_step_layer_and_wrong_microbatch_count() -> None:
    domain = _domain(
        dense_dp_rank=0,
        weights=[[0.8, 0.2]],
        outputs=[[[1.0, 0.0], [0.0, 1.0]]],
    )
    missing_cell = tuple(
        _invocation(
            global_batch_id=batch_id,
            microbatch_id=0,
            layer_no=layer_id,
            domains=(domain,),
        )
        for batch_id in (4, 5)
        for layer_id in (1, 3)
        if (batch_id, layer_id) != (5, 3)
    )
    with pytest.raises(
        ExpertContributionValidationError,
        match="Incomplete batch-by-layer grid",
    ) as missing_error:
        analyze_expert_contributions(
            missing_cell,
            expected_model_id="olmoe-real",
            expected_step_count=2,
            expected_layer_ids=(1, 3),
            expected_microbatches_per_step=1,
        )
    assert missing_error.value.validation.status == "failed"
    assert missing_error.value.validation.expected_global_batch_ids == (4, 5)
    assert missing_error.value.validation.observed_global_batch_ids == (4, 5)
    assert missing_error.value.validation.expected_global_moe_layer_ids == (1, 3)
    assert missing_error.value.validation.observed_global_moe_layer_ids == (1, 3)
    assert missing_error.value.validation.observed_microbatch_invocation_counts == (
        (4, 1, 1),
        (4, 3, 1),
        (5, 1, 1),
    )

    wrong_microbatch_count = (
        _invocation(
            global_batch_id=4,
            microbatch_id=0,
            layer_no=1,
            domains=(domain,),
        ),
        _invocation(
            global_batch_id=4,
            microbatch_id=1,
            layer_no=1,
            domains=(domain,),
        ),
    )
    with pytest.raises(
        ExpertContributionValidationError, match="microbatch invocation counts"
    ) as count_error:
        analyze_expert_contributions(
            wrong_microbatch_count,
            expected_model_id="olmoe-real",
            expected_step_count=1,
            expected_layer_ids=(1,),
            expected_microbatches_per_step=1,
        )
    assert count_error.value.validation.status == "failed"
    assert count_error.value.validation.observed_microbatch_invocation_counts == (
        (4, 1, 2),
    )


def test_grid_validation_rejects_nonconsecutive_or_short_step_ids() -> None:
    domain = _domain(
        dense_dp_rank=0,
        weights=[[0.8, 0.2]],
        outputs=[[[1.0, 0.0], [0.0, 1.0]]],
    )
    for batch_ids in ((10, 12), (10,)):
        invocations = tuple(
            _invocation(
                global_batch_id=batch_id,
                microbatch_id=0,
                layer_no=1,
                domains=(domain,),
            )
            for batch_id in batch_ids
        )
        with pytest.raises(ValueError, match="required consecutive range"):
            analyze_expert_contributions(
                invocations,
                expected_model_id="olmoe-real",
                expected_step_count=2,
                expected_layer_ids=(1,),
                expected_microbatches_per_step=1,
            )


def test_reader_requests_only_accepted_primary_forward_rows_and_preserves_rows() -> None:
    tensors = {
        act_name: torch.tensor([index], dtype=torch.int64)
        for index, act_name in enumerate(
            (
                "router_topk_expert_ids",
                "router_topk_weights",
                "moe_inverse_map",
                "moe_packed_weighted_output",
            )
        )
    }

    class FakeReader:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict[str, bool]]] = []

        def training_prefix_get(self, prefix_key: tuple, **kwargs):
            self.calls.append((prefix_key, kwargs))
            act_name = prefix_key[1]
            return [(('source', act_name), tensors[act_name])]

    reader = FakeReader()
    rows = read_primary_forward_rows(reader, model_id="olmoe-real")

    assert [call[0] for call in reader.calls] == [
        ("olmoe-real", act_name, "fwd", "train") for act_name in tensors
    ]
    assert all(
        options
        == {
            "return_full_key_tuple": True,
            "include_all_attempts": False,
            "include_all_invocations": False,
        }
        for _, options in reader.calls
    )
    for act_name, tensor in tensors.items():
        assert rows[act_name][0][1] is tensor


def test_writes_exact_csv_columns_and_validation_json(tmp_path: Path) -> None:
    invocation = _invocation(
        global_batch_id=9,
        microbatch_id=0,
        layer_no=3,
        domains=(
            _domain(
                dense_dp_rank=0,
                weights=[[0.8, 0.2]],
                outputs=[[[1.0, 0.0], [0.0, 0.5]]],
            ),
        ),
    )
    rows, validation = analyze_expert_contributions(
        (invocation,),
        expected_model_id="olmoe-real",
        expected_step_count=1,
        expected_layer_ids=(3,),
        expected_microbatches_per_step=1,
    )
    csv_path = tmp_path / "nested" / "expert_contribution.csv"
    json_path = tmp_path / "nested" / "validation.json"

    write_expert_contribution_outputs(
        rows,
        validation,
        csv_path=csv_path,
        validation_json_path=json_path,
    )

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert tuple(csv_rows[0]) == CSV_COLUMNS
    assert csv_rows[0]["global_moe_layer"] == "3"
    assert csv_rows[0]["valid_token_count"] == "1"
    assert csv_rows[0]["magnitude_tie_token_count"] == "0"
    assert json.loads(csv_rows[0]["router_rank_dominance_counts"]) == [1, 0]
    assert csv_rows[0]["top_weight_tie_token_count"] == "0"
    assert float(csv_rows[0]["top_route_alignment_q1"]) == pytest.approx(
        2 / math.sqrt(5)
    )
    assert float(csv_rows[0]["top_route_alignment_median"]) == pytest.approx(
        2 / math.sqrt(5)
    )
    assert float(csv_rows[0]["top_route_alignment_q3"]) == pytest.approx(
        2 / math.sqrt(5)
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["status"] == "passed"
    assert payload["output_row_count"] == 1
    assert payload["valid_token_count"] == 1
    assert payload["expected_global_batch_ids"] == [9]
    assert payload["observed_global_batch_ids"] == [9]
    assert payload["expected_global_moe_layer_ids"] == [3]
    assert payload["observed_global_moe_layer_ids"] == [3]
    assert payload["observed_microbatch_invocation_counts"] == [[9, 3, 1]]
