from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from dmi.api.v1 import HookPointV1, RecordType, TransportType

from dmi_megatron_integration.adapter import MegatronHookBinding
from dmi_megatron_integration.hooks.selection import parse_hook_selection
from dmi_megatron_integration.hooks.specs import (
    DimSpec,
    HookInputLayout,
    HookPhase,
    MegatronMetadataField,
    MegatronHookSpec,
    MegatronOutputSpec,
    ShardPolicy,
)
from dmi_megatron_integration.metadata_context import (
    DMIMetadataContext,
    LocalMetadataPropagator,
)
from dmi_megatron_integration.schedule_runtime import (
    MegatronScheduleRuntime,
    get_active_megatron_schedule_runtime,
    set_active_megatron_schedule_runtime,
)
from dmi_megatron_integration.records.format import required_record_metadata_fields
from dmi_megatron_integration.startup import (
    MegatronDMIConfig,
    MegatronRankContext,
    _MetadataRequirementReport,
    _active_hooks_for_rank,
    _gather_metadata_requirement_reports,
    _local_metadata_requirement_report,
    _make_hook,
    _metadata_field_specs_from_requirements,
    _megatron_hook_spec,
    _install_moe_inverse_map_hooks,
    _vocab_logits_dtype,
    _apply_recompute_hook_policy,
    _resolve_dataset_provenance_modes,
    _resolve_metadata_requirements,
    _router_logits_dtype,
    resolve_megatron_dmi_config,
    resolve_model_id,
    setup_megatron_dmi,
)
from dmi_megatron_integration.topology.ep_topology_manifest import (
    load_ep_topology_manifest,
)


CONSTANT_PROVENANCE = (
    "train=constant-zero,valid=constant-zero,test=constant-zero"
)


class FakeDist:
    def __init__(
        self,
        *,
        initialized=True,
        rank=0,
        world_size=1,
        all_gather_results=None,
    ):
        self.initialized = initialized
        self.rank = rank
        self.world_size = world_size
        self.all_gather_results = (
            None
            if all_gather_results is None
            else [list(result) for result in all_gather_results]
        )
        self.broadcasts = []
        self.barriers = 0
        self.all_gathers = []

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialized

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def get_process_group_ranks(self, group):
        return list(group)

    def all_gather_object(self, output, value):
        self.all_gathers.append(value)
        if self.all_gather_results is not None:
            if not self.all_gather_results:
                raise ValueError("FakeDist has no controlled all_gather_object result")
            result = self.all_gather_results.pop(0)
            if len(result) != len(output):
                raise ValueError("FakeDist controlled all_gather_object result has wrong size")
            output[:] = result
            return
        if self.world_size != 1 or len(output) != 1:
            raise ValueError("FakeDist all_gather_object requires a controlled result")
        output[0] = value

    def broadcast_object_list(self, obj, src):
        self.broadcasts.append((list(obj), src))
        if obj[0] is None:
            obj[0] = "broadcast-model"

    def barrier(self):
        self.barriers += 1


class FakeParallelState:
    def __init__(
        self,
        *,
        tp_rank=0,
        ep_rank=0,
        cp_rank=0,
        pp_rank=0,
        dp_world=1,
        vp_world=None,
        pp_world=1,
        tp_world=1,
        cp_world=1,
    ):
        self.tp_rank = tp_rank
        self.ep_rank = ep_rank
        self.cp_rank = cp_rank
        self.pp_rank = pp_rank
        self.dp_world = dp_world
        self.vp_world = vp_world
        self.pp_world = pp_world
        self.tp_world = tp_world
        self.cp_world = cp_world

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def get_expert_model_parallel_rank(self):
        return self.ep_rank

    def get_context_parallel_rank(self):
        return self.cp_rank

    def get_pipeline_model_parallel_rank(self):
        return self.pp_rank

    def get_data_parallel_world_size(self, with_context_parallel=True):
        del with_context_parallel
        return self.dp_world

    def get_data_parallel_rank(self):
        return 0

    def get_virtual_pipeline_model_parallel_world_size(self):
        return self.vp_world

    def get_pipeline_model_parallel_world_size(self):
        return self.pp_world

    def get_tensor_model_parallel_world_size(self):
        return self.tp_world

    def get_context_parallel_world_size(self):
        return self.cp_world

    def get_tensor_model_parallel_group(self):
        return (0,)

    def get_pipeline_model_parallel_group(self):
        return (0,)

    def get_data_parallel_group(self):
        return (0,)

    def get_context_parallel_group(self):
        return (0,)

    def get_expert_model_parallel_group(self):
        return (0,)

    def get_expert_tensor_parallel_group(self, check_initialized=True):
        del check_initialized
        return (0,)

    def get_expert_tensor_and_model_parallel_group(self):
        return (0,)

    def get_expert_data_parallel_group(self):
        return (0,)


class FakeEngine:
    def __init__(self):
        self.closed = False
        self.flush_calls = 0
        self.record_formats = []
        self.record_runtime = SimpleNamespace()

    def create_record_runtime(self, record_format):
        self.record_formats.append(record_format)
        return self.record_runtime

    def close(self):
        self.closed = True

    def flush_and_wait(self, timeout_s=600.0):
        del timeout_s
        self.flush_calls += 1


def _fake_engine_factory(_cfg, _model_id, _record_format, _rank):
    return FakeEngine(), None


class FakeAdaptor:
    instances = []

    def __init__(self, engine, record_runtime, model_id, *, dims):
        self.engine = engine
        self.record_runtime = record_runtime
        self.model_id = model_id
        self.dims = dims
        self.attach_calls = []
        FakeAdaptor.instances.append(self)

    def attach_hooks(self, **kwargs):
        for binding in (
            *kwargs.get("model_hooks", ()),
            *kwargs.get("iteration_hooks", ()),
        ):
            binding.hook.spec = _megatron_hook_spec(binding.hook).resolve(self.dims)
        self.attach_calls.append(kwargs)

    def begin_attempt(self, **_kwargs):
        pass

    def end_attempt(self, **_kwargs):
        pass

    def set_current_iteration(self, _context):
        pass

    def clear_current_iteration(self):
        pass


class TinyModel(nn.Module):
    pass


class TinyGPTModel(nn.Module):
    def __init__(self, *, post_process: bool = True):
        super().__init__()
        self.post_process = bool(post_process)
        self.dmi_vocab_logits = None
        self.dmi_vocab_logits_topk = None


class TopKRouter(nn.Module):
    def __init__(self, layer_number: int = 1):
        super().__init__()
        self.layer_number = layer_number
        self.dmi_router_logits = None
        self.dmi_router_probs_mean = None
        self.dmi_router_token_entropy_mean = None
        self.dmi_pre_drop_token_count = None
        self.dmi_post_drop_token_count = None

    def _dmi_router_probs_mean_from_logits(self, *args):
        raise NotImplementedError

    def _dmi_router_logits_by_sample(self, *args):
        raise NotImplementedError

    def _dmi_router_token_entropy_mean_from_logits(self, *args):
        raise NotImplementedError

    def _dmi_expert_token_count_from_routing_map(self, *args):
        raise NotImplementedError


class TinyMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = TopKRouter(layer_number=3)


class MoEAlltoAllTokenDispatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.dmi_moe_inverse_map = None


class MoELayer(nn.Module):
    def __init__(self, layer_number: int = 3):
        super().__init__()
        self.layer_number = layer_number
        self.local_expert_indices = [0, 1, 2, 3]
        self.token_dispatcher = MoEAlltoAllTokenDispatcher()


class TinyMoEInternalsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = MoELayer()


def test_install_moe_inverse_map_hook_uses_execution_record():
    model = TinyMoEInternalsModel()

    _install_moe_inverse_map_hooks(model)

    hook = model.moe.dmi_moe_inverse_map
    assert isinstance(hook, HookPointV1)
    policy = _megatron_hook_spec(hook)
    assert policy.record_type is RecordType.PER_EXECUTION
    assert policy.need_token_range is False
    assert policy.binding_metadata_fields == frozenset()
    assert policy.shard_policy is ShardPolicy.GLOBAL_RANK_SHARDED
    assert model.moe.token_dispatcher.dmi_moe_inverse_map is hook


def test_per_execution_hook_records_global_physical_producer_rank():
    def binding(
        name: str,
        record_type: RecordType,
        shard_policy: ShardPolicy,
    ) -> MegatronHookBinding:
        return MegatronHookBinding(
            hook=_make_hook(
                MegatronHookSpec(
                    name=name,
                    layer_no=0,
                    outputs=(
                        MegatronOutputSpec(
                            name=f"{name}_value",
                            input_shape=(1,),
                            dtype=torch.float32,
                            transport_type=TransportType.IDENTITY,
                        ),
                    ),
                    shard_policy=shard_policy,
                    need_token_range=record_type is RecordType.PER_SAMPLE,
                    record_type=record_type,
                ),
                hook_phase=HookPhase.FWD,
            )
        )

    rank_ctx = MegatronRankContext(
        global_rank=7,
        tp_rank=0,
        tp_world_size=1,
        pp_rank=0,
        pp_world_size=1,
        dp_rank=0,
        dp_world_size=1,
        cp_rank=0,
        cp_world_size=1,
        ep_rank=1,
        ep_world_size=2,
        vp_rank=None,
        num_layers=1,
    )
    active = _active_hooks_for_rank(
        [
            binding(
                "execution",
                RecordType.PER_EXECUTION,
                ShardPolicy.GLOBAL_RANK_SHARDED,
            ),
            binding(
                "global_sample",
                RecordType.PER_SAMPLE,
                ShardPolicy.GLOBAL_RANK_SHARDED,
            ),
            binding("ep_sample", RecordType.PER_SAMPLE, ShardPolicy.EP_SHARDED),
        ],
        rank_ctx,
    )

    assert {
        _megatron_hook_spec(item.hook).name: item.record_shard_rank
        for item in active
    } == {
        "execution": 7,
        "global_sample": 7,
        "ep_sample": 1,
    }
    assert {
        _megatron_hook_spec(item.hook).name: item.record_dp_rank
        for item in active
    } == {
        "execution": -1,
        "global_sample": None,
        "ep_sample": None,
    }


def _policy_binding(
    selection_name: str,
    *,
    suppress_recompute: bool,
) -> MegatronHookBinding:
    hook = _make_hook(
        MegatronHookSpec(
            name=f"{selection_name}_hook",
            layer_no=0,
            outputs=(
                MegatronOutputSpec(
                    name=f"{selection_name}_value",
                    input_shape=(DimSpec.BATCH, 1),
                    dtype=torch.float32,
                ),
            ),
            preprocess=lambda value: value,
            enabled_by=frozenset({selection_name}),
        ),
        suppress_recompute=suppress_recompute,
        hook_phase=HookPhase.FWD,
    )
    return MegatronHookBinding(hook=hook)


def _metadata_report(
    *,
    global_rank: int = 0,
    dense_dp_rank: int = 0,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    pp_rank: int = 0,
    pp_world_size: int = 1,
    input_layout: HookInputLayout = HookInputLayout.SEQ_BATCH,
    max_batch_size: int = 2,
    segment_capacity: int | None = None,
    gpu_fields=(),
    cpu_record_fields=(),
    has_per_sample_records: bool = False,
) -> _MetadataRequirementReport:
    return _MetadataRequirementReport(
        global_rank=global_rank,
        dense_dp_rank=dense_dp_rank,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        pp_rank=pp_rank,
        pp_world_size=pp_world_size,
        input_layout=input_layout,
        max_batch_size=max_batch_size,
        segment_capacity=segment_capacity,
        gpu_fields=tuple(gpu_fields),
        cpu_record_fields=tuple(cpu_record_fields),
        has_per_sample_records=has_per_sample_records,
    )


def test_hook_binding_metadata_derives_from_preprocess_and_transport_consumers():
    identity = MegatronOutputSpec(
        name="identity",
        input_shape=(2, 1),
        dtype=torch.float32,
        transport_type=TransportType.IDENTITY,
    )
    prefix = MegatronOutputSpec(
        name="prefix",
        input_shape=(2, 1),
        dtype=torch.float32,
        transport_type=TransportType.PREFIX_STRIP,
    )
    sequence_pack = MegatronOutputSpec(
        name="sequence_pack",
        input_shape=(2, 1),
        dtype=torch.float32,
        transport_type=TransportType.SEQ_PREFIX_PACK,
    )
    segmented_pack = MegatronOutputSpec(
        name="segmented_pack",
        input_shape=(2, 1),
        dtype=torch.float32,
        transport_type=TransportType.SEGMENTED_PACK,
    )

    assert identity.transport_metadata_fields == frozenset()
    assert prefix.transport_metadata_fields == frozenset()
    assert sequence_pack.transport_metadata_fields == frozenset(
        {MegatronMetadataField.VALID_COUNT}
    )
    assert segmented_pack.transport_metadata_fields == frozenset(
        {MegatronMetadataField.SEGMENT_METADATA}
    )

    policy = MegatronHookSpec(
        name="consumer-derived",
        layer_no=0,
        outputs=(identity, sequence_pack),
        preprocess_metadata_fields={MegatronMetadataField.SEGMENT_METADATA},
    )
    assert policy.preprocess_metadata_fields == frozenset(
        {MegatronMetadataField.SEGMENT_METADATA}
    )
    assert policy.binding_metadata_fields == frozenset(
        {
            MegatronMetadataField.VALID_COUNT,
            MegatronMetadataField.SEGMENT_METADATA,
        }
    )

    with pytest.raises(TypeError, match="MegatronMetadataField"):
        MegatronHookSpec(
            name="invalid-metadata-field",
            layer_no=0,
            outputs=(identity,),
            preprocess_metadata_fields={"valid_count"},
        )


def test_local_metadata_report_uses_only_supplied_active_hooks():
    active = _make_hook(
        MegatronHookSpec(
            name="active",
            layer_no=0,
            outputs=(
                MegatronOutputSpec(
                    name="active_value",
                    input_shape=(2, 1),
                    dtype=torch.float32,
                ),
            ),
            need_token_range=False,
        ),
        hook_phase=HookPhase.FWD,
    )
    inactive = _make_hook(
        MegatronHookSpec(
            name="inactive",
            layer_no=1,
            outputs=(
                MegatronOutputSpec(
                    name="inactive_value",
                    input_shape=(2, 1),
                    dtype=torch.float32,
                    transport_type=TransportType.SEQ_PREFIX_PACK,
                ),
            ),
            need_token_range=False,
        ),
        hook_phase=HookPhase.FWD,
    )
    inactive.enabled = False
    supplied_active_hooks = [
        binding
        for binding in (MegatronHookBinding(active), MegatronHookBinding(inactive))
        if binding.hook.enabled
    ]
    rank_ctx = MegatronRankContext(
        global_rank=0,
        tp_rank=0,
        tp_world_size=1,
        pp_rank=0,
        pp_world_size=1,
        dp_rank=0,
        dp_world_size=1,
        cp_rank=0,
        cp_world_size=1,
        ep_rank=0,
        ep_world_size=1,
        vp_rank=None,
        num_layers=2,
    )

    report = _local_metadata_requirement_report(
        supplied_active_hooks,
        rank_ctx=rank_ctx,
        input_layout=HookInputLayout.SEQ_BATCH,
        max_batch_size=2,
        segment_capacity=None,
    )

    assert [binding.hook for binding in supplied_active_hooks] == [active]
    assert report.gpu_fields == ()
    assert report.cpu_record_fields == ()
    assert report.has_per_sample_records is True


def test_world_metadata_report_union_keeps_rank_local_gpu_visibility():
    rank0 = _metadata_report(
        global_rank=0,
        pp_rank=0,
        pp_world_size=2,
        input_layout=HookInputLayout.PACKED_SEGMENTED,
        segment_capacity=4,
        gpu_fields=(MegatronMetadataField.VALID_COUNT,),
        has_per_sample_records=True,
    )
    rank1 = _metadata_report(
        global_rank=1,
        pp_rank=1,
        pp_world_size=2,
        input_layout=HookInputLayout.PACKED_SEGMENTED,
        segment_capacity=4,
        gpu_fields=(MegatronMetadataField.SEGMENT_METADATA,),
        has_per_sample_records=True,
    )
    dist = FakeDist(
        rank=0,
        world_size=2,
        all_gather_results=[(rank0, rank1)],
    )

    reports = _gather_metadata_requirement_reports(rank0, dist_module=dist)
    rank0_requirements = _resolve_metadata_requirements(
        reports,
        local_report=rank0,
        dataset_provenance_modes={phase: "constant-zero" for phase in ("train", "valid", "test")},
    )
    rank1_requirements = _resolve_metadata_requirements(
        reports,
        local_report=rank1,
        dataset_provenance_modes={phase: "constant-zero" for phase in ("train", "valid", "test")},
    )

    expected_wire = (
        MegatronMetadataField.VALID_COUNT,
        MegatronMetadataField.SEGMENT_METADATA,
    )
    assert rank0_requirements.wire_fields == expected_wire
    assert rank1_requirements.wire_fields == expected_wire
    assert rank0_requirements.local_gpu_fields == frozenset(
        {MegatronMetadataField.VALID_COUNT}
    )
    assert rank1_requirements.local_gpu_fields == frozenset(
        {MegatronMetadataField.SEGMENT_METADATA}
    )

    rank0_specs = {
        spec.name: spec
        for spec in _metadata_field_specs_from_requirements(
            rank0_requirements,
            segment_capacity=4,
        )
    }
    rank1_specs = {
        spec.name: spec
        for spec in _metadata_field_specs_from_requirements(
            rank1_requirements,
            segment_capacity=4,
        )
    }
    assert tuple(rank0_specs) == tuple(rank1_specs) == (
        "valid_count",
        "segment_metadata",
    )
    assert {name: spec.gpu_visible for name, spec in rank0_specs.items()} == {
        "valid_count": True,
        "segment_metadata": False,
    }
    assert {name: spec.gpu_visible for name, spec in rank1_specs.items()} == {
        "valid_count": False,
        "segment_metadata": True,
    }


@pytest.mark.parametrize(
    ("remote_layout", "remote_segment_capacity"),
    [
        (HookInputLayout.SEQ_BATCH, 4),
        (HookInputLayout.PACKED_SEGMENTED, 8),
    ],
)
def test_metadata_report_domain_rejects_layout_or_capacity_mismatch(
    remote_layout,
    remote_segment_capacity,
):
    local = _metadata_report(
        global_rank=0,
        pp_rank=0,
        pp_world_size=2,
        input_layout=HookInputLayout.PACKED_SEGMENTED,
        segment_capacity=4,
    )
    remote = _metadata_report(
        global_rank=1,
        pp_rank=1,
        pp_world_size=2,
        input_layout=remote_layout,
        segment_capacity=remote_segment_capacity,
    )

    with pytest.raises(ValueError, match="layout or capacity"):
        _resolve_metadata_requirements(
            (local, remote),
            local_report=local,
            dataset_provenance_modes={
                phase: "constant-zero" for phase in ("train", "valid", "test")
            },
        )


def test_dataset_id_requires_dynamic_provenance_and_per_sample_record_in_dp_domain():
    local = _metadata_report(global_rank=0, pp_rank=0, pp_world_size=2)
    same_domain_per_sample = _metadata_report(
        global_rank=1,
        pp_rank=1,
        pp_world_size=2,
        has_per_sample_records=True,
    )
    dynamic_modes = {
        "train": "dynamic",
        "valid": "constant-zero",
        "test": "constant-zero",
    }
    constant_modes = {
        phase: "constant-zero" for phase in ("train", "valid", "test")
    }

    dynamic = _resolve_metadata_requirements(
        (local, same_domain_per_sample),
        local_report=local,
        dataset_provenance_modes=dynamic_modes,
    )
    constant = _resolve_metadata_requirements(
        (local, same_domain_per_sample),
        local_report=local,
        dataset_provenance_modes=constant_modes,
    )
    other_domain_per_sample = _metadata_report(
        global_rank=2,
        dense_dp_rank=1,
        has_per_sample_records=True,
    )
    different_domain = _resolve_metadata_requirements(
        (local, other_domain_per_sample),
        local_report=local,
        dataset_provenance_modes=dynamic_modes,
    )

    assert MegatronMetadataField.DATASET_ID in dynamic.wire_fields
    assert MegatronMetadataField.DATASET_ID not in constant.wire_fields
    assert MegatronMetadataField.DATASET_ID not in different_domain.wire_fields


def _install_fake_megatron_blend_resolver(monkeypatch, result) -> None:
    megatron_module = ModuleType("megatron")
    training_module = ModuleType("megatron.training")
    utils_module = ModuleType("megatron.training.utils")
    utils_module.get_blend_and_blend_per_split = lambda _args: result
    megatron_module.training = training_module
    training_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.training", training_module)
    monkeypatch.setitem(sys.modules, "megatron.training.utils", utils_module)


def teardown_function():
    set_active_megatron_schedule_runtime(None)
    FakeAdaptor.instances.clear()


def test_config_resolution_uses_cli_over_env_and_defaults():
    args = SimpleNamespace(
        dmi_enable=True,
        dmi_hook_selection=None,
        dmi_model_id="cli-model",
        dmi_db_host=None,
        dmi_db_port=9440,
        dmi_db_database=None,
        dmi_clickhouse_table=None,
        dmi_ch_parallelism=None,
        dmi_ring_payload_mb=None,
        dmi_ring_pinned_mb=8,
        dmi_ring_task_entries=None,
        dmi_flush_every_n_train_iters=3,
        dmi_vocab_logits_top_k=None,
        dmi_topology_manifest_path=None,
    )
    env = {
        "DMI_ENABLE": "0",
        "DMI_HOOK_SELECTION": "env-hook",
        "DMI_MODEL_ID": "env-model",
        "DMI_DB_HOST": "localhost",
        "DMI_RING_PINNED_MB": "16",
        "DMI_FLUSH_EVERY_N_TRAIN_ITERS": "5",
        "DMI_VOCAB_LOGITS_TOP_K": "100",
        "DMI_TOPOLOGY_MANIFEST_PATH": "topology.json",
    }

    cfg = resolve_megatron_dmi_config(args, environ=env)

    assert cfg.enabled is True
    assert cfg.hook_selection == "env-hook"
    assert cfg.model_id == "cli-model"
    assert cfg.db_host == "localhost"
    assert cfg.db_port == 9440
    assert cfg.ring_pinned_mb == 8
    assert cfg.ring_payload_mb == 4096
    assert cfg.flush_every_n_train_iters == 3
    assert cfg.vocab_logits_top_k == 100
    assert cfg.topology_manifest_path == "topology.json"


def test_explicit_config_overrides_cli_and_env():
    args = SimpleNamespace(dmi_enable=True)
    explicit = MegatronDMIConfig(enabled=False, model_id="explicit")

    cfg = resolve_megatron_dmi_config(args, explicit=explicit, environ={"DMI_ENABLE": "1"})

    assert cfg is explicit


def test_resolve_model_id_generates_and_broadcasts():
    dist = FakeDist(rank=0)
    seen = []

    model_id = resolve_model_id(
        MegatronDMIConfig(enabled=True),
        dist_module=dist,
        environ={"SLURM_JOB_ID": "123"},
        printer=seen.append,
    )

    assert model_id == "dmi-megatron-123"
    assert dist.broadcasts == [(["dmi-megatron-123"], 0)]
    assert "generated model_id=dmi-megatron-123" in seen[0]


def test_setup_disabled_is_noop():
    handle = setup_megatron_dmi(
        [TinyModel()],
        args=SimpleNamespace(dmi_enable=None),
        explicit_config=MegatronDMIConfig(enabled=False),
    )

    assert handle is None
    assert get_active_megatron_schedule_runtime() is None


def test_setup_moe_reconstruction_hook_requires_topology_manifest_path():
    engine_calls = []

    def engine_factory(_cfg, _model_id, _record_format, _rank):
        engine_calls.append(True)
        return FakeEngine(), None

    with pytest.raises(ValueError, match="DMI_TOPOLOGY_MANIFEST_PATH"):
        setup_megatron_dmi(
            [TinyMoEInternalsModel()],
            args=SimpleNamespace(global_batch_size=1, micro_batch_size=1),
            model_config=SimpleNamespace(),
            explicit_config=MegatronDMIConfig(
                enabled=True,
                model_id="missing-manifest",
                hook_selection="moe-inverse-map",
            ),
            parallel_state_module=FakeParallelState(),
            dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda model: model,
            engine_factory=engine_factory,
            device="cpu",
        )

    assert engine_calls == []


def test_setup_writes_frozen_ep_topology_manifest(tmp_path):
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    path = tmp_path / "topology.json"
    dist = FakeDist(initialized=True)
    handle = setup_megatron_dmi(
        [TinyMoEInternalsModel()],
        args=SimpleNamespace(global_batch_size=1, micro_batch_size=1),
        model_config=SimpleNamespace(
            num_moe_experts=4,
            sequence_parallel=False,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=False,
            moe_expert_capacity_factor=None,
            moe_token_dropping=False,
            moe_pad_expert_input_to_capacity=False,
        ),
        explicit_config=MegatronDMIConfig(
            enabled=True,
            model_id="manifest-run",
            hook_selection="moe-inverse-map",
            dataset_provenance_mode=CONSTANT_PROVENANCE,
            topology_manifest_path=str(path),
        ),
        parallel_state_module=FakeParallelState(),
        dist_module=dist,
        unwrap_fn=lambda model: model,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )
    try:
        manifest = load_ep_topology_manifest(path)
        assert manifest.model_id == "manifest-run"
        assert manifest.local_expert_order_by_ep_rank == ((0, 1, 2, 3),)
        assert [
            (placement.layer_no, placement.pp_rank, placement.scope_id)
            for placement in manifest.layer_placements
        ] == [(2, 0, 0)]
        assert len(dist.all_gathers) == 2
    finally:
        handle.close()


def test_setup_enabled_builds_runtime_and_attaches_model():
    model = [TinyModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=8, micro_batch_size=2)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(num_moe_experts=4),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(dp_world=2, vp_world=3),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    assert handle.model_id == "run"
    assert get_active_megatron_schedule_runtime() is handle.schedule_runtime
    assert handle.current_phase_tensor.device.type == "cpu"
    assert int(handle.current_phase_tensor.item()) == HookPhase.FWD.value
    assert handle.schedule_runtime.current_phase_tensor is handle.current_phase_tensor
    assert len(FakeAdaptor.instances) == 1
    adaptor = FakeAdaptor.instances[0]
    assert adaptor.record_runtime is handle.engine.record_runtime
    assert len(handle.engine.record_formats) == 1
    assert adaptor.dims[DimSpec.BATCH] == 2
    assert adaptor.dims[DimSpec.NUM_EXPERTS] == 4
    attach_kwargs = adaptor.attach_calls[0]
    assert attach_kwargs["model_hooks"] == []
    assert len(attach_kwargs["iteration_hooks"]) == 1
    assert attach_kwargs["iteration_hooks"][0].hook.spec.name == "iteration_attempt_status"
    assert attach_kwargs["metadata_context"] is runtime_contexts[0]
    assert attach_kwargs["current_phase_tensor"] is handle.current_phase_tensor
    assert not hasattr(model[0], "dmi_lm_per_sample_loss")

    handle.close()
    assert get_active_megatron_schedule_runtime() is None
    assert handle.engine.closed is True


def test_setup_hidden_states_resolves_seq_and_hidden_dims():
    model = [TinyModel()]

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="hidden-states",
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2, seq_length=16)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(hidden_size=32),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(dp_world=1),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    adaptor = FakeAdaptor.instances[0]
    assert adaptor.dims[DimSpec.BATCH] == 2
    assert adaptor.dims[DimSpec.SEQ] == 16
    assert adaptor.dims[DimSpec.HIDDEN] == 32
    assert DimSpec.NUM_EXPERTS not in adaptor.dims
    assert adaptor.attach_calls[0]["model_hooks"] == []

    handle.close()


def test_setup_vocab_logits_installs_last_stage_raw_identity_hook():
    model = [TinyGPTModel(post_process=True)]

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="vocab-logits",
        model_id="vocab-run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(
        global_batch_size=4,
        micro_batch_size=2,
        seq_length=16,
        padded_vocab_size=128,
    )
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(params_dtype=torch.bfloat16),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    adaptor = FakeAdaptor.instances[0]
    assert adaptor.dims[DimSpec.BATCH] == 2
    assert adaptor.dims[DimSpec.SEQ] == 16
    assert adaptor.dims[DimSpec.VOCAB] == 128
    hook = model[0].dmi_vocab_logits
    assert isinstance(hook, HookPointV1)
    assert hook.spec is not None
    assert hook.spec.name == "vocab_logits"
    assert hook.spec.outputs[0].name == "vocab_logits"
    policy = _megatron_hook_spec(hook)
    assert policy.need_token_range is False
    assert policy.outputs[0].dtype is torch.bfloat16
    assert policy.outputs[0].input_shape == (
        DimSpec.BATCH,
        DimSpec.SEQ,
        DimSpec.VOCAB,
    )
    assert adaptor.attach_calls[0]["model_hooks"][0].hook is hook

    handle.close()


def test_setup_vocab_logits_topk_installs_two_output_fixed_k_hook():
    model = [TinyGPTModel(post_process=True)]

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    handle = setup_megatron_dmi(
        model,
        args=SimpleNamespace(
            global_batch_size=4,
            micro_batch_size=2,
            seq_length=16,
            padded_vocab_size=128,
        ),
        model_config=SimpleNamespace(params_dtype=torch.bfloat16),
        explicit_config=MegatronDMIConfig(
            enabled=True,
            hook_selection="vocab-logits-topk",
            vocab_logits_top_k=100,
            model_id="vocab-topk-run",
            dataset_provenance_mode=CONSTANT_PROVENANCE,
        ),
        parallel_state_module=FakeParallelState(),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    adaptor = FakeAdaptor.instances[0]
    assert adaptor.dims[DimSpec.BATCH] == 2
    assert adaptor.dims[DimSpec.SEQ] == 16
    assert DimSpec.VOCAB not in adaptor.dims
    hook = model[0].dmi_vocab_logits_topk
    assert isinstance(hook, HookPointV1)
    assert hook.spec is not None
    assert hook.spec.name == "vocab_logits_topk"
    assert [output.name for output in hook.spec.outputs] == [
        "vocab_logits_topk_values",
        "vocab_logits_topk_indices",
    ]
    policy = _megatron_hook_spec(hook)
    assert policy.need_token_range is False
    assert [output.dtype for output in policy.outputs] == [
        torch.bfloat16,
        torch.int32,
    ]
    assert all(
        output.input_shape == (DimSpec.BATCH, DimSpec.SEQ, 100)
        for output in policy.outputs
    )
    assert adaptor.attach_calls[0]["model_hooks"][0].hook is hook

    handle.close()


@pytest.mark.parametrize(
    ("hook_selection", "top_k", "message"),
    [
        ("vocab-logits-topk", None, "requires --dmi-vocab-logits-top-k"),
        ("vocab-logits-topk", 0, "must satisfy"),
        ("vocab-logits-topk", 129, "must satisfy"),
        ("vocab-logits", 100, "configured without selecting"),
    ],
)
def test_setup_vocab_logits_topk_rejects_invalid_selection_contract(
    hook_selection,
    top_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        setup_megatron_dmi(
            [TinyGPTModel(post_process=True)],
            args=SimpleNamespace(
                global_batch_size=2,
                micro_batch_size=1,
                seq_length=4,
                padded_vocab_size=128,
            ),
            model_config=SimpleNamespace(params_dtype=torch.bfloat16),
            explicit_config=MegatronDMIConfig(
                enabled=True,
                hook_selection=hook_selection,
                vocab_logits_top_k=top_k,
                model_id="bad-top-k",
            ),
            parallel_state_module=FakeParallelState(),
            dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda x: x,
        )


@pytest.mark.parametrize(
    ("parallel_state", "message"),
    [
        (FakeParallelState(tp_world=2), "tensor-model-parallel size 1"),
        (FakeParallelState(cp_world=2), "context-parallel size 1"),
    ],
)
@pytest.mark.parametrize(
    ("hook_selection", "top_k"),
    [
        ("vocab-logits", None),
        ("vocab-logits-topk", 4),
    ],
)
def test_setup_vocab_logits_rejects_unsupported_parallelism(
    parallel_state,
    message,
    hook_selection,
    top_k,
):
    with pytest.raises(NotImplementedError, match=message):
        setup_megatron_dmi(
            [TinyGPTModel(post_process=True)],
            args=SimpleNamespace(
                global_batch_size=2,
                micro_batch_size=1,
                seq_length=4,
                padded_vocab_size=16,
            ),
            model_config=SimpleNamespace(params_dtype=torch.bfloat16),
            explicit_config=MegatronDMIConfig(
                enabled=True,
                hook_selection=hook_selection,
                vocab_logits_top_k=top_k,
                model_id="bad-parallel",
            ),
            parallel_state_module=parallel_state,
            dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda x: x,
        )


def test_vocab_logits_dtype_requires_megatron_parameter_dtype():
    assert _vocab_logits_dtype(SimpleNamespace(params_dtype=torch.float32)) is torch.float32
    with pytest.raises(ValueError, match="params_dtype"):
        _vocab_logits_dtype(SimpleNamespace())


def test_setup_loss_summary_installs_only_selected_hook():
    model = [TinyModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="loss-summary",
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(dp_world=1),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    assert hasattr(model[0], "dmi_lm_per_sample_loss")
    assert model[0].dmi_lm_per_sample_loss.suppress_recompute is False
    loss_policy = _megatron_hook_spec(model[0].dmi_lm_per_sample_loss)
    assert loss_policy.need_token_range is False
    assert loss_policy.binding_metadata_fields == frozenset()
    assert FakeAdaptor.instances[0].dims == {DimSpec.BATCH: 2}
    assert len(FakeAdaptor.instances[0].attach_calls[0]["model_hooks"]) == 1
    assert runtime_contexts[0].field_specs == {}

    handle.close()


def test_setup_router_health_hooks_installs_entropy_and_expert_counts():
    model = [TinyMoEModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="router-entropy,expert-counts",
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(num_moe_experts=64),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(dp_world=1),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    router = model[0].router
    assert handle is not None
    assert _megatron_hook_spec(router.dmi_router_token_entropy_mean).layer_no == 2
    assert router.dmi_router_token_entropy_mean.spec.outputs[0].name == "router_token_entropy_mean"
    assert router.dmi_pre_drop_token_count.spec.outputs[0].name == "pre_drop_token_count"
    assert router.dmi_post_drop_token_count.spec.outputs[0].name == "post_drop_token_count"
    assert FakeAdaptor.instances[0].dims[DimSpec.NUM_EXPERTS] == 64
    assert "valid_count" in runtime_contexts[0].field_specs

    handle.close()


def test_setup_router_logits_installs_raw_identity_hook_without_valid_count() -> None:
    model = [TinyMoEModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="router-logits",
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2, seq_length=16)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(
            num_moe_experts=64,
            moe_router_dtype="fp32",
            params_dtype=torch.bfloat16,
        ),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(dp_world=1),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    hook = model[0].router.dmi_router_logits
    assert isinstance(hook, HookPointV1)
    policy = _megatron_hook_spec(hook)
    assert policy.need_token_range is False
    assert policy.binding_metadata_fields == frozenset()
    assert hook.spec.outputs[0].transport_type is TransportType.IDENTITY
    assert policy.outputs[0].dtype is torch.float32
    assert policy.outputs[0].input_shape == (
        DimSpec.BATCH,
        DimSpec.SEQ,
        DimSpec.NUM_EXPERTS,
    )
    assert model[0].router.dmi_router_probs_mean is None
    assert FakeAdaptor.instances[0].dims == {
        DimSpec.BATCH: 2,
        DimSpec.NUM_EXPERTS: 64,
        DimSpec.SEQ: 16,
    }
    assert len(FakeAdaptor.instances[0].attach_calls[0]["model_hooks"]) == 1
    assert "valid_count" not in runtime_contexts[0].field_specs

    handle.close()


@pytest.mark.parametrize(
    ("configured", "params_dtype", "expected"),
    [
        ("fp32", torch.bfloat16, torch.float32),
        ("fp64", torch.bfloat16, torch.float64),
        (None, torch.bfloat16, torch.bfloat16),
    ],
)
def test_router_logits_dtype_matches_megatron_gating_policy(
    configured, params_dtype, expected
) -> None:
    model_config = SimpleNamespace(
        moe_router_dtype=configured,
        params_dtype=params_dtype,
    )

    assert _router_logits_dtype(model_config) is expected


def test_router_logits_dtype_rejects_unsupported_configuration() -> None:
    with pytest.raises(ValueError, match="Unsupported Megatron moe_router_dtype"):
        _router_logits_dtype(
            SimpleNamespace(moe_router_dtype="fp16", params_dtype=torch.float16)
        )

    with pytest.raises(ValueError, match="requires model_config.params_dtype"):
        _router_logits_dtype(SimpleNamespace(moe_router_dtype=None))


def test_loss_summary_inactive_on_non_emitting_rank_uses_no_metadata_fields():
    model = [TinyModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    cfg = MegatronDMIConfig(
        enabled=True,
        hook_selection="loss-summary",
        model_id="run",
        dataset_provenance_mode=CONSTANT_PROVENANCE,
    )
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2)
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(),
        explicit_config=cfg,
        parallel_state_module=FakeParallelState(tp_rank=1, pp_rank=1, pp_world=2),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    assert hasattr(model[0], "dmi_lm_per_sample_loss")
    assert model[0].dmi_lm_per_sample_loss.enabled is False
    assert runtime_contexts[0].field_specs == {}
    assert FakeAdaptor.instances[0].attach_calls[0]["model_hooks"] == []

    handle.close()


def test_setup_publishes_stable_required_metadata_field_names():
    model = [TinyModel()]
    runtime_contexts = []

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        runtime_contexts.append(context)
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    args = SimpleNamespace(
        global_batch_size=4,
        micro_batch_size=2,
        sft=True,
        dmi_packed_max_conversations_per_row=1,
        context_parallel_size=1,
    )
    handle = setup_megatron_dmi(
        model,
        args=args,
        model_config=SimpleNamespace(),
        explicit_config=MegatronDMIConfig(
            enabled=True,
            hook_selection="loss-summary",
            model_id="run",
            dataset_provenance_mode=(
                "train=dynamic,valid=constant-zero,test=constant-zero"
            ),
        ),
        parallel_state_module=FakeParallelState(),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    assert model[0].dmi_lm_per_sample_loss.enabled is True
    assert len(FakeAdaptor.instances[0].attach_calls[0]["model_hooks"]) == 1
    assert tuple(runtime_contexts[0].field_specs) == (
        "valid_count",
        "segment_metadata",
        "dataset_id",
    )
    assert {
        name: spec.gpu_visible
        for name, spec in runtime_contexts[0].field_specs.items()
    } == {
        "valid_count": False,
        "segment_metadata": True,
        "dataset_id": False,
    }
    assert args.dmi_required_metadata_fields == (
        "valid_count",
        "segment_metadata",
        "dataset_id",
    )

    handle.close()


def test_hook_selection_parser_only_parses_selected_names():
    assert parse_hook_selection("router-summary,loss-summary") == {
        "router-summary",
        "loss-summary",
    }
    assert parse_hook_selection(None) == {"router-summary"}
    with pytest.raises(ValueError, match="empty DMI hook selection"):
        parse_hook_selection("router-summary,")


@pytest.mark.parametrize(
    (
        "record_type",
        "need_token_range",
        "transport_type",
        "dynamic_provenance",
        "expected",
    ),
    [
        (RecordType.PER_SAMPLE, False, TransportType.IDENTITY, False, frozenset()),
        (
            RecordType.PER_SAMPLE,
            True,
            TransportType.IDENTITY,
            False,
            frozenset({MegatronMetadataField.VALID_COUNT}),
        ),
        (
            RecordType.PER_SAMPLE,
            False,
            TransportType.PREFIX_STRIP,
            False,
            frozenset({MegatronMetadataField.VALID_COUNT}),
        ),
        (
            RecordType.PER_SAMPLE,
            False,
            TransportType.SEQ_PREFIX_PACK,
            False,
            frozenset({MegatronMetadataField.VALID_COUNT}),
        ),
        (
            RecordType.PER_SAMPLE,
            False,
            TransportType.SEGMENTED_PACK,
            False,
            frozenset({MegatronMetadataField.VALID_COUNT}),
        ),
        (
            RecordType.PER_SAMPLE,
            False,
            TransportType.IDENTITY,
            True,
            frozenset({MegatronMetadataField.DATASET_ID}),
        ),
        (
            RecordType.PER_EXECUTION,
            True,
            TransportType.PREFIX_STRIP,
            True,
            frozenset(),
        ),
    ],
)
def test_record_metadata_requirements_are_consumer_derived(
    record_type,
    need_token_range,
    transport_type,
    dynamic_provenance,
    expected,
):
    assert required_record_metadata_fields(
        record_type=record_type,
        need_token_range=need_token_range,
        transport_type=transport_type,
        dynamic_dataset_provenance=dynamic_provenance,
    ) == expected


def test_setup_registers_grad_norm_as_coordinator_iteration_hook():
    model = [TinyModel()]

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs["max_num_microbatches"],
            max_batch_size=kwargs["max_batch_size"],
            num_scopes=kwargs["num_scopes"],
            field_specs=kwargs["field_specs"],
            device="cpu",
        )
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=kwargs["host_engine"],
        )

    handle = setup_megatron_dmi(
        model,
        args=SimpleNamespace(global_batch_size=4, micro_batch_size=2),
        model_config=SimpleNamespace(),
        explicit_config=MegatronDMIConfig(
            enabled=True,
            hook_selection="grad-norm",
            model_id="run",
        ),
        parallel_state_module=FakeParallelState(dp_world=1),
        dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x,
        engine_factory=_fake_engine_factory,
        runtime_factory=runtime_factory,
        adaptor_cls=FakeAdaptor,
        device="cpu",
    )

    assert handle is not None
    assert handle.grad_norm_hook is not None
    bindings = FakeAdaptor.instances[0].attach_calls[0]["iteration_hooks"]
    assert [binding.hook.spec.name for binding in bindings] == [
        "iteration_attempt_status",
        "grad_norm",
    ]
    assert all(binding.record_dp_rank == -1 for binding in bindings)
    assert all(binding.record_shard_rank == -1 for binding in bindings)
    handle.close()


def test_router_weights_rejects_dp_greater_than_one_before_engine_creation():
    engine_calls = []

    def engine_factory(_cfg, _model_id, _record_format, _rank):
        engine_calls.append(True)
        return FakeEngine(), None

    with pytest.raises(NotImplementedError, match="data-parallel world size exactly 1"):
        setup_megatron_dmi(
            [TinyMoEModel()],
            args=SimpleNamespace(global_batch_size=4, micro_batch_size=2),
            model_config=SimpleNamespace(num_moe_experts=4, hidden_size=8),
            explicit_config=MegatronDMIConfig(
                enabled=True,
                hook_selection="router-weights",
                model_id="run",
            ),
            parallel_state_module=FakeParallelState(dp_world=2),
            dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda x: x,
            engine_factory=engine_factory,
            adaptor_cls=FakeAdaptor,
            device="cpu",
        )
    assert engine_calls == []


def test_recompute_hook_policy_preserves_defaults_and_applies_both_overrides():
    retain = _policy_binding("retain-me", suppress_recompute=True)
    suppress = _policy_binding("suppress-me", suppress_recompute=False)
    unchanged = _policy_binding("unchanged", suppress_recompute=False)

    _apply_recompute_hook_policy(
        [retain, suppress, unchanged],
        selected_names={"retain-me", "suppress-me", "unchanged"},
        recompute_names_raw="retain-me",
        no_recompute_names_raw="suppress-me",
    )

    assert retain.hook.suppress_recompute is False
    assert suppress.hook.suppress_recompute is True
    assert unchanged.hook.suppress_recompute is False


@pytest.mark.parametrize(
    ("selected_names", "recompute_names", "no_recompute_names", "message"),
    [
        ({"selected"}, "missing", None, "cannot enable unselected hooks"),
        ({"selected"}, "selected", "selected", "occur in both lists"),
        ({"selected"}, "selected,selected", None, "duplicate"),
        ({"selected"}, "selected,", None, "empty"),
    ],
)
def test_recompute_hook_policy_rejects_invalid_lists(
    selected_names,
    recompute_names,
    no_recompute_names,
    message,
):
    binding = _policy_binding("selected", suppress_recompute=True)
    with pytest.raises(ValueError, match=message):
        _apply_recompute_hook_policy(
            [binding],
            selected_names=selected_names,
            recompute_names_raw=recompute_names,
            no_recompute_names_raw=no_recompute_names,
        )


def test_recompute_hook_policy_does_not_enable_or_silently_ignore_hooks():
    unselected = _policy_binding("other", suppress_recompute=True)
    unselected.hook.enabled = False

    with pytest.raises(ValueError, match="resolve to no selected HookPointV1"):
        _apply_recompute_hook_policy(
            [unselected],
            selected_names={"selected"},
            recompute_names_raw="selected",
            no_recompute_names_raw=None,
        )

    assert unselected.hook.enabled is False
    assert unselected.hook.suppress_recompute is True


def test_standard_dataset_topology_auto_detects_modes_per_phase(monkeypatch):
    _install_fake_megatron_blend_resolver(
        monkeypatch,
        (
            None,
            (
                (["train-a", "train-b"], [0.7, 0.3]),
                (["valid-a"], [1.0]),
                None,
            ),
        ),
    )
    provider = SimpleNamespace(dmi_standard_dataset_provider=True)

    modes = _resolve_dataset_provenance_modes(
        SimpleNamespace(mock_data=False, multiple_validation_sets=False),
        provider,
        configured_mode="auto",
        has_per_sample_hooks=True,
    )

    assert modes == {
        "train": "dynamic",
        "valid": "constant-zero",
        "test": "constant-zero",
    }


def test_dataset_provenance_auto_requires_standard_provider():
    with pytest.raises(ValueError, match="custom providers must configure"):
        _resolve_dataset_provenance_modes(
            SimpleNamespace(mock_data=False, multiple_validation_sets=False),
            object(),
            configured_mode="auto",
            has_per_sample_hooks=True,
        )


def test_no_per_sample_hooks_need_no_dataset_topology_resolution():
    modes = _resolve_dataset_provenance_modes(
        SimpleNamespace(),
        object(),
        configured_mode="auto",
        has_per_sample_hooks=False,
    )
    assert set(modes.values()) == {"constant-zero"}
