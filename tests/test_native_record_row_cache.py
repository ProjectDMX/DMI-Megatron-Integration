"""Reusing physical rows must never reuse execution identities or dynamic sizes."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dmi.api.v1 import HookOutput, ProducerPlanEntry, RecordType, TransportSpec, TransportType
from dmi_megatron_integration.records.format import MegatronRecordFormat
from dmi_megatron_integration.records.metadata import MegatronRecordMetadata


def setup(monkeypatch):
    fmt = MegatronRecordFormat('cache', producer_rank=3)
    original = fmt._row_specs
    calls = []
    def counted(metadata, entry):
        calls.append((metadata, entry))
        return original(metadata, entry)
    monkeypatch.setattr(fmt, '_row_specs', counted)
    encode = fmt._bind_native_encoder(SimpleNamespace(_compile_record_layout=lambda *x: x))
    spec = TransportSpec('probe', transport_type=TransportType.SEQ_PREFIX_PACK,
                         feature_bytes=16, output_shape=(-1, 4))
    entry = ProducerPlanEntry.from_output(output_id=65536, output_spec=spec,
                                         output=HookOutput(torch.empty(8, 2, 4)))
    md = MegatronRecordMetadata('m', 'p', 'fwd', 'train', 1, 0, 0, 0, 0, 0,
                               valid_counts=(3, 2), dataset_ids=(1, 2))
    return fmt, encode, entry, md, original, calls


def test_row_hit_keeps_execution_identity_and_expected_count_fresh(monkeypatch):
    fmt, encode, entry, md, reference, calls = setup(monkeypatch)
    fmt.count_records = True
    packets = []
    for i, phase in enumerate(('train', 'valid', 'test', 'train')):
        current = replace(md, model_id=f'm{i}', act_name=f'p{i}', direction=('bwd' if i==3 else 'fwd'),
                          phase=phase, global_batch_id=i+1, dp_rank=i, microbatch_id=i,
                          layer_no=i, shard_rank=i, attempt_id=i, invocation_id=i)
        packet = encode(current, replace(entry))
        assert packet[1] == fmt._common_coordinates(current)
        assert packet[2] == reference(current, entry)
        assert fmt.take_expected_count(phase, i+1, i) == 2
        packets.append(packet)
    assert len(calls) == 1
    assert all(p[2] is packets[0][2] for p in packets)
    assert len({p[1] for p in packets}) == 4


@pytest.mark.parametrize('field,value', [
    ('valid_counts', (1, 4)), ('valid_counts', (0, 0)),
    ('dataset_ids', (3, 4)), ('dataset_ids', ()), ('token_start', 128),
])
def test_dynamic_row_coordinates_invalidate_latest_layout(monkeypatch, field, value):
    fmt, encode, entry, md, reference, calls = setup(monkeypatch)
    for current in (md, replace(md, **{field:value}), md):
        assert encode(current, entry)[2] == reference(current, entry)
    assert len(calls) == 3


@pytest.mark.parametrize('change', [
    {'dtype':torch.float16},
    {'input_shape':(12, 2, 4), 'reservation_upper_bytes':384},
    {'output_shape':(-1, 2), 'transport_args':(8,)},
    {'record_type':RecordType.PER_EXECUTION, 'transport_type':TransportType.SEGMENTED_PACK},
])
def test_changed_producer_description_uses_original_validation(monkeypatch, change):
    fmt, encode, entry, md, reference, calls = setup(monkeypatch)
    encode(md, entry)
    other = replace(entry, **change)
    other_md = replace(md, valid_counts=(), dataset_ids=()) if other.record_type is RecordType.PER_EXECUTION else md
    assert encode(other_md, other)[2] == reference(other_md, other)
    assert len(calls) == 2


def test_invalid_change_is_not_masked_and_other_outputs_are_separate(monkeypatch):
    fmt, encode, entry, md, reference, calls = setup(monkeypatch)
    first = encode(md, entry)
    other = replace(entry, output_id=65537)
    assert encode(md, other)[3] == 65537
    assert encode(md, entry)[2] is first[2]
    assert len(calls) == 2
    with pytest.raises(ValueError, match='dataset_ids length'):
        encode(replace(md, dataset_ids=(1,)), entry)
    with pytest.raises(ValueError, match='row mismatch'):
        encode(md, replace(entry, output_shape=(4, 4)))
    with pytest.raises(ValueError, match='attempt_id'):
        encode(replace(md, attempt_id=-2), entry)
    assert encode(md, entry)[2] is first[2]


def test_manual_entry_with_mutable_shapes_does_not_reuse_stale_rows(monkeypatch):
    fmt, encode, entry, md, reference, calls = setup(monkeypatch)
    shape = list(entry.output_shape)
    args = list(entry.transport_args)
    manual = replace(entry, input_shape=list(entry.input_shape),
                     output_shape=shape, transport_args=args)
    first = encode(md, manual)
    shape[1] = 2
    args[0] = 8
    second = encode(md, manual)
    assert second[2] == reference(md, manual)
    assert first[2][0][4] == (3, 4)
    assert second[2][0][4] == (3, 2)
    assert first[2][0][2] == 48 and second[2][0][2] == 24
    assert len(calls) == 2
