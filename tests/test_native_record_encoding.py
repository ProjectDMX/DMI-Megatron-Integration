"""Built-in native rows retain the generic encoder's dynamic identities/values."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from dmi.api.v1 import HookOutput, OutputStorage, ProducerPlanEntry, RecordType, TransportSpec, TransportType
from dmi.transport import native
from dmi_megatron_integration.records.format import MegatronRecordFormat
from dmi_megatron_integration.records.metadata import MegatronRecordMetadata


def bind(fmt):
    backend = native._load_extension()
    schema = backend._compile_record_schema(fmt.schema)
    return fmt._bind_native_encoder(SimpleNamespace(_compile_record_layout=lambda *args:
        backend._compile_record_layout(schema, *args)))


@pytest.mark.parametrize("storage,dtype", [(OutputStorage.TENSOR, torch.float32),
    (OutputStorage.SCALAR_FLOAT, torch.float32), (OutputStorage.SCALAR_INT, torch.int64)])
@pytest.mark.parametrize("record_type", list(RecordType))
def test_native_materialized_rows_match_generic(tmp_path, storage, dtype, record_type):
    fmt = MegatronRecordFormat("native_parity", producer_rank=7)
    fmt.count_records = True
    encode = bind(fmt)
    dirs = [tmp_path / name for name in ("generic", "native")]
    sinks = [native.DropRecordSink(fmt.schema, str(path), 0) for path in dirs]
    try:
        for iteration, counts in enumerate(((2, 0, 3), (0, 3, 2), (0, 0, 0), (1, 1, 1))):
            payload = torch.arange(3, dtype=dtype).reshape(3, 1)
            spec = TransportSpec("probe", record_type=record_type, storage=storage)
            entry = ProducerPlanEntry.from_output(output_id=65536, output_spec=spec,
                output=HookOutput(payload))
            if record_type is not RecordType.PER_SAMPLE:
                payload = payload[:1].contiguous()
                entry = replace(entry, input_shape=(1, 1), output_shape=(1, 1),
                    reservation_upper_bytes=payload.nbytes)
            metadata = MegatronRecordMetadata("model", "probe", "fwd", "valid", iteration,
                2, 3, 4, 5, 8, valid_counts=counts if record_type is RecordType.PER_SAMPLE else (),
                dataset_ids=(10, 20, 30) if record_type is RecordType.PER_SAMPLE else (),
                attempt_id=2, invocation_id=iteration)
            for sink, encoder in zip(sinks, (fmt.encode, encode)):
                descriptor = encoder(metadata, entry)
                sink.submit(descriptor, payload.view(torch.uint8).flatten())
                assert fmt.take_expected_count("valid", iteration, 2) == (
                    sum(c > 0 for c in counts) if record_type is RecordType.PER_SAMPLE else 1)
    finally:
        for sink in sinks:
            sink.close()
    def rows(path):
        return [(event["layout"], event["rows"]) for event in map(json.loads,
            (path / "rank_00000/events.jsonl").read_text().splitlines()) if event["type"] == "record"]
    assert rows(dirs[0]) == rows(dirs[1])


@pytest.mark.parametrize("transport", [TransportType.SEQ_PREFIX_PACK, TransportType.SEGMENTED_PACK])
def test_packed_native_offsets_change_with_same_total(transport):
    fmt = MegatronRecordFormat("packed")
    encode = bind(fmt)
    spec = TransportSpec("probe", transport_type=transport, feature_bytes=16, output_shape=(-1, 4))
    entry = ProducerPlanEntry.from_output(output_id=65536, output_spec=spec,
        output=HookOutput(torch.empty(8, 2, 4)))
    packets = []
    for counts in ((3, 2), (1, 4), (0, 0)):
        md = MegatronRecordMetadata("m", "p", "fwd", "train", 1, 0, 0, 0, 0, 4,
            valid_counts=counts, dataset_ids=(1, 2))
        packet = encode(md, entry)
        reference = fmt.encode(md, entry)
        assert [(r[1], r[2], r[4]) for r in packet[2]] == [
            (r[-1].offset_bytes, r[-1].nbytes, r[-1].shape) for r in reference.rows]
        packets.append(packet)
    assert packets[0][2][1][1] == 48
    assert packets[1][2][1][1] == 16
    assert packets[2][2] == ()


def test_custom_format_is_not_bypassed():
    class Custom(MegatronRecordFormat):
        pass
    assert bind(Custom("custom")) is None
    fmt = MegatronRecordFormat("custom")
    fmt.encode = lambda *args: None
    assert bind(fmt) is None


def test_native_does_not_construct_public_descriptor(monkeypatch):
    import dmi_megatron_integration.records.format as module
    fmt = MegatronRecordFormat("direct")
    encode = bind(fmt)
    entry = ProducerPlanEntry.from_output(output_id=65536, output_spec=TransportSpec("probe"),
        output=HookOutput(torch.empty(1, 4)))
    md = MegatronRecordMetadata("m", "p", "fwd", "train", 1, 0, 0, 0, 0, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("old object construction")
    monkeypatch.setattr(module, "PayloadSlice", forbidden)
    monkeypatch.setattr(module, "RecordDescriptor", forbidden)
    assert len(encode(md, entry)[2]) == 1
