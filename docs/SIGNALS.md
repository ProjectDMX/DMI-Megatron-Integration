# Storage-side Signals

Signals are implemented in `src/dmi_megatron_integration/signals/`. They run on a
CPU worker. Existing Hook configuration controls capture and preprocessing;
Signals do not accept hook inputs or run inside the training process.

## Run the worker

```bash
python -m dmi_megatron_integration.signals monitoring.yaml \
  --run-id RUN --database default --table dmi_training_tensors \
  --producer-ranks 0 1 --num-layers 16
```

Use the same model/run ID and base table as capture. Supply **all capture runtime
ranks**, including ranks with zero payloads, from the known launch topology.
`--num-layers` resolves negative layer indices. Connection options accept
`--host`, `--port`, `--user`, and `--database`; password comes from
`DMX_DB_PASSWORD`. Put user transform modules on `PYTHONPATH`. The worker prints
Grafana SELECT statements for persisted outputs at startup.

One worker owns a run's Signal configuration. This implementation does not
coordinate concurrent workers or promise exactly-once application writes across
a process crash. Successful roots are remembered within the worker process;
`cache.mode: once` results survive worker restarts in ClickHouse.

## YAML and transform contract

See [`examples/signals/monitoring.yaml`](../examples/signals/monitoring.yaml) and
[`example_transforms.py`](../examples/signals/example_transforms.py).

- Each `inputs` item is one structured SELECT and one positional function
  argument, containing a list of rows. Fields support `row.name` and `row['name']`.
- `from: {table: name}` and `from: {signal: name}` use different namespaces.
  Inputs can join multiple real/virtual sources with explicit aliases and `on`
  comparisons. Virtual references must resolve to `on_demand` producers,
  including when used in joins. Forward references work; cycles are rejected.
- `select` accepts column names, `*`, `alias.*`, and
  `{column: alias.column, as: result_name}`. Name an entire logical Tensor column
  to select its dtype/shape/bytes representation and decode it into a CPU tensor.
- `where` accepts equality or `{eq, ne, lt, le, gt, ge, in}` predicates. Values
  are bound as parameters. `order_by` is a list of column names; otherwise row
  order is unspecified.
- `scope.event: current` restricts captured inputs to the root's iteration and
  accepted attempt, or phase batch range. `scope.phase` and `scope.layers`
  filter stored coordinates. Omitted layers mean all layers. `[-1]` means the
  final zero-based global layer.
- `scope.training_iteration: {through: k}` is inclusive. For captured validation
  records, the query uses iteration metadata to map phase-local batch IDs to
  training iterations. It includes advancing attempts only. Each producer has
  its own explicit input scopes; consumer filters are not inherited.
- Return a tuple/list with **one row collection per `output` mapping**. A row is
  either a mapping with exactly the declared fields or a positional sequence in
  column order. A single-output transform returns `(rows,)`.
- Outputs can mix `table:` and `signal:` destinations. Ordinary virtual results
  stay in memory. Output columns are explicit; the framework does not inject
  execution coordinates into user rows. Carry any desired coordinates through
  the numerical function's output.
- Supported result types: String, Bool, signed/unsigned 8/16/32/64-bit integers,
  Float32/64, Array/Nullable of supported SQL types, and Tensor. Tensor expands
  into `<name>_dtype`, `<name>_shape`, `<name>_bytes`; returned tensors must be CPU
  tensors. Scalars/arrays use the integration's ClickHouse writer.

The root trigger guarantees readiness for all recursively needed inputs by
contract. No additional dependency readiness traversal runs. The worker invokes
on-demand transforms only when requested and reuses each producer's full result
within that root execution.

Supported root events are `iteration_end`, `validation_end`, and `phase_end`.
Optional `phase`, `every_n_iterations`, and inclusive `training_iteration_min`
filter root events. For training-only iteration monitoring, specify
`phase: train`. `on_demand` is explicit and cannot launch as a root.

## One-shot cache

Declare `cache: {mode: once}` on an on-demand Signal. Each output has a
`<output_name>_cache_once` table containing the declared columns plus
`run_id String` and `signal_name String`. The key is `(run_id, signal_name)`;
there is no configuration hash. Keep that Signal's configuration fixed in a run.

The worker caches the full result before consumer filtering. An internal
`<base>_signal_cache_once_complete` table records successful completion and row
counts, including empty outputs. Partial writes without a completion marker are
cleared before a retry. A cache hit bypasses the producer's input queries and
numerical function. Ordinary virtual results are never persisted this way.

## Capture metadata and readiness

Capture schema version 3 adds `producer_rank` to tensor and scalar records and
adds `<base>_iteration_metadata` and `<base>_phase_metadata`. The existing
replicated `iteration_attempt_status` scalar remains for older readers; it is
excluded from payload readiness counts.

Every rank publishes one iteration metadata entry per attempt. `status=1`
advances, `0` retries, `-1` aborts; `weights_updated` separately records optimizer
success. Validation emits attempt zero with no weight update. Counts are taken
from scheduled record descriptors **after sample splitting**, including scalar
captures, and not from successful database writes. Graph-plan capture itself
is excluded; live graph replay contributes. Initial/resume snapshots are outside
attempt counting.

Iteration readiness uses one repeatedly polled SQL statement: all expected
ranks must agree on the advancing attempt, duplicate metadata must agree, and
unique payload counts must exactly equal the expected counts. Phase readiness
uses one statement over the declared batch range, including **all attempts**;
only duplicate insertion of the same identity is deduplicated. A late earlier
batch therefore prevents phase readiness even when the final batch is stored.
Training publishes metadata asynchronously and does not poll the database.

Use a **new base table** for schema-v3 capture. Old tables are not silently
altered, and historical runs cannot supply the new completeness metadata.
`MegatronTrainingReader` continues reading legacy schema-v2 payload tables.

## Reconstruction

Configure reconstruction as an on-demand Signal with virtual outputs. Functions
are shared by operation; a separate implementation per hook is unnecessary.
Use `transform: builtin:<function>` with these positional inputs:

| Function | Inputs | Hook outputs and operation |
| --- | --- | --- |
| `merge_token_shards` | payload rows | Hidden states; final hidden states; MoE inputs; raw router logits. Concatenate explicit token intervals. |
| `merge_vocab_shards` | payload rows; capture topology | Raw vocabulary logits. Concatenate vocabulary partitions in TP order. |
| `merge_weight_shards` | payload rows; capture topology | Q, K, and router parameter weights. Merge FSDP shards, replica-assigned slices, and TP partitions into complete matrices. |
| `merge_vocab_topk` | value rows; index rows; capture topology | Merge shard-local candidates into global top-k values and vocabulary IDs. |
| `merge_routing_shards` | payload rows; capture topology | Selected expert IDs; routing weights. Concatenate TP sequence slices and crop to the valid-token interval. |
| `merge_token_means` | payload rows | Router probability mean; token entropy mean. Weight means by each interval's valid-token count. |
| `sum_token_counts` | payload rows | Pre/post-drop expert counts. Sum counts over disjoint token intervals. |
| `reconstruct_expert_outputs` | payload rows; EP manifest | Weighted expert outputs + inverse map + selected expert IDs. Restore source-token order and combine ETP partial outputs. |

These names replace the earlier `reconstruct_hidden_states`,
`reconstruct_vocab_logits`, `reconstruct_router_weights`, and `reconstruct_ep`
builtins. See [`examples/signals/reconstruction.yaml`](../examples/signals/reconstruction.yaml)
for shared token merging and the single `weight_matrices` declaration. Applications consume their
virtual outputs using `from: {signal: ...}`.

### Output contract and grouping

The six single-payload `merge_*`/`sum_*` functions return `(rows,)`. Every row has
`model_id`, `phase`, `global_batch_id`, `attempt_id`, `microbatch_id`, `layer_no`,
`direction`, `dp_rank`, `dataset_id`, `sample_index`, `invocation_id`, `act_name`,
`token_start`, `token_end`, and `value`. `value` is a CPU tensor, except scalar
means/counts remain scalars. Declare the appropriate output type in YAML.

A call can contain multiple hook outputs. The functions group by output name and
all execution/sample identities before merging. DP samples remain separate; PP
layers retain their global layer numbers. Physical `producer_rank` and
`shard_rank` are removed from the resulting rows. Retries and distinct
invocations are never merged. Duplicate delivery of the same shard is ignored;
conflicting duplicates, internal interval gaps/overlaps, and incompatible
shapes fail rather than silently combining incorrect records.

`merge_vocab_topk` takes one matched value/index output pair. It returns the same
coordinates except `act_name`, and two tensors: `values` and `indices`, each
`token × k`. Use `row["values"]` because `values` is also a dictionary method.
K is the captured local K. Ties among captured candidates prefer lower global
IDs; ties discarded by local GPU top-k cannot be recovered. Raw/top-k vocabulary
merging retains the captured sequence padding: these hooks do not capture
valid-token ranges (`token_start/end` are record coordinates, not a padding mask).

`reconstruct_expert_outputs` retains its source-sample output contract:
`weighted_outputs` (`token × top_k × hidden`), `expert_ids` (`token × top_k`),
`token_indices`, and execution/sample coordinates. Capture `router-topk-expert-ids`
independently of `router-topk-weights`; the existing `router-topk` enables both.
CKA additionally reads actual routing weights through `merge_routing_shards`.
No raw router logits are needed to recover routing.

### Layout metadata

Capture topology is read from `<base>_capture_topology`; EP capture is not needed
for the TP helpers. Its `vocab_partition_size Int64` is the actual vocabulary
width presented to an active vocabulary hook before top-k selection: padded
vocabulary size divided by TP size for sharded output, the full padded size for
replicated output, and zero when the rank has no active vocabulary capture.
`merge_vocab_topk` requires this field to recover global IDs. It also lets the
vocabulary helpers distinguish replicated capture from missing TP shards.

For hidden states and the other token-interval hooks, SP-on capture uses the
existing local sequence slice. With SP off, capture selects separate slices
from the replicated tensor; the same merge applies. A single replicated record
passes through the interval helper. Root readiness and input queries must supply
all relevant rows; interval checks alone cannot detect a missing final slice.
The current capture/reconstruction paths do not provide general CP support.

Topology rows contain `manifest_json`. When EP capture freezes its topology
manifest, rank zero also stores it in `<base>_topology_manifest`. The on-demand
query can read that table explicitly. The supported EP path remains CP=1,
all-to-all, non-fused permutation, dropless and unpadded.

`regroup_ep_by_expert(rows)` is an optional Python helper retaining source-token
identities. CKA must intersect those identities so both expert matrices contain
the same tokens. Unweighting requires nonzero captured weights and ordinary
output weighting; BF16 reconstruction has rounding error. Numerical CKA,
pathway, LiD and selection logic belong to application transforms.

## Tests

CPU unit and real ClickHouse tests cover YAML validation, joins across real and
virtual sources, multiple arguments/outputs, cache reuse after restart, empty
caches, tensor storage, history cutoffs, retry/phase readiness and native sink
metadata publication. CPU synthetic EP tests compare three-input reconstruction
against the existing four-input oracle and compare matched-token CKA against
known unweighted expert outputs. These are not GPU training benchmarks.

```bash
DMI_TEST_CLICKHOUSE=1 PYTHONPATH=src python -m pytest -q \
  tests/test_signals.py tests/test_signal_capture_metadata.py \
  tests/test_signal_reconstruction.py tests/test_signal_shared_reconstruction.py
```

### Complete weight matrices

`weight_matrices` exposes the on-demand virtual output `complete_weight_matrices`.
It uses `merge_weight_shards(payload_rows, topology_rows)` for Q, K, and router
weights together. Filter `act_name` when an analysis needs only some weights.
Each result has `model_id`, `phase`, `global_batch_id`, `attempt_id`, `layer_no`,
`act_name`, and CPU tensor `value`. Q/K are projection-output × hidden-input;
router weights are expert × hidden-input. There is no rank or sample dimension.
The root Signal guarantees input readiness; the helper does not persist another
copy of the tensors. Missing ranks/byte ranges and conflicting duplicates fail.

Weight snapshots are captured **before** the optimizer update, after the final
forward/backward attempt. Skipped updates still have a snapshot. Initial/resume
snapshots use attempt ID -1, separate from training attempt snapshots. Capture
uses physical model-weight shards for FSDP and assigned slices of replicas for
non-FSDP. Quantized weights are unsupported. The payload is packed UInt8 bytes;
`capture_topology.weight_layout_json` records dtype, complete shape and rank
fragment offsets. Use the merged output, not the raw bytes, for matrix analyses.
The main tensor table schema is unchanged. For an existing schema-v3 capture
table, add the metadata column before starting a new weight-capture run:

```sql
ALTER TABLE dmi_training_tensors_capture_topology
ADD COLUMN IF NOT EXISTS weight_layout_json String DEFAULT '';
```

Use your actual capture-topology table name. Startup validates the schema; it
does not migrate existing tables automatically. Adding this column does not
retroactively supply layouts for old captures. Old full-matrix captures remain
readable with the legacy `merge_projection_shards` helper.

### Sampling expert outputs by source

Pass `--dmi-hook-config hooks.yaml` (or set `DMI_HOOK_CONFIG`) alongside the
usual hook selection. For example:

```yaml
hooks:
  moe_packed_weighted_output:
    source_sampling:
      function: dmi_megatron_integration.hooks.source_sampling.round_robin
      args:
        count: 2
        offset: 0
```

`count` sources are selected from each expert dispatch group. Source positions
are ordered by source expert-TP rank, then source EP rank. With `U = ETP * EP`,
the first position is `((global_batch_id - 1) * count + offset) % U`; the next
`count - 1` positions wrap around. EDP groups apply the same rule independently.
The same iteration, including its retries and recomputation, keeps its selection.
Omitting `source_sampling` retains full capture. Configuring it on a hook that
does not support it emits a warning and ignores that block.

A custom selector is a dotted Python callable accepting `iteration`,
`num_sources`, and its YAML `args`. It must deterministically return a fixed,
nonzero number of distinct source positions in `[0, num_sources)`. The callable
must be available both during capture and reconstruction. Arguments and the
iteration convention are persisted in the existing EP topology manifest.

The hook preprocesses `[local_expert, source]` row counts into GPU segment
ranges. The existing segmented producer copies those ranges directly into the
ring; it does not first gather activations into another tensor. Reservation
uses the exact selected CPU counts from Megatron's existing metadata transfer.
For one local expert, those counts join the same transfer and synchronization.
Destinations with no selected rows still publish an empty record.

`reconstruct_expert_outputs` reads the policy from the manifest and returns only
selected source tokens, retaining their original token and sample identities.
Keep the inverse-map and selected-expert-ID hooks enabled on every producer.
Destination ETP partial outputs are still captured and summed; source sampling
does not discard destination contributions. DP samples remain separate and PP
layers retain their global identity. Missing policy metadata means full capture,
so older manifests remain readable.

Both variants require CP=1 and dropless, unpadded, non-fused AlltoAll dispatch.
The expert-output hook stays eager; surrounding attention, router, and MoE
preprocessing may use supported partial CUDA graphs. Full expert-output graph
capture is not supported. Inactive hooks do not run the selector or retain/copy
additional source counts.


### Capture-layer selection

`--dmi-layer-indices 0 4 7` captures layer-based hooks only at those zero-based
**global** layer numbers, across pipeline stages and virtual pipeline chunks.
Indices must be nonempty, unique, and in range; their order is immaterial.
This option requires `--dmi-layer-stride 1`. Without explicit indices, the
existing stride behavior is unchanged. Python configuration uses
`MegatronDMIConfig(layer_indices=(0, 4, 7))`.

The selection intersects hook-specific layer restrictions and applies to both
activation hooks and weight capture, before graph recording. Unlayered outputs
(final residuals, vocabulary logits, losses, and status records) retain their
existing placement. Phase-specific hook selection still applies independently.
A rank or virtual pipeline chunk may have no retained layer hooks. Full and
sampled EP reconstruction uses matching retained layers with the complete
topology manifest. A storage-side Signal filter cannot enable an uncaptured
layer. Layer selection is fixed at startup.
