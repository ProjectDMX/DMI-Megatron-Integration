# DMI-Megatron-Integration

DMI integration for the ProjectDMX Megatron-LM-DMI fork based on NVIDIA Megatron-LM `core_v0.17.1`.

The integration package version is `0.17.1` and requires DMI `>=1.2.0,<2.0`. The matching Megatron fork is pinned as the `third_party/megatron-lm` Git submodule.

For a source setup, clone this repository recursively and follow the [installation guide](docs/install.md). The guide records the tested Python, PyTorch, CUDA, and Transformer Engine stack and installs the pinned fork rather than an unrelated `megatron-core` release.

## Hook selection by training phase

`--dmi-hook-selection` supplies the common selection. Override a phase with
`--dmi-train-hook-selection`, `--dmi-valid-hook-selection`, or
`--dmi-test-hook-selection`. Each override replaces the common selection for
that phase; omission inherits it. Use `none` for an empty selection.

For example:

```bash
--dmi-hook-selection router-logits,q-weights,k-weights \
--dmi-valid-hook-selection hidden-states,router-logits,loss-summary \
--dmi-test-hook-selection none
```

Disabled hooks are skipped before preprocessing and record preparation. Hooks
needed by any phase are installed at startup. This selection is fixed before
graph capture. It filters existing firing sites: weights still emit only at
initialization and before training updates, and gradient norms remain training
signals. Phase/attempt metadata remains enabled even with `none`.

If phase selections differ, local layer/partial CUDA graphs run validation and
test eagerly instead of reusing training graphs. Rank 0 warns once at the first
fallback. Training retains its cached graphs. Identical phase selections retain
existing behavior. TE graphs already use the non-TE-graph path for evaluation;
full-iteration graphs keep their separate phase-specific captures. Existing GPU
recomputation gates are unchanged.

## Per-token loss hook

Select `--dmi-hook-selection token-loss` with DMI enabled. This separate hook
records `lm_per_token_loss` directly after Megatron's cross-entropy returns
`[batch, sequence]` losses. Each sample record contains its sequence of losses,
before masking or averaging; the recorded token range identifies its valid
prefix. For ordinary next-token cross-entropy, `log P(target) = -token_loss`.
No normalization is recomputed, and the existing `loss-summary` mean/count
outputs remain independently selectable (for example, `token-loss,loss-summary`).

The hook emits on TP rank 0 of the last PP stage for every DP replica, including
folded EP layouts. It currently supports dense batches with CP=1. The raw loss
tensor keeps Megatron's output dtype; the hook performs no cast.

## Parameter-weight hooks

Select `--dmi-hook-selection q-weights,k-weights,router-weights` (each is
independently selectable). Capture runs once **before** the optimizer update,
after the final forward/backward attempt, outside CUDA graphs. It records the
weights used by that iteration even when the optimizer skips its update.
Initial/resume snapshots have the separate attempt identity -1.

Each rank captures only assigned bytes: retained model-weight shards for FSDP,
or disjoint slices among actual replicas for non-FSDP. No additional all-gather
is performed. TP Q/K layout is preserved, including TP greater than KV head
count and empty local K portions. Capture does not depend on microbatch count.
Non-quantized Megatron FSDP and Torch FSDP2 storage mappings are implemented;
upstream restrictions on distributed/graph combinations still apply. Gated
attention and quantized FP8/FP4 weights remain unsupported.

The raw records contain packed byte fragments. Use the single on-demand
`weight_matrices` Signal (`builtin:merge_weight_shards`) to obtain complete Q,
K, and router matrices through `complete_weight_matrices`. It merges rank
contributions using `capture_topology.weight_layout_json`, retaining iteration,
attempt, layer and weight name. See [Signal configuration](docs/SIGNALS.md) and
[the example](examples/signals/reconstruction.yaml). The main tensor table schema
is unchanged. Existing schema-v3 capture tables need the additional
`weight_layout_json String` column in their `capture_topology` table before a
new weight-capture run; old full-matrix captures use the legacy reader.

## Recurring D2H windows

With DMI enabled, opt in with `--dmi-recurring-d2h-windows` (or
`DMI_RECURRING_D2H_WINDOWS=true`). The default is off; PP=1 forces it off.
This integration supports eager, non-interleaved 1F1B with the standard PP
communicator. Interleaved/VPP runs warn once per process and use ordinary
batching. Multi-module pipelines are rejected when windows are active.
PP with full-iteration CUDA Graphs is not supported by this integration: the
vendored Megatron benchmark skips it because pipeline communication fails
during capture.

Each rank installs its pattern before the first training schedule call and
redefines it when `(PP size, PP rank, microbatch count)` changes. A window opens
before an eligible receive or combined send/receive and closes at the next
forward/backward computation entry. DMI-core learns transfer sizes within
these windows; terminal capacity fallback continues with ordinary batching.

| CLI option | Default |
|---|---|
| `--dmi-d2h-window-minimum-record-probe-retry-interval-occurrences` | 4 |
| `--dmi-d2h-window-timing-revalidation-retry-interval-occurrences` | 4 |
| `--dmi-d2h-window-capacity-flush-fallback-threshold` | 3 |
| `--dmi-d2h-window-capacity-flush-count-reset-interval-periods` | 32 |
| `--dmi-d2h-window-debug` | off |

Each option also has an uppercase environment equivalent with hyphens replaced
by underscores, such as
`DMI_D2H_WINDOW_TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES`. Explicit
`MegatronDMIConfig` takes precedence over CLI, environment, then defaults.
Boolean environment values `0` and `false` disable the setting. The reset
interval is measured in full pattern periods (`32 * T` by default).

Debugging prints rank-local pattern definitions with iteration/attempt,
`(P, r, M)`, window count, period, and acceptance, plus DMI-core transfer
decisions and completion results. Enable it with `--dmi-d2h-window-debug` or
`DMI_D2H_WINDOW_DEBUG=true`.
