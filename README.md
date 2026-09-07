# DMI-Megatron-Integration

DMI integration for the ProjectDMX Megatron-LM-DMI fork based on NVIDIA Megatron-LM `core_v0.17.1`.

The integration package version is `0.17.1` and requires DMI `>=1.2.0,<2.0`. The matching Megatron fork is pinned as the `third_party/megatron-lm` Git submodule.

For a source setup, clone this repository recursively and follow the [installation guide](docs/install.md). The guide records the tested Python, PyTorch, CUDA, and Transformer Engine stack and installs the pinned fork rather than an unrelated `megatron-core` release.

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
