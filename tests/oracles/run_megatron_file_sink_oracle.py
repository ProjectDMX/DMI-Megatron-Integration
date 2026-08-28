"""Run tiny Megatron training with the test-only encoded-record file sink."""

from __future__ import annotations

import os
import time
from functools import partial

import pretrain_gpt as base_pretrain

import dmi_megatron_integration.startup as dmi_startup
from tests.support.megatron_file_sink import (
    MegatronTestFileSink,
    MegatronTestRecordEngine,
)


def _file_engine_factory(_cfg, _model_id, _record_format, _rank):
    root_dir = os.environ.get("DMI_TEST_FILE_SINK_DIR")
    if not root_dir:
        raise RuntimeError("DMI_TEST_FILE_SINK_DIR is required")
    sink = MegatronTestFileSink(root_dir)
    return MegatronTestRecordEngine(sink), sink


def main() -> None:
    original_setup = dmi_startup.setup_megatron_dmi

    def setup_with_file_sink(*args, **kwargs):
        if kwargs.get("engine_factory") is not None:
            raise RuntimeError("file-sink oracle received an existing engine factory")
        kwargs["engine_factory"] = _file_engine_factory
        return original_setup(*args, **kwargs)

    dmi_startup.setup_megatron_dmi = setup_with_file_sink

    now = time.time()
    base_pretrain.set_startup_timestamps(program_start=now, main_entry=now)
    base_pretrain.train_valid_test_datasets_provider.is_distributed = True
    base_pretrain.train_valid_test_datasets_provider.dmi_standard_dataset_provider = True

    base_pretrain.pretrain(
        base_pretrain.train_valid_test_datasets_provider,
        partial(base_pretrain.model_provider, base_pretrain.gpt_builder),
        base_pretrain.ModelType.encoder_or_decoder,
        base_pretrain.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=(
            base_pretrain.add_modelopt_args
            if base_pretrain.has_nvidia_modelopt
            else None
        ),
        store=None,
        get_embedding_ranks=base_pretrain.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
