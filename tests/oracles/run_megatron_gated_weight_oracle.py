"""Retain physical model QKV tensors before DMI weight snapshots for comparison."""
import os
from pathlib import Path
import runpy

import torch
from dmi_megatron_integration.startup import MegatronDMIHandle


def main():
    output = Path(os.environ['DMI_WEIGHT_ORACLE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    original = MegatronDMIHandle.emit_qk_weights

    def observe(handle, *, model_state_iteration_id, allow_zero=False):
        from megatron.core import parallel_state
        reference = {}
        for capture in handle.weight_captures:
            if capture.act_name == 'query_projection_weight':
                # Full physical fused parameter, before using any Q/K selection map.
                reference[capture.layer_no] = capture.source.get().detach().cpu().clone()
        torch.save(dict(tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                        weights=reference), output / (
            f'rank{torch.distributed.get_rank()}_step{model_state_iteration_id}.pt'))
        return original(handle, model_state_iteration_id=model_state_iteration_id,
                        allow_zero=allow_zero)

    MegatronDMIHandle.emit_qk_weights = observe
    runpy.run_module('pretrain_gpt', run_name='__main__')


if __name__ == '__main__':
    main()
