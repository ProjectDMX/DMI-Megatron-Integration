"""Resolve declared Signal inputs to existing capture hooks before training starts.

This module does not import Megatron, initialize CUDA, or import user transforms.
It follows declared inputs, not the Python implementation of an analysis. Query
filters remain storage-side: capture is a conservative union of dependencies.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import re

from .config import parse_config
from .definition import identifier, sources
from ..hooks.selection import parse_hook_selection, resolve_phase_hook_selections


# Physical output names emitted by the startup hook installers. Multiple outputs
# may belong to one hook; resolving either output enables that existing producer.
OBSERVATION_HOOKS = {
    'router_probs_mean': 'router-summary',
    'router_logits': 'router-logits',
    'router_token_entropy_mean': 'router-entropy',
    'pre_drop_token_count': 'expert-counts',
    'post_drop_token_count': 'expert-counts',
    'router_topk_expert_ids': 'router-topk-expert-ids',
    'router_topk_weights': 'router-topk-weights',
    'hidden_states': 'hidden-states',
    'hook_resid_final': 'resid_final',
    'moe_input': 'moe-input',
    'moe_inverse_map': 'moe-inverse-map',
    'moe_packed_weighted_output': 'moe-packed-weighted-output',
    'moe_tp_complete_output': 'moe-tp-complete-output',
    'moe_tp_complete_output_token_indices': 'moe-tp-complete-output',
    'moe_tp_complete_output_expert_counts': 'moe-tp-complete-output',
    'vocab_logits': 'vocab-logits',
    'vocab_logits_topk_values': 'vocab-logits-topk',
    'vocab_logits_topk_indices': 'vocab-logits-topk',
    'lm_per_token_loss': 'token-loss',
    'lm_per_sample_loss': 'loss-summary',
    'lm_per_sample_loss_token_count': 'loss-summary',
    'grad_norm': 'grad-norm',
    'query_projection_weight': 'q-weights',
    'key_projection_weight': 'k-weights',
    'router_projection_weight': 'router-weights',
}
PHASES = frozenset(('train', 'valid', 'test'))


def _unloaded_transform(*args):
    raise RuntimeError('Capture planning does not execute analysis functions')


def _validate_transform(reference):
    if not isinstance(reference, str) or not re.fullmatch(
        r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*', reference
    ):
        raise ValueError('transform must be module:function or builtin:name')
    return _unloaded_transform


def _literal_names(predicate, label):
    """Require finite, explicit observation names; never guess a wildcard's cost."""
    if isinstance(predicate, str):
        values = [predicate]
    elif isinstance(predicate, Mapping) and set(predicate) == {'eq'}:
        values = [predicate['eq']]
    elif isinstance(predicate, Mapping) and set(predicate) == {'in'}:
        values = predicate['in']
        if not isinstance(values, list) or not values:
            raise ValueError(f'{label}: in requires a nonempty list')
    else:
        raise ValueError(f'{label}: use an explicit act_name string, eq, or in list')
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f'{label}: observation names must be nonempty strings')
    return set(values)


def _phase(value, label):
    if not isinstance(value, str) or value not in PHASES:
        raise ValueError(f'{label}: phase must be train, valid, or test')
    return frozenset((value,))


def compile_capture_plan(document, *, base_table='dmi_training_tensors'):
    """Return phase -> hook names for all root Signals and reachable helpers.

    Only the configured payload tables require hooks. Other tables are existing
    external/derived inputs or runtime metadata. Every payload-table input must
    name its observations explicitly. Unused on-demand helpers are not captured.
    Unscoped/history inputs capture all phases unless an explicit phase narrows
    them; a root trigger's cadence never reduces capture cadence.
    """
    identifier(base_table)
    registry = parse_config(document, transform_resolver=_validate_transform)
    payload_tables = {base_table, base_table + '_scalar_float', base_table + '_scalar_int'}
    plan = {phase: set() for phase in sorted(PHASES)}
    visited = set()

    def visit(signal, event_phases):
        key = (signal.name, event_phases)
        if key in visited:
            return
        visited.add(key)
        for query in signal.inputs:
            query_sources = sources(query)
            payload_sources = [source for source in query_sources if source.get('table') in payload_tables]
            scope = query.get('scope', {})
            phases = event_phases if scope.get('event') == 'current' else PHASES
            if 'phase' in scope:
                phases = phases & _phase(scope['phase'], signal.name)
            where = query.get('where', {})
            if not isinstance(where, Mapping):
                raise ValueError(f'{signal.name}: where must be a mapping')
            for source in query_sources:
                if 'signal' in source:
                    producer, _ = registry.outputs[source['signal']]
                    # Helpers execute against the root event, not the filtered
                    # rows of their caller. Do not push filters through Python.
                    visit(producer, event_phases)
                    continue
                if source not in payload_sources:
                    continue
                alias = source.get('as', source['table'])
                predicate = where.get(alias + '.act_name')
                if predicate is None and len(payload_sources) == 1:
                    predicate = where.get('act_name')
                names = _literal_names(predicate, f'{signal.name}/{alias}')
                unknown = names - OBSERVATION_HOOKS.keys() - {'iteration_attempt_status'}
                if unknown:
                    raise ValueError(f'{signal.name}: no registered capture hook for {sorted(unknown)}')
                hooks = {OBSERVATION_HOOKS[name] for name in names if name in OBSERVATION_HOOKS}
                for phase in phases:
                    plan[phase].update(hooks)

    roots = [signal for signal in registry.signals.values() if signal.trigger['event'] != 'on_demand']
    if not roots:
        raise ValueError('Signal capture configuration requires at least one root Signal')
    for root in roots:
        phases = PHASES
        if root.trigger['event'] == 'validation_end':
            phases = frozenset(('valid',))
        if 'phase' in root.trigger:
            phases = phases & _phase(root.trigger['phase'], root.name)
        if not phases:
            raise ValueError(f'{root.name}: trigger phase conflicts with its event')
        visit(root, phases)
    if not any(plan.values()):
        raise ValueError('No capture hooks resolved; check payload table names and root inputs')
    return {phase: frozenset(hooks) for phase, hooks in plan.items()}


def load_capture_plan(path, *, base_table='dmi_training_tensors'):
    import yaml
    with open(path) as stream:
        return compile_capture_plan(yaml.safe_load(stream), base_table=base_table)


def merge_capture_selections(plan, common='none', *, train=None, valid=None, test=None):
    """Union explicit capture selections with YAML requirements, phase by phase."""
    # Custom hook names can only be validated after the model is available.
    # Preserve them here; setup_megatron_dmi performs that validation as usual.
    declared = set().union(*(parse_hook_selection(value) for value in
                            (common, train, valid, test) if value is not None))
    manual = resolve_phase_hook_selections(
        common, train=train, valid=valid, test=test, additional_names=declared,
    )
    return {phase: ','.join(sorted(manual[phase] | plan[phase])) or 'none' for phase in plan}


def configure_signal_capture_args(args, *, environ=None):
    """Enable capture early in Megatron startup, before its DMI enable gates."""
    import os
    environ = os.environ if environ is None else environ
    path = getattr(args, 'dmi_signal_config', None) or environ.get('DMI_SIGNAL_CONFIG')
    if not path:
        return args
    base_table = getattr(args, 'dmi_clickhouse_table', None) or environ.get('DMI_CLICKHOUSE_TABLE', 'dmi_training_tensors')
    plan = load_capture_plan(path, base_table=base_table)
    common = getattr(args, 'dmi_hook_selection', None)
    if common is None:
        common = environ.get('DMI_HOOK_SELECTION', 'none')
    selections = merge_capture_selections(plan, common, **{
        phase: getattr(args, f'dmi_{phase}_hook_selection', None) for phase in PHASES
    })
    args.dmi_enable = True
    args.dmi_signal_config = str(path)
    args.dmi_hook_selection = 'none'
    for phase, selection in selections.items():
        setattr(args, f'dmi_{phase}_hook_selection', selection)
    return args


def main():
    parser = argparse.ArgumentParser(description='Preview YAML-derived hooks without starting training.')
    parser.add_argument('config')
    parser.add_argument('--table', default='dmi_training_tensors')
    args = parser.parse_args()
    plan = load_capture_plan(args.config, base_table=args.table)
    print(json.dumps({phase: sorted(hooks) for phase, hooks in plan.items()}, indent=2))


if __name__ == '__main__':
    main()
