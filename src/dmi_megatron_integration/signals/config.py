"""YAML loading without connecting to storage or importing Megatron startup."""
import importlib
from pathlib import Path
from .definition import Signal, Output, identifier, check_keys, validate_query
from .registry import Registry


def resolve_transform(reference):
    if not isinstance(reference, str) or ':' not in reference:
        raise ValueError('transform must be module:function or builtin:name')
    module, name = reference.split(':', 1)
    identifier(name)
    if module == 'builtin':
        from ..materialization import reconstruction
        function = getattr(reconstruction, name, None)
    else:
        function = getattr(importlib.import_module(module), name, None)
    if not callable(function):
        raise ValueError(f'Transform is not callable: {reference}')
    return function


def parse_config(document):
    check_keys(document, ('signals',), 'configuration')
    if not isinstance(document.get('signals'), list):
        raise ValueError('signals must be a list')
    signals = []
    for item in document['signals']:
        check_keys(item, ('name', 'inputs', 'trigger', 'transform', 'output', 'cache'), 'Signal')
        name = identifier(item['name'])
        trigger = item['trigger']
        check_keys(trigger, ('event', 'phase', 'every_n_iterations', 'training_iteration_min'), 'trigger')
        if trigger.get('event') not in ('iteration_end', 'validation_end', 'phase_end', 'on_demand'):
            raise ValueError(f'{name}: invalid trigger event')
        for key in ('every_n_iterations', 'training_iteration_min'):
            if key in trigger and (type(trigger[key]) is not int or trigger[key] < (1 if key == 'every_n_iterations' else 0)):
                raise ValueError(f'{name}: invalid {key}')
        cache = item.get('cache', {})
        check_keys(cache, ('mode',), 'cache')
        if cache and (cache.get('mode') != 'once' or trigger['event'] != 'on_demand'):
            raise ValueError('cache.mode: once requires an on_demand Signal')
        if not isinstance(item.get('inputs'), list) or not isinstance(item.get('output'), list) or not item['output']:
            raise ValueError('inputs and nonempty output must be lists')
        for query in item['inputs']:
            validate_query(query)
        outputs = []
        for output in item['output']:
            check_keys(output, ('table', 'signal', 'columns'), 'output')
            if ('table' in output) == ('signal' in output):
                raise ValueError('output requires exactly one destination')
            kind = 'table' if 'table' in output else 'signal'
            columns = []
            for column in output['columns']:
                check_keys(column, ('name', 'type'), 'column')
                columns.append((identifier(column['name']), column['type']))
            if not columns or len({c[0] for c in columns}) != len(columns):
                raise ValueError('output requires nonempty, unique columns')
            if cache and {'run_id', 'signal_name'} & {c[0] for c in columns}:
                raise ValueError('cache columns run_id and signal_name are reserved')
            from .storage import physical_columns
            physical_columns(columns)  # Validate types before creating the worker.
            outputs.append(Output(kind, identifier(output[kind]), tuple(columns)))
        signals.append(Signal(name, tuple(item['inputs']), dict(trigger), resolve_transform(item['transform']), tuple(outputs), bool(cache)))
    return Registry(signals)


def load_config(path):
    import yaml
    with Path(path).open() as stream:
        return parse_config(yaml.safe_load(stream))
