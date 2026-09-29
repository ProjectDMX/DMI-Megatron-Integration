"""Storage-side Signal declarations. Importing this module has no runtime side effects."""
from dataclasses import dataclass
from collections.abc import Callable, Mapping
import re


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError(f'Invalid identifier: {value!r}')
    return value


def check_keys(data, allowed, label):
    if not isinstance(data, Mapping):
        raise ValueError(f'{label} must be a mapping')
    extra = set(data) - set(allowed)
    if extra:
        raise ValueError(f'{label}: unsupported fields {sorted(extra)}')


def sources(query):
    return [query['from']] + [join['from'] for join in query.get('joins', [])]


def validate_query(query):
    check_keys(query, ('from', 'select', 'joins', 'where', 'scope', 'order_by'), 'input')
    if not isinstance(query.get('select', ['*']), list):
        raise ValueError('select must be a list')
    for source in sources(query):
        check_keys(source, ('table', 'signal', 'as'), 'from')
        if ('table' in source) == ('signal' in source):
            raise ValueError('from requires exactly one of table or signal')
        identifier(source.get('table', source.get('signal')))
        if 'as' in source:
            identifier(source['as'])
    for join in query.get('joins', []):
        check_keys(join, ('type', 'from', 'on'), 'join')
        if join.get('type', 'inner') not in ('inner', 'left', 'right', 'full', 'cross'):
            raise ValueError('Unsupported join type')
        if join.get('type') != 'cross' and not join.get('on'):
            raise ValueError('join requires on conditions')
    scope = query.get('scope', {})
    check_keys(scope, ('event', 'phase', 'layers', 'training_iteration'), 'scope')
    if 'event' in scope and scope['event'] != 'current':
        raise ValueError('scope.event must be current')
    if 'training_iteration' in scope:
        check_keys(scope['training_iteration'], ('through', 'from'), 'training_iteration')
        for value in scope['training_iteration'].values():
            if type(value) is not int or value < 0:
                raise ValueError('training_iteration bounds must be nonnegative integers')
    if 'layers' in scope and (not isinstance(scope['layers'], list) or any(type(x) is not int for x in scope['layers'])):
        raise ValueError('layers must be a list of global layer indices')


@dataclass(frozen=True)
class Output:
    kind: str
    name: str
    columns: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Signal:
    name: str
    inputs: tuple[dict, ...]
    trigger: dict
    transform: Callable
    outputs: tuple[Output, ...]
    cache_once: bool = False


class Row(dict):
    """Query row with dictionary and attribute access; no tensor stacking."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
