"""Structured ClickHouse queries, virtual external tables, and derived Signal rows."""
from collections.abc import Mapping
import re
from .definition import Row, identifier


def quote(name):
    return '.'.join('`' + identifier(part) + '`' for part in name.split('.'))


def column_type(value):
    if value in {'Tensor', 'String', 'Bool', 'Float32', 'Float64'} or re.fullmatch(r'U?Int(8|16|32|64)', str(value)):
        return value
    match = re.fullmatch(r'(Array|Nullable)\((.*)\)', str(value))
    if match and column_type(match[2]) != 'Tensor':
        return value
    raise ValueError(f'Unsupported Signal column type: {value!r}')


def physical_columns(columns):
    result = []
    for name, dtype in columns:
        identifier(name)
        column_type(dtype)
        if dtype == 'Tensor':
            result.extend([(name + '_dtype', 'String'), (name + '_shape', 'Array(Int64)'), (name + '_bytes', 'String')])
        else:
            result.append((name, dtype))
    if len({name for name, _ in result}) != len(result):
        raise ValueError('Tensor serialization column names collide')
    return result


def normalize_rows(rows, columns):
    names = [name for name, _ in columns]
    result = []
    for row in rows:
        if isinstance(row, Mapping):
            if set(row) != set(names):
                raise ValueError(f'Output row columns must be exactly {names}, got {list(row)}')
            result.append(Row((name, row[name]) for name in names))
        else:
            if len(row) != len(names):
                raise ValueError('Output row width disagrees with columns')
            result.append(Row(zip(names, row)))
    return result


def _wire_value(value):
    # External tables share strings_as_bytes with the SELECT response. Encoding
    # strings explicitly also preserves binary tensor columns without ambiguity.
    if isinstance(value, str):
        return value.encode('utf-8')
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    return value


def encode_rows(rows, columns):
    result = []
    for row in normalize_rows(rows, columns):
        cells = []
        for name, dtype in columns:
            value = row[name]
            if dtype == 'Tensor':
                import torch
                if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                    raise ValueError('Storage transforms must return CPU tensors')
                value = value.detach().contiguous()
                cells.extend([str(value.dtype), list(value.shape), value.reshape(-1).view(torch.uint8).numpy().tobytes()])
            else:
                cells.append(value)
        result.append(tuple(_wire_value(cell) for cell in cells))
    return result


def decode_rows(data, columns):
    result = []
    names = [c[0] for c in columns]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate query result columns; use explicit select aliases")
    for values in data:
        row = Row(zip(names, values))
        # Support native captured dtype/shape/bytes and derived <name>_* fields.
        for name in list(names):
            if name == 'dtype' and {'shape', 'bytes'} <= set(names):
                stem, field = '', 'value'
            elif name.endswith('_dtype') and {name[:-6] + '_shape', name[:-6] + '_bytes'} <= set(names):
                stem, field = name[:-5], name[:-6]
            else:
                continue
            from ..records.reader import MegatronTrainingReader
            dtype = row[stem + 'dtype']
            shape = row[stem + 'shape']
            payload = row[stem + 'bytes']
            if not payload:
                import torch
                tensor = torch.empty(shape, dtype=MegatronTrainingReader.bytes_to_torch_dtype(dtype))
            else:
                tensor = MegatronTrainingReader.torch_decode(dtype, shape, payload).clone()
            row[field] = tensor
            for suffix in ('dtype', 'shape', 'bytes'):
                del row[stem + suffix]
        def text(value):
            if isinstance(value, bytes):
                return value.decode('utf-8')
            if isinstance(value, list):
                return [text(item) for item in value]
            return value
        for name, value in list(row.items()):
            row[name] = text(value)
        result.append(row)
    return result


class ClickHouseStorage:
    def __init__(self, client, *, database='default', base_table='offload', num_layers=None):
        self.client = client
        self.database = identifier(database)
        self.base_table = identifier(base_table)
        self.num_layers = num_layers
        self._schemas = {}

    def table(self, name):
        return quote(self.database) + '.' + quote(identifier(name))

    def schema(self, name):
        if name not in self._schemas:
            rows = self.client.execute('DESCRIBE TABLE ' + self.table(name))
            self._schemas[name] = tuple((row[0], row[1]) for row in rows)
        return self._schemas[name]

    def create(self, name, columns):
        physical = physical_columns(columns)
        ddl = ', '.join(quote(n) + ' ' + t for n, t in physical)
        self.client.execute(f'CREATE TABLE IF NOT EXISTS {self.table(name)} ({ddl}) ENGINE=MergeTree ORDER BY tuple()')
        if tuple(self.schema(name)) != tuple(physical):
            raise ValueError(f'Schema mismatch for Signal table {name}')

    def write(self, output, rows):
        self.create(output.name, output.columns)
        self._insert(output.name, rows, output.columns)

    def _insert(self, name, rows, columns):
        data = encode_rows(rows, columns)
        if data:
            names = ', '.join(quote(n) for n, _ in physical_columns(columns))
            self.client.execute(f'INSERT INTO {self.table(name)} ({names}) VALUES', data, settings={'async_insert': 0})

    def compile_query(self, query, event, virtuals):
        params, external, aliases = {}, [], {}
        def bind(value):
            key = f'p{len(params)}'
            params[key] = value
            return f'%({key})s'
        def source_sql(source):
            name = source.get('table', source.get('signal'))
            alias = source.get('as', name)
            identifier(alias)
            if alias in aliases:
                raise ValueError(f'Duplicate query alias: {alias}')
            if 'table' in source:
                schema = self.schema(name)
                sql = self.table(name)
            else:
                output, rows = virtuals[name]
                schema = physical_columns(output.columns)
                temporary = f'_signal_input_{len(external)}'
                external.append({'name': temporary, 'structure': schema, 'data': encode_rows(rows, output.columns)})
                sql = quote(temporary)
            aliases[alias] = (source, {n for n, _ in schema})
            return sql + ' AS ' + quote(alias)
        sql_from = source_sql(query['from'])
        for join in query.get('joins', []):
            kind = join.get('type', 'inner').upper()
            sql_from += f' {kind} JOIN ' + source_sql(join['from'])
            if kind != 'CROSS':
                predicates = []
                for condition in join['on']:
                    if condition['op'] not in ('=', '!=', '<', '<=', '>', '>='):
                        raise ValueError('Unsupported join comparison')
                    predicates.append(f"{quote(condition['left'])} {condition['op']} {quote(condition['right'])}")
                sql_from += ' ON ' + ' AND '.join(predicates)
        def selected(selection):
            field = selection['column'] if isinstance(selection, Mapping) else selection
            rename = selection.get('as') if isinstance(selection, Mapping) else None
            if isinstance(selection, Mapping) and set(selection) != {'column', 'as'}:
                raise ValueError('select mapping requires column and as')
            if field == '*':
                return ['*']
            if field.endswith('.*'):
                return [quote(field[:-2]) + '.*']
            parts = field.split('.')
            alias = parts[0] if len(parts) == 2 else None
            name = parts[-1]
            candidates = [(a, cols) for a, (_, cols) in aliases.items() if alias is None or a == alias]
            tensors = []
            for a, cols in candidates:
                prefix = '' if name == 'value' and {'dtype','shape','bytes'} <= cols else name + '_'
                if {prefix + x for x in ('dtype','shape','bytes')} <= cols:
                    tensors.append((a,prefix))
            if tensors:
                if len(tensors) != 1:
                    raise ValueError(f'Ambiguous tensor selection: {field}')
                a,prefix=tensors[0]
                target = (rename or name) + '_'
                return [quote(a + '.' + prefix + x) + ' AS ' + quote(target + x) for x in ('dtype','shape','bytes')]
            return [quote(field) + (' AS ' + quote(identifier(rename)) if rename else '')]
        conditions = []
        for name, value in query.get('where', {}).items():
            if isinstance(value, Mapping):
                for op, operand in value.items():
                    operators = {'eq': '=', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'in': 'IN'}
                    if op not in operators:
                        raise ValueError(f'Unknown predicate: {op}')
                    conditions.append(quote(name) + ' ' + operators[op] + ' ' + bind(tuple(operand) if op == 'in' else operand))
            elif value is None:
                conditions.append(quote(name) + ' IS NULL')
            else:
                conditions.append(quote(name) + ' = ' + bind(value))
        scope = query.get('scope', {})
        for alias, (source, columns) in aliases.items():
            def field(name):
                return quote(alias + '.' + name)
            if 'model_id' in columns:
                conditions.append(field('model_id') + ' = ' + bind(event.run_id))
            elif 'run_id' in columns:
                conditions.append(field('run_id') + ' = ' + bind(event.run_id))
            if 'phase' in scope and 'phase' in columns:
                conditions.append(field('phase') + ' = ' + bind(scope['phase']))
            if 'layers' in scope and 'layer_no' in columns:
                layers = []
                for index in scope['layers']:
                    if index < 0:
                        if self.num_layers is None:
                            raise ValueError('Negative layer indices require num_layers')
                        index += self.num_layers
                    if index < 0 or (self.num_layers is not None and index >= self.num_layers):
                        raise ValueError('Layer index outside model')
                    layers.append(index)
                conditions.append(field('layer_no') + ' IN ' + bind(tuple(layers)))
            if scope.get('event') == 'current':
                if 'phase' in columns:
                    conditions.append(field('phase') + ' = ' + bind(event.phase))
                if 'global_batch_id' in columns:
                    if event.event == 'iteration_end':
                        conditions.append(field('global_batch_id') + ' = ' + bind(event.global_batch_id))
                        if 'attempt_id' in columns:
                            conditions.append(field('attempt_id') + ' = ' + bind(event.attempt_id))
                    else:
                        conditions.extend([field('global_batch_id') + ' >= ' + bind(event.start), field('global_batch_id') + ' < ' + bind(event.end)])
                elif 'training_iteration_id' in columns:
                    conditions.append(field('training_iteration_id') + ' = ' + bind(event.training_iteration))
            if 'training_iteration' in scope:
                bounds = scope['training_iteration']
                if 'training_iteration_id' in columns:
                    target = field('training_iteration_id')
                elif {'phase', 'global_batch_id', 'model_id'} <= columns:
                    # Validation IDs are phase-local. Resolve history via metadata
                    # inside this same SQL, without a correlated subquery.
                    meta = self.table(self.base_table + '_iteration_metadata')
                    predicates = ['model_id = ' + bind(event.run_id), 'status = 1']
                    for bound, op in [('through', '<='), ('from', '>=')]:
                        if bound in bounds:
                            predicates.append('training_iteration_id ' + op + ' ' + bind(bounds[bound]))
                    keys = ['phase', 'global_batch_id'] + (['attempt_id'] if 'attempt_id' in columns else [])
                    conditions.append('tuple(' + ', '.join(field(k) for k in keys) + ') IN (SELECT ' + ', '.join(quote(k) for k in keys) + ' FROM ' + meta + ' WHERE ' + ' AND '.join(predicates) + ')')
                    continue
                else:
                    raise ValueError('training_iteration scope needs iteration metadata or a training_iteration_id column')
                for name, op in [('through', '<='), ('from', '>=')]:
                    if name in bounds:
                        conditions.append(target + ' ' + op + ' ' + bind(bounds[name]))
        sql = 'SELECT ' + ', '.join(expr for x in query.get('select', ['*']) for expr in selected(x)) + ' FROM ' + sql_from
        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)
        if query.get('order_by'):
            sql += ' ORDER BY ' + ', '.join(quote(x) for x in query['order_by'])
        return sql, params, external

    def read(self, query, event, virtuals):
        sql, params, external = self.compile_query(query, event, virtuals)
        data, columns = self.client.execute(sql, params, external_tables=external, with_column_types=True, settings={'strings_as_bytes': True})
        return decode_rows(data, columns)

    def grafana_sql(self, output):
        if output.kind != 'table':
            raise ValueError('Grafana can query persisted outputs only')
        return 'SELECT ' + ', '.join(quote(n) for n, _ in physical_columns(output.columns)) + ' FROM ' + self.table(output.name)

    def _cache_marker(self):
        name = self.base_table + '_signal_cache_once_complete'
        self.create(name, (('run_id','String'), ('signal_name','String'), ('counts','Array(UInt64)')))
        return name

    def load_cache(self, signal, run_id):
        marker = self._cache_marker()
        key = {'run': run_id, 'signal': signal.name}
        rows = self.client.execute(f'SELECT counts FROM {self.table(marker)} WHERE run_id=%(run)s AND signal_name=%(signal)s', key)
        if not rows:
            return None
        counts = rows[-1][0]
        result = []
        if len(counts) != len(signal.outputs):
            raise ValueError('Cache output count mismatch')
        for output, count in zip(signal.outputs, counts):
            name = output.name + '_cache_once'
            self.create(name, (('run_id','String'), ('signal_name','String')) + output.columns)
            names = ', '.join(quote(n) for n, _ in physical_columns(output.columns))
            data, columns = self.client.execute(f'SELECT {names} FROM {self.table(name)} WHERE run_id=%(run)s AND signal_name=%(signal)s', key, with_column_types=True, settings={'strings_as_bytes': True})
            if len(data) != count:
                raise RuntimeError('Completed cache has inconsistent row count')
            result.append(decode_rows(data, columns))
        return tuple(result)

    def save_cache(self, signal, run_id, results):
        # A single worker owns a run. Completion is published only after all outputs,
        # including empty outputs. Clear remnants of any interrupted previous attempt.
        marker = self._cache_marker()
        for output, rows in zip(signal.outputs, results):
            name = output.name + '_cache_once'
            columns = (('run_id','String'), ('signal_name','String')) + output.columns
            self.create(name, columns)
            self.client.execute(f'ALTER TABLE {self.table(name)} DELETE WHERE run_id=%(run)s AND signal_name=%(signal)s', {'run':run_id,'signal':signal.name}, settings={'mutations_sync':2})
            self._insert(name, [dict(row, run_id=run_id, signal_name=signal.name) for row in rows], columns)
        self._insert(marker, [dict(run_id=run_id, signal_name=signal.name, counts=[len(x) for x in results])], (('run_id','String'),('signal_name','String'),('counts','Array(UInt64)')))
