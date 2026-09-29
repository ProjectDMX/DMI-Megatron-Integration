"""Root lifecycle readiness. On-demand nodes have no independent readiness gate."""
from dataclasses import dataclass, replace
from .storage import quote


@dataclass(frozen=True)
class Event:
    run_id: str
    event: str
    phase: str
    training_iteration: int
    global_batch_id: int = 0
    eval_index: int = 0
    attempt_id: int = 0
    start: int = 0
    end: int = 0


def matches(trigger, event):
    name = trigger['event']
    if name == 'on_demand':
        return False
    if name != event.event and not (name == 'phase_end' and event.event == 'validation_end'):
        return False
    if trigger.get('phase', event.phase) != event.phase:
        return False
    if event.training_iteration < trigger.get('training_iteration_min', 0):
        return False
    return event.training_iteration % trigger.get('every_n_iterations', 1) == 0


def readiness_query(storage, event, expected_ranks):
    """One statement: metadata consensus + filtered unique payload counts per rank."""
    ranks = tuple(sorted(set(expected_ranks)))
    if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError('Expected producer ranks must be explicitly configured')
    iteration = event.event == 'iteration_end'
    params = dict(run=event.run_id, phase=event.phase, batch=event.global_batch_id, eval=event.eval_index, training=event.training_iteration, ranks=ranks)
    if iteration:
        table = storage.table(storage.base_table + '_iteration_metadata')
        metadata = f'''SELECT producer_rank, countDistinct(tuple(attempt_id, expected_tensor_count)) AS variants,
            any(attempt_id) AS selected_attempt, any(expected_tensor_count) AS expected,
            any(global_batch_id) AS start, any(global_batch_id)+1 AS end, count() AS present
            FROM {table} WHERE model_id=%(run)s AND phase=%(phase)s AND global_batch_id=%(batch)s
            AND eval_index=%(eval)s AND status=1 GROUP BY producer_rank'''
        attempt_filter = 'AND p.attempt_id=m.selected_attempt'
    else:
        table = storage.table(storage.base_table + '_phase_metadata')
        metadata = f'''SELECT producer_rank, countDistinct(tuple(global_batch_id_start, global_batch_id_end, expected_tensor_count)) AS variants,
            toInt32(0) AS selected_attempt, any(expected_tensor_count) AS expected,
            any(global_batch_id_start) AS start, any(global_batch_id_end) AS end, count() AS present
            FROM {table} WHERE model_id=%(run)s AND phase=%(phase)s AND eval_index=%(eval)s
            AND training_iteration_id=%(training)s AND boundary_type='exit' GROUP BY producer_rank'''
        attempt_filter = ''
    from ..records.schema import TRAINING_ROW_COORDINATE_COLUMN_NAMES
    identity = ', '.join('p.' + quote(name) for name in TRAINING_ROW_COORDINATE_COLUMN_NAMES)
    selects = []
    for suffix in ('', '_scalar_float', '_scalar_int'):
        table = storage.table(storage.base_table + suffix)
        selects.append(f"SELECT * EXCEPT(value) FROM {table}" if suffix else f"SELECT * EXCEPT(dtype, shape, bytes) FROM {table}")
    # All metadata is independent of insertion order. Distinct attempts remain part
    # of identity, including at phase end; only duplicate delivery collapses.
    sql = f'''WITH m AS ({metadata}),
      payload AS ({' UNION ALL '.join(selects)}),
      counts AS (
        SELECT p.producer_rank AS producer_rank, uniqExact(tuple({identity})) AS actual
        FROM payload p INNER JOIN m ON p.producer_rank=m.producer_rank
        WHERE p.model_id=%(run)s AND p.phase=%(phase)s
        AND p.global_batch_id>=m.start AND p.global_batch_id<m.end {attempt_filter}
        AND p.act_name != 'iteration_attempt_status' GROUP BY p.producer_rank
      ), ranks AS (SELECT arrayJoin(%(ranks)s) AS producer_rank)
      SELECT toUInt8(count() > 0 AND min(m.present>0 AND m.variants=1 AND ifNull(c.actual,0)=m.expected)
         AND uniqExact(m.selected_attempt)=1 AND uniqExact(tuple(m.start,m.end))=1) AS ready,
         any(m.selected_attempt) AS attempt_id, any(m.start) AS start, any(m.end) AS end
      FROM ranks r LEFT JOIN m ON r.producer_rank=m.producer_rank
      LEFT JOIN counts c ON r.producer_rank=c.producer_rank'''
    params['ranks'] = list(ranks)
    return sql, params


class EventPoller:
    def __init__(self, storage, *, run_id, expected_ranks):
        self.storage = storage
        self.run_id = run_id
        self.expected_ranks = tuple(expected_ranks)

    def candidates(self):
        s = self.storage
        rows = s.client.execute(f'''SELECT DISTINCT phase, training_iteration_id, global_batch_id, eval_index
            FROM {s.table(s.base_table + '_iteration_metadata')}
            WHERE model_id=%(run)s AND status=1 ORDER BY training_iteration_id, phase, global_batch_id''', {'run':self.run_id})
        events = [Event(self.run_id, 'iteration_end', phase, training, batch, evaluation) for phase, training, batch, evaluation in rows]
        rows = s.client.execute(f'''SELECT DISTINCT phase, training_iteration_id, eval_index
            FROM {s.table(s.base_table + '_phase_metadata')}
            WHERE model_id=%(run)s AND boundary_type='exit' ORDER BY training_iteration_id, phase, eval_index''', {'run':self.run_id})
        events.extend(Event(self.run_id, 'validation_end' if phase == 'valid' else 'phase_end', phase, training, eval_index=evaluation) for phase, training, evaluation in rows)
        return events

    def ready(self, event):
        query, params = readiness_query(self.storage, event, self.expected_ranks)
        rows = self.storage.client.execute(query, params)
        if not rows or not rows[0][0]:
            return None
        _, attempt, start, end = rows[0]
        return replace(event, attempt_id=attempt, start=start, end=end)
