"""CPU-only Signal worker. Root triggers provide recursive input readiness."""
import time
from .definition import sources
from .events import matches
from .storage import normalize_rows


class SignalWorker:
    def __init__(self, registry, storage):
        self.registry = registry
        self.storage = storage
        self._completed = set()

    def execute(self, name, event):
        root = self.registry.signals[name]
        if root.trigger['event'] == 'on_demand':
            raise ValueError('An on-demand Signal cannot be a root')
        if not matches(root.trigger, event):
            return None
        memo = {}
        def invoke(signal):
            if signal.name in memo:
                return memo[signal.name]
            cached = self.storage.load_cache(signal, event.run_id) if signal.cache_once else None
            if cached is not None:
                memo[signal.name] = cached
                return cached
            arguments = []
            for query in signal.inputs:
                virtuals = {}
                for source in sources(query):
                    if 'signal' in source:
                        producer, index = self.registry.outputs[source['signal']]
                        virtuals[source['signal']] = (producer.outputs[index], invoke(producer)[index])
                arguments.append(self.storage.read(query, event, virtuals))
            returned = signal.transform(*arguments)
            if not isinstance(returned, (tuple, list)) or len(returned) != len(signal.outputs):
                raise ValueError(f'{signal.name}: return one row collection per output mapping')
            results = tuple(normalize_rows(rows, output.columns) for rows, output in zip(returned, signal.outputs))
            for output, rows in zip(signal.outputs, results):
                if output.kind == 'table':
                    self.storage.write(output, rows)
            if signal.cache_once:
                self.storage.save_cache(signal, event.run_id, results)
            memo[signal.name] = results
            return results
        return invoke(root)

    def poll_once(self, poller):
        executed = []
        for candidate in poller.candidates():
            pending = [s for s in self.registry.signals.values() if matches(s.trigger, candidate) and (s.name, candidate) not in self._completed]
            if not pending:
                continue
            event = poller.ready(candidate)
            if event is None:
                continue
            for signal in pending:
                self.execute(signal.name, event)
                self._completed.add((signal.name, candidate))
                executed.append((signal.name, event))
        return executed

    def run(self, poller, *, interval=0.1, stop=None):
        if interval <= 0:
            raise ValueError('Polling interval must be positive')
        while stop is None or not stop.is_set():
            self.poll_once(poller)
            if stop is None:
                time.sleep(interval)
            else:
                stop.wait(interval)
