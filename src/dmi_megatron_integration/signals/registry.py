"""Resolve virtual output names independently of physical table names."""
from .definition import sources


class Registry:
    def __init__(self, signals):
        self.signals = {}
        self.outputs = {}
        for signal in signals:
            if signal.name in self.signals:
                raise ValueError(f'Duplicate Signal: {signal.name}')
            self.signals[signal.name] = signal
            for index, output in enumerate(signal.outputs):
                if output.kind == 'signal':
                    if output.name in self.outputs:
                        raise ValueError(f'Duplicate virtual output: {output.name}')
                    self.outputs[output.name] = (signal, index)
        dependencies = {}
        for signal in signals:
            dependencies[signal.name] = []
            for query in signal.inputs:
                for source in sources(query):
                    if 'signal' not in source:
                        continue
                    name = source['signal']
                    if name not in self.outputs:
                        raise ValueError(f'{signal.name}: unresolved virtual output {name}')
                    producer, _ = self.outputs[name]
                    event = producer.trigger['event']
                    if event != 'on_demand':
                        raise ValueError(f'{signal.name}: output {name} producer {producer.name} has trigger {event}; on_demand required')
                    dependencies[signal.name].append(producer.name)
        visited, active = set(), set()
        def visit(name):
            if name in active:
                raise ValueError(f'Signal dependency cycle at {name}')
            if name in visited:
                return
            active.add(name)
            for dependency in dependencies[name]:
                visit(dependency)
            active.remove(name)
            visited.add(name)
        for name in dependencies:
            visit(name)
