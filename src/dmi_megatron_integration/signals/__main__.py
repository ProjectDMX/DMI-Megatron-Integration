"""Run with python -m dmi_megatron_integration.signals CONFIG --run-id RUN ..."""
import argparse
import os
from .config import load_config
from .events import EventPoller
from .storage import ClickHouseStorage
from .worker import SignalWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--producer-ranks', type=int, nargs='+', required=True)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--host', default=os.environ.get('DMI_DB_HOST', os.environ.get('DMX_DB_HOST', 'localhost')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('DMI_DB_PORT', os.environ.get('DMX_DB_PORT', '9000'))))
    parser.add_argument('--database', default=os.environ.get('DMI_DB_DATABASE', os.environ.get('DMX_DB_DATABASE', 'default')))
    parser.add_argument('--table', default=os.environ.get('DMI_CLICKHOUSE_TABLE', 'dmi_training_tensors'))
    parser.add_argument('--user', default=os.environ.get('DMX_DB_USER', 'default'))
    parser.add_argument('--poll-interval', type=float, default=0.1)
    args = parser.parse_args()
    registry = load_config(args.config)
    from clickhouse_driver import Client
    client = Client(host=args.host, port=args.port, user=args.user, password=os.environ.get('DMX_DB_PASSWORD', ''), database=args.database)
    storage = ClickHouseStorage(client, database=args.database, base_table=args.table, num_layers=args.num_layers)
    poller = EventPoller(storage, run_id=args.run_id, expected_ranks=args.producer_ranks)
    for signal in registry.signals.values():
        for output in signal.outputs:
            if output.kind == 'table':
                print(f'{signal.name}: {storage.grafana_sql(output)}', flush=True)
    try:
        SignalWorker(registry, storage).run(poller, interval=args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()
