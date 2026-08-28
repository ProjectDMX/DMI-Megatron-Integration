"""Deterministic windowed mixture selection and exact-window decision delivery."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping

from dmi_megatron_integration.dynamic_mixture_primitives import (
    DecisionHTTPClient,
    DecisionPrefetcher,
    MixtureDecision,
    SelectionWindow,
    WindowSelector,
    normalize_weights,
)


_DECISION_STORE_STATE_VERSION = 1


class DecisionStore:
    """Immutable exact-window decision map with condition-variable waiting."""

    def __init__(
        self,
        *,
        on_publish: Callable[[MixtureDecision], None] | None = None,
        on_request: Callable[[str, int, str, str], None] | None = None,
    ) -> None:
        self._condition = threading.Condition()
        self._decisions: dict[tuple[str, int], MixtureDecision] = {}
        self._on_publish = on_publish
        self._on_request = on_request

    def publish(self, decision: MixtureDecision) -> None:
        self._install(decision, persist=True)

    def _install(self, decision: MixtureDecision, *, persist: bool) -> None:
        key = (decision.run_id, decision.effective_window_id)
        with self._condition:
            existing = self._decisions.get(key)
            if existing is not None:
                if existing != decision:
                    raise RuntimeError(f"conflicting decision for {key}")
                return
            self._decisions[key] = decision
            if persist and self._on_publish is not None:
                self._on_publish(decision)
            self._condition.notify_all()

    def wait(
        self,
        run_id: str,
        effective_window_id: int,
        *,
        timeout_s: float,
        client_id: str = "",
    ) -> MixtureDecision | None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        key = (str(run_id), int(effective_window_id))
        deadline = time.monotonic() + float(timeout_s)
        if self._on_request is not None:
            self._on_request(key[0], key[1], str(client_id), "WAIT")
        with self._condition:
            while key not in self._decisions:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    if self._on_request is not None:
                        self._on_request(key[0], key[1], str(client_id), "TIMEOUT")
                    return None
                self._condition.wait(remaining)
            decision = self._decisions[key]
        if self._on_request is not None:
            self._on_request(key[0], key[1], str(client_id), "DELIVERED")
        return decision

    def get(self, run_id: str, effective_window_id: int) -> MixtureDecision | None:
        with self._condition:
            return self._decisions.get((str(run_id), int(effective_window_id)))

    def state_dict(
        self,
        *,
        run_id: str,
        pending_effective_window_id: int | None,
    ) -> dict[str, object]:
        pending = (
            None
            if pending_effective_window_id is None
            else self.get(run_id, pending_effective_window_id)
        )
        return {
            "schema_version": _DECISION_STORE_STATE_VERSION,
            "pending_decision": None if pending is None else pending.to_dict(),
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        run_id: str,
    ) -> MixtureDecision | None:
        expected = {"schema_version", "pending_decision"}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"decision-store state fields mismatch: missing={missing}, extra={extra}"
            )
        if int(state["schema_version"]) != _DECISION_STORE_STATE_VERSION:
            raise ValueError(
                f"unsupported decision-store state version: {state['schema_version']!r}"
            )
        raw = state["pending_decision"]
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError("pending decision must be a mapping")
        rebound = dict(raw)
        rebound["run_id"] = str(run_id)
        decision = MixtureDecision.from_dict(rebound)
        # The pending decision was persisted by the parent segment. The child
        # needs it in memory for its first window, but does not own another row.
        self._install(decision, persist=False)
        return decision


class DecisionHTTPServer:
    """Threaded HTTP wrapper around :class:`DecisionStore`."""

    def __init__(
        self,
        host: str,
        port: int,
        store: DecisionStore,
        *,
        checkpoint_callback: Callable[[int, int, float], dict[str, object]]
        | None = None,
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        if port < 0 or port > 65535:
            raise ValueError("port must be in [0, 65535]")
        self.store = store
        self.checkpoint_callback = checkpoint_callback
        self.shutdown_callback = shutdown_callback
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/healthz":
                    self._json(HTTPStatus.OK, {"status": "ready"})
                    return
                if parsed.path != "/v1/decision":
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    run_id = query["run_id"][0]
                    window_id = int(query["effective_window_id"][0])
                    client_id = query.get("client_id", [""])[0]
                    timeout_s = float(query.get("timeout_s", ["60"])[0])
                    timeout_s = min(timeout_s, 300.0)
                    decision = outer.store.wait(
                        run_id,
                        window_id,
                        timeout_s=timeout_s,
                        client_id=client_id,
                    )
                except (KeyError, ValueError, TypeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if decision is None:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self._json(HTTPStatus.OK, decision.to_dict())

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/v1/checkpoint":
                    if outer.checkpoint_callback is None:
                        self._json(
                            HTTPStatus.NOT_IMPLEMENTED,
                            {"error": "checkpoint snapshots are not enabled"},
                        )
                        return
                    try:
                        body = self._read_json()
                        result = outer.checkpoint_callback(
                            int(body["checkpoint_iteration"]),
                            int(body["installed_window_id"]),
                            float(body.get("timeout_s", 600.0)),
                        )
                    except (KeyError, ValueError, TypeError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                        return
                    except TimeoutError as error:
                        self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": str(error)})
                        return
                    except RuntimeError as error:
                        self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                        return
                    self._json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/v1/shutdown":
                    if outer.shutdown_callback is None:
                        self._json(
                            HTTPStatus.NOT_IMPLEMENTED,
                            {"error": "graceful shutdown is not enabled"},
                        )
                        return
                    outer.shutdown_callback()
                    self._json(HTTPStatus.OK, {"status": "shutting_down"})
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def log_message(self, format: str, *args: object) -> None:
                return

            def _read_json(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise TypeError("request body must be a JSON object")
                return value

            def _json(self, status: HTTPStatus, value: dict[str, object]) -> None:
                payload = json.dumps(value, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("decision HTTP server is already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dmi-mixture-http",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
