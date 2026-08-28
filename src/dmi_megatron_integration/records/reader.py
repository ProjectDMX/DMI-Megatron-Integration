"""Megatron schema-v2 queries layered on the public DMI ClickHouse reader."""

from __future__ import annotations

import re
from typing import Any, Literal, Sequence

from dmi.api.v1 import CHClickhouseDriverReadOnly

from .schema import (
    TRAINING_PRIMARY_KEY_COLUMN_NAMES,
    TRAINING_ROW_COORDINATE_COLUMN_NAMES,
    TRAINING_SCHEMA_VERSION,
)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRAINING_COORDINATE_TYPES = (
    ("model_id", "String"),
    ("act_name", "String"),
    ("direction", "String"),
    ("phase", "String"),
    ("global_batch_id", "Int64"),
    ("dp_rank", "Int32"),
    ("microbatch_id", "Int32"),
    ("sample_index", "Int32"),
    ("layer_no", "Int32"),
    ("shard_rank", "Int32"),
    ("token_start", "Int64"),
    ("token_end", "Int64"),
    ("attempt_id", "Int32"),
    ("invocation_id", "Int32"),
    ("dataset_id", "Int32"),
)


class MegatronTrainingReader(CHClickhouseDriverReadOnly):
    """Read the established Megatron tables through public DMI storage access."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        username: str = "default",
        password: str = "",
        database: str = "default",
        table: str = "offload",
        secure: bool = False,
        client_settings: dict[str, str | int | bool] | None = None,
        decode_strings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            table=table,
            secure=secure,
            client_settings=client_settings,
            primary_key_column_names=TRAINING_ROW_COORDINATE_COLUMN_NAMES,
            order_by_column_names=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
            value_column_names=("dtype", "shape", "bytes"),
            decode_strings=decode_strings,
            **kwargs,
        )
        self._training_decode_strings = bool(decode_strings)
        self._validated_training_tables: set[tuple[str, str]] = set()

    @staticmethod
    def _training_ident(value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid ClickHouse identifier: {value!r}")
        return value

    @classmethod
    def _quoted(cls, value: str) -> str:
        return f"`{cls._training_ident(value)}`"

    @staticmethod
    def _decode_cell(value: Any) -> Any:
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="surrogateescape")
        return value

    def _select(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        return super().custom_select(query, params)

    def _ensure_training_schema(self, table: str, *, value_kind: str) -> None:
        table = self._training_ident(table)
        cache_key = (table, value_kind)
        if cache_key in self._validated_training_tables:
            return
        if value_kind == "tensor":
            value_columns = (
                ("dtype", "String"),
                ("shape", "Array(Int64)"),
                ("bytes", "String"),
            )
        elif value_kind == "float":
            value_columns = (("value", "Float64"),)
        elif value_kind == "int":
            value_columns = (("value", "Int64"),)
        else:
            raise ValueError(f"unknown Megatron training value kind: {value_kind!r}")

        rows = self._select(
            "SELECT name, type FROM system.columns "
            "WHERE database = %(database)s AND table = %(table)s "
            "ORDER BY position",
            {"database": self.database, "table": table},
        )
        actual = tuple(
            (self._decode_cell(row[0]), self._decode_cell(row[1]))
            for row in rows
        )
        expected = _TRAINING_COORDINATE_TYPES + value_columns
        if actual != expected:
            raise RuntimeError(
                f"Megatron training schema mismatch for {self.database}.{table}: "
                f"expected version {TRAINING_SCHEMA_VERSION}"
            )

        key_rows = self._select(
            "SELECT primary_key, sorting_key FROM system.tables "
            "WHERE database = %(database)s AND name = %(table)s",
            {"database": self.database, "table": table},
        )
        if len(key_rows) != 1:
            raise RuntimeError(f"could not validate Megatron training table key for {table}")
        expected_key = ",".join(TRAINING_PRIMARY_KEY_COLUMN_NAMES)
        actual_keys = tuple(self._normalize_key(value) for value in key_rows[0])
        if actual_keys != (expected_key, expected_key):
            raise RuntimeError(
                f"Megatron training key mismatch for {self.database}.{table}"
            )
        self._validated_training_tables.add(cache_key)

    @classmethod
    def _normalize_key(cls, value: Any) -> str:
        decoded = str(cls._decode_cell(value))
        return "".join(
            character
            for character in decoded
            if character not in "`() " and not character.isspace()
        )

    def _accepted_training_filter(self, *, scalar_int_table: str) -> str:
        table = self._training_ident(scalar_int_table)
        return (
            "(`phase` != 'train' OR `global_batch_id` = 0 OR "
            "(`model_id`, `global_batch_id`, `attempt_id`) IN ("
            "SELECT `model_id`, `global_batch_id`, `attempt_id` "
            f"FROM `{self.database}`.`{table}` "
            "WHERE `act_name` = 'iteration_attempt_status' "
            "AND `direction` = 'iter' AND `phase` = 'train' "
            "AND `invocation_id` = 0 AND `dataset_id` = -1 AND `value` = 1))"
        )

    def _training_attempt_statuses(
        self,
        model_id: Any,
        *,
        scalar_table: str,
    ) -> dict[int, dict[int, int]]:
        scalar_table = self._training_ident(scalar_table)
        self._ensure_training_schema(scalar_table, value_kind="int")
        rows = self._select(
            f"""
            SELECT global_batch_id, attempt_id, value,
                   dp_rank, microbatch_id, sample_index, layer_no, shard_rank,
                   token_start, token_end, invocation_id, dataset_id
            FROM `{self.database}`.`{scalar_table}`
            WHERE model_id = %(model_id)s
              AND act_name = 'iteration_attempt_status'
              AND direction = 'iter'
              AND phase = 'train'
            ORDER BY global_batch_id, attempt_id
            """,
            {"model_id": model_id},
        )
        statuses: dict[int, dict[int, int]] = {}
        expected_identity = (-1, -1, -1, -1, -1, 0, 1, 0, -1)
        for row in rows:
            global_batch_id, attempt_id, value = map(int, row[:3])
            identity = tuple(int(item) for item in row[3:])
            if identity != expected_identity:
                raise RuntimeError(
                    "invalid iteration_attempt_status row identity: "
                    f"gid={global_batch_id}, attempt={attempt_id}, "
                    f"identity={identity}"
                )
            if value not in {-1, 0, 1}:
                raise RuntimeError(
                    f"invalid attempt status value {value} for "
                    f"gid={global_batch_id}, attempt={attempt_id}"
                )
            by_attempt = statuses.setdefault(global_batch_id, {})
            if attempt_id in by_attempt:
                raise RuntimeError(
                    f"duplicate attempt status for gid={global_batch_id}, "
                    f"attempt={attempt_id}"
                )
            by_attempt[attempt_id] = value
        return statuses

    @staticmethod
    def _accepted_attempts_from_statuses(
        statuses: dict[int, dict[int, int]],
    ) -> dict[int, int]:
        accepted: dict[int, int] = {}
        for global_batch_id, by_attempt in statuses.items():
            accepted_ids = [
                attempt_id
                for attempt_id, value in by_attempt.items()
                if value == 1
            ]
            if len(accepted_ids) > 1:
                raise RuntimeError(
                    "multiple accepted attempts for training iteration "
                    f"{global_batch_id}: {accepted_ids}"
                )
            if accepted_ids:
                accepted[global_batch_id] = accepted_ids[0]
            elif -1 not in by_attempt.values():
                raise RuntimeError(
                    f"completed iteration {global_batch_id} has no accepted attempt"
                )
        return accepted

    def _validate_normal_training_attempts(
        self,
        *,
        table: str,
        model_id: Any,
        prefix_filters: Sequence[str],
        params: dict[str, Any],
        scalar_int_table: str,
    ) -> None:
        table = self._training_ident(table)
        filters = [*prefix_filters, "`phase` = 'train'", "`global_batch_id` != 0"]
        rows = self._select(
            f"SELECT DISTINCT `global_batch_id` "
            f"FROM `{self.database}`.`{table}` "
            f"WHERE {' AND '.join(filters)} "
            f"ORDER BY `global_batch_id`",
            params,
        )
        represented_iterations = {int(row[0]) for row in rows}
        if not represented_iterations:
            return

        statuses = self._training_attempt_statuses(
            model_id,
            scalar_table=scalar_int_table,
        )
        accepted = self._accepted_attempts_from_statuses(statuses)
        missing = sorted(
            global_batch_id
            for global_batch_id in represented_iterations
            if global_batch_id not in accepted
            and -1 not in statuses.get(global_batch_id, {}).values()
        )
        if missing:
            raise RuntimeError(
                "training payload iterations have no accepted attempt status: "
                f"{missing}"
            )

    @staticmethod
    def _prefix_parts(prefix_key: tuple[Any, ...]) -> tuple[
        tuple[str, ...], dict[str, Any], list[str]
    ]:
        if not isinstance(prefix_key, tuple):
            raise TypeError(f"prefix_key must be a tuple, got {type(prefix_key)!r}")
        if not prefix_key:
            raise ValueError("training prefix requires a non-empty prefix_key")
        if len(prefix_key) > len(TRAINING_PRIMARY_KEY_COLUMN_NAMES):
            raise ValueError(
                f"prefix_key too long: got {len(prefix_key)} "
                f"max {len(TRAINING_PRIMARY_KEY_COLUMN_NAMES)}"
            )
        names = TRAINING_PRIMARY_KEY_COLUMN_NAMES[: len(prefix_key)]
        params = {name: prefix_key[index] for index, name in enumerate(names)}
        filters = [f"`{name}` = %({name})s" for name in names]
        return names, params, filters

    def training_prefix_get(
        self,
        prefix_key: tuple[Any, ...],
        *,
        return_full_key_tuple: bool = True,
        include_all_attempts: bool = False,
        include_all_invocations: bool = False,
    ) -> list[Any]:
        """Read tensor rows, normally retaining accepted attempt/invocation zero."""

        self._ensure_training_schema(self.table, value_kind="tensor")
        _, params, prefix_filters = self._prefix_parts(prefix_key)
        value_columns = ("dtype", "shape", "bytes")
        selected_columns = (
            TRAINING_ROW_COORDINATE_COLUMN_NAMES + value_columns
            if return_full_key_tuple
            else value_columns
        )
        filters = list(prefix_filters)
        if not include_all_attempts:
            self._validate_normal_training_attempts(
                table=self.table,
                model_id=prefix_key[0],
                prefix_filters=prefix_filters,
                params=params,
                scalar_int_table=f"{self.table}_scalar_int",
            )
            filters.append(
                self._accepted_training_filter(
                    scalar_int_table=f"{self.table}_scalar_int"
                )
            )
        if not include_all_invocations:
            filters.append("`invocation_id` = 0")
        selected = ", ".join(self._quoted(name) for name in selected_columns)
        sql = f"SELECT {selected} FROM `{self.database}`.`{self.table}`"
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY " + ", ".join(
            self._quoted(name) for name in TRAINING_PRIMARY_KEY_COLUMN_NAMES
        )
        rows = self._select(sql, params)
        if return_full_key_tuple:
            coordinate_count = len(TRAINING_ROW_COORDINATE_COLUMN_NAMES)
            result = []
            for row in rows:
                coordinates = tuple(row[:coordinate_count])
                if self._training_decode_strings:
                    coordinates = tuple(self._decode_cell(cell) for cell in coordinates)
                result.append(
                    (coordinates, self.torch_decode(*row[coordinate_count:]))
                )
            return result
        return [self.torch_decode(*row) for row in rows]

    def training_raw_prefix_get(
        self,
        prefix_key: tuple[Any, ...],
        *,
        return_full_key_tuple: bool = True,
    ) -> list[Any]:
        return self.training_prefix_get(
            prefix_key,
            return_full_key_tuple=return_full_key_tuple,
            include_all_attempts=True,
            include_all_invocations=True,
        )

    def training_scalar_prefix_get(
        self,
        prefix_key: tuple[Any, ...],
        *,
        scalar_kind: Literal["float", "int"],
        return_full_key_tuple: bool = True,
        table: str | None = None,
        include_all_attempts: bool = False,
        include_all_invocations: bool = False,
    ) -> list[Any]:
        """Read typed scalar rows by the frozen training-key prefix."""

        if scalar_kind not in ("float", "int"):
            raise ValueError("scalar_kind must be 'float' or 'int'")
        _, params, prefix_filters = self._prefix_parts(prefix_key)
        suffix = "scalar_float" if scalar_kind == "float" else "scalar_int"
        scalar_table = self._training_ident(table or f"{self.table}_{suffix}")
        self._ensure_training_schema(scalar_table, value_kind=scalar_kind)
        selected_columns = (
            TRAINING_ROW_COORDINATE_COLUMN_NAMES + ("value",)
            if return_full_key_tuple
            else ("value",)
        )
        filters = list(prefix_filters)
        if not include_all_attempts:
            self._validate_normal_training_attempts(
                table=scalar_table,
                model_id=prefix_key[0],
                prefix_filters=prefix_filters,
                params=params,
                scalar_int_table=f"{self.table}_scalar_int",
            )
            filters.append(
                self._accepted_training_filter(
                    scalar_int_table=f"{self.table}_scalar_int"
                )
            )
        if not include_all_invocations:
            filters.append("`invocation_id` = 0")
        selected = ", ".join(self._quoted(name) for name in selected_columns)
        sql = f"SELECT {selected} FROM `{self.database}`.`{scalar_table}`"
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY " + ", ".join(
            self._quoted(name) for name in TRAINING_PRIMARY_KEY_COLUMN_NAMES
        )
        rows = self._select(sql, params)
        if return_full_key_tuple:
            coordinate_count = len(TRAINING_ROW_COORDINATE_COLUMN_NAMES)
            result = []
            for row in rows:
                coordinates = tuple(row[:coordinate_count])
                if self._training_decode_strings:
                    coordinates = tuple(self._decode_cell(cell) for cell in coordinates)
                result.append((coordinates, row[coordinate_count]))
            return result
        return [row[0] for row in rows]

    def training_scalar_raw_prefix_get(
        self,
        prefix_key: tuple[Any, ...],
        *,
        scalar_kind: Literal["float", "int"],
        return_full_key_tuple: bool = True,
        table: str | None = None,
    ) -> list[Any]:
        return self.training_scalar_prefix_get(
            prefix_key,
            scalar_kind=scalar_kind,
            return_full_key_tuple=return_full_key_tuple,
            table=table,
            include_all_attempts=True,
            include_all_invocations=True,
        )

    def training_accepted_attempts(
        self,
        model_id: str,
        *,
        phase: str = "train",
        table: str | None = None,
    ) -> dict[int, int]:
        """Return the unique accepted attempt for each completed iteration."""

        if phase != "train":
            raise ValueError("attempt status is defined only for the train phase")
        scalar_table = self._training_ident(table or f"{self.table}_scalar_int")
        return self._accepted_attempts_from_statuses(
            self._training_attempt_statuses(model_id, scalar_table=scalar_table)
        )

    def training_eval_boundaries(
        self,
        model_id: str,
        *,
        training_iteration_id: int | None = None,
        phase: str | None = None,
        eval_index: int | None = None,
        table: str | None = None,
    ) -> list[tuple[Any, ...]]:
        """Return validation/test boundary rows for ``model_id``."""

        boundary_table = self._training_ident(
            table or f"{self.table}_eval_phase_boundary"
        )
        filters = ["`model_id` = %(model_id)s"]
        params: dict[str, Any] = {"model_id": model_id}
        if training_iteration_id is not None:
            filters.append("`training_iteration_id` = %(training_iteration_id)s")
            params["training_iteration_id"] = int(training_iteration_id)
        if phase is not None:
            filters.append("`phase` = %(phase)s")
            params["phase"] = str(phase)
        if eval_index is not None:
            filters.append("`eval_index` = %(eval_index)s")
            params["eval_index"] = int(eval_index)
        rows = self._select(
            f"SELECT `model_id`, `training_iteration_id`, `phase`, "
            f"`eval_index`, `boundary_type`, `next_global_batch_id` "
            f"FROM `{self.database}`.`{boundary_table}` "
            f"WHERE {' AND '.join(filters)} "
            f"ORDER BY `training_iteration_id`, `phase`, `eval_index`, "
            f"`next_global_batch_id`, `boundary_type`",
            params,
        )
        if not self._training_decode_strings:
            return rows
        return [
            tuple(
                self._decode_cell(cell) if index in (0, 2, 4) else cell
                for index, cell in enumerate(row)
            )
            for row in rows
        ]


__all__ = [
    "MegatronTrainingReader",
    "TRAINING_PRIMARY_KEY_COLUMN_NAMES",
    "TRAINING_ROW_COORDINATE_COLUMN_NAMES",
    "TRAINING_SCHEMA_VERSION",
]
