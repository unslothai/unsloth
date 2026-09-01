# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Project-scoped durable sequential graphs.

Graphs are orchestration records. They do not contain a second model or agent
runtime. Loop and model nodes submit work to the existing background-agent
manager, while tool nodes use the existing MCP transport.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from string import Formatter
from typing import Any, Callable, Optional

from core.inference.mcp_client import (
    call_tool_sync,
    oauth_credential_binding,
    parse_server_headers,
)
from storage import mcp_servers_db

from .common import AgentWorkspaceError, now_ms
from .state import connection


_GRAPH_STATUSES = frozenset(
    {
        "queued",
        "running",
        "pausing",
        "paused",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "interrupted",
    }
)
_NODE_STATUSES = frozenset({"running", "paused", "cancelled", "completed", "failed", "interrupted"})
_APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
_NODE_TYPES = frozenset({"input", "loop", "model", "tool", "condition", "approval", "output"})
_MAX_NODES = 100
_MAX_EDGES = 200
_MAX_JSON_BYTES = 256 * 1024
_MAX_GRAPH_DOCUMENT_BYTES = 512 * 1024
_MAX_RUN_OUTPUT_BYTES = 1024 * 1024
_MAX_RUN_SECONDS = 24 * 60 * 60
_MAX_NODE_SECONDS = 2 * 60 * 60
_MAX_NODE_ATTEMPTS = 10
_MAX_RETRY_BACKOFF_MS = 60_000
_MAX_RUN_ITERATIONS = 10_000
_MAX_RUN_OUTPUT_TOKENS = 1_048_576
_UNSET = object()


class _GraphNodeTimeout(AgentWorkspaceError):
    pass


class _GraphToolEffectUncertain(AgentWorkspaceError):
    pass


class _GraphLoopEffectUncertain(AgentWorkspaceError):
    pass


class _RunControl:
    """Thread-safe wakeup with a reason visible to native node adapters."""

    _PRIORITY = {"pause": 0, "cancel": 1, "budget": 2, "shutdown": 3}

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: Optional[str] = None

    def request(self, reason: str) -> None:
        if reason not in self._PRIORITY:
            raise ValueError("Unknown graph run control reason.")
        with self._lock:
            if self._reason is None or self._PRIORITY[reason] >= self._PRIORITY[self._reason]:
                self._reason = reason
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def should_cancel_work(self) -> bool:
        with self._lock:
            return self._event.is_set() and self._reason != "pause"


class _CombinedEvent:
    def __init__(self, *events: Any):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def should_cancel_work(self) -> bool:
        for event in self._events:
            if not event.is_set():
                continue
            check = getattr(event, "should_cancel_work", None)
            if check is None or check():
                return True
        return False


def _validate_retry_policy(value: Any, node_type: str, config: dict, node_id: str) -> dict:
    if value is None:
        return {"maxAttempts": 1, "backoffMs": 0, "retryOn": ["error", "timeout"]}
    if not isinstance(value, dict) or set(value) - {"maxAttempts", "backoffMs", "retryOn"}:
        raise AgentWorkspaceError(f"Graph node '{node_id}' retry policy is invalid.")
    retry_on = value.get("retryOn", ["error", "timeout"])
    if (
        not isinstance(retry_on, list)
        or not retry_on
        or any(item not in {"error", "timeout"} for item in retry_on)
    ):
        raise AgentWorkspaceError(f"Graph node '{node_id}' retryOn is invalid.")
    policy = {
        "maxAttempts": _bounded_int(
            value.get("maxAttempts", 1),
            f"Graph node '{node_id}' maxAttempts",
            1,
            _MAX_NODE_ATTEMPTS,
        ),
        "backoffMs": _bounded_int(
            value.get("backoffMs", 0),
            f"Graph node '{node_id}' backoffMs",
            0,
            _MAX_RETRY_BACKOFF_MS,
        ),
        "retryOn": sorted(set(retry_on)),
    }
    if policy["maxAttempts"] > 1:
        if node_type == "approval":
            raise AgentWorkspaceError("Approval nodes cannot retry automatically.")
        if (
            node_type in {"loop", "model"}
            and (config.get("runtime") or {}).get("permissionMode") != "off"
        ):
            raise AgentWorkspaceError(
                "Loop and model retries require a runtime with permissionMode 'off'."
            )
        if node_type == "tool" and config.get("sideEffecting", True):
            raise AgentWorkspaceError(
                "Side-effecting tool nodes fail closed and cannot retry automatically."
            )
    return policy


def _json(value: Any, *, limit: int, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} must be JSON serializable.") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise AgentWorkspaceError(f"{label} is too large.")
    return encoded


def _load(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _string(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
    maximum: int = 512,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return value.strip()


def _authored_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return value


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise AgentWorkspaceError(f"{label} is invalid.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} is invalid.") from exc
    if result < minimum or result > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AgentWorkspaceError(f"{label} is invalid.")
    return value


def _node_id(value: Any) -> str:
    node_id = _string(value, "Graph node ID", maximum = 128)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,127}", node_id) is None:
        raise AgentWorkspaceError("Graph node ID is invalid.")
    return node_id


def _validate_runtime(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise AgentWorkspaceError(f"{label} must be an object.")
    allowed = {
        "kind",
        "model",
        "providerId",
        "permissionMode",
        "reasoningEffort",
        "maxOutputTokens",
    }
    if set(value) - allowed:
        raise AgentWorkspaceError(f"{label} contains unsupported fields.")
    kind = value.get("kind")
    if kind not in {"local", "provider"}:
        raise AgentWorkspaceError(f"{label}.kind is invalid.")
    model = _string(value.get("model"), f"{label}.model", maximum = 512)
    permission = value.get("permissionMode")
    if permission not in {"off", "full"}:
        raise AgentWorkspaceError(f"{label}.permissionMode is invalid.")
    result = {
        "kind": kind,
        "model": model,
        "permissionMode": permission,
        "maxOutputTokens": _bounded_int(
            value.get("maxOutputTokens", 8192), f"{label}.maxOutputTokens", 1, 32768
        ),
    }
    for key in ("providerId", "reasoningEffort"):
        if value.get(key) is not None:
            result[key] = _string(value[key], f"{label}.{key}", maximum = 256)
    if kind == "provider" and not result.get("providerId"):
        raise AgentWorkspaceError(f"{label}.providerId is required for provider runtimes.")
    return result


def _validate_node(node: Any) -> dict:
    if not isinstance(node, dict) or set(node) - {
        "id",
        "type",
        "config",
        "label",
        "retryPolicy",
    }:
        raise AgentWorkspaceError("Graph node is invalid.")
    node_id = _node_id(node.get("id"))
    node_type = node.get("type")
    if node_type not in _NODE_TYPES:
        raise AgentWorkspaceError(f"Graph node '{node_id}' has an invalid type.")
    config = node.get("config") or {}
    if not isinstance(config, dict):
        raise AgentWorkspaceError(f"Graph node '{node_id}' config must be an object.")
    allowed: set[str]
    normalized: dict[str, Any] = {}
    if node_type == "input":
        allowed = {"name"}
        normalized["name"] = _string(config.get("name", "input"), "Input name", maximum = 128)
    elif node_type in {"loop", "model"}:
        allowed = {"instruction", "prompt", "runtime", "timeoutSeconds"}
        text_key = "instruction" if node_type == "loop" else "prompt"
        normalized[text_key] = _authored_text(
            config.get(text_key),
            f"{node_type} prompt",
            maximum = 32768,
        )
        if config.get("runtime") is None:
            raise AgentWorkspaceError(f"{node_type} runtime is required.")
        normalized["runtime"] = _validate_runtime(config["runtime"], f"{node_type} runtime")
        normalized["timeoutSeconds"] = _bounded_int(
            config.get("timeoutSeconds", _MAX_NODE_SECONDS),
            f"{node_type}.timeoutSeconds",
            1,
            _MAX_NODE_SECONDS,
        )
    elif node_type == "tool":
        allowed = {
            "serverId",
            "toolName",
            "arguments",
            "timeoutSeconds",
            "sideEffecting",
            "idempotencyKey",
        }
        normalized["serverId"] = _string(config.get("serverId"), "Tool serverId", maximum = 128)
        normalized["toolName"] = _string(config.get("toolName"), "Tool name", maximum = 256)
        arguments = config.get("arguments", {})
        if not isinstance(arguments, dict):
            raise AgentWorkspaceError("Tool arguments must be an object.")
        _json(arguments, limit = 64 * 1024, label = "Tool arguments")
        normalized["arguments"] = arguments
        normalized["timeoutSeconds"] = _bounded_int(
            config.get("timeoutSeconds", 300), "Tool timeoutSeconds", 1, _MAX_NODE_SECONDS
        )
        normalized["sideEffecting"] = _boolean(
            config.get("sideEffecting", True), "Tool sideEffecting"
        )
        if config.get("idempotencyKey") is not None:
            normalized["idempotencyKey"] = _string(
                config["idempotencyKey"], "Tool idempotencyKey", maximum = 512
            )
        if normalized["sideEffecting"] and not normalized.get("idempotencyKey"):
            raise AgentWorkspaceError(
                "Side-effecting tool nodes require an idempotencyKey template."
            )
    elif node_type == "condition":
        allowed = {"path", "operator", "value"}
        normalized["path"] = _string(config.get("path"), "Condition path", maximum = 512)
        operator = config.get("operator", "truthy")
        if operator not in {"truthy", "falsy", "exists", "equals", "notEquals"}:
            raise AgentWorkspaceError("Condition operator is invalid.")
        normalized["operator"] = operator
        if operator in {"equals", "notEquals"} and "value" not in config:
            raise AgentWorkspaceError("Condition value is required for equality operators.")
        if "value" in config:
            _json(config["value"], limit = 32 * 1024, label = "Condition value")
            normalized["value"] = config["value"]
    elif node_type == "approval":
        allowed = {"title", "description"}
        normalized["title"] = _string(
            config.get("title", "Approval required"), "Approval title", maximum = 500
        )
        normalized["description"] = str(config.get("description", ""))[:4000]
    else:
        allowed = {"name", "path"}
        normalized["name"] = _string(config.get("name", node_id), "Output name", maximum = 128)
        if config.get("path") is not None:
            normalized["path"] = _string(config["path"], "Output path", maximum = 512)
    if set(config) - allowed:
        raise AgentWorkspaceError(f"Graph node '{node_id}' config contains unsupported fields.")
    result = {"id": node_id, "type": node_type, "config": normalized}
    result["retryPolicy"] = _validate_retry_policy(
        node.get("retryPolicy"), node_type, normalized, node_id
    )
    if node.get("label") is not None:
        result["label"] = _string(node["label"], "Graph node label", maximum = 200)
    return result


def validate_graph_spec(spec: dict) -> dict:
    """Validate and canonicalize one immutable graph revision."""
    if not isinstance(spec, dict):
        raise AgentWorkspaceError("Graph definition must be an object.")
    allowed = {
        "name",
        "description",
        "metadata",
        "inputSchema",
        "outputSchema",
        "nodes",
        "edges",
        "permissions",
        "limits",
    }
    if set(spec) - allowed:
        raise AgentWorkspaceError("Graph definition contains unsupported fields.")
    name = _string(spec.get("name"), "Graph name", maximum = 200)
    description = str(spec.get("description", ""))[:4000]
    metadata = spec.get("metadata", {})
    if not isinstance(metadata, dict):
        raise AgentWorkspaceError("Graph metadata must be an object.")
    _json(metadata, limit = 64 * 1024, label = "Graph metadata")
    input_schema = spec.get("inputSchema", {"type": "object"})
    output_schema = spec.get("outputSchema", {"type": "object"})
    if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
        raise AgentWorkspaceError("Graph schemas must be objects.")
    _validate_schema_definition(input_schema, "Graph input")
    _validate_schema_definition(output_schema, "Graph output")
    if input_schema.get("type") != "object":
        raise AgentWorkspaceError("Graph input schema must describe an object.")
    _json(input_schema, limit = 64 * 1024, label = "Graph input schema")
    _json(output_schema, limit = 64 * 1024, label = "Graph output schema")
    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes or len(raw_nodes) > _MAX_NODES:
        raise AgentWorkspaceError(f"Graph must contain 1 to {_MAX_NODES} nodes.")
    nodes = [_validate_node(node) for node in raw_nodes]
    node_ids = [node["id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise AgentWorkspaceError("Graph node IDs must be unique.")
    by_id = set(node_ids)
    raw_edges = spec.get("edges", [])
    if not isinstance(raw_edges, list) or len(raw_edges) > _MAX_EDGES:
        raise AgentWorkspaceError(f"Graph must contain at most {_MAX_EDGES} edges.")
    edges = []
    incoming = {node_id: 0 for node_id in by_id}
    outgoing: dict[str, list[dict]] = {node_id: [] for node_id in by_id}
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict) or set(raw_edge) - {"from", "to", "when"}:
            raise AgentWorkspaceError("Graph edge is invalid.")
        source = _string(raw_edge.get("from"), "Graph edge source", maximum = 128)
        target = _string(raw_edge.get("to"), "Graph edge target", maximum = 128)
        if source not in by_id or target not in by_id:
            raise AgentWorkspaceError("Graph edges must reference existing nodes.")
        if source == target:
            raise AgentWorkspaceError("Graph cannot contain self edges.")
        when = raw_edge.get("when")
        if when is not None and when not in {"true", "false", "default"}:
            raise AgentWorkspaceError("Graph edge condition is invalid.")
        source_type = next(node["type"] for node in nodes if node["id"] == source)
        if source_type != "condition" and when is not None:
            raise AgentWorkspaceError("Only condition nodes may have conditional edges.")
        edge = {"from": source, "to": target}
        if when is not None:
            edge["when"] = when
        edges.append(edge)
        incoming[target] += 1
        outgoing[source].append(edge)
    inputs = [node for node in nodes if node["type"] == "input"]
    if len(inputs) != 1 or incoming[inputs[0]["id"]] != 0:
        raise AgentWorkspaceError("Graph must have exactly one root input node.")
    if any(count > 1 for count in incoming.values()):
        raise AgentWorkspaceError("Graph joins are not supported in the sequential graph version.")
    for node in nodes:
        node_edges = outgoing[node["id"]]
        if node["type"] == "condition":
            if len(node_edges) not in {1, 2}:
                raise AgentWorkspaceError("Condition nodes need one or two outgoing edges.")
            conditions = [edge.get("when") for edge in node_edges]
            if len(node_edges) == 1 and conditions[0] not in {None, "default"}:
                raise AgentWorkspaceError(
                    "A single condition edge must be unconditional or default."
                )
            if len(node_edges) == 2 and set(conditions) != {"true", "false"}:
                raise AgentWorkspaceError("Two condition edges must be true and false.")
        elif node["type"] == "output" and node_edges:
            raise AgentWorkspaceError("Output nodes must be terminal.")
        elif len(node_edges) > 1:
            raise AgentWorkspaceError(f"Node '{node['id']}' has too many outgoing edges.")
    terminals = [node for node in nodes if not outgoing[node["id"]]]
    if not terminals or any(node["type"] != "output" for node in terminals):
        raise AgentWorkspaceError("Every graph path must terminate at an output node.")
    visited: set[str] = set()
    stack = [inputs[0]["id"]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(edge["to"] for edge in outgoing[current])
    if visited != by_id:
        raise AgentWorkspaceError("Graph contains unreachable nodes.")
    indegrees = dict(incoming)
    queue = [node_id for node_id, count in indegrees.items() if count == 0]
    visited_count = 0
    while queue:
        current = queue.pop()
        visited_count += 1
        for edge in outgoing[current]:
            indegrees[edge["to"]] -= 1
            if indegrees[edge["to"]] == 0:
                queue.append(edge["to"])
    if visited_count != len(nodes):
        raise AgentWorkspaceError("Graph cannot contain cycles.")
    permissions = spec.get("permissions", {})
    if not isinstance(permissions, dict) or set(permissions) - {"allowedToolServerIds"}:
        raise AgentWorkspaceError("Graph permissions are invalid.")
    allowed_tools = permissions.get("allowedToolServerIds", [])
    if not isinstance(allowed_tools, list) or any(
        not isinstance(item, str) for item in allowed_tools
    ):
        raise AgentWorkspaceError("Graph allowedToolServerIds is invalid.")
    limits = spec.get("limits", {})
    if not isinstance(limits, dict) or set(limits) - {
        "maxNodes",
        "maxRunSeconds",
        "maxOutputBytes",
        "maxIterations",
        "maxOutputTokens",
    }:
        raise AgentWorkspaceError("Graph limits are invalid.")
    normalized_limits = {
        "maxNodes": _bounded_int(
            limits.get("maxNodes", len(nodes)), "Graph maxNodes", 1, _MAX_NODES
        ),
        "maxRunSeconds": _bounded_int(
            limits.get("maxRunSeconds", 3600), "Graph maxRunSeconds", 1, _MAX_RUN_SECONDS
        ),
        "maxOutputBytes": _bounded_int(
            limits.get("maxOutputBytes", _MAX_RUN_OUTPUT_BYTES),
            "Graph maxOutputBytes",
            1024,
            _MAX_RUN_OUTPUT_BYTES,
        ),
        "maxIterations": _bounded_int(
            limits.get("maxIterations", max(len(nodes), 100)),
            "Graph maxIterations",
            1,
            _MAX_RUN_ITERATIONS,
        ),
        "maxOutputTokens": _bounded_int(
            limits.get("maxOutputTokens", 262_144),
            "Graph maxOutputTokens",
            1,
            _MAX_RUN_OUTPUT_TOKENS,
        ),
    }
    if normalized_limits["maxNodes"] < len(nodes):
        raise AgentWorkspaceError("Graph maxNodes cannot be lower than the node count.")
    if normalized_limits["maxIterations"] < len(nodes):
        raise AgentWorkspaceError("Graph maxIterations cannot be lower than the node count.")
    result = {
        "name": name,
        "description": description,
        "metadata": metadata,
        "inputSchema": input_schema,
        "outputSchema": output_schema,
        "nodes": nodes,
        "edges": edges,
        "permissions": {"allowedToolServerIds": sorted(set(allowed_tools))},
        "limits": normalized_limits,
    }
    _json(result, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition")
    return result


def _table_exists(conn, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _create_tool_effects_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_graph_tool_effects (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            graph_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            arguments_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            UNIQUE(project_id, server_id, tool_name, idempotency_key)
        )
        """
    )


def _migrate_tool_effects_schema(conn) -> None:
    legacy_table = "agent_graph_tool_effects_legacy"
    legacy_exists = _table_exists(conn, legacy_table)
    foreign_tables = {
        row[2]
        for row in conn.execute("PRAGMA foreign_key_list(agent_graph_tool_effects)").fetchall()
    }
    needs_rebuild = bool(foreign_tables & {"agent_graphs", "agent_graph_runs"})
    if not needs_rebuild and not legacy_exists:
        return
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP INDEX IF EXISTS idx_agent_graph_tool_effects_run")
        if needs_rebuild:
            if legacy_exists:
                raise AgentWorkspaceError(
                    "Graph tool-effect migration found conflicting legacy tables."
                )
            conn.execute(
                "ALTER TABLE agent_graph_tool_effects RENAME TO agent_graph_tool_effects_legacy"
            )
            legacy_exists = True
            _create_tool_effects_table(conn)
        if legacy_exists:
            conflict = conn.execute(
                """
                SELECT 1
                FROM agent_graph_tool_effects_legacy AS legacy
                JOIN agent_graph_tool_effects AS current
                  ON current.id = legacy.id
                  OR (
                    current.project_id = legacy.project_id
                    AND current.server_id = legacy.server_id
                    AND current.tool_name = legacy.tool_name
                    AND current.idempotency_key = legacy.idempotency_key
                  )
                WHERE NOT (
                    current.id IS legacy.id
                    AND current.project_id IS legacy.project_id
                    AND current.graph_id IS legacy.graph_id
                    AND current.run_id IS legacy.run_id
                    AND current.node_id IS legacy.node_id
                    AND current.server_id IS legacy.server_id
                    AND current.tool_name IS legacy.tool_name
                    AND current.idempotency_key IS legacy.idempotency_key
                    AND current.arguments_hash IS legacy.arguments_hash
                    AND current.status IS legacy.status
                    AND current.output_json IS legacy.output_json
                    AND current.error IS legacy.error
                    AND current.created_at IS legacy.created_at
                    AND current.updated_at IS legacy.updated_at
                    AND current.completed_at IS legacy.completed_at
                )
                LIMIT 1
                """
            ).fetchone()
            if conflict is not None:
                raise AgentWorkspaceError(
                    "Graph tool-effect migration found conflicting idempotency receipts."
                )
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_graph_tool_effects(
                    id, project_id, graph_id, run_id, node_id, server_id, tool_name,
                    idempotency_key, arguments_hash, status, output_json, error,
                    created_at, updated_at, completed_at
                )
                SELECT
                    id, project_id, graph_id, run_id, node_id, server_id, tool_name,
                    idempotency_key, arguments_hash, status, output_json, error,
                    created_at, updated_at, completed_at
                FROM agent_graph_tool_effects_legacy
                """
            )
            missing = conn.execute(
                """
                SELECT 1
                FROM agent_graph_tool_effects_legacy AS legacy
                LEFT JOIN agent_graph_tool_effects AS current ON current.id = legacy.id
                WHERE current.id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if missing is not None:
                raise AgentWorkspaceError("Graph tool-effect receipt migration was incomplete.")
            conn.execute("DROP TABLE agent_graph_tool_effects_legacy")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_graph_tool_effects_run "
            "ON agent_graph_tool_effects(run_id, node_id, created_at)"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_schema(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_graphs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            current_revision INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(project_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graphs_project
            ON agent_graphs(project_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS agent_graph_revisions (
            graph_id TEXT NOT NULL REFERENCES agent_graphs(id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            document_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY(graph_id, revision)
        );
        CREATE TABLE IF NOT EXISTS agent_graph_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            graph_id TEXT NOT NULL REFERENCES agent_graphs(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            resource_bindings_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT,
            error TEXT,
            current_node_id TEXT,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            retry_of_run_id TEXT REFERENCES agent_graph_runs(id) ON DELETE SET NULL,
            idempotency_key TEXT,
            iteration_count INTEGER NOT NULL DEFAULT 0,
            reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
            pause_requested INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_runs_project
            ON agent_graph_runs(project_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS agent_graph_node_executions (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT,
            output_json TEXT,
            checkpoint_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            UNIQUE(run_id, node_id, attempt)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_node_runs
            ON agent_graph_node_executions(run_id, created_at);
        CREATE TABLE IF NOT EXISTS agent_graph_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            node_id TEXT,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(run_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS agent_graph_approvals (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            decision TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(run_id, node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_approvals_project
            ON agent_graph_approvals(project_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS agent_graph_tool_effects (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            graph_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            arguments_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            UNIQUE(project_id, server_id, tool_name, idempotency_key)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_tool_effects_run
            ON agent_graph_tool_effects(run_id, node_id, created_at);
        """
    )
    _migrate_tool_effects_schema(conn)
    run_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_graph_runs)").fetchall()}
    if "iteration_count" not in run_columns:
        conn.execute(
            "ALTER TABLE agent_graph_runs ADD COLUMN iteration_count INTEGER NOT NULL DEFAULT 0"
        )
    if "reserved_output_tokens" not in run_columns:
        conn.execute(
            "ALTER TABLE agent_graph_runs ADD COLUMN reserved_output_tokens INTEGER NOT NULL DEFAULT 0"
        )
    if "resource_bindings_json" not in run_columns:
        conn.execute(
            "ALTER TABLE agent_graph_runs ADD COLUMN resource_bindings_json TEXT NOT NULL DEFAULT '{}'"
        )
    execution_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(agent_graph_node_executions)").fetchall()
    }
    if "checkpoint_json" not in execution_columns:
        conn.execute("ALTER TABLE agent_graph_node_executions ADD COLUMN checkpoint_json TEXT")
    index_columns = [
        row[2]
        for row in conn.execute("PRAGMA index_info(idx_agent_graph_runs_idempotency)").fetchall()
    ]
    if index_columns != ["project_id", "graph_id", "idempotency_key"]:
        conn.execute("DROP INDEX IF EXISTS idx_agent_graph_runs_idempotency")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_graph_runs_idempotency "
            "ON agent_graph_runs(project_id, graph_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
    conn.commit()


def _conn():
    conn = connection()
    _ensure_schema(conn)
    return conn


def _revision_document(row) -> dict:
    document = _load(row["document_json"], {})
    nodes = []
    for raw_node in document.get("nodes", []):
        node = dict(raw_node)
        node.setdefault(
            "retryPolicy",
            {"maxAttempts": 1, "backoffMs": 0, "retryOn": ["error", "timeout"]},
        )
        config = dict(node.get("config") or {})
        if node.get("type") == "tool":
            config.setdefault("sideEffecting", True)
        node["config"] = config
        nodes.append(node)
    document["nodes"] = nodes
    limits = dict(document.get("limits") or {})
    limits.setdefault("maxNodes", max(1, len(nodes)))
    limits.setdefault("maxRunSeconds", 3600)
    limits.setdefault("maxOutputBytes", _MAX_RUN_OUTPUT_BYTES)
    limits.setdefault("maxIterations", max(len(nodes), 100))
    limits.setdefault("maxOutputTokens", 262_144)
    document["limits"] = limits
    return {
        **document,
        "graphId": row["graph_id"],
        "projectId": row["project_id"],
        "revision": row["revision"],
        "createdAt": row["created_at"],
    }


def _graph(conn, row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "description": row["description"],
        "currentRevision": row["current_revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def create_graph(project_id: str, spec: dict) -> dict:
    document = validate_graph_spec(spec)
    graph_id = str(uuid.uuid4())
    current = now_ms()
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO agent_graphs(id, project_id, name, description, current_revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (graph_id, project_id, document["name"], document["description"], current, current),
        )
        conn.execute(
            "INSERT INTO agent_graph_revisions(graph_id, project_id, revision, document_json, created_at) "
            "VALUES (?, ?, 1, ?, ?)",
            (
                graph_id,
                project_id,
                _json(document, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition"),
                current,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graphs WHERE id = ?", (graph_id,)).fetchone()
        return _graph(conn, row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise AgentWorkspaceError("A graph with this name already exists in the project.") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_graph(project_id: str, graph_id: str, spec: dict, *, expected_revision: int) -> dict:
    document = validate_graph_spec(spec)
    current = now_ms()
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph not found.")
        if row["current_revision"] != expected_revision:
            raise AgentWorkspaceError("Graph changed in another session. Refresh and retry.")
        revision = int(row["current_revision"]) + 1
        conn.execute(
            "INSERT INTO agent_graph_revisions(graph_id, project_id, revision, document_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                graph_id,
                project_id,
                revision,
                _json(document, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition"),
                current,
            ),
        )
        conn.execute(
            "UPDATE agent_graphs SET name = ?, description = ?, current_revision = ?, updated_at = ? WHERE id = ?",
            (document["name"], document["description"], revision, current, graph_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM agent_graphs WHERE id = ?", (graph_id,)).fetchone()
        return _graph(conn, updated)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise AgentWorkspaceError("A graph with this name already exists in the project.") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph(project_id: str, graph_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        ).fetchone()
        return _graph(conn, row) if row else None
    finally:
        conn.close()


def list_graphs(project_id: str) -> list[dict]:
    conn = _conn()
    try:
        return [
            _graph(conn, row)
            for row in conn.execute(
                "SELECT * FROM agent_graphs WHERE project_id = ? ORDER BY updated_at DESC, id",
                (project_id,),
            ).fetchall()
        ]
    finally:
        conn.close()


def delete_graph(project_id: str, graph_id: str) -> None:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id FROM agent_graphs WHERE id = ? AND project_id = ?",
            (graph_id, project_id),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph not found.")
        active = conn.execute(
            "SELECT 1 FROM agent_graph_runs WHERE graph_id = ? AND status IN "
            "('queued', 'running', 'pausing', 'paused', 'cancelling') LIMIT 1",
            (graph_id,),
        ).fetchone()
        if active is not None:
            raise AgentWorkspaceError("Stop active graph runs before deleting this graph.")
        conn.execute(
            "DELETE FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph_revision(
    project_id: str,
    graph_id: str,
    revision: Optional[int] = None,
) -> Optional[dict]:
    conn = _conn()
    try:
        if revision is None:
            row = conn.execute(
                "SELECT r.* FROM agent_graph_revisions r JOIN agent_graphs g ON g.id = r.graph_id "
                "WHERE r.project_id = ? AND r.graph_id = ? AND r.revision = g.current_revision",
                (project_id, graph_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM agent_graph_revisions WHERE project_id = ? AND graph_id = ? AND revision = ?",
                (project_id, graph_id, revision),
            ).fetchone()
        return _revision_document(row) if row else None
    finally:
        conn.close()


def list_graph_revisions(
    project_id: str,
    graph_id: str,
    limit: int = 100,
) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT graph_id, project_id, revision, document_json, created_at "
            "FROM agent_graph_revisions WHERE project_id = ? AND graph_id = ? "
            "ORDER BY revision DESC LIMIT ?",
            (project_id, graph_id, max(1, min(limit, 500))),
        ).fetchall()
        return [
            {
                "graphId": row["graph_id"],
                "projectId": row["project_id"],
                "revision": row["revision"],
                "name": _load(row["document_json"], {}).get("name", ""),
                "description": _load(row["document_json"], {}).get("description", ""),
                "createdAt": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def _run(row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "graphId": row["graph_id"],
        "revision": row["revision"],
        "input": _load(row["input_json"], {}),
        "output": _load(row["output_json"], None),
        "error": row["error"],
        "currentNodeId": row["current_node_id"],
        "status": row["status"],
        "attempt": row["attempt"],
        "retryOfRunId": row["retry_of_run_id"],
        "idempotencyKey": row["idempotency_key"],
        "iterationCount": row["iteration_count"],
        "reservedOutputTokens": row["reserved_output_tokens"],
        "pauseRequested": bool(row["pause_requested"]),
        "cancelRequested": bool(row["cancel_requested"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def _insert_graph_event(
    conn: sqlite3.Connection,
    run_id: str,
    event_type: str,
    *,
    node_id: Optional[str] = None,
    payload: Optional[dict] = None,
    created_at: Optional[int] = None,
) -> dict:
    event_type = _string(event_type, "Graph event type", maximum = 128)
    payload = payload or {}
    sequence_row = conn.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM agent_graph_events WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    sequence = int(sequence_row["next"])
    event_id = str(uuid.uuid4())
    current = now_ms() if created_at is None else created_at
    conn.execute(
        "INSERT INTO agent_graph_events(id, run_id, sequence, event_type, node_id, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            run_id,
            sequence,
            event_type,
            node_id,
            _json(payload, limit = 64 * 1024, label = "Graph event"),
            current,
        ),
    )
    return {
        "id": event_id,
        "runId": run_id,
        "sequence": sequence,
        "type": event_type,
        "nodeId": node_id,
        "payload": payload,
        "createdAt": current,
    }


def _resource_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii = False,
        allow_nan = False,
        sort_keys = True,
        separators = (",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mcp_resource_binding(server_id: str, server: Any = _UNSET) -> dict:
    server = mcp_servers_db.get_server(server_id) if server is _UNSET else server
    if server is None or not server.get("is_enabled"):
        raise AgentWorkspaceError(
            f"MCP server '{server_id}' is unavailable. Connect it before starting this graph."
        )
    url = str(server.get("url") or "")
    if not url:
        raise AgentWorkspaceError(f"MCP server '{server_id}' has no endpoint.")
    headers = parse_server_headers(server)
    if headers is None and isinstance(server.get("headers"), dict):
        headers = server["headers"]
    if server.get("headers_json") and headers is None:
        raise AgentWorkspaceError(f"MCP server '{server_id}' has invalid saved headers.")
    use_oauth = bool(server.get("use_oauth"))
    oauth_binding = None
    if use_oauth:
        try:
            oauth_binding = oauth_credential_binding(url)
        except Exception as exc:
            raise AgentWorkspaceError(
                f"MCP server '{server_id}' OAuth account could not be bound. Reconnect it first."
            ) from exc
        if oauth_binding is None:
            raise AgentWorkspaceError(
                f"MCP server '{server_id}' OAuth account is not connected. Connect it before starting this graph."
            )
    return {
        "type": "mcp",
        "serverId": server_id,
        "configurationDigest": _resource_digest(
            {
                "url": url,
                "headers": headers or {},
                "useOauth": use_oauth,
            }
        ),
        "oauthCredentialDigest": oauth_binding,
    }


def _capture_graph_resource_bindings(document: dict) -> dict:
    from .inference_executor import capture_runtime_snapshot

    bindings: dict[str, dict] = {}
    for node in document.get("nodes", []):
        node_type = node.get("type")
        config = node.get("config") or {}
        if node_type in {"loop", "model"}:
            runtime = config.get("runtime")
            if runtime is None:
                raise AgentWorkspaceError(
                    "This graph revision predates durable runtime selection. Save a new revision before running it."
                )
            bindings[node["id"]] = {
                "type": "runtime",
                "snapshot": capture_runtime_snapshot(runtime),
            }
        elif node_type == "tool":
            bindings[node["id"]] = _mcp_resource_binding(config["serverId"])
    return {"version": 1, "nodes": bindings}


def _graph_node_resource_binding(run_id: str, node_id: str, expected_type: str) -> dict:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT resource_bindings_json FROM agent_graph_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        bindings = _load(row["resource_bindings_json"], {})
    finally:
        conn.close()
    binding = (bindings.get("nodes") or {}).get(node_id) if isinstance(bindings, dict) else None
    if (
        not isinstance(bindings, dict)
        or bindings.get("version") != 1
        or not isinstance(binding, dict)
        or binding.get("type") != expected_type
    ):
        raise AgentWorkspaceError(
            "This graph run predates durable resource binding. Start a new run from a current revision."
        )
    return binding


def _bound_mcp_server(run_id: str, node_id: str, server_id: str) -> dict:
    expected = _graph_node_resource_binding(run_id, node_id, "mcp")
    current_server = mcp_servers_db.get_server(server_id)
    current = _mcp_resource_binding(server_id, current_server)
    if current != expected:
        raise AgentWorkspaceError(
            "The MCP server endpoint or credential changed after this run was created. Start a new run."
        )
    return current_server


def create_graph_run(
    project_id: str,
    graph_id: str,
    input_data: Any,
    *,
    revision: Optional[int] = None,
    idempotency_key: Optional[str] = None,
    retry_of_run_id: Optional[str] = None,
    attempt: int = 1,
) -> dict:
    if not isinstance(input_data, dict):
        raise AgentWorkspaceError("Graph run input must be an object.")
    encoded_input = _json(input_data, limit = _MAX_JSON_BYTES, label = "Graph run input")
    attempt = _bounded_int(attempt, "Graph run attempt", 1, 1000)
    conn = _conn()
    try:
        graph = conn.execute(
            "SELECT current_revision FROM agent_graphs WHERE id = ? AND project_id = ?",
            (graph_id, project_id),
        ).fetchone()
        if graph is None:
            raise AgentWorkspaceError("Graph not found.")
        selected_revision = int(graph["current_revision"] if revision is None else revision)
        document_row = conn.execute(
            "SELECT document_json FROM agent_graph_revisions WHERE graph_id = ? AND project_id = ? AND revision = ?",
            (graph_id, project_id, selected_revision),
        ).fetchone()
        if document_row is None:
            raise AgentWorkspaceError("Graph revision not found.")
        document = _load(document_row["document_json"], {})
        _validate_schema_value(input_data, document.get("inputSchema", {}), "Graph input")
        if idempotency_key is not None:
            idempotency_key = _string(idempotency_key, "Graph idempotency key", maximum = 256)
            existing = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? AND graph_id = ? AND idempotency_key = ?",
                (project_id, graph_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_input = _json(
                    _load(existing["input_json"], {}),
                    limit = _MAX_JSON_BYTES,
                    label = "Existing graph run input",
                )
                if (
                    int(existing["revision"]) != selected_revision
                    or existing_input != encoded_input
                ):
                    raise AgentWorkspaceError(
                        "Graph idempotency key was already used with different input or revision."
                    )
                return _run(existing)
        resource_bindings = _capture_graph_resource_bindings(document)
        conn.execute("BEGIN IMMEDIATE")
        revision_still_exists = conn.execute(
            "SELECT 1 FROM agent_graph_revisions WHERE graph_id = ? AND project_id = ? AND revision = ?",
            (graph_id, project_id, selected_revision),
        ).fetchone()
        if revision_still_exists is None:
            raise AgentWorkspaceError("Graph revision not found.")
        if idempotency_key is not None:
            existing = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? AND graph_id = ? AND idempotency_key = ?",
                (project_id, graph_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_input = _json(
                    _load(existing["input_json"], {}),
                    limit = _MAX_JSON_BYTES,
                    label = "Existing graph run input",
                )
                if (
                    int(existing["revision"]) != selected_revision
                    or existing_input != encoded_input
                ):
                    raise AgentWorkspaceError(
                        "Graph idempotency key was already used with different input or revision."
                    )
                conn.commit()
                return _run(existing)
        run_id = str(uuid.uuid4())
        current = now_ms()
        conn.execute(
            "INSERT INTO agent_graph_runs(id, project_id, graph_id, revision, input_json, resource_bindings_json, "
            "status, attempt, retry_of_run_id, idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
            (
                run_id,
                project_id,
                graph_id,
                selected_revision,
                encoded_input,
                _json(
                    resource_bindings,
                    limit = _MAX_GRAPH_DOCUMENT_BYTES,
                    label = "Graph resource bindings",
                ),
                attempt,
                retry_of_run_id,
                idempotency_key,
                current,
                current,
            ),
        )
        _insert_graph_event(
            conn,
            run_id,
            "run.created",
            payload = {"revision": selected_revision, "attempt": attempt},
            created_at = current,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph_run(project_id: str, run_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graph_runs WHERE id = ? AND project_id = ?", (run_id, project_id)
        ).fetchone()
        return _run(row) if row else None
    finally:
        conn.close()


def list_graph_runs(
    project_id: str,
    graph_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    conn = _conn()
    try:
        if graph_id is None:
            rows = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, max(1, min(limit, 500))),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? AND graph_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, graph_id, max(1, min(limit, 500))),
            ).fetchall()
        return [_run(row) for row in rows]
    finally:
        conn.close()


def _list_active_graph_runs(project_id: str) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_graph_runs WHERE project_id = ? AND status IN "
            "('queued', 'running', 'pausing', 'paused', 'cancelling') ORDER BY created_at, id",
            (project_id,),
        ).fetchall()
        return [_run(row) for row in rows]
    finally:
        conn.close()


def claim_graph_run(run_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = now_ms()
        cursor = conn.execute(
            "UPDATE agent_graph_runs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (current, current, run_id),
        )
        if not cursor.rowcount:
            conn.commit()
            return None
        _insert_graph_event(conn, run_id, "run.started", created_at = current)
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_graph_run(
    run_id: str,
    *,
    status: Optional[str] = None,
    output: Any = None,
    error: Optional[str] = None,
    current_node_id: Any = _UNSET,
) -> Optional[dict]:
    if status is not None and status not in _GRAPH_STATUSES:
        raise AgentWorkspaceError("Invalid graph run status.")
    if output is not None:
        _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph run output")
    assignments = ["updated_at = ?"]
    values: list[Any] = [now_ms()]
    if status is not None:
        assignments.append("status = ?")
        values.append(status)
        if status in {"cancelled", "completed", "failed", "interrupted"}:
            assignments.append("completed_at = ?")
            values.append(now_ms())
    if output is not None:
        assignments.append("output_json = ?")
        values.append(_json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph run output"))
    if error is not None:
        assignments.append("error = ?")
        values.append(str(error)[:8000])
    if current_node_id is not _UNSET:
        assignments.append("current_node_id = ?")
        values.append(current_node_id)
    values.append(run_id)
    conn = _conn()
    try:
        conn.execute(f"UPDATE agent_graph_runs SET {', '.join(assignments)} WHERE id = ?", values)
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None
    finally:
        conn.close()


def finish_graph_run(
    run_id: str,
    status: str,
    event_type: str,
    *,
    output: Any = _UNSET,
    error: Optional[str] = None,
    current_node_id: Any = _UNSET,
    node_id: Optional[str] = None,
    payload: Optional[dict] = None,
    respect_control_requests: bool = False,
) -> dict:
    """Commit a run state transition and its durable event together."""
    if status not in _GRAPH_STATUSES:
        raise AgentWorkspaceError("Invalid graph run status.")
    encoded_output = (
        _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph run output")
        if output is not _UNSET
        else None
    )
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] in {"cancelled", "completed", "failed", "interrupted"}:
            conn.commit()
            return _run(row)
        resolved_status = status
        resolved_event_type = event_type
        resolved_error = error
        if row["cancel_requested"] or row["status"] == "cancelling":
            resolved_status = "cancelled"
            resolved_event_type = "run.cancelled"
            resolved_error = "Graph run cancelled."
            payload = {"error": resolved_error}
            encoded_output = None
        elif respect_control_requests and (row["pause_requested"] or row["status"] == "pausing"):
            resolved_status = "paused"
            resolved_event_type = "run.paused"
            encoded_output = None
        current = now_ms()
        assignments = ["status = ?", "updated_at = ?"]
        values: list[Any] = [resolved_status, current]
        if resolved_status in {"cancelled", "completed", "failed", "interrupted"}:
            assignments.append("completed_at = ?")
            values.append(current)
        if encoded_output is not None:
            assignments.append("output_json = ?")
            values.append(encoded_output)
        if resolved_error is not None:
            assignments.append("error = ?")
            values.append(str(resolved_error)[:8000])
        if current_node_id is not _UNSET:
            assignments.append("current_node_id = ?")
            values.append(current_node_id)
        if resolved_status == "cancelled":
            assignments.extend(["cancel_requested = 1", "pause_requested = 0"])
        values.append(run_id)
        conn.execute(
            f"UPDATE agent_graph_runs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        _insert_graph_event(
            conn,
            run_id,
            resolved_event_type,
            node_id = node_id,
            payload = payload,
            created_at = current,
        )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def consume_graph_budget(
    run_id: str,
    *,
    max_iterations: int,
    max_output_tokens: int,
    iterations: int = 0,
    output_tokens: int = 0,
) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT iteration_count, reserved_output_tokens FROM agent_graph_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        next_iterations = int(row["iteration_count"]) + max(0, int(iterations))
        next_tokens = int(row["reserved_output_tokens"]) + max(0, int(output_tokens))
        if next_iterations > max_iterations:
            raise AgentWorkspaceError("Graph iteration budget exhausted.")
        if next_tokens > max_output_tokens:
            raise AgentWorkspaceError("Graph output token budget exhausted.")
        conn.execute(
            "UPDATE agent_graph_runs SET iteration_count = ?, reserved_output_tokens = ?, updated_at = ? "
            "WHERE id = ?",
            (next_iterations, next_tokens, now_ms(), run_id),
        )
        conn.commit()
        return {
            "iterationCount": next_iterations,
            "reservedOutputTokens": next_tokens,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reserve_graph_task_output_tokens(
    run_id: str,
    execution_id: str,
    reservation_id: str,
    *,
    max_output_tokens: int,
    output_tokens: int,
) -> dict:
    """Reserve one task's maximum output and checkpoint it in the same commit."""
    reservation_id = _string(reservation_id, "Graph task reservation ID", maximum = 256)
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        execution = conn.execute(
            "SELECT checkpoint_json FROM agent_graph_node_executions WHERE id = ? AND run_id = ?",
            (execution_id, run_id),
        ).fetchone()
        if execution is None:
            raise AgentWorkspaceError("Graph node execution not found.")
        checkpoint = _load(execution["checkpoint_json"], {})
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        if checkpoint.get("outputTokenReservationId") == reservation_id:
            conn.commit()
            return checkpoint
        run = conn.execute(
            "SELECT reserved_output_tokens FROM agent_graph_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise AgentWorkspaceError("Graph run not found.")
        next_tokens = int(run["reserved_output_tokens"]) + max(0, int(output_tokens))
        if next_tokens > max_output_tokens:
            raise AgentWorkspaceError("Graph output token budget exhausted.")
        checkpoint = {**checkpoint, "outputTokenReservationId": reservation_id}
        encoded_checkpoint = _json(
            checkpoint,
            limit = 64 * 1024,
            label = "Graph node checkpoint",
        )
        current = now_ms()
        conn.execute(
            "UPDATE agent_graph_runs SET reserved_output_tokens = ?, updated_at = ? WHERE id = ?",
            (next_tokens, current, run_id),
        )
        conn.execute(
            "UPDATE agent_graph_node_executions SET checkpoint_json = ? WHERE id = ?",
            (encoded_checkpoint, execution_id),
        )
        conn.commit()
        return checkpoint
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def request_graph_pause(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] == "queued":
            status = "paused"
        elif row["status"] == "running":
            status = "pausing"
        else:
            raise AgentWorkspaceError("Only queued or running graph runs can be paused.")
        current = now_ms()
        conn.execute(
            "UPDATE agent_graph_runs SET status = ?, pause_requested = 1, updated_at = ? WHERE id = ?",
            (status, current, run_id),
        )
        if status == "paused":
            _insert_graph_event(
                conn,
                run_id,
                "run.paused",
                node_id = row["current_node_id"],
                created_at = current,
            )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def resume_graph_run(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["cancel_requested"]:
            raise AgentWorkspaceError("A cancelled graph run cannot be resumed.")
        if row["status"] not in {"paused", "interrupted"}:
            raise AgentWorkspaceError("Only paused or interrupted graph runs can be resumed.")
        conn.execute(
            "UPDATE agent_graph_runs SET status = 'queued', pause_requested = 0, "
            "error = NULL, completed_at = NULL, updated_at = ? WHERE id = ?",
            (now_ms(), run_id),
        )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def request_graph_cancel(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] == "queued":
            status = "cancelled"
        elif row["status"] == "paused":
            status = "cancelled"
        elif row["status"] in {"running", "pausing"}:
            status = "cancelling"
        else:
            raise AgentWorkspaceError("Only active graph runs can be cancelled.")
        completed_at = now_ms() if status == "cancelled" else None
        if completed_at is None:
            conn.execute(
                "UPDATE agent_graph_runs SET status = ?, cancel_requested = 1, pause_requested = 0, updated_at = ? WHERE id = ?",
                (status, now_ms(), run_id),
            )
        else:
            conn.execute(
                "UPDATE agent_graph_runs SET status = ?, cancel_requested = 1, pause_requested = 0, "
                "error = COALESCE(error, 'Graph run cancelled.'), updated_at = ?, completed_at = ? "
                "WHERE id = ?",
                (status, completed_at, completed_at, run_id),
            )
            conn.execute(
                "UPDATE agent_graph_node_executions SET status = 'cancelled', "
                "error = COALESCE(error, 'Graph run cancelled.'), completed_at = ? "
                "WHERE run_id = ? AND status IN ('running', 'paused')",
                (completed_at, run_id),
            )
            _insert_graph_event(
                conn,
                run_id,
                "run.cancelled",
                node_id = row["current_node_id"],
                created_at = completed_at,
            )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def admit_node_execution(
    run_id: str,
    node: dict,
    input_value: Any,
    attempt: int,
    *,
    max_iterations: int,
    checkpoint: Optional[dict] = None,
) -> dict:
    """Atomically spend one iteration, create its execution, and record admission."""
    attempt = _bounded_int(attempt, "Graph node attempt", 1, _MAX_NODE_ATTEMPTS)
    encoded_input = _json(
        input_value,
        limit = _MAX_RUN_OUTPUT_BYTES,
        label = "Graph node input",
    )
    encoded_checkpoint = (
        _json(checkpoint, limit = 64 * 1024, label = "Graph node checkpoint")
        if checkpoint is not None
        else None
    )
    execution_id = str(uuid.uuid4())
    current = now_ms()
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            "SELECT status, iteration_count, pause_requested, cancel_requested "
            "FROM agent_graph_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise AgentWorkspaceError("Graph run not found.")
        if run["status"] != "running" or run["pause_requested"] or run["cancel_requested"]:
            raise AgentWorkspaceError("Graph run control changed before node admission.")
        next_iterations = int(run["iteration_count"]) + 1
        if next_iterations > max_iterations:
            raise AgentWorkspaceError("Graph iteration budget exhausted.")
        conn.execute(
            "UPDATE agent_graph_runs SET iteration_count = ?, current_node_id = ?, updated_at = ? "
            "WHERE id = ?",
            (next_iterations, node["id"], current, run_id),
        )
        conn.execute(
            "INSERT INTO agent_graph_node_executions(id, run_id, node_id, node_type, attempt, "
            "status, input_json, checkpoint_json, created_at, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
            (
                execution_id,
                run_id,
                node["id"],
                node["type"],
                attempt,
                encoded_input,
                encoded_checkpoint,
                current,
                current,
            ),
        )
        _insert_graph_event(
            conn,
            run_id,
            "node.started",
            node_id = node["id"],
            payload = {"type": node["type"], "attempt": attempt},
            created_at = current,
        )
        conn.commit()
        return {
            "id": execution_id,
            "runId": run_id,
            "nodeId": node["id"],
            "nodeType": node["type"],
            "attempt": attempt,
            "status": "running",
            "checkpoint": checkpoint,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_node_execution(
    run_id: str,
    node: dict,
    input_value: Any,
    attempt: int,
    *,
    checkpoint: Optional[dict] = None,
) -> dict:
    execution_id = str(uuid.uuid4())
    current = now_ms()
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO agent_graph_node_executions(id, run_id, node_id, node_type, attempt, status, input_json, checkpoint_json, created_at, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
            (
                execution_id,
                run_id,
                node["id"],
                node["type"],
                attempt,
                _json(input_value, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node input"),
                (
                    _json(checkpoint, limit = 64 * 1024, label = "Graph node checkpoint")
                    if checkpoint is not None
                    else None
                ),
                current,
                current,
            ),
        )
        conn.commit()
        return {
            "id": execution_id,
            "runId": run_id,
            "nodeId": node["id"],
            "nodeType": node["type"],
            "attempt": attempt,
            "status": "running",
            "checkpoint": checkpoint,
        }
    finally:
        conn.close()


def finish_node_execution(
    execution_id: str,
    status: str,
    *,
    output: Any = None,
    error: Optional[str] = None,
) -> None:
    if status not in _NODE_STATUSES:
        raise AgentWorkspaceError("Invalid graph node status.")
    if output is not None:
        _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node output")
    conn = _conn()
    try:
        conn.execute(
            "UPDATE agent_graph_node_executions SET status = ?, output_json = ?, error = ?, "
            "completed_at = ? WHERE id = ? AND status != 'completed'",
            (
                status,
                _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node output")
                if output is not None
                else None,
                str(error)[:8000] if error else None,
                now_ms(),
                execution_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def complete_node_execution(
    run_id: str,
    execution_id: str,
    node_id: str,
    attempt: int,
    output: Any,
    next_node_id: Optional[str],
) -> None:
    """Commit node output, completion event, and run cursor atomically."""
    encoded_output = _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node output")
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = now_ms()
        cursor = conn.execute(
            "UPDATE agent_graph_node_executions SET status = 'completed', output_json = ?, "
            "error = NULL, completed_at = ? WHERE id = ? AND run_id = ? AND status = 'running'",
            (encoded_output, current, execution_id, run_id),
        )
        if cursor.rowcount != 1:
            raise AgentWorkspaceError("Graph node completion state changed concurrently.")
        _insert_graph_event(
            conn,
            run_id,
            "node.completed",
            node_id = node_id,
            payload = {"attempt": attempt},
            created_at = current,
        )
        conn.execute(
            "UPDATE agent_graph_runs SET current_node_id = ?, updated_at = ? WHERE id = ?",
            (next_node_id, current, run_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_node_checkpoint(execution_id: str, checkpoint: dict) -> None:
    encoded = _json(checkpoint, limit = 64 * 1024, label = "Graph node checkpoint")
    conn = _conn()
    try:
        conn.execute(
            "UPDATE agent_graph_node_executions SET checkpoint_json = ? WHERE id = ?",
            (encoded, execution_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_node_executions(project_id: str, run_id: str) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT e.* FROM agent_graph_node_executions e JOIN agent_graph_runs r ON r.id = e.run_id "
            "WHERE e.run_id = ? AND r.project_id = ? ORDER BY e.created_at, e.id",
            (run_id, project_id),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "runId": row["run_id"],
                "nodeId": row["node_id"],
                "nodeType": row["node_type"],
                "attempt": row["attempt"],
                "status": row["status"],
                "input": _load(row["input_json"], None),
                "output": _load(row["output_json"], None),
                "checkpoint": _load(row["checkpoint_json"], None),
                "error": row["error"],
                "createdAt": row["created_at"],
                "startedAt": row["started_at"],
                "completedAt": row["completed_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def append_graph_event(
    run_id: str,
    event_type: str,
    *,
    node_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        event = _insert_graph_event(
            conn,
            run_id,
            event_type,
            node_id = node_id,
            payload = payload,
        )
        conn.commit()
        return event
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_graph_events(
    project_id: str,
    run_id: str,
    after: int = 0,
    limit: int = 500,
) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT e.* FROM agent_graph_events e JOIN agent_graph_runs r ON r.id = e.run_id "
            "WHERE e.run_id = ? AND r.project_id = ? AND e.sequence > ? ORDER BY e.sequence LIMIT ?",
            (run_id, project_id, max(0, after), max(1, min(limit, 1000))),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "runId": row["run_id"],
                "sequence": row["sequence"],
                "type": row["event_type"],
                "nodeId": row["node_id"],
                "payload": _load(row["payload_json"], {}),
                "createdAt": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_or_create_approval(project_id: str, run_id: str, node: dict) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE run_id = ? AND node_id = ?",
            (run_id, node["id"]),
        ).fetchone()
        if row is None:
            approval_id = str(uuid.uuid4())
            current = now_ms()
            config = node["config"]
            conn.execute(
                "INSERT INTO agent_graph_approvals(id, project_id, run_id, node_id, title, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    approval_id,
                    project_id,
                    run_id,
                    node["id"],
                    config["title"],
                    config["description"],
                    current,
                    current,
                ),
            )
            _insert_graph_event(
                conn,
                run_id,
                "approval.required",
                node_id = node["id"],
                payload = {"approvalId": approval_id},
                created_at = current,
            )
            row = conn.execute(
                "SELECT * FROM agent_graph_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        conn.commit()
        return _approval(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _approval(row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "runId": row["run_id"],
        "nodeId": row["node_id"],
        "title": row["title"],
        "description": row["description"],
        "status": row["status"],
        "decision": row["decision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def get_graph_approval(project_id: str, run_id: str, approval_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE id = ? AND project_id = ? AND run_id = ?",
            (approval_id, project_id, run_id),
        ).fetchone()
        return _approval(row) if row else None
    finally:
        conn.close()


def list_graph_approvals(project_id: str, run_id: str) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT a.* FROM agent_graph_approvals a JOIN agent_graph_runs r ON r.id = a.run_id "
            "WHERE a.project_id = ? AND a.run_id = ? ORDER BY a.created_at, a.id",
            (project_id, run_id),
        ).fetchall()
        return [_approval(row) for row in rows]
    finally:
        conn.close()


def decide_graph_approval(project_id: str, run_id: str, approval_id: str, decision: str) -> dict:
    if decision not in {"approved", "rejected"}:
        raise AgentWorkspaceError("Approval decision is invalid.")
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE id = ? AND project_id = ? AND run_id = ?",
            (approval_id, project_id, run_id),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph approval not found.")
        if row["status"] != "pending":
            raise AgentWorkspaceError("Graph approval has already been decided.")
        run = conn.execute(
            "SELECT status, current_node_id, cancel_requested FROM agent_graph_runs "
            "WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
        if run is None:
            raise AgentWorkspaceError("Graph run not found.")
        if (
            run["status"] in {"cancelling", "cancelled", "completed", "failed"}
            or run["cancel_requested"]
        ):
            raise AgentWorkspaceError("This graph run is no longer awaiting approval.")
        if run["current_node_id"] not in {None, row["node_id"]}:
            raise AgentWorkspaceError("This graph run is no longer at the approval node.")
        current = now_ms()
        conn.execute(
            "UPDATE agent_graph_approvals SET status = ?, decision = ?, updated_at = ? WHERE id = ?",
            (decision, decision, current, approval_id),
        )
        _insert_graph_event(
            conn,
            run_id,
            "approval.decided",
            node_id = row["node_id"],
            payload = {"approvalId": approval_id, "decision": decision},
            created_at = current,
        )
        result = _approval(
            conn.execute(
                "SELECT * FROM agent_graph_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return result


def _begin_tool_effect(
    run: dict, node: dict, arguments: dict, idempotency_key: str
) -> tuple[str, Optional[str], Any]:
    encoded_arguments = _json(arguments, limit = 64 * 1024, label = "Tool arguments")
    arguments_hash = hashlib.sha256(encoded_arguments.encode("utf-8")).hexdigest()
    config = node["config"]
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM agent_graph_tool_effects WHERE project_id = ? AND server_id = ? "
            "AND tool_name = ? AND idempotency_key = ?",
            (
                run["projectId"],
                config["serverId"],
                config["toolName"],
                idempotency_key,
            ),
        ).fetchone()
        if existing is not None:
            if existing["arguments_hash"] != arguments_hash:
                raise AgentWorkspaceError(
                    "Tool idempotency key was already used with different arguments."
                )
            if existing["status"] == "completed":
                conn.commit()
                return "cached", existing["id"], _load(existing["output_json"], None)
            raise _GraphToolEffectUncertain(
                "Tool effect state is uncertain. Verify the external system before using a new idempotency key."
            )
        effect_id = str(uuid.uuid4())
        current = now_ms()
        conn.execute(
            "INSERT INTO agent_graph_tool_effects(id, project_id, graph_id, run_id, node_id, "
            "server_id, tool_name, idempotency_key, arguments_hash, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
            (
                effect_id,
                run["projectId"],
                run["graphId"],
                run["id"],
                node["id"],
                config["serverId"],
                config["toolName"],
                idempotency_key,
                arguments_hash,
                current,
                current,
            ),
        )
        conn.commit()
        return "dispatch", effect_id, None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _complete_tool_effect(effect_id: str, output: Any) -> None:
    encoded = _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Tool output")
    conn = _conn()
    try:
        current = now_ms()
        conn.execute(
            "UPDATE agent_graph_tool_effects SET status = 'completed', output_json = ?, "
            "updated_at = ?, completed_at = ? WHERE id = ? AND status = 'running'",
            (encoded, current, current, effect_id),
        )
        conn.commit()
    finally:
        conn.close()


def _mark_tool_effect_uncertain(effect_id: str, error: str) -> None:
    conn = _conn()
    try:
        conn.execute(
            "UPDATE agent_graph_tool_effects SET status = 'uncertain', error = ?, updated_at = ? "
            "WHERE id = ? AND status = 'running'",
            (str(error)[:8000], now_ms(), effect_id),
        )
        conn.commit()
    finally:
        conn.close()


def recover_graph_runs() -> int:
    """Fence in-flight graph records after a process restart."""
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = now_ms()
        active_runs = conn.execute(
            "SELECT id, current_node_id, status, cancel_requested FROM agent_graph_runs "
            "WHERE status IN ('queued', 'running', 'pausing', 'cancelling')"
        ).fetchall()
        conn.execute(
            "UPDATE agent_graph_tool_effects SET status = 'uncertain', "
            "error = COALESCE(error, 'Studio restarted while the tool effect was active.'), updated_at = ? "
            "WHERE status = 'running'",
            (current,),
        )
        for run in active_runs:
            cancelled = bool(run["cancel_requested"]) or run["status"] == "cancelling"
            status = "cancelled" if cancelled else "interrupted"
            error = (
                "Graph run cancellation completed after Studio restarted."
                if cancelled
                else "Studio restarted while the graph was active."
            )
            conn.execute(
                "UPDATE agent_graph_runs SET status = ?, error = COALESCE(error, ?), "
                "pause_requested = 0, updated_at = ?, completed_at = ? WHERE id = ?",
                (status, error, current, current, run["id"]),
            )
            conn.execute(
                "UPDATE agent_graph_node_executions SET status = ?, error = COALESCE(error, ?), "
                "completed_at = ? WHERE run_id = ? AND status = 'running'",
                (status, error, current, run["id"]),
            )
            _insert_graph_event(
                conn,
                run["id"],
                "run." + status,
                node_id = run["current_node_id"],
                payload = {"error": error},
                created_at = current,
            )
        conn.commit()
        return len(active_runs)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _graph_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _render(
    value: Any,
    context: dict,
    *,
    exact_values: bool = False,
) -> Any:
    if isinstance(value, str):
        try:
            parsed = list(Formatter().parse(value))
            if exact_values and len(parsed) == 1:
                literal, field, format_spec, conversion = parsed[0]
                if literal == "" and field and not format_spec and not conversion:
                    return _graph_path(context, field)
            rendered = []
            for literal, field, format_spec, conversion in parsed:
                rendered.append(literal)
                if field is None:
                    continue
                if not field or format_spec or conversion:
                    raise ValueError("unsupported graph template field")
                rendered.append(json.dumps(_graph_path(context, field), ensure_ascii = False))
            return "".join(rendered)
        except (ValueError, IndexError) as exc:
            raise AgentWorkspaceError("Graph template references an invalid context path.") from exc
    if isinstance(value, dict):
        return {
            key: _render(item, context, exact_values = exact_values) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render(item, context, exact_values = exact_values) for item in value]
    return value


def _condition(value: Any, config: dict) -> bool:
    operator = config["operator"]
    if operator == "truthy":
        return bool(value)
    if operator == "falsy":
        return not bool(value)
    if operator == "exists":
        return value is not None
    if operator == "equals":
        return value == config.get("value")
    return value != config.get("value")


def _validate_schema_value(value: Any, schema: dict, label: str) -> None:
    """Validate the bounded JSON-schema subset used by graph inputs and outputs."""
    if not isinstance(schema, dict):
        raise AgentWorkspaceError(f"{label} schema is invalid.")
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected is not None and expected not in type_matches:
        raise AgentWorkspaceError(f"{label} schema type is invalid.")
    if expected is not None and not type_matches[expected]:
        raise AgentWorkspaceError(f"{label} does not match the graph schema.")
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise AgentWorkspaceError(f"{label} schema required fields are invalid.")
        missing = [item for item in required if item not in value]
        if missing:
            raise AgentWorkspaceError(f"{label} is missing required fields: {', '.join(missing)}.")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise AgentWorkspaceError(f"{label} schema properties are invalid.")
        if schema.get("additionalProperties") is False:
            unknown = [key for key in value if key not in properties]
            if unknown:
                raise AgentWorkspaceError(
                    f"{label} contains unsupported fields: {', '.join(unknown)}."
                )
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_value(value[key], child_schema, f"{label}.{key}")
    elif isinstance(value, list) and schema.get("items") is not None:
        if not isinstance(schema["items"], dict):
            raise AgentWorkspaceError(f"{label} schema items are invalid.")
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], f"{label}[{index}]")


def _validate_schema_definition(schema: dict, label: str) -> None:
    if not isinstance(schema, dict):
        raise AgentWorkspaceError(f"{label} schema is invalid.")
    expected = schema.get("type")
    if expected is not None and expected not in {
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
    }:
        raise AgentWorkspaceError(f"{label} schema type is invalid.")
    allowed = {"type"}
    if expected == "object":
        allowed.update({"properties", "required", "additionalProperties"})
    elif expected == "array":
        allowed.add("items")
    if set(schema) - allowed:
        raise AgentWorkspaceError(f"{label} schema contains unsupported fields.")
    if expected == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise AgentWorkspaceError(f"{label} schema is invalid.")
        if "additionalProperties" in schema and not isinstance(
            schema["additionalProperties"], bool
        ):
            raise AgentWorkspaceError(f"{label} schema additionalProperties is invalid.")
        if any(not isinstance(item, str) or item not in properties for item in required):
            raise AgentWorkspaceError(f"{label} schema required fields are invalid.")
        for key, child in properties.items():
            if not isinstance(key, str):
                raise AgentWorkspaceError(f"{label} schema property names are invalid.")
            _validate_schema_definition(child, f"{label}.{key}")
    if expected == "array" and schema.get("items") is not None:
        _validate_schema_definition(schema["items"], f"{label} items")


class GraphLoopAdapter:
    """Adapter boundary for the existing durable background-agent runtime."""

    def run(
        self,
        project_id: str,
        instruction: str,
        runtime: Optional[dict],
        cancel_event: Any,
        *,
        runtime_snapshot: Optional[dict] = None,
        checkpoint: Optional[dict] = None,
        checkpoint_callback: Optional[Callable[[dict], None]] = None,
        before_start: Optional[Callable[[str], None]] = None,
    ) -> dict:
        from .background import manager as background_manager

        task = None
        needs_start = False
        checkpoint_task_id = str((checkpoint or {}).get("backgroundTaskId") or "")
        if checkpoint_task_id:
            task = background_manager_task(checkpoint_task_id)
            if task and task["projectId"] != project_id:
                raise AgentWorkspaceError("Graph loop checkpoint belongs to another project.")
            if (
                task
                and runtime_snapshot is not None
                and (task.get("payload") or {}).get("runtime") != runtime_snapshot
            ):
                raise AgentWorkspaceError(
                    "Graph loop checkpoint runtime does not match the pinned run resource."
                )
            if task and task["status"] == "completed":
                result = task.get("result") or {}
                if checkpoint_callback:
                    checkpoint_callback(
                        {
                            "backgroundTaskId": task["id"],
                            "status": "completed",
                            "toolIterations": int(result.get("toolEvents") or 0) // 2,
                        }
                    )
                return result
            if task and task["status"] in {"failed", "cancelled", "interrupted"}:
                if (runtime or {}).get("permissionMode") != "off":
                    raise _GraphLoopEffectUncertain(
                        "The prior Loop task stopped after it may have changed the workspace. Inspect the project before starting a new run."
                    )
                task = None
                checkpoint_task_id = ""
            elif task and task["status"] == "queued":
                needs_start = True
        if task is None:
            allocated_task_id = checkpoint_task_id or str(uuid.uuid4())
            if checkpoint_callback:
                checkpoint_callback(
                    {
                        "backgroundTaskId": allocated_task_id,
                        "status": "allocated",
                        "toolIterations": 0,
                    }
                )
            if before_start:
                before_start(allocated_task_id)
            task = background_manager.enqueue_agent(
                project_id,
                instruction,
                task_id = allocated_task_id,
                runtime_selection = runtime if runtime_snapshot is None else None,
                runtime_snapshot = runtime_snapshot,
                start = False,
            )
            needs_start = True
            if checkpoint_callback:
                checkpoint_callback(
                    {
                        "backgroundTaskId": task["id"],
                        "status": "queued",
                        "toolIterations": 0,
                    }
                )
        if needs_start:
            task = background_manager.start(task["id"])
        last_status = None
        cancel_sent = False
        while True:
            current = background_manager_task(task["id"])
            if current is None:
                raise AgentWorkspaceError("Graph loop task disappeared before completion.")
            if current["status"] != last_status and checkpoint_callback:
                result = current.get("result") or {}
                checkpoint_callback(
                    {
                        "backgroundTaskId": task["id"],
                        "status": current["status"],
                        "toolIterations": int(result.get("toolEvents") or 0) // 2,
                    }
                )
                last_status = current["status"]
            if current["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                if current["status"] != "completed":
                    raise AgentWorkspaceError(current.get("error") or "Graph loop failed.")
                return current.get("result") or {}
            should_cancel = getattr(cancel_event, "should_cancel_work", cancel_event.is_set)
            if should_cancel() and not cancel_sent:
                background_manager.cancel(task["id"])
                cancel_sent = True
            time.sleep(0.025)


def background_manager_task(task_id: str) -> Optional[dict]:
    from .state import get_background_task
    return get_background_task(task_id)


class GraphRunManager:
    """Durable graph coordinator that delegates node work to existing runtimes."""

    def __init__(
        self,
        max_workers: int = 2,
        loop_adapter: Optional[GraphLoopAdapter] = None,
    ):
        self._executor = ThreadPoolExecutor(
            max_workers = max_workers, thread_name_prefix = "studio-graph-run"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._cancellations: dict[str, _RunControl] = {}
        self._deleting_projects: set[str] = set()
        self._stopping = threading.Event()
        self.loop_adapter = loop_adapter or GraphLoopAdapter()

    def begin_project_deletion(self, project_id: str) -> None:
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is already in progress.")
            self._deleting_projects.add(project_id)

    def finish_project_deletion(self, project_id: str) -> None:
        with self._lock:
            self._deleting_projects.discard(project_id)

    def enqueue(
        self,
        project_id: str,
        graph_id: str,
        input_data: dict,
        *,
        revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        start: bool = True,
    ) -> dict:
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
            run = create_graph_run(
                project_id, graph_id, input_data, revision = revision, idempotency_key = idempotency_key
            )
        if start and run["status"] == "queued":
            return self.start(run["id"])
        return run

    def start(self, run_id: str) -> dict:
        with self._lock:
            if self._stopping.is_set():
                raise AgentWorkspaceError("The graph coordinator is shutting down.")
            existing = self._get_any(run_id)
            if existing is None:
                raise AgentWorkspaceError("Graph run not found.")
            if existing["projectId"] in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
            claimed = claim_graph_run(run_id)
            if claimed is None:
                run = self._get_any(run_id) or existing
                if run["status"] != "running":
                    raise AgentWorkspaceError("Only queued graph runs can be started.")
                return run
            cancel_event = _RunControl()
            self._cancellations[run_id] = cancel_event
            try:
                future = self._executor.submit(self._run, run_id, cancel_event)
            except Exception as exc:
                self._cancellations.pop(run_id, None)
                finish_graph_run(
                    run_id,
                    "failed",
                    "run.failed",
                    error = "The graph coordinator could not start this run.",
                    payload = {"error": "The graph coordinator could not start this run."},
                )
                raise AgentWorkspaceError(
                    "The graph coordinator could not start this run."
                ) from exc
            self._futures[run_id] = future
        future.add_done_callback(lambda _future: self._forget(run_id))
        return claimed

    def _get_any(self, run_id: str) -> Optional[dict]:
        conn = _conn()
        try:
            row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
            return _run(row) if row else None
        finally:
            conn.close()

    def pause(self, run_id: str) -> dict:
        run = request_graph_pause(run_id)
        event = self._cancellations.get(run_id)
        if event:
            event.request("pause")
        return run

    def resume(self, run_id: str) -> dict:
        existing = self._get_any(run_id)
        if existing is None:
            raise AgentWorkspaceError("Graph run not found.")
        with self._lock:
            if existing["projectId"] in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
        run = resume_graph_run(run_id)
        return self.start(run_id)

    def cancel(self, run_id: str) -> dict:
        run = request_graph_cancel(run_id)
        event = self._cancellations.get(run_id)
        if event:
            event.request("cancel")
        return run

    def retry(
        self,
        project_id: str,
        run_id: str,
        *,
        start: bool = True,
    ) -> dict:
        previous = get_graph_run(project_id, run_id)
        if previous is None:
            raise AgentWorkspaceError("Graph run not found.")
        if previous["status"] not in {"failed", "cancelled", "interrupted"}:
            raise AgentWorkspaceError("Only stopped graph runs can be retried.")
        graph = get_graph_revision(project_id, previous["graphId"], previous["revision"])
        if graph is None:
            raise AgentWorkspaceError("Pinned graph revision is unavailable.")
        executed_nodes = {item["nodeId"] for item in list_node_executions(project_id, run_id)}
        unsafe_replays = [
            node["id"]
            for node in graph["nodes"]
            if node["id"] in executed_nodes
            and node["type"] in {"loop", "model"}
            and (node["config"].get("runtime") or {}).get("permissionMode") != "off"
        ]
        if unsafe_replays:
            raise AgentWorkspaceError(
                "This run used a Loop or model node that may have changed the workspace. Inspect the project and start a new run instead of replaying it."
            )
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
            run = create_graph_run(
                project_id,
                previous["graphId"],
                previous["input"],
                revision = previous["revision"],
                idempotency_key = f"graph-retry:{run_id}",
                retry_of_run_id = run_id,
                attempt = int(previous["attempt"]) + 1,
            )
            if run["retryOfRunId"] != run_id:
                raise AgentWorkspaceError("Graph retry idempotency key collision.")
        return self.start(run["id"]) if start and run["status"] == "queued" else run

    def _forget(self, run_id: str) -> None:
        with self._lock:
            self._futures.pop(run_id, None)
            self._cancellations.pop(run_id, None)

    def prepare_for_app_exit(self, timeout_seconds: float = 10) -> None:
        self._stopping.set()
        with self._lock:
            events = list(self._cancellations.values())
            futures = list(self._futures.values())
        for event in events:
            event.request("shutdown")
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout = remaining)
            except Exception:
                pass
        recover_graph_runs()

    def cancel_project_runs_and_wait(
        self,
        project_id: str,
        timeout_seconds: float = 30,
    ) -> list[dict]:
        active = _list_active_graph_runs(project_id)
        for run in active:
            try:
                self.cancel(run["id"])
            except AgentWorkspaceError:
                pass
        with self._lock:
            futures = [self._futures[run["id"]] for run in active if run["id"] in self._futures]
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentWorkspaceError("Timed out while stopping project graph runs.")
            try:
                future.result(timeout = remaining)
            except Exception as exc:
                raise AgentWorkspaceError("Failed while stopping a project graph run.") from exc
        remaining_runs = _list_active_graph_runs(project_id)
        if remaining_runs:
            raise AgentWorkspaceError("Project still has active graph runs.")
        return [
            get_graph_run(project_id, run["id"])
            for run in active
            if get_graph_run(project_id, run["id"])
        ]

    def _run(self, run_id: str, cancel_event: _RunControl) -> None:
        run = self._get_any(run_id)
        if run is None:
            return
        graph = get_graph_revision(run["projectId"], run["graphId"], run["revision"])
        budget_expired = threading.Event()
        max_run_seconds = int((graph or {}).get("limits", {}).get("maxRunSeconds", 3600))
        started_at = int(run.get("startedAt") or now_ms())
        remaining = max_run_seconds - max(0.0, (now_ms() - started_at) / 1000)

        def exhaust_budget() -> None:
            budget_expired.set()
            cancel_event.request("budget")

        timer = None
        if remaining <= 0:
            exhaust_budget()
        else:
            timer = threading.Timer(remaining, exhaust_budget)
            timer.daemon = True
            timer.start()
        try:
            self._run_impl(run_id, cancel_event, budget_expired)
        finally:
            if timer is not None:
                timer.cancel()

    def _run_impl(
        self, run_id: str, cancel_event: _RunControl, budget_expired: threading.Event
    ) -> None:
        run = self._get_any(run_id)
        if run is None:
            return
        try:
            graph = get_graph_revision(run["projectId"], run["graphId"], run["revision"])
            if graph is None:
                raise AgentWorkspaceError("Pinned graph revision is unavailable.")
            nodes = {node["id"]: node for node in graph["nodes"]}
            edges = graph["edges"]
            executions = list_node_executions(run["projectId"], run_id)
            context = {"input": run["input"], "nodes": {}, "previous": None}
            completed_by_node: dict[str, dict] = {}
            for execution in executions:
                if execution["status"] != "completed":
                    continue
                existing = completed_by_node.get(execution["nodeId"])
                if existing is None or int(execution["attempt"]) > int(existing["attempt"]):
                    completed_by_node[execution["nodeId"]] = execution
            current_node_id = next(node["id"] for node in graph["nodes"] if node["type"] == "input")
            completed_path: list[str] = []
            while current_node_id in completed_by_node:
                if current_node_id in completed_path:
                    raise AgentWorkspaceError("Graph execution history contains a cycle.")
                completed_execution = completed_by_node[current_node_id]
                completed_path.append(current_node_id)
                context["nodes"][current_node_id] = completed_execution["output"]
                context["previous"] = completed_execution["output"]
                current_node_id = self._next_node(
                    current_node_id,
                    completed_execution["output"],
                    nodes,
                    edges,
                )
            if set(completed_path) != set(completed_by_node):
                raise AgentWorkspaceError(
                    "Graph completed execution history does not match the pinned revision."
                )
            if run.get("currentNodeId") != current_node_id:
                update_graph_run(run_id, current_node_id = current_node_id)
            node_count = len(completed_path)
            while current_node_id is not None:
                run = self._get_any(run_id) or run
                if budget_expired.is_set():
                    raise AgentWorkspaceError("Graph run budget exhausted.")
                if self._stopping.is_set() and cancel_event.is_set():
                    finish_graph_run(
                        run_id,
                        "interrupted",
                        "run.interrupted",
                        error = "Studio stopped while the graph was active.",
                        current_node_id = current_node_id,
                        node_id = current_node_id,
                    )
                    return
                if run["cancelRequested"] or (cancel_event.is_set() and not run["pauseRequested"]):
                    finish_graph_run(
                        run_id,
                        "cancelled",
                        "run.cancelled",
                        error = "Graph run cancelled.",
                    )
                    return
                if (
                    run["pauseRequested"]
                    or run["status"] == "pausing"
                    or (cancel_event.is_set() and run["status"] == "paused")
                ):
                    finish_graph_run(
                        run_id,
                        "paused",
                        "run.paused",
                        current_node_id = current_node_id,
                        node_id = current_node_id,
                    )
                    return
                if node_count >= graph["limits"]["maxNodes"]:
                    raise AgentWorkspaceError("Graph node budget exhausted.")
                node = nodes[current_node_id]
                update_graph_run(run_id, current_node_id = current_node_id)
                prior_attempts = sorted(
                    [item for item in executions if item["nodeId"] == current_node_id],
                    key = lambda item: int(item["attempt"]),
                )
                seed_checkpoint = next(
                    (
                        item.get("checkpoint")
                        for item in reversed(prior_attempts)
                        if item.get("checkpoint")
                    ),
                    None,
                )
                retry_policy = node["retryPolicy"]
                failed_attempts = [item for item in prior_attempts if item["status"] == "failed"]
                failed_attempt_count = len(failed_attempts)
                if failed_attempt_count >= retry_policy["maxAttempts"]:
                    failure_error = (
                        failed_attempts[-1].get("error")
                        or "Graph node retry attempts were exhausted."
                    )
                    finish_graph_run(
                        run_id,
                        "failed",
                        "run.failed",
                        error = failure_error,
                        current_node_id = current_node_id,
                        node_id = current_node_id,
                        payload = {"error": str(failure_error)[:1000]},
                    )
                    return
                attempt = max((int(item["attempt"]) for item in prior_attempts), default = 0) + 1
                while True:
                    execution = admit_node_execution(
                        run_id,
                        node,
                        context["previous"],
                        attempt,
                        max_iterations = graph["limits"]["maxIterations"],
                        checkpoint = seed_checkpoint,
                    )
                    checkpoint_holder = {"value": seed_checkpoint}
                    node_timed_out = threading.Event()
                    node_timer = threading.Timer(
                        int(node["config"].get("timeoutSeconds", _MAX_NODE_SECONDS)),
                        node_timed_out.set,
                    )
                    node_timer.daemon = True
                    node_timer.start()
                    try:
                        output = self._execute_node(
                            run,
                            graph,
                            node,
                            context,
                            _CombinedEvent(cancel_event, node_timed_out),
                            execution["id"],
                            checkpoint_holder,
                        )
                        _json(
                            output,
                            limit = graph["limits"]["maxOutputBytes"],
                            label = "Graph node output",
                        )
                        current_after_node = self._get_any(run_id) or run
                        if self._stopping.is_set():
                            raise AgentWorkspaceError(
                                "Studio stopped while the graph node was finishing."
                            )
                        if budget_expired.is_set():
                            raise AgentWorkspaceError("Graph run budget exhausted.")
                        if current_after_node["cancelRequested"]:
                            raise AgentWorkspaceError("Graph run cancelled.")
                        if node_timed_out.is_set():
                            raise _GraphNodeTimeout("Graph node timeout exceeded.")
                        if cancel_event.is_set() and current_after_node["pauseRequested"]:
                            finish_node_execution(execution["id"], "paused", output = output)
                            finish_graph_run(
                                run_id,
                                "paused",
                                "node.paused",
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                            )
                            return
                        next_node_id = self._next_node(current_node_id, output, nodes, edges)
                        complete_node_execution(
                            run_id,
                            execution["id"],
                            current_node_id,
                            attempt,
                            output,
                            next_node_id,
                        )
                        executions.append(
                            {
                                "nodeId": current_node_id,
                                "status": "completed",
                                "output": output,
                                "checkpoint": checkpoint_holder["value"],
                            }
                        )
                        context["nodes"][current_node_id] = output
                        context["previous"] = output
                        node_count += 1
                        current_node_id = next_node_id
                        break
                    except Exception as exc:
                        effective_exc: Exception = exc
                        if node_timed_out.is_set() and not isinstance(exc, _GraphNodeTimeout):
                            effective_exc = _GraphNodeTimeout("Graph node timeout exceeded.")
                        persisted_execution = next(
                            (
                                item
                                for item in list_node_executions(run["projectId"], run_id)
                                if item["id"] == execution["id"]
                            ),
                            None,
                        )
                        if persisted_execution and persisted_execution["status"] == "completed":
                            persisted_output = persisted_execution["output"]
                            executions.append(persisted_execution)
                            context["nodes"][current_node_id] = persisted_output
                            context["previous"] = persisted_output
                            node_count += 1
                            current_node_id = self._next_node(
                                current_node_id,
                                persisted_output,
                                nodes,
                                edges,
                            )
                            break
                        current = self._get_any(run_id) or run
                        if self._stopping.is_set():
                            finish_node_execution(
                                execution["id"], "interrupted", error = str(effective_exc)
                            )
                            finish_graph_run(
                                run_id,
                                "interrupted",
                                "run.interrupted",
                                error = "Studio stopped while the graph was active.",
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                            )
                            return
                        if budget_expired.is_set():
                            finish_node_execution(
                                execution["id"], "failed", error = "Graph run budget exhausted."
                            )
                            raise AgentWorkspaceError("Graph run budget exhausted.") from exc
                        if current["cancelRequested"]:
                            finish_node_execution(
                                execution["id"], "cancelled", error = str(effective_exc)
                            )
                            finish_graph_run(
                                run_id,
                                "cancelled",
                                "run.cancelled",
                                error = str(effective_exc),
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                                payload = {"error": str(effective_exc)[:1000]},
                            )
                            return
                        if current["pauseRequested"]:
                            finish_node_execution(
                                execution["id"], "paused", error = str(effective_exc)
                            )
                            finish_graph_run(
                                run_id,
                                "paused",
                                "node.paused",
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                            )
                            return
                        finish_node_execution(execution["id"], "failed", error = str(effective_exc))
                        executions.append(
                            {
                                "nodeId": current_node_id,
                                "status": "failed",
                                "output": None,
                                "checkpoint": checkpoint_holder["value"],
                            }
                        )
                        failed_attempt_count += 1
                        retry_kind = (
                            "timeout" if isinstance(effective_exc, _GraphNodeTimeout) else "error"
                        )
                        retryable = (
                            failed_attempt_count < retry_policy["maxAttempts"]
                            and retry_kind in retry_policy["retryOn"]
                            and not isinstance(effective_exc, _GraphToolEffectUncertain)
                            and "budget exhausted" not in str(effective_exc).lower()
                        )
                        if not retryable:
                            finish_graph_run(
                                run_id,
                                "failed",
                                "run.failed",
                                error = str(effective_exc),
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                                payload = {"error": str(effective_exc)[:1000]},
                            )
                            return
                        append_graph_event(
                            run_id,
                            "node.retrying",
                            node_id = current_node_id,
                            payload = {
                                "attempt": attempt,
                                "nextAttempt": attempt + 1,
                                "reason": retry_kind,
                                "backoffMs": retry_policy["backoffMs"],
                            },
                        )
                        if cancel_event.wait(retry_policy["backoffMs"] / 1000):
                            current = self._get_any(run_id) or run
                            if self._stopping.is_set():
                                status = "interrupted"
                                finish_error = "Studio stopped during graph retry backoff."
                            elif budget_expired.is_set():
                                status = "failed"
                                finish_error = "Graph run budget exhausted."
                            elif current["cancelRequested"]:
                                status = "cancelled"
                                finish_error = "Graph retry cancelled."
                            elif current["pauseRequested"]:
                                status = "paused"
                                finish_error = "Graph retry paused."
                            else:
                                status = "cancelled"
                                finish_error = "Graph retry cancelled."
                            finish_graph_run(
                                run_id,
                                status,
                                "run." + status,
                                error = finish_error,
                                current_node_id = current_node_id,
                                node_id = current_node_id,
                            )
                            return
                        attempt += 1
                        seed_checkpoint = checkpoint_holder["value"]
                    finally:
                        node_timer.cancel()
            final_output = context["previous"]
            _json(
                final_output,
                limit = graph["limits"]["maxOutputBytes"],
                label = "Graph output",
            )
            _validate_schema_value(final_output, graph.get("outputSchema", {}), "Graph output")
            if self._stopping.is_set():
                finish_graph_run(
                    run_id,
                    "interrupted",
                    "run.interrupted",
                    error = "Studio stopped while the graph was finishing.",
                )
                return
            if budget_expired.is_set():
                raise AgentWorkspaceError("Graph run budget exhausted.")
            finish_graph_run(
                run_id,
                "completed",
                "run.completed",
                output = final_output,
                current_node_id = None,
                respect_control_requests = True,
            )
        except Exception as exc:
            current = self._get_any(run_id)
            if current and current["status"] not in {
                "cancelled",
                "completed",
                "failed",
                "interrupted",
            }:
                finish_graph_run(
                    run_id,
                    "failed",
                    "run.failed",
                    error = str(exc),
                    payload = {"error": str(exc)[:1000]},
                )

    def _execute_node(
        self,
        run: dict,
        graph: dict,
        node: dict,
        context: dict,
        cancel_event: Any,
        execution_id: str,
        checkpoint_holder: dict,
    ) -> Any:
        node_type = node["type"]
        config = node["config"]
        if node_type == "input":
            return context["input"]
        if node_type in {"loop", "model"}:
            template = config["instruction"] if node_type == "loop" else config["prompt"]
            instruction = _render(template, context)
            resource_binding = _graph_node_resource_binding(run["id"], node["id"], "runtime")
            runtime_snapshot = resource_binding.get("snapshot")
            if not isinstance(runtime_snapshot, dict):
                raise AgentWorkspaceError("The graph runtime binding is invalid.")

            def save_checkpoint(checkpoint: dict) -> None:
                merged_checkpoint = dict(checkpoint)
                previous_checkpoint = checkpoint_holder.get("value")
                if (
                    isinstance(previous_checkpoint, dict)
                    and previous_checkpoint.get("outputTokenReservationId")
                    and not merged_checkpoint.get("outputTokenReservationId")
                ):
                    merged_checkpoint["outputTokenReservationId"] = previous_checkpoint[
                        "outputTokenReservationId"
                    ]
                checkpoint_holder["value"] = merged_checkpoint
                update_node_checkpoint(execution_id, merged_checkpoint)

            def reserve_tokens(reservation_id: str) -> None:
                runtime = config.get("runtime") or {}
                checkpoint_holder["value"] = reserve_graph_task_output_tokens(
                    run["id"],
                    execution_id,
                    reservation_id,
                    max_output_tokens = graph["limits"]["maxOutputTokens"],
                    output_tokens = int(runtime.get("maxOutputTokens") or 8192),
                )

            run_parameters = inspect.signature(self.loop_adapter.run).parameters
            if "checkpoint" not in run_parameters:
                if runtime_snapshot.get("kind") == "provider":
                    raise AgentWorkspaceError(
                        "This Loop adapter cannot enforce the pinned provider resource."
                    )
                reserve_tokens(f"legacy:{execution_id}")
                return self.loop_adapter.run(
                    run["projectId"], instruction, config.get("runtime"), cancel_event
                )
            run_kwargs = {
                "checkpoint": checkpoint_holder["value"],
                "checkpoint_callback": save_checkpoint,
                "before_start": reserve_tokens,
            }
            if "runtime_snapshot" in run_parameters:
                run_kwargs["runtime_snapshot"] = runtime_snapshot
            return self.loop_adapter.run(
                run["projectId"],
                instruction,
                config.get("runtime"),
                cancel_event,
                **run_kwargs,
            )
        if node_type == "tool":
            allowed = graph["permissions"].get("allowedToolServerIds", [])
            if config["serverId"] not in allowed:
                raise AgentWorkspaceError("Graph tool is not permitted by this graph revision.")
            server = _bound_mcp_server(run["id"], node["id"], config["serverId"])
            arguments = _render(config["arguments"], context, exact_values = True)
            server_url = server["url"]
            server_headers = parse_server_headers(server)
            server_oauth = bool(server.get("use_oauth"))

            def config_current() -> bool:
                try:
                    current = _bound_mcp_server(run["id"], node["id"], config["serverId"])
                except AgentWorkspaceError:
                    return False
                return (
                    current.get("url") == server_url
                    and parse_server_headers(current) == server_headers
                    and bool(current.get("use_oauth")) == server_oauth
                )

            def dispatch() -> Any:
                result = call_tool_sync(
                    server_url,
                    server_headers,
                    config["toolName"],
                    arguments,
                    timeout = config["timeoutSeconds"],
                    use_oauth = server_oauth,
                    cancel_event = cancel_event,
                    scope = f"graph:{run['id']}",
                    config_check = config_current,
                )
                if isinstance(result, str) and result.startswith("Error:"):
                    if " timed out" in result:
                        raise _GraphNodeTimeout(result)
                    raise AgentWorkspaceError(result)
                return result

            if not config["sideEffecting"]:
                return dispatch()
            if not config.get("idempotencyKey"):
                raise AgentWorkspaceError(
                    "This pinned tool node predates effect idempotency. Save a new graph revision with an idempotencyKey before running it."
                )
            idempotency_key = _string(
                _render(config["idempotencyKey"], context),
                "Tool idempotency key",
                maximum = 512,
            )
            effect_status, effect_id, cached_output = _begin_tool_effect(
                run, node, arguments, idempotency_key
            )
            if effect_status == "cached":
                append_graph_event(
                    run["id"],
                    "tool.effect.reused",
                    node_id = node["id"],
                    payload = {"idempotencyKey": idempotency_key},
                )
                return cached_output
            try:
                output = dispatch()
                _complete_tool_effect(str(effect_id), output)
                return output
            except Exception as exc:
                _mark_tool_effect_uncertain(str(effect_id), str(exc))
                raise _GraphToolEffectUncertain(
                    "Tool effect may have occurred. Automatic replay was blocked."
                ) from exc
        if node_type == "condition":
            return _condition(_graph_path(context, config["path"]), config)
        if node_type == "approval":
            approval = get_or_create_approval(run["projectId"], run["id"], node)
            while approval["status"] == "pending":
                current = self._get_any(run["id"]) or run
                if current["cancelRequested"]:
                    raise AgentWorkspaceError("Graph approval was cancelled.")
                if current["pauseRequested"] or cancel_event.is_set():
                    raise AgentWorkspaceError("Graph approval was paused.")
                time.sleep(0.05)
                approval = (
                    get_graph_approval(run["projectId"], run["id"], approval["id"]) or approval
                )
            if approval["status"] == "rejected":
                raise AgentWorkspaceError("Graph approval was rejected.")
            return {"approvalId": approval["id"], "decision": "approved"}
        if node_type == "output":
            return (
                _graph_path(context, config["path"]) if config.get("path") else context["previous"]
            )
        raise AgentWorkspaceError("Unsupported graph node type.")

    @staticmethod
    def _next_node(
        node_id: str, output: Any, nodes: dict[str, dict], edges: list[dict]
    ) -> Optional[str]:
        outgoing = [edge for edge in edges if edge["from"] == node_id]
        if not outgoing:
            return None
        if nodes[node_id]["type"] != "condition":
            return outgoing[0]["to"]
        if len(outgoing) == 1 and outgoing[0].get("when") in {None, "default"}:
            return outgoing[0]["to"]
        wanted = "true" if bool(output) else "false"
        for edge in outgoing:
            if edge.get("when") == wanted:
                return edge["to"]
        for edge in outgoing:
            if edge.get("when") == "default":
                return edge["to"]
        return None


manager = GraphRunManager()


__all__ = [
    "GraphLoopAdapter",
    "GraphRunManager",
    "append_graph_event",
    "create_graph",
    "create_graph_run",
    "delete_graph",
    "decide_graph_approval",
    "get_graph",
    "get_graph_approval",
    "get_graph_revision",
    "get_graph_run",
    "list_graph_events",
    "list_graph_revisions",
    "list_graph_approvals",
    "list_graph_runs",
    "list_graphs",
    "list_node_executions",
    "manager",
    "recover_graph_runs",
    "update_graph",
    "update_node_checkpoint",
    "validate_graph_spec",
]
