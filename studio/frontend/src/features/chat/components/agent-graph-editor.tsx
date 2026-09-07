// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Background,
  type Connection,
  Controls,
  type Edge,
  type EdgeChange,
  MiniMap,
  type Node,
  type NodeChange,
  ReactFlow,
  applyNodeChanges,
} from "@xyflow/react";
import { Plus, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";
import type {
  AgentGraphDocument,
  AgentGraphEdge,
  AgentGraphNode,
  AgentGraphNodeType,
} from "../api/agent-workspace-api";
import { AgentGraphAlert } from "./agent-graph-ui";
import {
  connectGraphEdge,
  graphConnectionIsValid,
  graphEdgeId,
  reconnectGraphEdge,
  replaceGraphEdge,
} from "./agent-graph-workflow-state";

const NODE_TYPES: AgentGraphNodeType[] = [
  "input",
  "loop",
  "model",
  "tool",
  "condition",
  "approval",
  "output",
];

const NODE_COLORS: Record<AgentGraphNodeType, string> = {
  input: "#2563eb",
  loop: "#7c3aed",
  model: "#9333ea",
  tool: "#ea580c",
  condition: "#ca8a04",
  approval: "#db2777",
  output: "#059669",
};

const STATUS_COLORS: Record<string, string> = {
  queued: "#64748b",
  running: "#2563eb",
  paused: "#d97706",
  cancelled: "#64748b",
  completed: "#16a34a",
  failed: "#dc2626",
  interrupted: "#ea580c",
};

type SlothNodeData = {
  label: string;
  nodeType: AgentGraphNodeType;
  status?: string;
};

type SlothFlowNode = Node<SlothNodeData>;
type SlothFlowEdge = Edge<{ when?: AgentGraphEdge["when"] }>;

function isPosition(value: unknown): value is { x: number; y: number } {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return typeof candidate.x === "number" && typeof candidate.y === "number";
}

function positions(
  document: AgentGraphDocument,
): Record<string, { x: number; y: number }> {
  const raw = document.metadata?.nodePositions;
  if (!raw || typeof raw !== "object") {
    return {};
  }
  return Object.fromEntries(
    Object.entries(raw).filter(
      (entry): entry is [string, { x: number; y: number }] =>
        isPosition(entry[1]),
    ),
  );
}

function flowNodes(
  document: AgentGraphDocument,
  statuses: Record<string, string>,
): SlothFlowNode[] {
  const savedPositions = positions(document);
  return document.nodes.map((node, index) => {
    const status = statuses[node.id];
    const color = NODE_COLORS[node.type];
    const statusColor = status
      ? (STATUS_COLORS[status] ?? color)
      : `${color}88`;
    const label = node.label || `${node.type}: ${node.id}`;
    return {
      id: node.id,
      position: savedPositions[node.id] ?? {
        x: index * 210,
        y: (index % 2) * 110,
      },
      data: {
        label: status ? `${label} · ${status}` : label,
        nodeType: node.type,
        status,
      },
      style: {
        border: `1px solid ${statusColor}`,
        borderLeft: `5px solid ${color}`,
        borderRadius: 12,
        background: "var(--card)",
        color: "var(--foreground)",
        fontSize: 12,
        width: 165,
        boxShadow: status === "running" ? `0 0 0 3px ${color}33` : undefined,
      },
    };
  });
}

function flowEdges(
  document: AgentGraphDocument,
  selectedEdgeIds: ReadonlySet<string>,
): SlothFlowEdge[] {
  return document.edges.map((edge) => ({
    id: graphEdgeId(edge),
    source: edge.from,
    target: edge.to,
    label: edge.when,
    data: { when: edge.when },
    selected: selectedEdgeIds.has(graphEdgeId(edge)),
    animated: false,
  }));
}

function defaultConfig(type: AgentGraphNodeType): Record<string, unknown> {
  if (type === "input") {
    return { name: "input" };
  }
  if (type === "loop") {
    return {
      instruction: "Work on {input}",
      runtime: {
        kind: "local",
        model: "",
        permissionMode: "off",
        maxOutputTokens: 8192,
      },
      timeoutSeconds: 7200,
    };
  }
  if (type === "model") {
    return {
      prompt: "Respond to {previous}",
      runtime: {
        kind: "local",
        model: "",
        permissionMode: "off",
        maxOutputTokens: 8192,
      },
      timeoutSeconds: 7200,
    };
  }
  if (type === "tool") {
    return {
      serverId: "",
      toolName: "",
      arguments: {},
      timeoutSeconds: 300,
      sideEffecting: true,
      idempotencyKey: "",
    };
  }
  if (type === "condition") {
    return { path: "previous", operator: "truthy" };
  }
  if (type === "approval") {
    return { title: "Approval required", description: "" };
  }
  return { name: "output" };
}

function nextNodeId(
  document: AgentGraphDocument,
  type: AgentGraphNodeType,
): string {
  let index = 1;
  while (document.nodes.some((node) => node.id === `${type}_${index}`)) {
    index += 1;
  }
  return `${type}_${index}`;
}

function asConfig(value: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

export function AgentGraphEditor({
  document,
  onChange,
  onDraftStateChange,
  nodeStatuses = {},
  disabled = false,
}: {
  document: AgentGraphDocument;
  onChange: (document: AgentGraphDocument) => void;
  onDraftStateChange?: (hasDrafts: boolean) => void;
  nodeStatuses?: Record<string, string>;
  disabled?: boolean;
}) {
  const nodes = useMemo(
    () => flowNodes(document, nodeStatuses),
    [document, nodeStatuses],
  );
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<Set<string>>(
    () => new Set(),
  );
  const edges = useMemo(
    () => flowEdges(document, selectedEdgeIds),
    [document, selectedEdgeIds],
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    document.nodes[0]?.id ?? null,
  );
  const selectedNode = useMemo(
    () => document.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [document.nodes, selectedNodeId],
  );
  const [configDrafts, setConfigDrafts] = useState<Record<string, string>>({});
  const [configError, setConfigError] = useState<string | null>(null);
  const [schemaDrafts, setSchemaDrafts] = useState<{
    input?: string;
    output?: string;
  }>({});
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const sourceNodes = useMemo(
    () => document.nodes.filter((node) => node.type !== "output"),
    [document.nodes],
  );
  const targetNodes = useMemo(
    () => document.nodes.filter((node) => node.type !== "input"),
    [document.nodes],
  );
  const [newEdgeSource, setNewEdgeSource] = useState(
    () => sourceNodes[0]?.id ?? "",
  );
  const [newEdgeTarget, setNewEdgeTarget] = useState(
    () => targetNodes[0]?.id ?? "",
  );
  const [newEdgeWhen, setNewEdgeWhen] =
    useState<NonNullable<AgentGraphEdge["when"]>>("default");
  const [edgeError, setEdgeError] = useState<string | null>(null);
  const effectiveNewEdgeSource = sourceNodes.some(
    (node) => node.id === newEdgeSource,
  )
    ? newEdgeSource
    : (sourceNodes[0]?.id ?? "");
  const effectiveNewEdgeTarget = targetNodes.some(
    (node) => node.id === newEdgeTarget,
  )
    ? newEdgeTarget
    : (targetNodes[0]?.id ?? "");
  const hasDrafts =
    Object.keys(configDrafts).length > 0 ||
    schemaDrafts.input !== undefined ||
    schemaDrafts.output !== undefined;
  useEffect(() => {
    onDraftStateChange?.(hasDrafts);
  }, [hasDrafts, onDraftStateChange]);
  const configText = selectedNode
    ? (configDrafts[selectedNode.id] ??
      JSON.stringify(selectedNode.config, null, 2))
    : "{}";

  function updateNode(
    nodeId: string,
    update: (node: AgentGraphNode) => AgentGraphNode,
  ) {
    onChange({
      ...document,
      nodes: document.nodes.map((node) =>
        node.id === nodeId ? update(node) : node,
      ),
    });
  }

  function handleNodesChange(changes: NodeChange<SlothFlowNode>[]) {
    if (disabled) {
      return;
    }
    const removed = new Set(
      changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id),
    );
    if (removed.size > 0) {
      setConfigDrafts((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([nodeId]) => !removed.has(nodeId)),
        ),
      );
      onChange({
        ...document,
        nodes: document.nodes.filter((node) => !removed.has(node.id)),
        edges: document.edges.filter(
          (edge) => !(removed.has(edge.from) || removed.has(edge.to)),
        ),
      });
      if (selectedNodeId && removed.has(selectedNodeId)) {
        setSelectedNodeId(
          document.nodes.find((node) => !removed.has(node.id))?.id ?? null,
        );
      }
      return;
    }
    const next = applyNodeChanges(changes, nodes);
    if (changes.some((change) => change.type === "position")) {
      onChange({
        ...document,
        metadata: {
          ...(document.metadata ?? {}),
          nodePositions: Object.fromEntries(
            next.map((node) => [node.id, node.position]),
          ),
        },
      });
    }
  }

  function handleEdgesChange(changes: EdgeChange<SlothFlowEdge>[]) {
    if (disabled) {
      return;
    }
    const selectionChanges = changes.filter(
      (
        change,
      ): change is Extract<EdgeChange<SlothFlowEdge>, { type: "select" }> =>
        change.type === "select",
    );
    if (selectionChanges.length > 0) {
      setSelectedEdgeIds((current) => {
        const next = new Set(current);
        for (const change of selectionChanges) {
          if (change.selected) {
            next.add(change.id);
          } else {
            next.delete(change.id);
          }
        }
        return next;
      });
    }
    const removed = new Set(
      changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id),
    );
    if (removed.size === 0) {
      return;
    }
    setSelectedEdgeIds((current) => {
      const next = new Set(current);
      for (const edgeId of removed) {
        next.delete(edgeId);
      }
      return next;
    });
    onChange({
      ...document,
      edges: document.edges.filter((edge) => !removed.has(graphEdgeId(edge))),
    });
  }

  function reconnect(oldEdge: SlothFlowEdge, connection: Connection) {
    if (disabled || !connection.source || !connection.target) {
      return;
    }
    const nextEdges = reconnectGraphEdge(
      document.edges,
      document.nodes,
      oldEdge.id,
      connection.source,
      connection.target,
    );
    if (nextEdges === document.edges) {
      setEdgeError(
        "That edge would create a cycle, join, duplicate, or invalid branch.",
      );
      return;
    }
    setEdgeError(null);
    onChange({ ...document, edges: nextEdges });
  }

  function connect(connection: Connection) {
    if (disabled || !connection.source || !connection.target) {
      return;
    }
    const nextEdges = connectGraphEdge(
      document.edges,
      document.nodes,
      connection.source,
      connection.target,
    );
    if (nextEdges === document.edges) {
      setEdgeError(
        "That edge would create a cycle, join, duplicate, or invalid branch.",
      );
      return;
    }
    setEdgeError(null);
    onChange({ ...document, edges: nextEdges });
  }

  function addNode(type: AgentGraphNodeType) {
    if (
      disabled ||
      (type === "input" && document.nodes.some((node) => node.type === "input"))
    ) {
      return;
    }
    const id = nextNodeId(document, type);
    const nextNode: AgentGraphNode = {
      id,
      type,
      label: `${type[0].toUpperCase()}${type.slice(1)}`,
      config: defaultConfig(type),
      retryPolicy: {
        maxAttempts: 1,
        backoffMs: 0,
        retryOn: ["error", "timeout"],
      },
    };
    const currentPositions = positions(document);
    onChange({
      ...document,
      nodes: [...document.nodes, nextNode],
      metadata: {
        ...(document.metadata ?? {}),
        nodePositions: {
          ...currentPositions,
          [id]: { x: document.nodes.length * 190, y: 140 },
        },
      },
    });
    setSelectedNodeId(id);
    setConfigError(null);
  }

  function removeNode(nodeId: string) {
    if (disabled) {
      return;
    }
    onChange({
      ...document,
      nodes: document.nodes.filter((node) => node.id !== nodeId),
      edges: document.edges.filter(
        (edge) => edge.from !== nodeId && edge.to !== nodeId,
      ),
    });
    setConfigDrafts((current) => {
      const next = { ...current };
      delete next[nodeId];
      return next;
    });
    setSelectedNodeId(
      document.nodes.find((node) => node.id !== nodeId)?.id ?? null,
    );
  }

  function applyConfig() {
    if (!selectedNode) {
      return;
    }
    const config = asConfig(configText);
    if (!config) {
      setConfigError("Config must be a JSON object.");
      return;
    }
    setConfigError(null);
    updateNode(selectedNode.id, (node) => ({ ...node, config }));
    setConfigDrafts((current) => {
      const next = { ...current };
      delete next[selectedNode.id];
      return next;
    });
  }

  function replaceEdge(index: number, replacement: AgentGraphEdge) {
    const nextEdges = replaceGraphEdge(
      document.edges,
      document.nodes,
      index,
      replacement,
    );
    if (nextEdges === document.edges) {
      setEdgeError(
        "That edge would create a cycle, join, duplicate, or invalid branch.",
      );
      return;
    }
    setEdgeError(null);
    onChange({ ...document, edges: nextEdges });
  }

  function updateEdge(index: number, when: AgentGraphEdge["when"] | undefined) {
    const edge = document.edges[index];
    if (!edge) return;
    replaceEdge(index, {
      from: edge.from,
      to: edge.to,
      ...(when ? { when } : {}),
    });
  }

  function updateEdgeEndpoint(
    index: number,
    endpoint: "from" | "to",
    nodeId: string,
  ) {
    const edge = document.edges[index];
    if (!edge) return;
    const from = endpoint === "from" ? nodeId : edge.from;
    const source = document.nodes.find((node) => node.id === from);
    replaceEdge(index, {
      from,
      to: endpoint === "to" ? nodeId : edge.to,
      ...(source?.type === "condition" && edge.when ? { when: edge.when } : {}),
    });
  }

  function addEdge() {
    const source = document.nodes.find(
      (node) => node.id === effectiveNewEdgeSource,
    );
    const when =
      source?.type === "condition" && newEdgeWhen !== "default"
        ? newEdgeWhen
        : source?.type === "condition" && newEdgeWhen === "default"
          ? "default"
          : undefined;
    const nextEdges = connectGraphEdge(
      document.edges,
      document.nodes,
      effectiveNewEdgeSource,
      effectiveNewEdgeTarget,
      when,
    );
    if (nextEdges === document.edges) {
      setEdgeError(
        "That edge would create a cycle, join, duplicate, or invalid branch.",
      );
      return;
    }
    setEdgeError(null);
    onChange({ ...document, edges: nextEdges });
  }

  function updateLimit(
    name: keyof AgentGraphDocument["limits"],
    value: string,
  ) {
    onChange({
      ...document,
      limits: { ...document.limits, [name]: Number(value) },
    });
  }

  function applySchemas() {
    const input = asConfig(
      schemaDrafts.input ?? JSON.stringify(document.inputSchema, null, 2),
    );
    const output = asConfig(
      schemaDrafts.output ?? JSON.stringify(document.outputSchema, null, 2),
    );
    if (!(input && output)) {
      setSchemaError("Input and output schemas must be JSON objects.");
      return;
    }
    setSchemaError(null);
    setSchemaDrafts({});
    onChange({ ...document, inputSchema: input, outputSchema: output });
  }

  return (
    <div className="grid gap-3 xl:grid-cols-[150px_minmax(0,1fr)_280px]">
      <aside className="rounded-xl border border-border/60 bg-background/45 p-2">
        <p className="px-1 text-[11px] font-medium">Node palette</p>
        <div className="mt-2 grid gap-1.5">
          {NODE_TYPES.map((type) => (
            <Button
              key={type}
              type="button"
              size="xs"
              variant="outline"
              className="justify-start"
              onClick={() => addNode(type)}
              disabled={
                disabled ||
                (type === "input" &&
                  document.nodes.some((node) => node.type === "input"))
              }
            >
              <Plus style={{ color: NODE_COLORS[type] }} /> {type}
            </Button>
          ))}
        </div>
        <p className="mt-3 px-1 text-[10px] leading-relaxed text-muted-foreground">
          Connect handles on the canvas. Sequential graphs reject joins, cycles,
          and unreachable nodes.
        </p>
      </aside>

      <div className="h-[440px] overflow-hidden rounded-xl border border-border/60 bg-background/45">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={connect}
          onReconnect={reconnect}
          isValidConnection={(connection) =>
            Boolean(
              connection.source &&
                connection.target &&
                graphConnectionIsValid(
                  document.edges,
                  document.nodes,
                  connection.source,
                  connection.target,
                ),
            )
          }
          onNodeClick={(_, node) => {
            setSelectedNodeId(node.id);
            setConfigError(null);
          }}
          nodesDraggable={!disabled}
          nodesConnectable={!disabled}
          edgesReconnectable={!disabled}
          elementsSelectable={true}
          fitView={true}
          fitViewOptions={{ padding: 0.2 }}
          deleteKeyCode={disabled ? null : ["Backspace", "Delete"]}
        >
          <Background gap={18} size={1} />
          <MiniMap
            pannable={true}
            zoomable={true}
            nodeColor={(node) =>
              NODE_COLORS[(node.data as SlothNodeData).nodeType]
            }
          />
          <Controls showInteractive={!disabled} />
        </ReactFlow>
      </div>

      <aside className="space-y-2 rounded-xl border border-border/60 bg-background/45 p-3">
        <div>
          <label className="text-[11px] font-medium" htmlFor="sloth-graph-name">
            Graph name
          </label>
          <input
            id="sloth-graph-name"
            value={document.name}
            onChange={(event) =>
              onChange({ ...document, name: event.target.value })
            }
            disabled={disabled}
            className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
          />
        </div>
        <div>
          <label
            className="text-[11px] font-medium"
            htmlFor="sloth-graph-description"
          >
            Description
          </label>
          <Textarea
            id="sloth-graph-description"
            value={document.description}
            onChange={(event) =>
              onChange({ ...document, description: event.target.value })
            }
            disabled={disabled}
            className="mt-1 min-h-16 text-xs"
          />
        </div>
        <div className="border-t border-border/60 pt-2">
          <p className="text-[11px] font-medium">Graph budgets</p>
          <div className="mt-1 grid grid-cols-2 gap-2">
            {(
              [
                ["maxNodes", "Node count"],
                ["maxRunSeconds", "Run seconds"],
                ["maxIterations", "Iterations"],
                ["maxOutputTokens", "Output tokens"],
                ["maxOutputBytes", "Output bytes"],
              ] as const
            ).map(([name, label]) => (
              <label className="text-[10px] text-muted-foreground" key={name}>
                {label}
                <input
                  type="number"
                  min={1}
                  value={document.limits[name]}
                  onChange={(event) => updateLimit(name, event.target.value)}
                  disabled={disabled}
                  className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
                />
              </label>
            ))}
          </div>
          <label
            className="mt-2 block text-[10px] text-muted-foreground"
            htmlFor="sloth-tool-servers"
          >
            Allowed tool server IDs
          </label>
          <input
            id="sloth-tool-servers"
            value={(document.permissions.allowedToolServerIds ?? []).join(", ")}
            onChange={(event) =>
              onChange({
                ...document,
                permissions: {
                  allowedToolServerIds: event.target.value
                    .split(",")
                    .map((value) => value.trim())
                    .filter(Boolean),
                },
              })
            }
            disabled={disabled}
            placeholder="filesystem, github"
            className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
          />
          <details className="mt-2">
            <summary className="cursor-pointer text-[11px] font-medium">
              Input and output schemas
            </summary>
            <label
              className="mt-2 block text-[10px] text-muted-foreground"
              htmlFor="sloth-input-schema"
            >
              Input schema
            </label>
            <Textarea
              id="sloth-input-schema"
              value={
                schemaDrafts.input ??
                JSON.stringify(document.inputSchema, null, 2)
              }
              onChange={(event) =>
                setSchemaDrafts((current) => ({
                  ...current,
                  input: event.target.value,
                }))
              }
              disabled={disabled}
              className="mt-1 min-h-24 font-mono text-[10px]"
              spellCheck={false}
            />
            <label
              className="mt-2 block text-[10px] text-muted-foreground"
              htmlFor="sloth-output-schema"
            >
              Output schema
            </label>
            <Textarea
              id="sloth-output-schema"
              value={
                schemaDrafts.output ??
                JSON.stringify(document.outputSchema, null, 2)
              }
              onChange={(event) =>
                setSchemaDrafts((current) => ({
                  ...current,
                  output: event.target.value,
                }))
              }
              disabled={disabled}
              className="mt-1 min-h-24 font-mono text-[10px]"
              spellCheck={false}
            />
            {schemaError ? (
              <AgentGraphAlert className="mt-1 text-[10px] text-destructive">
                {schemaError}
              </AgentGraphAlert>
            ) : null}
            <Button
              type="button"
              size="xs"
              variant="outline"
              className="mt-2"
              onClick={applySchemas}
              disabled={disabled}
            >
              Apply schemas
            </Button>
          </details>
        </div>
        {selectedNode ? (
          <div className="border-t border-border/60 pt-2">
            <div className="flex items-center gap-2">
              <span
                className="size-2.5 rounded-full"
                style={{ background: NODE_COLORS[selectedNode.type] }}
              />
              <span className="min-w-0 flex-1 truncate text-xs font-medium">
                {selectedNode.id}
              </span>
              <Button
                type="button"
                size="icon-xs"
                variant="ghost"
                onClick={() => removeNode(selectedNode.id)}
                disabled={disabled}
                aria-label={`Delete ${selectedNode.id}`}
              >
                <Trash2 />
              </Button>
            </div>
            <label
              className="mt-2 block text-[11px] font-medium"
              htmlFor="sloth-node-label"
            >
              Label
            </label>
            <input
              id="sloth-node-label"
              value={selectedNode.label ?? ""}
              onChange={(event) =>
                updateNode(selectedNode.id, (node) => ({
                  ...node,
                  label: event.target.value,
                }))
              }
              disabled={disabled}
              className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
            />
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="text-[10px] text-muted-foreground">
                Attempts
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={selectedNode.retryPolicy?.maxAttempts ?? 1}
                  onChange={(event) =>
                    updateNode(selectedNode.id, (node) => ({
                      ...node,
                      retryPolicy: {
                        maxAttempts: Number(event.target.value),
                        backoffMs: node.retryPolicy?.backoffMs ?? 0,
                        retryOn: node.retryPolicy?.retryOn ?? [
                          "error",
                          "timeout",
                        ],
                      },
                    }))
                  }
                  disabled={disabled}
                  className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
                />
              </label>
              <label className="text-[10px] text-muted-foreground">
                Backoff ms
                <input
                  type="number"
                  min={0}
                  max={60000}
                  value={selectedNode.retryPolicy?.backoffMs ?? 0}
                  onChange={(event) =>
                    updateNode(selectedNode.id, (node) => ({
                      ...node,
                      retryPolicy: {
                        maxAttempts: node.retryPolicy?.maxAttempts ?? 1,
                        backoffMs: Number(event.target.value),
                        retryOn: node.retryPolicy?.retryOn ?? [
                          "error",
                          "timeout",
                        ],
                      },
                    }))
                  }
                  disabled={disabled}
                  className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
                />
              </label>
            </div>
            <label
              className="mt-2 block text-[11px] font-medium"
              htmlFor="sloth-node-config"
            >
              Node config and mappings
            </label>
            <Textarea
              id="sloth-node-config"
              value={configText}
              onChange={(event) =>
                setConfigDrafts((current) => ({
                  ...current,
                  [selectedNode.id]: event.target.value,
                }))
              }
              disabled={disabled}
              className="mt-1 min-h-40 font-mono text-[10px]"
              spellCheck={false}
            />
            {configError ? (
              <AgentGraphAlert className="mt-1 text-[10px] text-destructive">
                {configError}
              </AgentGraphAlert>
            ) : null}
            <Button
              type="button"
              size="xs"
              variant="outline"
              className="mt-2"
              onClick={applyConfig}
              disabled={disabled}
            >
              Apply config
            </Button>
          </div>
        ) : (
          <p className="text-[11px] text-muted-foreground">
            Select a node to configure it.
          </p>
        )}
      </aside>

      <div className="xl:col-start-2 xl:col-span-2 rounded-xl border border-border/60 bg-background/45 p-3">
        <p className="text-[11px] font-medium">Edge mappings</p>
        <div className="mt-2 grid gap-2 rounded-lg bg-muted/25 p-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_auto]">
          <label className="text-[10px] text-muted-foreground">
            Source node
            <select
              aria-label="New edge source node"
              value={effectiveNewEdgeSource}
              onChange={(event) => {
                setNewEdgeSource(event.target.value);
                setEdgeError(null);
              }}
              disabled={disabled || sourceNodes.length === 0}
              className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
            >
              {sourceNodes.map((node) => (
                <option value={node.id} key={node.id}>
                  {node.id}
                </option>
              ))}
            </select>
          </label>
          <label className="text-[10px] text-muted-foreground">
            Target node
            <select
              aria-label="New edge target node"
              value={effectiveNewEdgeTarget}
              onChange={(event) => {
                setNewEdgeTarget(event.target.value);
                setEdgeError(null);
              }}
              disabled={disabled || targetNodes.length === 0}
              className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
            >
              {targetNodes.map((node) => (
                <option value={node.id} key={node.id}>
                  {node.id}
                </option>
              ))}
            </select>
          </label>
          <label className="text-[10px] text-muted-foreground">
            Branch
            <select
              aria-label="New edge branch"
              value={newEdgeWhen}
              onChange={(event) => {
                setNewEdgeWhen(
                  event.target.value as NonNullable<AgentGraphEdge["when"]>,
                );
                setEdgeError(null);
              }}
              disabled={
                disabled ||
                document.nodes.find(
                  (node) => node.id === effectiveNewEdgeSource,
                )?.type !== "condition"
              }
              className="mt-1 h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground"
            >
              <option value="default">default</option>
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </label>
          <Button
            type="button"
            size="xs"
            variant="outline"
            className="self-end"
            onClick={addEdge}
            disabled={
              disabled ||
              effectiveNewEdgeSource.length === 0 ||
              effectiveNewEdgeTarget.length === 0
            }
          >
            <Plus /> Add edge
          </Button>
        </div>
        {edgeError ? (
          <AgentGraphAlert className="mt-2 text-[10px] text-destructive">
            {edgeError}
          </AgentGraphAlert>
        ) : null}
        <div className="mt-2 grid gap-1.5">
          {document.edges.map((edge, index) => {
            const source = document.nodes.find((node) => node.id === edge.from);
            return (
              <div
                key={`${graphEdgeId(edge)}:${index}`}
                className="flex items-center gap-2 text-[11px]"
              >
                <select
                  aria-label={`Source for edge ${index + 1}`}
                  value={edge.from}
                  onChange={(event) =>
                    updateEdgeEndpoint(index, "from", event.target.value)
                  }
                  disabled={disabled}
                  className="h-7 min-w-0 rounded-md border border-input bg-background px-2 text-[11px]"
                >
                  {sourceNodes.map((node) => (
                    <option value={node.id} key={node.id}>
                      {node.id}
                    </option>
                  ))}
                </select>
                <span className="text-muted-foreground">to</span>
                <select
                  aria-label={`Target for edge ${index + 1}`}
                  value={edge.to}
                  onChange={(event) =>
                    updateEdgeEndpoint(index, "to", event.target.value)
                  }
                  disabled={disabled}
                  className="h-7 min-w-0 rounded-md border border-input bg-background px-2 text-[11px]"
                >
                  {targetNodes.map((node) => (
                    <option value={node.id} key={node.id}>
                      {node.id}
                    </option>
                  ))}
                </select>
                {source?.type === "condition" ? (
                  <select
                    aria-label={`Branch from ${edge.from} to ${edge.to}`}
                    value={edge.when ?? "default"}
                    onChange={(event) =>
                      updateEdge(
                        index,
                        event.target.value as AgentGraphEdge["when"],
                      )
                    }
                    disabled={disabled}
                    className="h-7 rounded-md border border-input bg-background px-2 text-[11px]"
                  >
                    <option value="default">default</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : null}
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  onClick={() => {
                    setEdgeError(null);
                    onChange({
                      ...document,
                      edges: document.edges.filter(
                        (_, edgeIndex) => edgeIndex !== index,
                      ),
                    });
                  }}
                  disabled={disabled}
                  aria-label={`Delete edge from ${edge.from} to ${edge.to}`}
                >
                  <Trash2 />
                </Button>
              </div>
            );
          })}
          {document.edges.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">
              Connect nodes on the canvas.
            </p>
          ) : null}
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground">
          Templates can read input, previous, and nodes.&lt;node-id&gt; paths,
          for example {"{input.task}"}.
        </p>
      </div>
    </div>
  );
}
