// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge } from "@xyflow/react";
import type {
  RecipeExecutionBatch,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../../execution-types";
import type { NodeConfig } from "../../types";
import { extractRefs } from "../refs";

const ACTIVE_STATUSES: ReadonlySet<RecipeExecutionStatus> = new Set([
  "pending",
  "running",
  "active",
  "cancelling",
]);
const FRESH_PENDING_WINDOW_MS = 60_000;

const DONE_UPSTREAM_KINDS: ReadonlySet<NodeConfig["kind"]> = new Set([
  "sampler",
  "seed",
  "expression",
  "llm",
  "model_config",
  "model_provider",
  "tool_config",
]);

export type GraphRuntimeVisualState = {
  executionLocked: boolean;
  runningNodeId: string | null;
  doneNodeIds: Set<string>;
  activeEdgeIds: Set<string>;
  batch: RecipeExecutionBatch | null;
};

function isAuxEdge(edge: Edge): boolean {
  return edge.source.startsWith("aux-") || edge.target.startsWith("aux-");
}

function collectTemplateRefs(config: NodeConfig | null): Set<string> {
  if (!config) {
    return new Set();
  }
  const refs = new Set<string>();
  if (config.kind === "llm") {
    for (const ref of extractRefs(config.prompt ?? "")) {
      refs.add(ref.trim());
    }
    for (const ref of extractRefs(config.system_prompt ?? "")) {
      refs.add(ref.trim());
    }
    if (typeof config.output_format === "string") {
      for (const ref of extractRefs(config.output_format)) {
        refs.add(ref.trim());
      }
    }
    return refs;
  }
  if (config.kind === "expression") {
    for (const ref of extractRefs(config.expr ?? "")) {
      refs.add(ref.trim());
    }
  }
  return refs;
}

function isReversedRuntimeReferenceEdge(input: {
  edge: Edge;
  runningNodeId: string;
  runningTemplateRefs: Set<string>;
  configs: Record<string, NodeConfig>;
}): boolean {
  const { edge, runningNodeId, runningTemplateRefs, configs } = input;
  if (edge.source !== runningNodeId) {
    return false;
  }
  const targetName = configs[edge.target]?.name?.trim() ?? "";
  return Boolean(targetName && runningTemplateRefs.has(targetName));
}

function hasLiveExecutionSignal(execution: RecipeExecutionRecord): boolean {
  if (execution.lastEventId !== null) {
    return true;
  }
  if (execution.current_column !== null) {
    return true;
  }
  if (execution.progress !== null || execution.column_progress !== null) {
    return true;
  }
  return Boolean(execution.batch?.idx ?? execution.batch?.total);
}

export function pickLatestActiveExecution(
  executions: RecipeExecutionRecord[],
): RecipeExecutionRecord | null {
  const now = Date.now();
  for (const execution of executions) {
    if (!ACTIVE_STATUSES.has(execution.status)) {
      continue;
    }
    if (!execution.jobId) {
      continue;
    }
    if (execution.finishedAt !== null) {
      continue;
    }

    const liveSignal = hasLiveExecutionSignal(execution);
    if (!liveSignal && execution.status === "pending") {
      const ageMs = Math.max(0, now - execution.createdAt);
      if (ageMs > FRESH_PENDING_WINDOW_MS) {
        continue;
      }
    }
    if (!liveSignal && execution.status !== "pending") {
      continue;
    }

    return execution;
  }
  return null;
}

export function deriveGraphRuntimeVisualState(input: {
  activeExecution: RecipeExecutionRecord | null;
  configs: Record<string, NodeConfig>;
  edges: Edge[];
}): GraphRuntimeVisualState {
  const { activeExecution, configs, edges } = input;
  if (!activeExecution) {
    return {
      executionLocked: false,
      runningNodeId: null,
      doneNodeIds: new Set(),
      activeEdgeIds: new Set(),
      batch: null,
    };
  }

  const nameToNodeId = new Map<string, string>();
  for (const config of Object.values(configs)) {
    const name = config.name.trim();
    if (!name) {
      continue;
    }
    nameToNodeId.set(name, config.id);
  }

  const doneNodeIds = new Set<string>();
  for (const columnName of activeExecution.completed_columns) {
    const nodeId = nameToNodeId.get(columnName.trim());
    if (nodeId) {
      doneNodeIds.add(nodeId);
    }
  }

  const runningNodeId = activeExecution.current_column
    ? nameToNodeId.get(activeExecution.current_column.trim()) ?? null
    : null;
  if (runningNodeId) {
    doneNodeIds.delete(runningNodeId);
  }

  const activeEdgeIds = new Set<string>();
  if (runningNodeId) {
    const runningConfig = configs[runningNodeId] ?? null;
    const runningTemplateRefs = collectTemplateRefs(runningConfig);
    for (const ref of runningTemplateRefs) {
      const refNodeId = nameToNodeId.get(ref);
      if (refNodeId && refNodeId !== runningNodeId) {
        doneNodeIds.add(refNodeId);
      }
    }
    for (const upstreamNodeId of collectUpstreamDoneNodeIds({
      rootNodeId: runningNodeId,
      edges,
      configs,
    })) {
      doneNodeIds.add(upstreamNodeId);
    }
    for (const edge of edges) {
      if (isAuxEdge(edge)) {
        continue;
      }
      if (edge.target === runningNodeId) {
        activeEdgeIds.add(edge.id);
        continue;
      }
      if (
        isReversedRuntimeReferenceEdge({
          edge,
          runningNodeId,
          runningTemplateRefs,
          configs,
        })
      ) {
        activeEdgeIds.add(edge.id);
      }
    }
  }

  const batch =
    activeExecution.batch &&
    typeof activeExecution.batch.total === "number" &&
    activeExecution.batch.total > 1
      ? activeExecution.batch
      : null;

  return {
    executionLocked: true,
    runningNodeId,
    doneNodeIds,
    activeEdgeIds,
    batch,
  };
}

function collectUpstreamDoneNodeIds(input: {
  rootNodeId: string;
  edges: Edge[];
  configs: Record<string, NodeConfig>;
}): Set<string> {
  const { rootNodeId, edges, configs } = input;
  const incoming = new Map<string, string[]>();
  for (const edge of edges) {
    if (isAuxEdge(edge)) {
      continue;
    }
    const list = incoming.get(edge.target) ?? [];
    list.push(edge.source);
    incoming.set(edge.target, list);
  }

  const visited = new Set<string>();
  const queue = [rootNodeId];
  let queueIndex = 0;
  const doneNodeIds = new Set<string>();
  while (queueIndex < queue.length) {
    const current = queue[queueIndex];
    queueIndex += 1;
    if (!current || visited.has(current)) {
      continue;
    }
    visited.add(current);
    const sources = incoming.get(current) ?? [];
    for (const sourceId of sources) {
      if (!visited.has(sourceId)) {
        queue.push(sourceId);
      }
      const config = configs[sourceId];
      if (config && DONE_UPSTREAM_KINDS.has(config.kind)) {
        doneNodeIds.add(sourceId);
      }
    }
  }

  return doneNodeIds;
}
