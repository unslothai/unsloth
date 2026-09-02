// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  AgentGraphEdge,
  AgentGraphNode,
  AgentGraphRun,
} from "../api/agent-workspace-api";

type GraphMutationTarget = {
  projectId: string;
  graphId: string | null;
  runId?: string | null;
};

export type GraphRunActions = {
  start: boolean;
  resume: boolean;
  pause: boolean;
  cancel: boolean;
  retry: boolean;
};

let activeDraftNavigationGuard: (() => boolean) | null = null;
let authorizedRouteNavigation = false;

export function graphDraftIsPending(
  documentDirty: boolean,
  editorHasDrafts: boolean,
  jsonDraftDirty: boolean,
  runInputDirty = false,
): boolean {
  return documentDirty || editorHasDrafts || jsonDraftDirty || runInputDirty;
}

export function graphResponseIsCurrent(
  requestGeneration: number,
  currentGeneration: number,
  requestedGraphId: string,
  selectedGraphId: string | null,
): boolean {
  return (
    requestGeneration === currentGeneration &&
    requestedGraphId === selectedGraphId
  );
}

export function graphLoadCanApply(
  requestGeneration: number,
  currentGeneration: number,
  requestedGraphId: string,
  selectedGraphId: string | null,
  hasUnsavedDraft: boolean,
): boolean {
  return (
    !hasUnsavedDraft &&
    graphResponseIsCurrent(
      requestGeneration,
      currentGeneration,
      requestedGraphId,
      selectedGraphId,
    )
  );
}

export function graphEditorIsReady(
  selectedGraphId: string | null,
  loadedGraphId: string | null,
): boolean {
  return selectedGraphId === null || selectedGraphId === loadedGraphId;
}

export function graphMutationIsCurrent(
  target: GraphMutationTarget,
  current: GraphMutationTarget,
): boolean {
  return (
    target.projectId === current.projectId &&
    target.graphId === current.graphId &&
    (target.runId === undefined || target.runId === current.runId)
  );
}

export function graphRunMatchesEditor(
  run: Pick<AgentGraphRun, "graphId" | "revision"> | null,
  graphId: string | null,
  revision: number,
): boolean {
  return Boolean(run && run.graphId === graphId && run.revision === revision);
}

export function graphRunResponseIsCurrent(
  requestGeneration: number,
  currentGeneration: number,
  requestedGraphId: string,
  selectedGraphId: string | null,
  requestedRunId: string,
  selectedRunId: string | null,
): boolean {
  return (
    graphResponseIsCurrent(
      requestGeneration,
      currentGeneration,
      requestedGraphId,
      selectedGraphId,
    ) && requestedRunId === selectedRunId
  );
}

export function graphRunActions(status: string): GraphRunActions {
  return {
    start: status === "queued",
    resume: status === "paused" || status === "interrupted",
    pause: status === "queued" || status === "running",
    cancel: ["queued", "running", "pausing", "paused"].includes(status),
    retry: ["failed", "cancelled", "interrupted"].includes(status),
  };
}

export function graphEdgeId(edge: AgentGraphEdge): string {
  return `${edge.from}:${edge.to}:${edge.when || "next"}`;
}

function graphOutgoingEdgesAreValid(
  node: Pick<AgentGraphNode, "type"> | undefined,
  outgoing: AgentGraphEdge[],
): boolean {
  if (node?.type === "condition") {
    if (outgoing.length > 2) {
      return false;
    }
    if (outgoing.length === 1) {
      return [undefined, "default"].includes(outgoing[0]?.when);
    }
    if (outgoing.length === 2) {
      return (
        new Set(outgoing.map((edge) => edge.when)).size === 2 &&
        outgoing.every((edge) => edge.when === "true" || edge.when === "false")
      );
    }
    return true;
  }
  if (node?.type === "output") {
    return outgoing.length === 0;
  }
  return outgoing.length <= 1;
}

function graphEdgesCreateCycle(
  edges: AgentGraphEdge[],
  source: string,
  target: string,
): boolean {
  const adjacency = new Map<string, string[]>();
  for (const edge of edges) {
    const targets = adjacency.get(edge.from) ?? [];
    targets.push(edge.to);
    adjacency.set(edge.from, targets);
  }
  const pending = [target];
  const visited = new Set<string>();
  while (pending.length > 0) {
    const current = pending.pop();
    if (!current || visited.has(current)) {
      continue;
    }
    if (current === source) {
      return true;
    }
    visited.add(current);
    pending.push(...(adjacency.get(current) ?? []));
  }
  return false;
}

export function graphConnectionIsValid(
  edges: AgentGraphEdge[],
  nodes: Pick<AgentGraphNode, "id" | "type">[],
  source: string,
  target: string,
  when?: AgentGraphEdge["when"],
  ignoredEdgeId?: string,
): boolean {
  const sourceNode = nodes.find((node) => node.id === source);
  const targetNode = nodes.find((node) => node.id === target);
  if (!(sourceNode && targetNode)) {
    return false;
  }
  if (
    source === target ||
    sourceNode.type === "output" ||
    targetNode.type === "input" ||
    (sourceNode.type !== "condition" && when !== undefined)
  ) {
    return false;
  }
  const ignoredEdge = edges.find(
    (edge) =>
      ignoredEdgeId !== undefined && graphEdgeId(edge) === ignoredEdgeId,
  );
  const retained = edges.filter(
    (edge) =>
      ignoredEdgeId === undefined || graphEdgeId(edge) !== ignoredEdgeId,
  );
  const candidate: AgentGraphEdge = {
    from: source,
    to: target,
    ...(when === undefined ? {} : { when }),
  };
  if (
    retained.some((edge) => graphEdgeId(edge) === graphEdgeId(candidate)) ||
    retained.some((edge) => edge.to === target)
  ) {
    return false;
  }
  const candidateEdges = [...retained, candidate];
  const affectedSources = new Set<string>([source]);
  if (ignoredEdge) {
    affectedSources.add(ignoredEdge.from);
  }
  for (const affectedSource of affectedSources) {
    const node = nodes.find((item) => item.id === affectedSource);
    const outgoing = candidateEdges.filter(
      (edge) => edge.from === affectedSource,
    );
    if (!graphOutgoingEdgesAreValid(node, outgoing)) {
      return false;
    }
  }
  return !graphEdgesCreateCycle(candidateEdges, source, target);
}

export function connectGraphEdge(
  edges: AgentGraphEdge[],
  nodes: Pick<AgentGraphNode, "id" | "type">[],
  source: string,
  target: string,
  when?: AgentGraphEdge["when"],
): AgentGraphEdge[] {
  const sourceNode = nodes.find((node) => node.id === source);
  const outgoing = edges.filter((edge) => edge.from === source);
  let retained = edges;
  if (
    sourceNode?.type === "condition" &&
    outgoing.length === 1 &&
    [undefined, "default"].includes(outgoing[0]?.when) &&
    (when === "true" || when === "false")
  ) {
    const opposite = when === "true" ? "false" : "true";
    retained = edges.map((edge) =>
      edge === outgoing[0] ? { ...edge, when: opposite } : edge,
    );
  }
  if (!graphConnectionIsValid(retained, nodes, source, target, when)) {
    return edges;
  }
  return [...retained, { from: source, to: target, ...(when ? { when } : {}) }];
}

export function replaceGraphEdge(
  edges: AgentGraphEdge[],
  nodes: Pick<AgentGraphNode, "id" | "type">[],
  edgeIndex: number,
  replacement: AgentGraphEdge,
): AgentGraphEdge[] {
  const previous = edges[edgeIndex];
  if (!previous) {
    return edges;
  }
  if (
    !graphConnectionIsValid(
      edges,
      nodes,
      replacement.from,
      replacement.to,
      replacement.when,
      graphEdgeId(previous),
    )
  ) {
    return edges;
  }
  return edges.map((edge, index) => (index === edgeIndex ? replacement : edge));
}

export function reconnectGraphEdge(
  edges: AgentGraphEdge[],
  nodes: Pick<AgentGraphNode, "id" | "type">[],
  oldEdgeId: string,
  source: string,
  target: string,
): AgentGraphEdge[] {
  const sourceIsCondition = nodes.some(
    (node) => node.id === source && node.type === "condition",
  );
  const edgeIndex = edges.findIndex((edge) => graphEdgeId(edge) === oldEdgeId);
  if (edgeIndex < 0) {
    return edges;
  }
  const previous = edges[edgeIndex];
  const replacement: AgentGraphEdge = {
    from: source,
    to: target,
    ...(sourceIsCondition && previous.when ? { when: previous.when } : {}),
  };
  return replaceGraphEdge(edges, nodes, edgeIndex, replacement);
}

export function graphRouteRetainsProjectEditor(
  next: { pathname: string; search: unknown },
  projectId: string,
): boolean {
  if (next.pathname !== "/chat") {
    return false;
  }
  const search =
    next.search && typeof next.search === "object"
      ? (next.search as Record<string, unknown>)
      : {};
  if (typeof search.compare === "string" && search.compare.length > 0) {
    return true;
  }
  return (
    search.project === projectId &&
    search.thread === undefined &&
    search.new === undefined
  );
}

export function agentGraphRouteNavigationShouldBlock(
  next: { pathname: string; search: unknown },
  projectId: string,
  confirmDraftDiscard: () => boolean,
): boolean {
  if (consumeAgentGraphDraftRouteAuthorization()) {
    return false;
  }
  if (graphRouteRetainsProjectEditor(next, projectId)) {
    return false;
  }
  return !confirmDraftDiscard();
}

export function registerAgentGraphDraftNavigationGuard(
  guard: () => boolean,
): () => void {
  activeDraftNavigationGuard = guard;
  return () => {
    if (activeDraftNavigationGuard === guard) {
      activeDraftNavigationGuard = null;
    }
  };
}

export function confirmAgentGraphDraftNavigation(): boolean {
  return activeDraftNavigationGuard?.() ?? true;
}

export function agentGraphProjectSubmitIsAllowed(
  graphEditorActive: boolean,
): boolean {
  return !graphEditorActive || confirmAgentGraphDraftNavigation();
}

export function authorizeAgentGraphDraftRouteNavigation(): boolean {
  if (!confirmAgentGraphDraftNavigation()) {
    return false;
  }
  if (activeDraftNavigationGuard) {
    authorizedRouteNavigation = true;
  }
  return true;
}

export function permitNextAgentGraphDraftRouteNavigation(): void {
  if (activeDraftNavigationGuard) {
    authorizedRouteNavigation = true;
  }
}

export function consumeAgentGraphDraftRouteAuthorization(): boolean {
  const authorized = authorizedRouteNavigation;
  authorizedRouteNavigation = false;
  return authorized;
}
