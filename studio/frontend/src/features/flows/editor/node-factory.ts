// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { defaultParams, getSpec } from "../stores/catalog-store";
import { newId, useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowEdge, FlowNode, PortSpec } from "../types";
import {
  FLOW_IN_HANDLE,
  FLOW_TARGET_PORT,
  checkConnection,
  flowOutHandle,
  inHandle,
  outHandle,
  wouldCreateCycle,
} from "./connection-rules";
import {
  REROUTE_IN_HANDLE,
  REROUTE_OUT_HANDLE,
  REROUTE_SIZE,
  REROUTE_TYPE,
  addReroute,
  connectThroughReroute,
  touchesReroute,
} from "./reroutes";

export const NODE_DRAG_MIME = "application/x-unsloth-flow-node";

export function createNode(
  type: string,
  pos: [number, number],
): FlowNode | null {
  const spec = getSpec(type);
  if (!spec) return null;
  return { id: newId("n"), type, v: spec.v, params: defaultParams(spec), pos };
}

export interface ConnectRequest {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export function tryConnect(req: ConnectRequest): string | null {
  const doc = useDocStore.getState();
  if (touchesReroute(doc.graph, req)) return connectThroughReroute(req);
  const sourceNode = doc.graph.nodes[req.source];
  const targetNode = doc.graph.nodes[req.target];
  if (!sourceNode || !targetNode) return "Missing node";
  const sourceSpec = getSpec(sourceNode.type);
  const targetSpec = getSpec(targetNode.type);
  const check = checkConnection(
    req.source,
    sourceSpec,
    req.sourceHandle,
    req.target,
    targetSpec,
    req.targetHandle,
  );
  if (!check.ok) return check.reason ?? "Cannot connect";
  if (
    check.kind === "data" &&
    wouldCreateCycle(
      Object.values(doc.graph.edges),
      req.source,
      req.target,
      check.targetPort,
    )
  ) {
    return "That would make a loop. Use For Each to repeat steps.";
  }
  const targetPort = targetSpec?.inputs.find((p) => p.id === check.targetPort);
  const edge: FlowEdge = {
    id: newId("e"),
    source: req.source,
    source_port: check.sourcePort as string,
    target: req.target,
    target_port: check.targetPort as string,
    kind: check.kind ?? "data",
  };
  doc.connect(edge, !targetPort?.multi);
  return null;
}

export function addNodeConnected(
  type: string,
  pos: [number, number],
  from?: {
    nodeId: string;
    handleId: string;
    side: "in" | "out";
    kind: "data" | "flow";
    port: PortSpec | null;
  },
  port?: string,
): FlowNode | null {
  if (type === REROUTE_TYPE) {
    const id = addReroute([
      Math.round(pos[0] + 130 - REROUTE_SIZE.width / 2),
      Math.round(pos[1] + 20 - REROUTE_SIZE.height / 2),
    ]);
    if (from?.kind === "data") {
      tryConnect(
        from.side === "out"
          ? {
              source: from.nodeId,
              sourceHandle: from.handleId,
              target: id,
              targetHandle: REROUTE_IN_HANDLE,
            }
          : {
              source: id,
              sourceHandle: REROUTE_OUT_HANDLE,
              target: from.nodeId,
              targetHandle: from.handleId,
            },
      );
    }
    useUiStore.getState().setSelection([id], []);
    return null;
  }
  const node = createNode(type, pos);
  if (!node) return null;
  useDocStore.getState().addNode(node);
  if (from && port) {
    if (from.kind === "flow") {
      if (from.side === "out") {
        tryConnect({
          source: from.nodeId,
          sourceHandle: from.handleId,
          target: node.id,
          targetHandle: FLOW_IN_HANDLE,
        });
      } else {
        tryConnect({
          source: node.id,
          sourceHandle: flowOutHandle(port),
          target: from.nodeId,
          targetHandle: FLOW_IN_HANDLE,
        });
      }
    } else if (from.side === "out") {
      tryConnect({
        source: from.nodeId,
        sourceHandle: from.handleId,
        target: node.id,
        targetHandle: inHandle(port),
      });
    } else {
      tryConnect({
        source: node.id,
        sourceHandle: outHandle(port),
        target: from.nodeId,
        targetHandle: from.handleId,
      });
    }
  }
  return node;
}

export { FLOW_TARGET_PORT };
