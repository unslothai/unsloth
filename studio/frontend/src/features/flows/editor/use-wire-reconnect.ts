// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Connection, useReactFlow } from "@xyflow/react";
import { useCallback, useMemo, useRef } from "react";
import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import type { RfEdge } from "./derive-rf";
import { tryConnect } from "./node-factory";

const EDGE_OPTIONS = { reconnectable: "target" as const };

/** Moves a wire to another input as one undo step; restores it if the new input refuses. */
export function reconnectWire(old: RfEdge, connection: Connection): string | null {
  const before = useDocStore.getState();
  if (!before.graph.edges[old.id]) return null;
  if (
    old.source === connection.source &&
    old.target === connection.target &&
    (old.sourceHandle ?? null) === (connection.sourceHandle ?? null) &&
    (old.targetHandle ?? null) === (connection.targetHandle ?? null)
  )
    return null;
  const { graph, past, future } = before;
  before.removeElements({ edges: [old.id] });
  const error = tryConnect({
    source: connection.source,
    sourceHandle: connection.sourceHandle ?? "",
    target: connection.target,
    targetHandle: connection.targetHandle ?? "",
  });
  if (error) {
    useDocStore.setState({ graph, past, future });
    return error;
  }
  // The removal and the new wire are one step: drop the snapshot between them.
  useDocStore.setState((s) =>
    s.past.length === past.length + 2 ? { past: s.past.slice(0, -1) } : {},
  );
  return null;
}

/** Drag a wire's input end to another port to move it, or off onto the canvas to remove it. */
export function useWireReconnect(enabled: boolean) {
  const { deleteElements } = useReactFlow();
  const moved = useRef(false);
  const onReconnectStart = useCallback(() => {
    moved.current = false;
  }, []);
  const onReconnect = useCallback((old: RfEdge, connection: Connection) => {
    moved.current = true;
    const error = reconnectWire(old, connection);
    if (error) toast.error(error);
  }, []);
  const onReconnectEnd = useCallback(
    (_: unknown, edge: RfEdge) => {
      if (!moved.current) void deleteElements({ edges: [{ id: edge.id }] });
      moved.current = false;
    },
    [deleteElements],
  );
  return useMemo(
    () =>
      enabled
        ? {
            onReconnect,
            onReconnectStart,
            onReconnectEnd,
            defaultEdgeOptions: EDGE_OPTIONS,
          }
        : {},
    [enabled, onReconnect, onReconnectStart, onReconnectEnd],
  );
}
