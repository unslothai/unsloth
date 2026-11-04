// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type Connection,
  type FinalConnectionState,
  useReactFlow,
} from "@xyflow/react";
import { useCallback, useMemo, useRef } from "react";
import { toast } from "sonner";
import { disconnectColumns } from "../recipe/recipe-actions";
import { COLUMN_TYPE } from "../recipe/recipe-columns";
import { HISTORY_LIMIT, useDocStore } from "../stores/doc-store";
import type { RfEdge } from "./derive-rf";
import { tryConnect } from "./node-factory";

const EDGE_OPTIONS = { reconnectable: "target" as const };
// A read-only canvas: no ×, no drag-off, nothing deletes a wire.
const LOCKED_OPTIONS = { reconnectable: false, deletable: false };

/** Moves a wire to another input as one undo step; restores it if the new input refuses. */
export function reconnectWire(
  old: RfEdge,
  connection: Connection,
): string | null {
  const before = useDocStore.getState();
  // A reroute's wire segments are not plain edges and do not reconnect.
  if (!before.graph.edges[old.id] || isWireSegment(before.graph, old.id))
    return null;
  if (
    old.source === connection.source &&
    old.target === connection.target &&
    (old.sourceHandle ?? null) === (connection.sourceHandle ?? null) &&
    (old.targetHandle ?? null) === (connection.targetHandle ?? null)
  )
    return null;
  const { graph, meta, past, future } = before;
  const wire = graph.edges[old.id];
  if (
    graph.nodes[wire.source]?.type === COLUMN_TYPE &&
    graph.nodes[wire.target]?.type === COLUMN_TYPE
  ) {
    // A wire between recipe columns is the reference in the target's prompt: take it out there,
    // or the reference sync draws the old wire again.
    if (!disconnectColumns([old.id]).length) {
      useDocStore.setState({ graph, past, future });
      return ""; // disconnectColumns already said why
    }
  }
  if (useDocStore.getState().graph.edges[old.id])
    useDocStore.getState().removeElements({ edges: [old.id] });
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
  // The removal and the new wire are one step: history goes back to the graph before the move,
  // whatever the bounded stack dropped in between.
  useDocStore.setState({
    past: [...past, { graph, meta }].slice(-HISTORY_LIMIT),
  });
  return null;
}

function isWireSegment(graph: object, id: string): boolean {
  const wires = (graph as { wires?: Record<string, unknown> }).wires;
  return !!wires?.[id];
}

/**
 * Drag a wire's end to another port to move it, or off onto the canvas to remove it.
 * `reconnecting` is true for the length of that drag, so the canvas can keep it out of Quick Add.
 */
export function useWireReconnect(enabled: boolean) {
  const { deleteElements } = useReactFlow();
  const moved = useRef(false);
  const reconnecting = useRef(false);
  const onReconnectStart = useCallback(() => {
    moved.current = false;
    reconnecting.current = true;
  }, []);
  const onReconnect = useCallback((old: RfEdge, connection: Connection) => {
    moved.current = true;
    const error = reconnectWire(old, connection);
    if (error) toast.error(error);
  }, []);
  const onReconnectEnd = useCallback(
    (
      _: unknown,
      edge: RfEdge,
      _handle: unknown,
      state?: FinalConnectionState,
    ) => {
      reconnecting.current = false;
      // Dropped on empty canvas: remove. Dropped on a port that refused it: keep the wire.
      if (!moved.current && !state?.toHandle)
        void deleteElements({ edges: [{ id: edge.id }] });
      moved.current = false;
    },
    [deleteElements],
  );
  const props = useMemo(
    () =>
      enabled
        ? {
            onReconnect,
            onReconnectStart,
            onReconnectEnd,
            defaultEdgeOptions: EDGE_OPTIONS,
          }
        : { defaultEdgeOptions: LOCKED_OPTIONS },
    [enabled, onReconnect, onReconnectStart, onReconnectEnd],
  );
  return { props, reconnecting };
}
