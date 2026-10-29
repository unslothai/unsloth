// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import {
  Background,
  BackgroundVariant,
  type Connection,
  ConnectionMode,
  Controls,
  type EdgeChange,
  type FinalConnectionState,
  MiniMap,
  type NodeChange,
  type OnConnectStartParams,
  ReactFlow,
  SelectionMode,
  useReactFlow,
} from "@xyflow/react";
import {
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { categoryColor, findPort, getSpec } from "../stores/catalog-store";
import { type DocGraph, useDocStore } from "../stores/doc-store";
import {
  type RerouteWire,
  rerouteDownstream,
  rerouteUpstream,
} from "../stores/reroute-graph";
import { useUiStore } from "../stores/ui-store";
import type { FlowEdge } from "../types";
import { dragHasFiles } from "./canvas-rules";
import {
  checkConnection,
  parseHandle,
  wouldCreateCycle,
} from "./connection-rules";
import {
  GROUP_PREFIX,
  NOTE_PREFIX,
  RfDeriver,
  type RfEdge,
  type RfNode,
  type WireInfo,
} from "./derive-rf";
import { useWireReconnect } from "./use-wire-reconnect";
import { addFileNodes } from "./file-drop";
import { FlowEdgeView } from "./flow-edge";
import { FlowNodeView } from "./flow-node";
import { GroupFrame, NoteFrame } from "./frame-nodes";
import { NODE_DRAG_MIME, addNodeConnected, tryConnect } from "./node-factory";
import { SelectedNodeToolbar } from "./node-toolbar";
import { RerouteNode } from "./reroute-node";
import {
  REROUTE_OUT_HANDLE,
  REROUTE_SIZE,
  addReroute,
  checkRerouteConnection,
  reroutePort,
  touchesReroute,
} from "./reroutes";
import { WireRouterController, setWireHover } from "./wire-routing";

const NODE_TYPES = {
  flowNode: FlowNodeView,
  flowGroup: GroupFrame,
  flowNote: NoteFrame,
  flowReroute: RerouteNode,
};
const EDGE_TYPES = { flowEdge: FlowEdgeView };
const SNAP_GRID: [number, number] = [16, 16];
const VISIBLE_ONLY_THRESHOLD = 300;
const PRO_OPTIONS = { hideAttribution: true };
const DELETE_KEYS = ["Delete", "Backspace"];
const MULTI_SELECT_KEYS = ["Meta", "Control", "Shift"];
const FIT_VIEW_OPTIONS = { padding: 0.2, maxZoom: 1.1 };

type Size = { width: number; height: number };
type Positions = Record<string, { x: number; y: number }>;

function portTypeOf(edge: FlowEdge): string {
  if (edge.kind === "flow") return "flow";
  const node = useDocStore.getState().graph.nodes[edge.source];
  const spec = node ? getSpec(node.type) : undefined;
  return findPort(spec, "out", edge.source_port)?.type ?? "any";
}

function wireInfo(wire: RerouteWire): WireInfo {
  const graph = useDocStore.getState().graph;
  const reroute = graph.reroutes[wire.source] ? wire.source : wire.target;
  const portType = graph.reroutes[wire.source]
    ? reroutePort(graph, wire.source).type
    : (findPort(
        getSpec(graph.nodes[wire.source]?.type ?? ""),
        "out",
        wire.source_port,
      )?.type ?? "any");
  const ends = new Set([wire.source, wire.target]);
  const upstream = rerouteUpstream(graph, reroute);
  if (upstream) ends.add(upstream.node);
  for (const ref of rerouteDownstream(graph, reroute)) ends.add(ref.node);
  return { portType, ends: [...ends] };
}

function wireUnder(x: number, y: number, ignore: string): string | null {
  for (const element of document.elementsFromPoint(x, y)) {
    const edge = element.closest(".react-flow__edge");
    const id = edge?.getAttribute("data-id");
    if (id && id !== ignore) return id;
  }
  return null;
}

function hasWires(graph: DocGraph, rerouteId: string): boolean {
  return graph.wireOrder.some(
    (id) =>
      graph.wires[id].source === rerouteId ||
      graph.wires[id].target === rerouteId,
  );
}

function minimapColor(node: RfNode): string {
  if (node.type === "flowReroute") return "#71717a";
  if (node.type !== "flowNode") return "#27272a";
  const doc = useDocStore.getState().graph.nodes[node.id];
  const spec = doc ? getSpec(doc.type) : undefined;
  return spec ? categoryColor(spec.category) : "#52525b";
}

function insideRect(
  x: number,
  y: number,
  rect: [number, number, number, number],
): boolean {
  return (
    x >= rect[0] &&
    y >= rect[1] &&
    x <= rect[0] + rect[2] &&
    y <= rect[1] + rect[3]
  );
}

function groupMembership(
  positions: Record<string, [number, number]>,
  sizes: Map<string, Size>,
): Record<string, string | null> {
  const { graph } = useDocStore.getState();
  const membership: Record<string, string | null> = {};
  for (const [id, pos] of Object.entries(positions)) {
    if (!graph.nodes[id]) continue;
    const width = sizes.get(id)?.width ?? 260;
    let best: { id: string; area: number } | null = null;
    for (const groupId of graph.groupOrder) {
      const key = GROUP_PREFIX + groupId;
      const g = graph.groups[groupId];
      const rect: [number, number, number, number] = positions[key]
        ? [positions[key][0], positions[key][1], g.rect[2], g.rect[3]]
        : g.rect;
      if (!insideRect(pos[0] + width / 2, pos[1] + 16, rect)) continue;
      const area = rect[2] * rect[3];
      if (!best || area < best.area) best = { id: groupId, area };
    }
    membership[id] = best?.id ?? null;
  }
  return membership;
}

export function FlowCanvas({ readOnly = false }: { readOnly?: boolean }) {
  const graph = useDocStore((s) => s.graph);
  const selectedNodes = useUiStore((s) => s.selectedNodes);
  const selectedEdges = useUiStore((s) => s.selectedEdges);
  const snapToGrid = useUiStore((s) => s.snapToGrid);
  const bus = useUiStore((s) => s.wireStyle === "bus");
  const flowId = useDocStore((s) => s.flowId);
  const { screenToFlowPosition, flowToScreenPosition } = useReactFlow();
  const { props: wireReconnect, reconnecting } =
    useWireReconnect(!readOnly);

  const deriver = useRef<RfDeriver | null>(null);
  deriver.current ??= new RfDeriver();
  const [drag, setDrag] = useState<Positions | null>(null);
  const dragMembers = useRef<{
    groupRfId: string;
    origin: { x: number; y: number };
    members: Map<string, { x: number; y: number }>;
  } | null>(null);
  const [sizes, setSizes] = useState<Map<string, Size>>(() => new Map());
  const connectStart = useRef<OnConnectStartParams | null>(null);
  const sizesRef = useRef(sizes);
  sizesRef.current = sizes;
  const dragRef = useRef<Positions | null>(null);
  dragRef.current = drag;
  const moving = useMemo(
    () => (drag ? new Set(Object.keys(drag)) : null),
    [drag],
  );

  const selectedSet = useMemo(
    () => new Set([...selectedNodes, ...selectedEdges]),
    [selectedNodes, selectedEdges],
  );
  const nodes = useMemo(
    () => (deriver.current as RfDeriver).nodes(graph, selectedSet, drag, sizes),
    [graph, selectedSet, drag, sizes],
  );
  const edges = useMemo(
    () =>
      (deriver.current as RfDeriver).edges(
        graph,
        selectedSet,
        portTypeOf,
        wireInfo,
      ),
    [graph, selectedSet],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<RfNode>[]) => {
      const drag = dragRef.current;
      const sizes = sizesRef.current;
      let nextDrag: Positions | null = null;
      let nextSizes: Map<string, Size> | null = null;
      let selection: { add: Set<string>; remove: Set<string> } | null = null;
      for (const change of changes) {
        if (change.type === "position" && change.position) {
          if (change.dragging) {
            nextDrag ??= { ...(drag ?? {}) };
            nextDrag[change.id] = change.position;
            const group = dragMembers.current;
            if (group && group.groupRfId === change.id) {
              const dx = change.position.x - group.origin.x;
              const dy = change.position.y - group.origin.y;
              for (const [memberId, origin] of group.members) {
                nextDrag[memberId] = { x: origin.x + dx, y: origin.y + dy };
              }
            }
          }
        } else if (change.type === "dimensions" && change.dimensions) {
          const current = (nextSizes ?? sizes).get(change.id);
          if (
            !current ||
            current.width !== change.dimensions.width ||
            current.height !== change.dimensions.height
          ) {
            nextSizes ??= new Map(sizes);
            nextSizes.set(change.id, {
              width: change.dimensions.width,
              height: change.dimensions.height,
            });
          }
        } else if (change.type === "select") {
          selection ??= { add: new Set(), remove: new Set() };
          (change.selected ? selection.add : selection.remove).add(change.id);
        }
      }
      if (nextDrag) setDrag(nextDrag);
      if (nextSizes) setSizes(nextSizes);
      if (selection) {
        const ui = useUiStore.getState();
        const next = new Set(ui.selectedNodes);
        for (const id of selection.remove) next.delete(id);
        for (const id of selection.add) next.add(id);
        ui.setSelection([...next], ui.selectedEdges);
      }
    },
    [],
  );

  const onEdgesChange = useCallback((changes: EdgeChange<RfEdge>[]) => {
    let selection: { add: Set<string>; remove: Set<string> } | null = null;
    for (const change of changes) {
      if (change.type === "select") {
        selection ??= { add: new Set(), remove: new Set() };
        (change.selected ? selection.add : selection.remove).add(change.id);
      }
    }
    if (selection) {
      const ui = useUiStore.getState();
      const next = new Set(ui.selectedEdges);
      for (const id of selection.remove) next.delete(id);
      for (const id of selection.add) next.add(id);
      ui.setSelection(ui.selectedNodes, [...next]);
    }
  }, []);

  const onNodeDragStart = useCallback(
    (_: ReactMouseEvent, node: RfNode) => {
      dragMembers.current = null;
      if (!node.id.startsWith(GROUP_PREFIX)) return;
      const doc = useDocStore.getState().graph;
      const group = doc.groups[node.id.slice(GROUP_PREFIX.length)];
      if (!group) return;
      const members = new Map<string, { x: number; y: number }>();
      const currentSizes = sizesRef.current;
      for (const id of doc.nodeOrder) {
        const n = doc.nodes[id];
        const size = currentSizes.get(id) ?? { width: 260, height: 120 };
        if (insideRect(n.pos[0] + size.width / 2, n.pos[1] + 20, group.rect)) {
          members.set(id, { x: n.pos[0], y: n.pos[1] });
        }
      }
      for (const id of doc.rerouteOrder) {
        const [x, y] = doc.reroutes[id].pos;
        if (insideRect(x, y, group.rect)) members.set(id, { x, y });
      }
      dragMembers.current = {
        groupRfId: node.id,
        origin: { x: group.rect[0], y: group.rect[1] },
        members,
      };
    },
    [],
  );

  const onNodeDragStop = useCallback(
    (_: unknown, node: RfNode) => {
      dragMembers.current = null;
      const current = dragRef.current;
      if (current) {
        const positions: Record<string, [number, number]> = {};
        for (const [id, p] of Object.entries(current))
          positions[id] = [Math.round(p.x), Math.round(p.y)];
        useDocStore
          .getState()
          .moveNodes(positions, groupMembership(positions, sizesRef.current));
      }
      setDrag(null);
      const doc = useDocStore.getState().graph;
      const reroute = doc.reroutes[node.id];
      if (!reroute || hasWires(doc, node.id)) return;
      // A free reroute dropped on a wire threads itself into it.
      const centre = flowToScreenPosition({
        x: reroute.pos[0] + REROUTE_SIZE.width / 2,
        y: reroute.pos[1] + REROUTE_SIZE.height / 2,
      });
      const wire = wireUnder(centre.x, centre.y, node.id);
      if (wire && (doc.wires[wire] || doc.edges[wire]?.kind === "data"))
        useDocStore.getState().addReroute(reroute, wire);
    },
    [flowToScreenPosition],
  );

  const onConnect = useCallback((connection: Connection) => {
    const error = tryConnect({
      source: connection.source,
      sourceHandle: connection.sourceHandle ?? "",
      target: connection.target,
      targetHandle: connection.targetHandle ?? "",
    });
    if (error) toast.error(error);
  }, []);

  const isValidConnection = useCallback((connection: Connection | RfEdge) => {
    const doc = useDocStore.getState().graph;
    if (touchesReroute(doc, connection)) {
      return checkRerouteConnection(doc, {
        source: connection.source,
        sourceHandle: connection.sourceHandle ?? "",
        target: connection.target,
        targetHandle: connection.targetHandle ?? "",
      }).ok;
    }
    const source = doc.nodes[connection.source];
    const target = doc.nodes[connection.target];
    if (!source || !target) return false;
    const check = checkConnection(
      connection.source,
      getSpec(source.type),
      connection.sourceHandle,
      connection.target,
      getSpec(target.type),
      connection.targetHandle,
    );
    if (!check.ok) return false;
    if (
      check.kind === "data" &&
      wouldCreateCycle(
        Object.values(doc.edges),
        connection.source,
        connection.target,
        check.targetPort,
      )
    ) {
      return false;
    }
    return true;
  }, []);

  const onConnectStart = useCallback(
    (_: unknown, params: OnConnectStartParams) => {
      connectStart.current = params;
    },
    [],
  );

  const onConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      const start = connectStart.current;
      connectStart.current = null;
      if (
        readOnly ||
        // Dropping a wire's end to remove it is not a new wire.
        reconnecting.current ||
        !start?.nodeId ||
        !start.handleId ||
        state.isValid ||
        state.toHandle
      )
        return;
      const target = event.target as Element | null;
      if (!target?.classList.contains("react-flow__pane")) return;
      const point = "changedTouches" in event ? event.changedTouches[0] : event;
      const graph = useDocStore.getState().graph;
      const fromReroute = !!graph.reroutes[start.nodeId];
      const handle = fromReroute
        ? {
            kind: "data" as const,
            side:
              start.handleId === REROUTE_OUT_HANDLE
                ? ("out" as const)
                : ("in" as const),
            port: "",
          }
        : parseHandle(start.handleId);
      if (!handle) return;
      const node = graph.nodes[start.nodeId];
      const spec = node ? getSpec(node.type) : undefined;
      const port = fromReroute
        ? reroutePort(graph, start.nodeId)
        : handle.kind === "data"
          ? (findPort(spec, handle.side, handle.port) ?? null)
          : null;
      const flow = screenToFlowPosition({ x: point.clientX, y: point.clientY });
      useUiStore.getState().setQuickAdd({
        screen: { x: point.clientX, y: point.clientY },
        flow: { x: flow.x - (handle.side === "out" ? 0 : 260), y: flow.y - 20 },
        from: {
          nodeId: start.nodeId,
          handleId: start.handleId,
          side: handle.side,
          kind: handle.kind,
          port,
        },
      });
    },
    [readOnly, screenToFlowPosition],
  );

  const onDelete = useCallback(
    ({
      nodes: rfNodes,
      edges: rfEdges,
    }: { nodes: RfNode[]; edges: RfEdge[] }) => {
      if (readOnly) return;
      const ids = {
        nodes: [] as string[],
        // removeElements keeps a deleted reroute's own wires out of this, so it can heal.
        edges: rfEdges.map((e) => e.id),
        groups: [] as string[],
        notes: [] as string[],
        reroutes: [] as string[],
      };
      for (const node of rfNodes) {
        if (node.type === "flowReroute") ids.reroutes.push(node.id);
        else if (node.id.startsWith(GROUP_PREFIX))
          ids.groups.push(node.id.slice(GROUP_PREFIX.length));
        else if (node.id.startsWith(NOTE_PREFIX))
          ids.notes.push(node.id.slice(NOTE_PREFIX.length));
        else ids.nodes.push(node.id);
      }
      useDocStore.getState().removeElements(ids);
      useUiStore.getState().setSelection([], []);
    },
    [readOnly],
  );

  const onDoubleClick = useCallback(
    (event: ReactMouseEvent) => {
      if (readOnly) return;
      const target = event.target as Element;
      if (!target.classList.contains("react-flow__pane")) return;
      const flow = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      useUiStore
        .getState()
        .setQuickAdd({ screen: { x: event.clientX, y: event.clientY }, flow });
    },
    [readOnly, screenToFlowPosition],
  );

  const onDragOver = useCallback((event: DragEvent) => {
    if (!event.dataTransfer.types.includes(NODE_DRAG_MIME)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const dropPoint = useRef<{ x: number; y: number } | null>(null);
  const wrapper = useRef<HTMLDivElement | null>(null);
  const fileDrop = useNativeFileDrop({
    disabled: readOnly,
    disabledReason: "This flow is read-only",
    onFiles: (files) => {
      const rect = wrapper.current?.getBoundingClientRect();
      const point =
        dropPoint.current ??
        (rect
          ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
          : { x: 0, y: 0 });
      dropPoint.current = null;
      return addFileNodes(files, screenToFlowPosition(point)).then(
        () => undefined,
      );
    },
  });
  const { ref: fileDropRef } = fileDrop;
  const setWrapper = useCallback(
    (element: HTMLDivElement | null) => {
      wrapper.current = element;
      fileDropRef(element);
    },
    [fileDropRef],
  );

  const onDrop = useCallback(
    (event: DragEvent) => {
      const type = event.dataTransfer.getData(NODE_DRAG_MIME);
      if (!type || readOnly) return;
      event.preventDefault();
      const flow = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const node = addNodeConnected(type, [
        Math.round(flow.x - 130),
        Math.round(flow.y - 20),
      ]);
      if (node) useUiStore.getState().setSelection([node.id], []);
    },
    [readOnly, screenToFlowPosition],
  );

  const onNodeDoubleClick = useCallback((_: ReactMouseEvent, node: RfNode) => {
    if (node.type === "flowNode") useUiStore.getState().openInspector(node.id);
  }, []);

  const onEdgeDoubleClick = useCallback(
    (event: ReactMouseEvent, edge: RfEdge) => {
      if (readOnly || edge.data?.kind !== "data") return;
      const flow = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const id = addReroute(
        [
          Math.round(flow.x - REROUTE_SIZE.width / 2),
          Math.round(flow.y - REROUTE_SIZE.height / 2),
        ],
        edge.id,
      );
      useUiStore.getState().setSelection([id], []);
    },
    [readOnly, screenToFlowPosition],
  );

  const onNodeMouseEnter = useCallback((_: ReactMouseEvent, node: RfNode) => {
    if (node.type === "flowNode" || node.type === "flowReroute")
      setWireHover(node.id);
  }, []);
  const onNodeMouseLeave = useCallback(() => setWireHover(null), []);

  return (
    <div
      ref={setWrapper}
      className="relative h-full w-full"
      onDoubleClick={onDoubleClick}
      onDragEnter={fileDrop.dragHandlers.onDragEnter}
      onDragOver={(event) => {
        if (dragHasFiles(Array.from(event.dataTransfer.types)))
          dropPoint.current = { x: event.clientX, y: event.clientY };
        fileDrop.dragHandlers.onDragOver(event);
      }}
      onDragLeave={fileDrop.dragHandlers.onDragLeave}
      onDrop={fileDrop.dragHandlers.onDrop}
      data-testid="flows-canvas"
    >
      <ReactFlow<RfNode, RfEdge>
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        onConnect={onConnect}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        onDelete={onDelete}
        {...wireReconnect}
        onNodeDoubleClick={onNodeDoubleClick}
        onEdgeDoubleClick={onEdgeDoubleClick}
        onNodeMouseEnter={bus ? onNodeMouseEnter : undefined}
        onNodeMouseLeave={bus ? onNodeMouseLeave : undefined}
        onDragOver={onDragOver}
        onDrop={onDrop}
        connectionMode={ConnectionMode.Strict}
        deleteKeyCode={readOnly ? null : DELETE_KEYS}
        multiSelectionKeyCode={MULTI_SELECT_KEYS}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag={false}
        panOnDrag={true}
        panActivationKeyCode="Space"
        zoomOnDoubleClick={false}
        snapToGrid={snapToGrid}
        snapGrid={SNAP_GRID}
        minZoom={0.08}
        maxZoom={2.5}
        onlyRenderVisibleElements={nodes.length > VISIBLE_ONLY_THRESHOLD}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        colorMode="dark"
        proOptions={PRO_OPTIONS}
        fitView={true}
        fitViewOptions={FIT_VIEW_OPTIONS}
        className="flows-canvas bg-zinc-950"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.2}
          color="#3f3f46"
        />
        <MiniMap
          pannable={true}
          zoomable={true}
          nodeColor={minimapColor}
          nodeStrokeWidth={0}
          maskColor="rgba(9,9,11,0.7)"
          className="!rounded-lg !border !border-white/10 !bg-zinc-900"
        />
        <Controls
          showInteractive={false}
          className="!rounded-lg !border !border-white/10 !shadow-none"
        />
        <SelectedNodeToolbar readOnly={readOnly} />
        {bus && (
          <WireRouterController scope={flowId ?? ""} moving={moving} />
        )}
      </ReactFlow>
    </div>
  );
}
