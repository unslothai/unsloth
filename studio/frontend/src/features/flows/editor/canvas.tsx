// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import { useTheme } from "@/features/settings/stores/theme-store";
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
import { useDocStore } from "../stores/doc-store";
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
} from "./derive-rf";
import { addFileNodes } from "./file-drop";
import {
  OpenSuperCards,
  SuperCardCombineToolbar,
  SuperCardView,
  useSuperCardView,
} from "./super-card-node";
import {
  SUPER_PREFIX,
  multiSelectKeys,
  superCardDragMembers,
} from "./super-cards";
import { FlowEdgeView } from "./flow-edge";
import { FlowNodeView } from "./flow-node";
import { GroupFrame, NoteFrame } from "./frame-nodes";
import { NODE_DRAG_MIME, addNodeConnected, tryConnect } from "./node-factory";
import { SelectedNodeToolbar } from "./node-toolbar";

const NODE_TYPES = {
  flowSuperCard: SuperCardView,
  flowNode: FlowNodeView,
  flowGroup: GroupFrame,
  flowNote: NoteFrame,
};
const EDGE_TYPES = { flowEdge: FlowEdgeView };
const SNAP_GRID: [number, number] = [16, 16];
const VISIBLE_ONLY_THRESHOLD = 300;
const PRO_OPTIONS = { hideAttribution: true };
const DELETE_KEYS = ["Delete", "Backspace"];
const MULTI_SELECT_KEYS = multiSelectKeys(
  typeof navigator === "undefined" ? "" : navigator.platform,
);
const FIT_VIEW_OPTIONS = { padding: 0.2, maxZoom: 1.1 };

type Size = { width: number; height: number };
type Positions = Record<string, { x: number; y: number }>;

function portTypeOf(edge: FlowEdge): string {
  if (edge.kind === "flow") return "flow";
  const node = useDocStore.getState().graph.nodes[edge.source];
  const spec = node ? getSpec(node.type) : undefined;
  return findPort(spec, "out", edge.source_port)?.type ?? "any";
}

function minimapColor(node: RfNode): string {
  if (node.type !== "flowNode") return "var(--muted)";
  const doc = useDocStore.getState().graph.nodes[node.id];
  const spec = doc ? getSpec(doc.type) : undefined;
  return spec ? categoryColor(spec.category) : "var(--muted-foreground)";
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
  const colorMode = useTheme().resolved;
  const graph = useDocStore((s) => s.graph);
  const selectedNodes = useUiStore((s) => s.selectedNodes);
  const selectedEdges = useUiStore((s) => s.selectedEdges);
  const snapToGrid = useUiStore((s) => s.snapToGrid);
  const { screenToFlowPosition } = useReactFlow();

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

  const selectedSet = useMemo(
    () => new Set([...selectedNodes, ...selectedEdges]),
    [selectedNodes, selectedEdges],
  );
  const nodes = useMemo(
    () => (deriver.current as RfDeriver).nodes(graph, selectedSet, drag, sizes),
    [graph, selectedSet, drag, sizes],
  );
  const edges = useMemo(
    () => (deriver.current as RfDeriver).edges(graph, selectedSet, portTypeOf),
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

  const view = useSuperCardView(nodes, edges, selectedSet);

  const onNodeDragStart = useCallback(
    (_: ReactMouseEvent, node: RfNode) => {
      dragMembers.current = null;
      const doc = useDocStore.getState().graph;
      const folded = superCardDragMembers(node, doc);
      if (folded) {
        dragMembers.current = {
          groupRfId: node.id,
          origin: node.position,
          members: folded,
        };
        return;
      }
      if (!node.id.startsWith(GROUP_PREFIX)) return;
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
      dragMembers.current = {
        groupRfId: node.id,
        origin: { x: group.rect[0], y: group.rect[1] },
        members,
      };
    },
    [],
  );

  const onNodeDragStop = useCallback(() => {
    const card = dragMembers.current?.groupRfId.startsWith(SUPER_PREFIX);
    dragMembers.current = null;
    const current = dragRef.current;
    if (current) {
      const positions: Record<string, [number, number]> = {};
      for (const [id, p] of Object.entries(current))
        positions[id] = [Math.round(p.x), Math.round(p.y)];
      // A super card moves its blocks without changing which group they are in.
      useDocStore
        .getState()
        .moveNodes(
          positions,
          card ? undefined : groupMembership(positions, sizesRef.current),
        );
    }
    setDrag(null);
  }, []);

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
        !start?.nodeId ||
        !start.handleId ||
        state.isValid ||
        state.toHandle
      )
        return;
      const target = event.target as Element | null;
      if (!target?.classList.contains("react-flow__pane")) return;
      const point = "changedTouches" in event ? event.changedTouches[0] : event;
      const handle = parseHandle(start.handleId);
      if (!handle) return;
      const node = useDocStore.getState().graph.nodes[start.nodeId];
      const spec = node ? getSpec(node.type) : undefined;
      const port =
        handle.kind === "data"
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
      const ids = {
        nodes: [] as string[],
        edges: rfEdges.map((e) => e.id),
        groups: [] as string[],
        notes: [] as string[],
      };
      for (const node of rfNodes) {
        if (node.id.startsWith(GROUP_PREFIX))
          ids.groups.push(node.id.slice(GROUP_PREFIX.length));
        else if (node.id.startsWith(NOTE_PREFIX))
          ids.notes.push(node.id.slice(NOTE_PREFIX.length));
        else ids.nodes.push(node.id);
      }
      useDocStore.getState().removeElements(ids);
      useUiStore.getState().setSelection([], []);
    },
    [],
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
        nodes={view.nodes}
        edges={view.edges}
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
        onNodeDoubleClick={onNodeDoubleClick}
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
        colorMode={colorMode}
        proOptions={PRO_OPTIONS}
        fitView={true}
        fitViewOptions={FIT_VIEW_OPTIONS}
        className="flows-canvas bg-background"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.2}
          color="color-mix(in oklab, var(--foreground) 28%, transparent)"
        />
        <MiniMap
          pannable={true}
          zoomable={true}
          nodeColor={minimapColor}
          nodeStrokeWidth={0}
          maskColor="color-mix(in oklab, var(--background) 70%, transparent)"
          className="!rounded-lg !border !border-border !bg-card"
        />
        <Controls
          showInteractive={false}
          className="!rounded-lg !border !border-border !shadow-none"
        />
        <SuperCardCombineToolbar readOnly={readOnly} />
        <OpenSuperCards readOnly={readOnly} />
        <SelectedNodeToolbar readOnly={readOnly} />
      </ReactFlow>
    </div>
  );
}
