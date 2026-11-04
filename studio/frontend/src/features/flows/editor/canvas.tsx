// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import { useTheme } from "@/features/settings/stores/theme-store";
import {
  Background,
  BackgroundVariant,
  type Connection,
  ConnectionMode,
  type EdgeChange,
  type FinalConnectionState,
  type NodeChange,
  type OnConnectStartParams,
  type ReactFlowState,
  ReactFlow,
  SelectionMode,
  useReactFlow,
  useStoreApi,
  useUpdateNodeInternals,
} from "@xyflow/react";
import {
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { findPort, getSpec } from "../stores/catalog-store";
import { type DocGraph, useDocStore } from "../stores/doc-store";
import {
  type RerouteWire,
  rerouteDownstream,
  rerouteUpstream,
} from "../stores/reroute-graph";
import { useUiStore } from "../stores/ui-store";
import type { FlowEdge, FlowGroup } from "../types";
import { CanvasControls } from "./canvas-controls";
import { FIT_MIN_ZOOM, dragHasFiles } from "./canvas-rules";
import {
  checkConnection,
  parseHandle,
  wouldCreateCycle,
} from "./connection-rules";
import {
  GROUP_PREFIX,
  NOTE_PREFIX,
  type Boxes,
  RfDeriver,
  type RfEdge,
  type RfNode,
  type WireInfo,
} from "./derive-rf";
import { watchRewiredOutputs } from "./stale-outputs";
import { useWireReconnect } from "./use-wire-reconnect";
import { wheelScrollsInside } from "./wheel-scroll";
import { addFileNodes } from "./file-drop";
import { rememberSizes } from "./layout";
import {
  OpenSuperCards,
  SuperCardCombineToolbar,
  SuperCardView,
  useSuperCardView,
} from "./super-card-node";
import {
  isApplePlatform,
  multiSelectKeys,
  SUPER_PREFIX,
  foldedGroups,
  type SuperCardData,
  boxCovers,
  readSuperCards,
  superCardDragMembers,
  toggledSelection,
} from "./super-cards";
import { FlowEdgeView } from "./flow-edge";
import { snapToWire, wireUnder } from "./wire-snap";
import { FlowNodeView } from "./flow-node";
import { GroupFrame, NoteFrame } from "./frame-nodes";
import { NODE_DRAG_MIME, addNodeConnected, tryConnect } from "./node-factory";
import {
  recipeConnectionError,
  removeElementsWithReferences,
} from "../recipe/recipe-actions";
import { COLUMN_TYPE, isRecipeGroup } from "../recipe/recipe-columns";
import { holdRecipeBlocks, ownRecipe } from "./recipe-drag";
import {
  parseSetupId,
  setupEdges,
  setupLayout,
  withSetupPosition,
} from "../recipe/recipe-setup";
import {
  RecipeSetupNode,
  SETUP_IN,
  SETUP_OUT,
  type SetupData,
} from "../recipe/recipe-setup-node";
import { GRID, placementPosition } from "./placement";
import { SelectedNodeToolbar } from "./node-toolbar";
import { RerouteNode } from "./reroute-node";
import {
  REROUTE_OUT_HANDLE,
  dirAlong,
  flippedStyle,
  rerouteSize,
  addReroute,
  checkRerouteConnection,
  reroutePort,
  touchesReroute,
} from "./reroutes";
import { WireRouterController, setWireHover } from "./wire-routing";

const NODE_TYPES = {
  flowSuperCard: SuperCardView,
  flowNode: FlowNodeView,
  flowGroup: GroupFrame,
  flowNote: NoteFrame,
  recipeSetup: RecipeSetupNode,
  flowReroute: RerouteNode,
};
const EDGE_TYPES = { flowEdge: FlowEdgeView };
const SNAP_GRID: [number, number] = [GRID, GRID];
// RF draws a dot at its tile corner plus the radius, less `offset`: offset = radius puts dots on multiples of GRID.
const DOT_SIZE = 1.2;
const VISIBLE_ONLY_THRESHOLD = 300;
const PRO_OPTIONS = { hideAttribution: true };
const DELETE_KEYS = ["Delete", "Backspace"];
const PLATFORM = typeof navigator === "undefined" ? "" : navigator.platform;
const MULTI_SELECT_KEYS = multiSelectKeys(PLATFORM);
const APPLE = isApplePlatform(PLATFORM);
const FIT_VIEW_OPTIONS = { padding: 0.2, maxZoom: 1.1, minZoom: FIT_MIN_ZOOM };

// React Flow reports box-selection changes before it stores the new rect, so read the pointer ourselves.
const pointer = { x: 0, y: 0 };
const pressed = { x: 0, y: 0 };

/** Whether the box from `from` to the pointer (client coords) encloses the whole frame `id`. */
function boxHoldsGroup(
  state: ReactFlowState<RfNode, RfEdge>,
  id: string,
  from: { x: number; y: number },
): boolean {
  const node = state.nodeLookup.get(id);
  const pane = state.domNode?.getBoundingClientRect();
  if (!node || !pane) return false;
  return boxCovers(
    {
      x: Math.min(pointer.x, from.x) - pane.left,
      y: Math.min(pointer.y, from.y) - pane.top,
      width: Math.abs(pointer.x - from.x),
      height: Math.abs(pointer.y - from.y),
    },
    state.transform,
    {
      ...node.internals.positionAbsolute,
      width: node.measured.width ?? 0,
      height: node.measured.height ?? 0,
    },
  );
}

/** A selection box drawn inside a group touches it; only a box around the whole group takes it. */
function boxSkips(state: ReactFlowState<RfNode, RfEdge>, id: string): boolean {
  if (!id.startsWith(GROUP_PREFIX) || !state.userSelectionRect) return false;
  return !boxHoldsGroup(state, id, pressed);
}

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

function hasWires(graph: DocGraph, rerouteId: string): boolean {
  return graph.wireOrder.some(
    (id) =>
      graph.wires[id].source === rerouteId ||
      graph.wires[id].target === rerouteId,
  );
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

export function groupMembership(
  positions: Record<string, [number, number]>,
  sizes: Map<string, Size>,
  // A recipe frame drawn taller than it is saved (its setup cards) holds what lands in the extra space.
  heights: Record<string, number> = {},
): Record<string, string | null> {
  const { graph, meta } = useDocStore.getState();
  // A frame folded into a closed card is hidden, so a block dropped where it was stays out of it.
  const folded = foldedGroups(meta.editor, graph);
  const membership: Record<string, string | null> = {};
  for (const [id, pos] of Object.entries(positions)) {
    if (!graph.nodes[id]) continue;
    const width = sizes.get(id)?.width ?? 260;
    let best: { id: string; area: number } | null = null;
    for (const groupId of graph.groupOrder) {
      if (folded.has(groupId) && graph.nodes[id].group !== groupId) continue;
      const key = GROUP_PREFIX + groupId;
      const g = graph.groups[groupId];
      const h = Math.max(g.rect[3], heights[groupId] ?? 0);
      const rect: [number, number, number, number] = positions[key]
        ? [positions[key][0], positions[key][1], g.rect[2], h]
        : [g.rect[0], g.rect[1], g.rect[2], h];
      if (!insideRect(pos[0] + width / 2, pos[1] + 16, rect)) continue;
      const area = rect[2] * rect[3];
      if (!best || area < best.area) best = { id: groupId, area };
    }
    membership[id] = best?.id ?? null;
    // A recipe's blocks only run inside their recipe: dragged out, they stay its own (the frame grows).
    const recipe = ownRecipe(graph, id);
    if (recipe) membership[id] = recipe;
  }
  return membership;
}


export function FlowCanvas({ readOnly = false }: { readOnly?: boolean }) {
  const colorMode = useTheme().resolved;
  const graph = useDocStore((s) => s.graph);
  const selectedNodes = useUiStore((s) => s.selectedNodes);
  const selectedEdges = useUiStore((s) => s.selectedEdges);
  const snapToGrid = useUiStore((s) => s.snapToGrid);
  const recipeView = useDocStore((s) => {
    const first = s.graph.groupOrder.find((id) =>
      isRecipeGroup(s.graph.groups[id]),
    );
    return first ? `${s.flowId}:${first}` : null;
  });
  const bus = useUiStore((s) => s.wireStyle === "bus");
  const pinnedBus = useDocStore((s) =>
    s.graph.wireOrder.some((id) => s.graph.wires[id].style === "bus"),
  );
  const flowId = useDocStore((s) => s.flowId);
  const { screenToFlowPosition, flowToScreenPosition, fitView } = useReactFlow();
  const store = useStoreApi<RfNode, RfEdge>();
  const { props: wireReconnect, reconnecting } =
    useWireReconnect(!readOnly);
  const updateNodeInternals = useRef(useUpdateNodeInternals());
  useEffect(() => (readOnly ? undefined : watchRewiredOutputs()), [readOnly]);

  const deriver = useRef<RfDeriver | null>(null);
  deriver.current ??= new RfDeriver();
  const [drag, setDrag] = useState<Positions | null>(null);
  // Every dragged frame or closed super card, keyed by its React Flow id, with its contents' start.
  const dragMembers = useRef<
    Map<
      string,
      {
        card: boolean;
        /** Blocks of a closed card carried along: they keep their groups, like a card drag. */
        keep?: string[];
        origin: { x: number; y: number };
        members: Map<string, { x: number; y: number }>;
      }
    >
  >(new Map());
  const [sizes, setSizes] = useState<Map<string, Size>>(() => new Map());
  const connectStart = useRef<OnConnectStartParams | null>(null);
  const sizesRef = useRef(sizes);
  sizesRef.current = sizes;
  useEffect(() => {
    rememberSizes(sizes);
    return () => rememberSizes(new Map());
  }, [sizes]);
  const dragRef = useRef<Positions | null>(null);
  dragRef.current = drag;
  // Sizes under a resize handle, drawn live and committed by the card on release.
  const [boxes, setBoxes] = useState<Boxes | null>(null);
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;
  const moving = useMemo(
    () =>
      drag ? new Set([...Object.keys(drag), ...Object.keys(boxes ?? {})]) : null,
    [drag, boxes],
  );

  const selectedSet = useMemo(
    () => new Set([...selectedNodes, ...selectedEdges]),
    [selectedNodes, selectedEdges],
  );
  const derivedNodes = useMemo(
    () =>
      (deriver.current as RfDeriver).nodes(
        graph,
        selectedSet,
        drag,
        sizes,
        boxes,
      ),
    [graph, selectedSet, drag, sizes, boxes],
  );
  const derivedEdges = useMemo(
    () =>
      (deriver.current as RfDeriver).edges(
        graph,
        selectedSet,
        portTypeOf,
        wireInfo,
      ),
    [graph, selectedSet],
  );
  const setupSelected = useUiStore((s) => s.recipeSetup);
  const setup = useMemo(
    () => setupLayout(graph, drag, sizes),
    [graph, drag, sizes],
  );
  const setupRef = useRef(setup);
  setupRef.current = setup;
  const nodes = useMemo(() => {
    if (!setup.elements.length) return derivedNodes;
    const grown = derivedNodes.map((node) => {
      const height = node.id.startsWith(GROUP_PREFIX)
        ? setup.heights[node.id.slice(GROUP_PREFIX.length)]
        : undefined;
      return height ? { ...node, style: { ...node.style, height } } : node;
    });
    const cards: RfNode[] = setup.elements.map(({ card, x, y }) => {
      const size = sizes.get(card.id);
      const seed = graph.groups[card.groupId]?.config?.seed_config as
        | { source?: { seed_type?: string } }
        | undefined;
      const data: SetupData = { card, seedType: seed?.source?.seed_type };
      return {
        id: card.id,
        type: "recipeSetup",
        position: { x, y },
        data: data as unknown as RfNode["data"],
        selected:
          setupSelected?.groupId === card.groupId &&
          setupSelected.kind === card.kind &&
          setupSelected.name === card.name,
        selectable: false,
        deletable: false,
        connectable: false,
        ...(size ? { measured: size } : {}),
      };
    });
    return [...grown, ...cards];
  }, [derivedNodes, setup, sizes, graph, setupSelected]);
  const edges = useMemo(() => {
    if (!setup.links.length) return derivedEdges;
    const links = setupEdges(setup.links, graph, {
      source: SETUP_OUT,
      target: SETUP_IN,
    });
    return [...derivedEdges, ...links];
  }, [derivedEdges, setup, graph]);

  // A recipe opens at a size you can read, not fitted to the whole flow.
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per opened recipe flow
  useEffect(() => {
    if (!recipeView) return;
    const doc = useDocStore.getState().graph;
    const groupId = doc.groupOrder.find((id) => isRecipeGroup(doc.groups[id]));
    if (!groupId) return;
    const ids = doc.nodeOrder.filter((id) => doc.nodes[id].group === groupId);
    const timer = window.setTimeout(
      () =>
        void fitView({
          nodes: [{ id: GROUP_PREFIX + groupId }, ...ids.map((id) => ({ id }))],
          padding: 0.08,
          minZoom: 0.55,
          maxZoom: 1,
        }),
      60,
    );
    return () => window.clearTimeout(timer);
  }, [recipeView]);

  const onNodesChange = useCallback(
    (changes: NodeChange<RfNode>[]) => {
      const drag = dragRef.current;
      const sizes = sizesRef.current;
      const boxes = boxesRef.current;
      let nextDrag: Positions | null = null;
      let nextSizes: Map<string, Size> | null = null;
      let nextBoxes: Boxes | null = null;
      let ended = false;
      const resizing = new Set(
        changes.flatMap((c) =>
          c.type === "dimensions" && c.resizing ? [c.id] : [],
        ),
      );
      let selection: { add: Set<string>; remove: Set<string> } | null = null;
      for (const change of changes) {
        if (change.type === "position" && change.position) {
          if (
            !change.dragging &&
            (resizing.has(change.id) || boxes?.[change.id])
          ) {
            // A resize from a top or left handle moves the card too.
            nextDrag ??= { ...(drag ?? {}) };
            nextDrag[change.id] = change.position;
          } else if (change.dragging) {
            nextDrag ??= { ...(drag ?? {}) };
            nextDrag[change.id] = change.position;
            const group = dragMembers.current.get(change.id);
            if (group) {
              const dx = change.position.x - group.origin.x;
              const dy = change.position.y - group.origin.y;
              for (const [memberId, origin] of group.members) {
                nextDrag[memberId] = { x: origin.x + dx, y: origin.y + dy };
              }
            }
          }
        } else if (change.type === "dimensions" && change.resizing) {
          nextBoxes ??= { ...(boxes ?? {}) };
          nextBoxes[change.id] = change.dimensions ?? nextBoxes[change.id];
          nextDrag ??= { ...(drag ?? {}) };
        } else if (
          change.type === "dimensions" &&
          change.resizing === false &&
          boxes?.[change.id]
        ) {
          nextBoxes ??= { ...boxes };
          delete nextBoxes[change.id];
          nextDrag ??= { ...(drag ?? {}) };
          delete nextDrag[change.id];
          ended = true;
        }
        // Every new size reaches `measured`, a resize's too: the DOM already has the final size
        // when the handle is released, so no later measurement would report it.
        if (change.type === "dimensions" && change.dimensions) {
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
          if (change.selected && boxSkips(store.getState(), change.id)) {
            // React Flow marks the frame selected in its own store before telling us; undo that
            // too, or dragging the boxed blocks would carry the frame and its other blocks.
            const internal = store.getState().nodeLookup.get(change.id);
            if (internal) internal.selected = false;
            continue;
          }
          selection ??= { add: new Set(), remove: new Set() };
          (change.selected ? selection.add : selection.remove).add(change.id);
        }
      }
      if (nextBoxes) {
        setBoxes(Object.keys(nextBoxes).length ? nextBoxes : null);
        // Handles move with a resize; React Flow re-reads their positions only when asked,
        // since the size it holds is already the new one.
        const resized = changes.flatMap((c) =>
          c.type === "dimensions" && c.resizing !== undefined ? [c.id] : [],
        );
        if (resized.length)
          requestAnimationFrame(() => updateNodeInternals.current(resized));
      }
      if (nextDrag)
        setDrag(ended && !Object.keys(nextDrag).length ? null : nextDrag);
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

  const view = useSuperCardView(nodes, edges, selectedSet, sizes);

  const onNodeDragStart = useCallback(
    (_: ReactMouseEvent, node: RfNode, dragged: RfNode[]) => {
      const doc = useDocStore.getState().graph;
      const currentSizes = sizesRef.current;
      dragMembers.current = new Map();
      for (const n of dragged.length ? dragged : [node]) {
        const folded = superCardDragMembers(n, doc);
        if (folded) {
          dragMembers.current.set(n.id, {
            card: true,
            origin: n.position,
            members: folded,
          });
          continue;
        }
        if (!n.id.startsWith(GROUP_PREFIX)) continue;
        const group = doc.groups[n.id.slice(GROUP_PREFIX.length)];
        if (!group) continue;
        const members = new Map<string, { x: number; y: number }>();
        for (const id of doc.nodeOrder) {
          const m = doc.nodes[id];
          const size = currentSizes.get(id) ?? { width: 260, height: 120 };
          if (insideRect(m.pos[0] + size.width / 2, m.pos[1] + 20, group.rect)) {
            members.set(id, { x: m.pos[0], y: m.pos[1] });
          }
        }
        for (const id of doc.rerouteOrder) {
          const [x, y] = doc.reroutes[id].pos;
          if (insideRect(x, y, group.rect)) members.set(id, { x, y });
        }
        // A closed card is one object: if the frame carries any of its blocks, it carries all of it.
        const keep: string[] = [];
        const { meta } = useDocStore.getState();
        for (const card of readSuperCards(meta.editor, doc)) {
          if (card.open || !card.members.some((id) => members.has(id))) continue;
          const rf = store.getState().nodeLookup.get(SUPER_PREFIX + card.id);
          const whole = rf ? superCardDragMembers(rf as unknown as RfNode, doc) : null;
          for (const [id, at] of whole ?? []) members.set(id, at);
          keep.push(...card.members);
        }
        dragMembers.current.set(n.id, {
          card: false,
          keep,
          origin: { x: group.rect[0], y: group.rect[1] },
          members,
        });
      }
    },
    [],
  );
  // React Flow only reports a box's changes when its picked set changes, so a frame dropped while the
  // box only touched it is never offered again; settle every frame against the final box.
  const onSelectionEnd = useCallback(() => {
    const state = store.getState();
    const ui = useUiStore.getState();
    const next = new Set(ui.selectedNodes);
    for (const id of state.nodeLookup.keys()) {
      if (!id.startsWith(GROUP_PREFIX)) continue;
      if (boxHoldsGroup(state, id, pressed)) next.add(id);
      else next.delete(id);
    }
    ui.setSelection([...next], ui.selectedEdges);
  }, [store]);


  const onNodeDragStop = useCallback(
    (_: unknown, node: RfNode) => {
      // A super card moves its blocks without changing which group they are in.
      const kept = new Set<string>();
      for (const group of dragMembers.current.values()) {
        if (group.card) for (const id of group.members.keys()) kept.add(id);
        for (const id of group.keep ?? []) kept.add(id);
      }
      dragMembers.current = new Map();
      const current = dragRef.current;
      if (current) {
        const positions: Record<string, [number, number]> = {};
        const doc = useDocStore.getState();
        const groups = new Map<string, FlowGroup>();
        for (const [id, p] of Object.entries(current)) {
          const card = parseSetupId(id);
          if (!card) {
            positions[id] = [Math.round(p.x), Math.round(p.y)];
            continue;
          }
          // Only a card dragged on its own keeps a spot; one carried with its frame follows the frame.
          if (current[GROUP_PREFIX + card.groupId]) continue;
          const group = groups.get(card.groupId) ?? doc.graph.groups[card.groupId];
          if (group) groups.set(card.groupId, withSetupPosition(group, id, [p.x, p.y]));
        }
        if (Object.keys(positions).length) {
          const membership = groupMembership(
            positions,
            sizesRef.current,
            setupRef.current.heights,
          );
          for (const id of kept) delete membership[id];
          doc.moveNodes(positions, membership);
          const held = new Map<string, Record<string, [number, number]>>();
          for (const [id, pos] of Object.entries(positions)) {
            const group = ownRecipe(doc.graph, id);
            if (group && membership[id] === group)
              held.set(group, { ...held.get(group), [id]: pos });
          }
          for (const [groupId, moved] of held) {
            // A frame dragged along with its blocks is placed by its own drag.
            if (current[GROUP_PREFIX + groupId]) continue;
            const group = groups.get(groupId) ?? doc.graph.groups[groupId];
            const grown = holdRecipeBlocks(group, moved, sizesRef.current);
            if (grown !== group) groups.set(groupId, grown);
          }
        }
        if (groups.size) doc.patchGraph({ groups: [...groups.values()] });
      }
      setDrag(null);
      const doc = useDocStore.getState().graph;
      const reroute = doc.reroutes[node.id];
      if (!reroute || hasWires(doc, node.id)) return;
      // A free reroute dropped on a wire threads itself into it.
      const size = rerouteSize(reroute.dir);
      const centre = flowToScreenPosition({
        x: reroute.pos[0] + size.width / 2,
        y: reroute.pos[1] + size.height / 2,
      });
      const wire = wireUnder(centre.x, centre.y, node.id);
      if (!(wire && (doc.wires[wire] || doc.edges[wire]?.kind === "data")))
        return;
      // Turned to run along the wire where it lands, and centred on it.
      const at = snapToWire(wire, {
        x: reroute.pos[0] + size.width / 2,
        y: reroute.pos[1] + size.height / 2,
      });
      const dir = at ? dirAlong(at.tangent) : (reroute.dir ?? "right");
      const fit = rerouteSize(dir);
      const { dir: _dir, ...rest } = reroute;
      useDocStore.getState().addReroute(
        {
          ...rest,
          ...(dir === "right" ? {} : { dir }),
          ...(at
            ? {
                pos: [
                  Math.round(at.x - fit.width / 2),
                  Math.round(at.y - fit.height / 2),
                ] as [number, number],
              }
            : {}),
        },
        wire,
      );
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
    if (recipeConnectionError(doc, connection.source, connection.target))
      return false;
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
      const { reroutes, groups, nodes, nodeOrder } = useDocStore.getState().graph;
      // A closed super card goes with its blocks and everything folded into it.
      const removed = rfNodes.flatMap((node) => {
        if (!node.id.startsWith(SUPER_PREFIX)) return [node.id];
        const data = node.data as unknown as SuperCardData;
        return [...data.memberIds, ...data.folded.map((f) => f.id)];
      });
      for (const id of removed) {
        if (reroutes[id]) ids.reroutes.push(id);
        else if (id.startsWith(GROUP_PREFIX)) {
          const groupId = id.slice(GROUP_PREFIX.length);
          ids.groups.push(groupId);
          // A recipe's columns mean nothing outside it: they go with the frame.
          if (groups[groupId]?.kind === "recipe")
            for (const member of nodeOrder)
              if (nodes[member].group === groupId) ids.nodes.push(member);
        } else if (id.startsWith(NOTE_PREFIX))
          ids.notes.push(id.slice(NOTE_PREFIX.length));
        else ids.nodes.push(id);
      }
      removeElementsWithReferences(ids);
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
      const node = addNodeConnected(
        type,
        placementPosition(flow, useUiStore.getState().snapToGrid),
      );
      if (node) useUiStore.getState().setSelection([node.id], []);
    },
    [readOnly, screenToFlowPosition],
  );

  const onNodeDoubleClick = useCallback((_: ReactMouseEvent, node: RfNode) => {
    if (node.type === "flowNode") useUiStore.getState().openInspector(node.id);
  }, []);

  const onNodeClick = useCallback((_: ReactMouseEvent, node: RfNode) => {
    const card = parseSetupId(node.id);
    if (!card) return;
    const ui = useUiStore.getState();
    ui.setSelection([], []);
    ui.setRecipeSetup(card);
  }, []);
  const onEdgeDoubleClick = useCallback(
    (event: ReactMouseEvent, edge: RfEdge) => {
      if (readOnly || edge.data?.kind !== "data") return;
      const ends = useDocStore.getState().graph.nodes;
      if (
        ends[edge.source]?.type === COLUMN_TYPE ||
        ends[edge.target]?.type === COLUMN_TYPE
      ) {
        toast.error("Recipe columns connect directly, not through a reroute");
        return;
      }
      const flow = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const at = snapToWire(edge.id, flow);
      const id = addReroute(at ?? flow, {
        splitId: edge.id,
        dir: at ? dirAlong(at.tangent) : undefined,
        style: event.shiftKey
          ? flippedStyle(
              useDocStore.getState().graph,
              edge.id,
              useUiStore.getState().wireStyle,
            )
          : undefined,
      });
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
      onPointerDownCapture={(event) => {
        pressed.x = pointer.x = event.clientX;
        pressed.y = pointer.y = event.clientY;
      }}
      onPointerMoveCapture={(event) => {
        pointer.x = event.clientX;
        pointer.y = event.clientY;
      }}
      onDragEnter={fileDrop.dragHandlers.onDragEnter}
      onDragOver={(event) => {
        if (dragHasFiles(Array.from(event.dataTransfer.types)))
          dropPoint.current = { x: event.clientX, y: event.clientY };
        fileDrop.dragHandlers.onDragOver(event);
      }}
      onDragLeave={fileDrop.dragHandlers.onDragLeave}
      onDrop={fileDrop.dragHandlers.onDrop}
      // Scroll a box under the pointer only while it can still move; otherwise the wheel zooms.
      onWheelCapture={(event) => {
        if (
          wrapper.current &&
          wheelScrollsInside(
            event.target as HTMLElement,
            wrapper.current,
            event.deltaX,
            event.deltaY,
            event.ctrlKey,
          )
        )
          event.stopPropagation();
      }}
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
        {...wireReconnect}
        onNodeDoubleClick={onNodeDoubleClick}
        onNodeClick={onNodeClick}
        onEdgeDoubleClick={onEdgeDoubleClick}
        onNodeMouseEnter={bus ? onNodeMouseEnter : undefined}
        onNodeMouseLeave={bus ? onNodeMouseLeave : undefined}
        onDragOver={onDragOver}
        onDrop={onDrop}
        connectionMode={ConnectionMode.Strict}
        deleteKeyCode={readOnly ? null : DELETE_KEYS}
        multiSelectionKeyCode={MULTI_SELECT_KEYS}
        selectionKeyCode={MULTI_SELECT_KEYS}
        onSelectionEnd={onSelectionEnd}
        onNodeContextMenu={(event, node) => {
          if (!APPLE || !event.ctrlKey) return;
          event.preventDefault();
          const ui = useUiStore.getState();
          ui.setSelection(toggledSelection(ui.selectedNodes, node.id), ui.selectedEdges);
        }}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag={false}
        panOnDrag={true}
        panActivationKeyCode="Space"
        zoomOnDoubleClick={false}
        // `nowheel` boxes would block zoom even with nothing to scroll; onWheelCapture decides.
        noWheelClassName="flows-wheel-unused"
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
          gap={GRID}
          size={DOT_SIZE}
          offset={DOT_SIZE / 2}
          color="color-mix(in oklab, var(--foreground) 14%, transparent)"
        />
        <SuperCardCombineToolbar readOnly={readOnly} />
        <OpenSuperCards readOnly={readOnly} />
        <CanvasControls readOnly={readOnly} />
        <SelectedNodeToolbar readOnly={readOnly} />
        {(bus || pinnedBus) && (
          <WireRouterController scope={flowId ?? ""} moving={moving} />
        )}
      </ReactFlow>
    </div>
  );
}
