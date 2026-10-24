// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  ReactFlowInstance,
  XYPosition,
} from "@xyflow/react";
import {
  type DragEvent as ReactDragEvent,
  type RefObject,
  useCallback,
  useMemo,
  useState,
} from "react";
import { RECIPE_BLOCK_DND_MIME, type RecipeBlockDragPayload } from "../components/block-sheet";
import type { SeedBlockType, ValidatorBlockType } from "../blocks/registry";
import type {
  LlmType,
  NodeConfig,
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
  SamplerType,
} from "../types";
import { logicalEdgeId, REROUTE_HEIGHT, REROUTE_WIDTH } from "../utils/graph/reroutes";
import { applyAuxNodeChanges, filterEdgeChangesByIds, filterNodeChangesByIds } from "../utils/reactflow-changes";
import type { RecipeGraphAuxNodeData } from "../components/recipe-graph-aux-node";
import { findWireAt } from "../components/graph/wire-routing";

const SUPPORTED_DRAG_KINDS: RecipeBlockDragPayload["kind"][] = [
  "reroute",
  "sampler",
  "seed",
  "llm",
  "validator",
  "expression",
  "note",
];

function parseRecipeBlockDragPayload(raw: string): RecipeBlockDragPayload | null {
  try {
    const parsed = JSON.parse(raw) as {
      kind?: RecipeBlockDragPayload["kind"];
      type?: RecipeBlockDragPayload["type"];
    };
    if (!parsed.kind || !parsed.type || !SUPPORTED_DRAG_KINDS.includes(parsed.kind)) {
      return null;
    }
    return { kind: parsed.kind, type: parsed.type } as RecipeBlockDragPayload;
  } catch {
    return null;
  }
}

type UseRecipeEditorGraphArgs = {
  nodes: RecipeBuilderNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  reactFlowInstance: ReactFlowInstance<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge> | null;
  flowContainerRef: RefObject<HTMLDivElement | null>;
  selectConfig: (id: string) => void;
  openConfig: (id: string) => void;
  onNodesChange: (changes: NodeChange<RecipeBuilderNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  setAuxNodePosition: (id: string, position: XYPosition) => void;
  reroutes: Record<string, XYPosition>;
  addReroute: (position: XYPosition, wireId?: string | null) => void;
  setReroutePosition: (id: string, position: XYPosition) => void;
  dropRerouteOnWire: (id: string, wireId: string) => void;
  removeReroutes: (ids: string[]) => void;
  addSamplerNode: (type: SamplerType, position?: XYPosition, openDialog?: boolean) => void;
  addSeedNode: (type: SeedBlockType, position?: XYPosition, openDialog?: boolean) => void;
  addLlmNode: (type: LlmType, position?: XYPosition, openDialog?: boolean) => void;
  addModelProviderNode: (position?: XYPosition, openDialog?: boolean) => void;
  addModelConfigNode: (position?: XYPosition, openDialog?: boolean) => void;
  addToolProfileNode: (position?: XYPosition, openDialog?: boolean) => void;
  addExpressionNode: (position?: XYPosition, openDialog?: boolean) => void;
  addValidatorNode: (
    type: ValidatorBlockType,
    position?: XYPosition,
    openDialog?: boolean,
  ) => void;
  addMarkdownNoteNode: (position?: XYPosition, openDialog?: boolean) => void;
};

type UseRecipeEditorGraphResult = {
  handleNodeClick: (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => void;
  handleNodeDoubleClick: (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => void;
  handleNodesChange: (
    changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[],
  ) => void;
  handleEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  handleDragOver: (event: ReactDragEvent<HTMLDivElement>) => void;
  handleDrop: (event: ReactDragEvent<HTMLDivElement>) => void;
  handleAddSamplerFromSheet: (type: SamplerType) => void;
  handleAddSeedFromSheet: (type: SeedBlockType) => void;
  handleAddLlmFromSheet: (type: LlmType) => void;
  handleAddModelProviderFromSheet: () => void;
  handleAddModelConfigFromSheet: () => void;
  handleAddToolProfileFromSheet: () => void;
  handleAddExpressionFromSheet: () => void;
  handleAddValidatorFromSheet: (
    type: ValidatorBlockType,
  ) => void;
  handleAddMarkdownNoteFromSheet: () => void;
  handleAddRerouteFromSheet: () => void;
  handleNodeDragStop: (_: unknown, node: Node) => void;
  selectedReroutes: Set<string>;
};

export function useRecipeEditorGraph({
  nodes,
  edges,
  configs,
  reactFlowInstance,
  flowContainerRef,
  selectConfig,
  openConfig,
  onNodesChange,
  onEdgesChange,
  setAuxNodePosition,
  reroutes,
  addReroute,
  setReroutePosition,
  dropRerouteOnWire,
  removeReroutes,
  addSamplerNode,
  addSeedNode,
  addLlmNode,
  addModelProviderNode,
  addModelConfigNode,
  addToolProfileNode,
  addExpressionNode,
  addValidatorNode,
  addMarkdownNoteNode,
}: UseRecipeEditorGraphArgs): UseRecipeEditorGraphResult {
  const baseNodeIds = useMemo(() => new Set(nodes.map((node) => node.id)), [nodes]);
  const baseEdgeIds = useMemo(() => new Set(edges.map((edge) => edge.id)), [edges]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      selectConfig(node.id);
    },
    [selectConfig],
  );

  const handleNodeDoubleClick = useCallback(
    (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      const nodeConfig = configs[node.id];
      if (nodeConfig?.kind === "markdown_note") {
        openConfig(node.id);
      }
    },
    [configs, openConfig],
  );

  const [selectedReroutes, setSelectedReroutes] = useState<Set<string>>(() => new Set());

  const applyRerouteChanges = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      const removed: string[] = [];
      const selection = new Map<string, boolean>();
      for (const change of changes) {
        if (!("id" in change && change.id in reroutes)) {
          continue;
        }
        if (change.type === "position" && change.position) {
          setReroutePosition(change.id, change.position);
        } else if (change.type === "remove") {
          removed.push(change.id);
        } else if (change.type === "select") {
          selection.set(change.id, change.selected);
        }
      }
      if (removed.length > 0) {
        removeReroutes(removed);
      }
      if (selection.size > 0) {
        setSelectedReroutes((current) => {
          const next = new Set(current);
          for (const [id, selected] of selection) {
            if (selected) {
              next.add(id);
            } else {
              next.delete(id);
            }
          }
          return next;
        });
      }
    },
    [removeReroutes, reroutes, setReroutePosition],
  );

  // A free reroute dropped on a wire threads into it, lined up with the wire.
  const handleNodeDragStop = useCallback(
    (_: unknown, node: Node) => {
      if (node.type !== "reroute") {
        return;
      }
      const center = {
        x: node.position.x + REROUTE_WIDTH / 2,
        y: node.position.y + REROUTE_HEIGHT / 2,
      };
      const wire = findWireAt(center, node.id);
      if (!wire) {
        return;
      }
      dropRerouteOnWire(node.id, wire.id);
      if (wire.y !== null) {
        setReroutePosition(node.id, { x: node.position.x, y: wire.y - REROUTE_HEIGHT / 2 });
      }
    },
    [dropRerouteOnWire, setReroutePosition],
  );

  const placeReroute = useCallback(
    (point: XYPosition) => {
      const wire = findWireAt(point, null);
      addReroute(
        {
          x: point.x - REROUTE_WIDTH / 2,
          y: (wire?.y ?? point.y) - REROUTE_HEIGHT / 2,
        },
        wire?.id ?? null,
      );
    },
    [addReroute],
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      applyRerouteChanges(changes);
      applyAuxNodeChanges(changes, { setAuxNodePosition });
      const next = filterNodeChangesByIds(
        changes as NodeChange<RecipeBuilderNode>[],
        baseNodeIds,
      );
      if (next.length) {
        onNodesChange(next);
      }
    },
    [applyRerouteChanges, baseNodeIds, onNodesChange, setAuxNodePosition],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      // Wires through reroutes are drawn as segments named after their edge.
      const next = filterEdgeChangesByIds(
        changes.map((change) =>
          "id" in change && !baseEdgeIds.has(change.id)
            ? { ...change, id: logicalEdgeId(change.id) }
            : change,
        ) as EdgeChange<Edge>[],
        baseEdgeIds,
      );
      if (next.length) {
        onEdgesChange(next);
      }
    },
    [baseEdgeIds, onEdgesChange],
  );

  const handleDragOver = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
    if (
      !event.dataTransfer.types.includes(RECIPE_BLOCK_DND_MIME) &&
      !event.dataTransfer.types.includes("text/plain")
    ) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const handleDrop = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!reactFlowInstance) {
        return;
      }
      const raw =
        event.dataTransfer.getData(RECIPE_BLOCK_DND_MIME) ||
        event.dataTransfer.getData("text/plain");
      if (!raw) {
        return;
      }
      const payload = parseRecipeBlockDragPayload(raw);
      if (!payload) {
        return;
      }
      event.preventDefault();
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      if (payload.kind === "reroute") {
        placeReroute(position);
        return;
      }
      if (payload.kind === "sampler") {
        addSamplerNode(payload.type as SamplerType, position, false);
        return;
      }
      if (payload.kind === "seed") {
        addSeedNode(payload.type as SeedBlockType, position, false);
        return;
      }
      if (payload.kind === "expression") {
        addExpressionNode(position, false);
        return;
      }
      if (payload.kind === "validator") {
        addValidatorNode(
          payload.type as ValidatorBlockType,
          position,
          false,
        );
        return;
      }
      if (payload.kind === "note") {
        addMarkdownNoteNode(position, false);
        return;
      }
      if (payload.type === "model_provider") {
        addModelProviderNode(position, false);
        return;
      }
      if (payload.type === "model_config") {
        addModelConfigNode(position, false);
        return;
      }
      if (payload.type === "tool_config") {
        addToolProfileNode(position, false);
        return;
      }
      addLlmNode(payload.type as LlmType, position, false);
    },
    [
      addExpressionNode,
      addLlmNode,
      addMarkdownNoteNode,
      addModelConfigNode,
      addModelProviderNode,
      addToolProfileNode,
      addSamplerNode,
      addSeedNode,
      addValidatorNode,
      placeReroute,
      reactFlowInstance,
    ],
  );

  const getViewportCenterPosition = useCallback(() => {
    if (!reactFlowInstance || !flowContainerRef.current) {
      return undefined;
    }
    const rect = flowContainerRef.current.getBoundingClientRect();
    return reactFlowInstance.screenToFlowPosition({
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    });
  }, [flowContainerRef, reactFlowInstance]);

  const handleAddSamplerFromSheet = useCallback(
    (type: SamplerType) => {
      addSamplerNode(type, getViewportCenterPosition());
    },
    [addSamplerNode, getViewportCenterPosition],
  );

  const handleAddSeedFromSheet = useCallback(
    (type: SeedBlockType) => {
      addSeedNode(type, getViewportCenterPosition());
    },
    [addSeedNode, getViewportCenterPosition],
  );

  const handleAddLlmFromSheet = useCallback(
    (type: LlmType) => {
      addLlmNode(type, getViewportCenterPosition());
    },
    [addLlmNode, getViewportCenterPosition],
  );

  const handleAddModelProviderFromSheet = useCallback(() => {
    addModelProviderNode(getViewportCenterPosition());
  }, [addModelProviderNode, getViewportCenterPosition]);

  const handleAddModelConfigFromSheet = useCallback(() => {
    addModelConfigNode(getViewportCenterPosition());
  }, [addModelConfigNode, getViewportCenterPosition]);

  const handleAddExpressionFromSheet = useCallback(() => {
    addExpressionNode(getViewportCenterPosition());
  }, [addExpressionNode, getViewportCenterPosition]);

  const handleAddToolProfileFromSheet = useCallback(() => {
    addToolProfileNode(getViewportCenterPosition());
  }, [addToolProfileNode, getViewportCenterPosition]);

  const handleAddValidatorFromSheet = useCallback(
    (type: ValidatorBlockType) => {
      addValidatorNode(type, getViewportCenterPosition());
    },
    [addValidatorNode, getViewportCenterPosition],
  );

  const handleAddMarkdownNoteFromSheet = useCallback(() => {
    addMarkdownNoteNode(getViewportCenterPosition());
  }, [addMarkdownNoteNode, getViewportCenterPosition]);

  const handleAddRerouteFromSheet = useCallback(() => {
    const center = getViewportCenterPosition();
    if (center) {
      placeReroute(center);
    }
  }, [getViewportCenterPosition, placeReroute]);

  return {
    handleNodeClick,
    handleNodeDoubleClick,
    handleNodesChange,
    handleEdgesChange,
    handleDragOver,
    handleDrop,
    handleAddSamplerFromSheet,
    handleAddSeedFromSheet,
    handleAddLlmFromSheet,
    handleAddModelProviderFromSheet,
    handleAddModelConfigFromSheet,
    handleAddToolProfileFromSheet,
    handleAddExpressionFromSheet,
    handleAddValidatorFromSheet,
    handleAddMarkdownNoteFromSheet,
    handleAddRerouteFromSheet,
    handleNodeDragStop,
    selectedReroutes,
  };
}
