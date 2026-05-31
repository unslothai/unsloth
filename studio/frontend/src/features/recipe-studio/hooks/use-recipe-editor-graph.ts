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
} from "react";
import { RECIPE_BLOCK_DND_MIME, type RecipeBlockDragPayload } from "../components/block-sheet";
import type { SeedBlockType } from "../blocks/registry";
import type {
  LlmType,
  NodeConfig,
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
  SamplerType,
} from "../types";
import { applyAuxNodeChanges, filterEdgeChangesByIds, filterNodeChangesByIds } from "../utils/reactflow-changes";
import type { RecipeGraphAuxNodeData } from "../components/recipe-graph-aux-node";

const SUPPORTED_DRAG_KINDS: RecipeBlockDragPayload["kind"][] = [
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
    return {
      kind: parsed.kind,
      type: parsed.type,
    };
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
  addSamplerNode: (type: SamplerType, position?: XYPosition, openDialog?: boolean) => void;
  addSeedNode: (type: SeedBlockType, position?: XYPosition, openDialog?: boolean) => void;
  addLlmNode: (type: LlmType, position?: XYPosition, openDialog?: boolean) => void;
  addModelProviderNode: (position?: XYPosition, openDialog?: boolean) => void;
  addModelConfigNode: (position?: XYPosition, openDialog?: boolean) => void;
  addToolProfileNode: (position?: XYPosition, openDialog?: boolean) => void;
  addExpressionNode: (position?: XYPosition, openDialog?: boolean) => void;
  addValidatorNode: (
    type: "validator_python" | "validator_sql" | "validator_oxc",
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
    type: "validator_python" | "validator_sql" | "validator_oxc",
  ) => void;
  handleAddMarkdownNoteFromSheet: () => void;
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

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      applyAuxNodeChanges(changes, { setAuxNodePosition });
      const next = filterNodeChangesByIds(
        changes as NodeChange<RecipeBuilderNode>[],
        baseNodeIds,
      );
      if (next.length) {
        onNodesChange(next);
      }
    },
    [baseNodeIds, onNodesChange, setAuxNodePosition],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const next = filterEdgeChangesByIds(changes, baseEdgeIds);
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
          payload.type as "validator_python" | "validator_sql" | "validator_oxc",
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
    (type: "validator_python" | "validator_sql" | "validator_oxc") => {
      addValidatorNode(type, getViewportCenterPosition());
    },
    [addValidatorNode, getViewportCenterPosition],
  );

  const handleAddMarkdownNoteFromSheet = useCallback(() => {
    addMarkdownNoteNode(getViewportCenterPosition());
  }, [addMarkdownNoteNode, getViewportCenterPosition]);

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
  };
}
