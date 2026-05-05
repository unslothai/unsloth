// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type Connection,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
  type XYPosition,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import { create } from "zustand";
import type {
  RecipeNode,
  RecipeProcessorConfig,
  LayoutDirection,
  LlmType,
  NodeConfig,
  SeedSourceType,
  SamplerType,
} from "../types";
import {
  getBlockDefinition,
  type BlockKind,
  type BlockType,
  type SeedBlockType,
} from "../blocks/registry";
import { deriveDisplayGraph } from "../utils/graph/derive-display-graph";
import { applyRecipeConnection, isValidRecipeConnection } from "../utils/graph";
import {
  HANDLE_IDS,
  normalizeRecipeHandleId,
  remapRecipeEdgeHandlesForLayout,
} from "../utils/handles";
import type { RecipeSnapshot } from "../utils/import";
import { getLayoutedElements } from "../utils/layout";
import {
  centerModelInfraNodes,
  optimizeModelInfraEdgeHandles,
} from "./helpers/model-infra-layout";
import { applyEdgeRemovals, applyNodeRemovals } from "./helpers/removals";
import {
  applyRenameToConfigs,
  applyLayoutDirectionToNodes,
  buildNodeUpdate,
  syncEdgesForConfigPatch,
  syncSubcategoryConfigsForCategoryUpdate,
  updateNodeData,
} from "./recipe-studio-helpers";

type SheetView =
  | "root"
  | "sampler"
  | "seed"
  | "llm"
  | "validator"
  | "expression"
  | "note"
  | "processor";

type RecipeStudioState = {
  nodes: RecipeNode[];
  edges: Edge[];
  auxNodePositions: Record<string, XYPosition>;
  llmAuxVisibility: Record<string, boolean>;
  configs: Record<string, NodeConfig>;
  processors: RecipeProcessorConfig[];
  sheetOpen: boolean;
  sheetView: SheetView;
  activeConfigId: string | null;
  dialogOpen: boolean;
  layoutDirection: LayoutDirection;
  executionLocked: boolean;
  nextId: number;
  nextY: number;
  fitViewTick: number;
  setSheetOpen: (open: boolean) => void;
  setSheetView: (view: SheetView) => void;
  setProcessors: (processors: RecipeProcessorConfig[]) => void;
  setDialogOpen: (open: boolean) => void;
  setExecutionLocked: (locked: boolean) => void;
  resetRecipe: () => void;
  selectConfig: (id: string) => void;
  openConfig: (id: string) => void;
  setLayoutDirection: (direction: LayoutDirection) => void;
  applyLayout: () => void;
  setLlmAuxVisibility: (id: string, visible: boolean) => void;
  addSamplerNode: (
    type: SamplerType,
    position?: XYPosition,
    openDialog?: boolean,
  ) => void;
  addSeedNode: (
    type: SeedBlockType,
    position?: XYPosition,
    openDialog?: boolean,
  ) => void;
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
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  setAuxNodePosition: (id: string, position: XYPosition) => void;
  onNodesChange: (changes: NodeChange<RecipeNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (connection: Connection) => void;
  isValidConnection: IsValidConnection;
};

const INITIAL_STATE = {
  nodes: [],
  edges: [],
  auxNodePositions: {},
  llmAuxVisibility: {},
  configs: {},
  processors: [],
  sheetOpen: false,
  sheetView: "root",
  activeConfigId: null,
  dialogOpen: false,
  layoutDirection: "LR",
  executionLocked: false,
  nextId: 3,
  nextY: 280,
  fitViewTick: 0,
} satisfies Pick<
  RecipeStudioState,
  | "nodes"
  | "edges"
  | "auxNodePositions"
  | "llmAuxVisibility"
  | "configs"
  | "processors"
  | "sheetOpen"
  | "sheetView"
  | "activeConfigId"
  | "dialogOpen"
  | "layoutDirection"
  | "executionLocked"
  | "nextId"
  | "nextY"
  | "fitViewTick"
>;

function buildAddedNodeState(
  state: RecipeStudioState,
  kind: BlockKind,
  type: BlockType,
  position?: XYPosition,
  openDialog = true,
): Partial<RecipeStudioState> | RecipeStudioState {
  const id = `n${state.nextId}`;
  const existing = Object.values(state.configs);
  const definition = getBlockDefinition(kind, type);
  if (!definition) {
    return state;
  }
  const config = definition.createConfig(id, existing);
  return buildNodeUpdate(
    state,
    config,
    state.layoutDirection,
    position,
    openDialog,
  );
}

function getAddedNodeContext(
  update: Partial<RecipeStudioState> | RecipeStudioState,
): {
  nodes: RecipeNode[];
  configs: Record<string, NodeConfig>;
  newNodeId: string;
} | null {
  const nodes = "nodes" in update ? update.nodes : null;
  const configs = "configs" in update ? update.configs : null;
  const newNodeId = "activeConfigId" in update ? update.activeConfigId : null;
  if (!(nodes && configs && newNodeId)) {
    return null;
  }
  return { nodes, configs, newNodeId };
}

function placeNodeNear(
  nodes: RecipeNode[],
  nodeId: string,
  anchorId: string,
  direction: LayoutDirection,
  relation: "before" | "after",
): RecipeNode[] {
  const anchor = nodes.find((node) => node.id === anchorId);
  if (!anchor) {
    return nodes;
  }
  const primaryOffset = relation === "before" ? -440 : 440;
  return nodes.map((node) => {
    if (node.id !== nodeId) {
      return node;
    }
    if (direction === "TB") {
      return {
        ...node,
        position: {
          x: anchor.position.x,
          y: anchor.position.y + primaryOffset,
        },
      };
    }
    return {
      ...node,
      position: {
        x: anchor.position.x + primaryOffset,
        y: anchor.position.y,
      },
    };
  });
}

function connectSemantic(
  edges: Edge[],
  configs: Record<string, NodeConfig>,
  sourceId: string,
  targetId: string,
  layoutDirection: LayoutDirection,
): { edges: Edge[]; configs: Record<string, NodeConfig> } {
  const result = applyRecipeConnection(
    {
      source: sourceId,
      sourceHandle: HANDLE_IDS.semanticOut,
      target: targetId,
      targetHandle: HANDLE_IDS.semanticIn,
    },
    configs,
    edges,
    layoutDirection,
  );
  return {
    edges: result.edges,
    configs: result.configs ?? configs,
  };
}

function isModelSemanticEdge(edge: Edge, configs: Record<string, NodeConfig>): boolean {
  const source = configs[edge.source];
  const target = configs[edge.target];
  return Boolean(
    source &&
      target &&
      ((source.kind === "model_provider" && target.kind === "model_config") ||
        (source.kind === "model_config" && target.kind === "llm") ||
        (source.kind === "tool_config" && target.kind === "llm")),
  );
}

export const useRecipeStudioStore = create<RecipeStudioState>((set, get) => ({
  ...INITIAL_STATE,
  setSheetOpen: (open) => set({ sheetOpen: open }),
  setSheetView: (view) => set({ sheetView: view }),
  setProcessors: (processors) =>
    set((state) => (state.executionLocked ? state : { processors })),
  setDialogOpen: (open) => set({ dialogOpen: open }),
  setExecutionLocked: (locked) => set({ executionLocked: locked }),
  resetRecipe: () => set(INITIAL_STATE),
  selectConfig: (id) => set({ activeConfigId: id, dialogOpen: false }),
  openConfig: (id) => set({ activeConfigId: id, dialogOpen: true }),
  setLayoutDirection: (direction) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      return {
        layoutDirection: direction,
        edges: state.edges.map((edge) => {
          if (isModelSemanticEdge(edge, state.configs)) {
            return {
              ...edge,
              sourceHandle: normalizeRecipeHandleId(edge.sourceHandle),
              targetHandle: normalizeRecipeHandleId(edge.targetHandle),
            };
          }
          return {
            ...edge,
            ...remapRecipeEdgeHandlesForLayout(edge, direction),
          };
        }),
        nodes: applyLayoutDirectionToNodes(
          state.nodes,
          state.configs,
          direction,
        ),
      };
    }),
  applyLayout: () =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const isTopBottom = state.layoutDirection === "TB";

      const displayGraph = deriveDisplayGraph({
        nodes: state.nodes,
        edges: state.edges,
        configs: state.configs,
        layoutDirection: state.layoutDirection,
        auxNodePositions: {},
        llmAuxVisibility: state.llmAuxVisibility,
      });
      const { nodes } = getLayoutedElements(displayGraph.nodes, displayGraph.edges, {
        direction: state.layoutDirection,
        nodesep: isTopBottom ? 120 : 80,
        ranksep: isTopBottom ? 140 : 80,
        configs: state.configs,
      });
      const layoutedPositions = new Map(
        nodes.map((node) => [node.id, node.position] as const),
      );
      const nextNodes = state.nodes.map((node) => {
        const position = layoutedPositions.get(node.id);
        if (!position) {
          return node;
        }
        return { ...node, position };
      });
      const centeredNodes = centerModelInfraNodes(
        nextNodes,
        state.edges,
        state.configs,
        state.layoutDirection,
      );
      const optimizedEdges = optimizeModelInfraEdgeHandles(
        state.edges,
        centeredNodes,
        state.configs,
        state.layoutDirection,
      );
      return {
        auxNodePositions: {},
        edges: optimizedEdges,
        nodes: applyLayoutDirectionToNodes(
          centeredNodes,
          state.configs,
          state.layoutDirection,
        ),
      };
    }),
  setLlmAuxVisibility: (id, visible) =>
    set((state) => {
      if (state.llmAuxVisibility[id] === visible) {
        return state;
      }
      return {
        llmAuxVisibility: {
          ...state.llmAuxVisibility,
          [id]: visible,
        },
      };
    }),
  addSamplerNode: (type, position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      return buildAddedNodeState(state, "sampler", type, position, openDialog);
    }),
  addSeedNode: (type, position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const existing = Object.values(state.configs).find(
        (config) => config.kind === "seed",
      );
      if (!existing) {
        return buildAddedNodeState(
          state,
          "seed",
          type,
          position,
          openDialog,
        );
      }
      let nextSourceType: SeedSourceType = "hf";
      if (type === "seed_local") {
        nextSourceType = "local";
      } else if (type === "seed_unstructured") {
        nextSourceType = "unstructured";
      } else if (type === "seed_github") {
        nextSourceType = "github_repo";
      }

      const nextConfig: typeof existing = {
        ...existing,
        seed_source_type: nextSourceType,
        hf_repo_id: "",
        hf_subset: "",
        hf_split: "",
        hf_path: "",
        hf_token: "",
        hf_endpoint: "https://huggingface.co",
        local_file_name: "",
        unstructured_file_ids: [],
        unstructured_file_names: [],
        unstructured_file_sizes: [],
        resolved_paths: [],
        seed_columns: [],
        seed_drop_columns: [],
        seed_preview_rows: [],
        unstructured_chunk_size: "1200",
        unstructured_chunk_overlap: "200",
        github_repo_slug: "",
        github_token: "",
        github_limit: "100",
        github_item_types: ["issues", "pulls"],
        github_include_comments: true,
        github_max_comments_per_item: "30",
      };
      return {
        configs: {
          ...state.configs,
          [existing.id]: nextConfig,
        },
        nodes: updateNodeData(
          state.nodes.map((node) => ({ ...node, selected: node.id === existing.id })),
          existing.id,
          nextConfig,
          state.layoutDirection,
        ),
        activeConfigId: existing.id,
        dialogOpen: openDialog,
      };
    }),
  addLlmNode: (type, position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const added = buildAddedNodeState(state, "llm", type, position, openDialog);
      const context = getAddedNodeContext(added);
      if (!context) {
        return added;
      }
      let { nodes, configs } = context;
      let edges = state.edges;
      const modelConfigs = Object.values(configs).filter(
        (config) => config.kind === "model_config",
      );
      if (modelConfigs.length === 1) {
        if (!position) {
          nodes = placeNodeNear(
            nodes,
            context.newNodeId,
            modelConfigs[0].id,
            state.layoutDirection,
            "after",
          );
        }
        const next = connectSemantic(
          edges,
          configs,
          modelConfigs[0].id,
          context.newNodeId,
          state.layoutDirection,
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
  addModelProviderNode: (position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const added = buildAddedNodeState(
        state,
        "llm",
        "model_provider",
        position,
        openDialog,
      );
      const context = getAddedNodeContext(added);
      if (!context) {
        return added;
      }
      let { nodes, configs } = context;
      let edges = state.edges;
      const unboundModelConfigs = Object.values(configs).filter(
        (config) =>
          config.kind === "model_config" &&
          !config.provider.trim(),
      );
      if (!position && unboundModelConfigs.length > 0) {
        nodes = placeNodeNear(
          nodes,
          context.newNodeId,
          unboundModelConfigs[0].id,
          state.layoutDirection,
          "before",
        );
      }
      if (unboundModelConfigs.length === 1) {
        const next = connectSemantic(
          edges,
          configs,
          context.newNodeId,
          unboundModelConfigs[0].id,
          state.layoutDirection,
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
  addModelConfigNode: (position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const added = buildAddedNodeState(
        state,
        "llm",
        "model_config",
        position,
        openDialog,
      );
      const context = getAddedNodeContext(added);
      if (!context) {
        return added;
      }
      let { nodes, configs } = context;
      let edges = state.edges;
      const providers = Object.values(configs).filter(
        (config) => config.kind === "model_provider",
      );
      const unboundLlms = Object.values(configs).filter(
        (config) => config.kind === "llm" && !config.model_alias.trim(),
      );
      if (!position && providers.length === 1) {
        nodes = placeNodeNear(
          nodes,
          context.newNodeId,
          providers[0].id,
          state.layoutDirection,
          "after",
        );
      } else if (!position && unboundLlms.length > 0) {
        nodes = placeNodeNear(
          nodes,
          context.newNodeId,
          unboundLlms[0].id,
          state.layoutDirection,
          "before",
        );
      }
      if (providers.length === 1) {
        const next = connectSemantic(
          edges,
          configs,
          providers[0].id,
          context.newNodeId,
          state.layoutDirection,
        );
        edges = next.edges;
        configs = next.configs;
      }
      if (unboundLlms.length === 1) {
        const next = connectSemantic(
          edges,
          configs,
          context.newNodeId,
          unboundLlms[0].id,
          state.layoutDirection,
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
  addToolProfileNode: (position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const added = buildAddedNodeState(
        state,
        "llm",
        "tool_config",
        position,
        openDialog,
      );
      const context = getAddedNodeContext(added);
      if (!context) {
        return added;
      }
      let { nodes, configs } = context;
      let edges = state.edges;
      const unboundLlms = Object.values(configs).filter(
        (config) => config.kind === "llm" && !(config.tool_alias?.trim()),
      );
      if (!position && unboundLlms.length > 0) {
        nodes = placeNodeNear(
          nodes,
          context.newNodeId,
          unboundLlms[0].id,
          state.layoutDirection,
          "before",
        );
      }
      if (unboundLlms.length === 1) {
        const next = connectSemantic(
          edges,
          configs,
          context.newNodeId,
          unboundLlms[0].id,
          state.layoutDirection,
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
  addExpressionNode: (position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      return buildAddedNodeState(
        state,
        "expression",
        "expression",
        position,
        openDialog,
      );
    }),
  addValidatorNode: (type, position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      return buildAddedNodeState(
        state,
        "validator",
        type,
        position,
        openDialog,
      );
    }),
  addMarkdownNoteNode: (position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      return buildAddedNodeState(
        state,
        "note",
        "markdown_note",
        position,
        openDialog,
      );
    }),
  loadRecipe: (snapshot) =>
    set((state) => ({
      configs: snapshot.configs,
      nodes: applyLayoutDirectionToNodes(
        snapshot.nodes,
        snapshot.configs,
        snapshot.layoutDirection,
      ),
      edges: snapshot.edges,
      processors: snapshot.processors,
      layoutDirection: snapshot.layoutDirection,
      nextId: snapshot.nextId,
      nextY: snapshot.nextY,
      auxNodePositions: snapshot.auxNodePositions ?? {},
      llmAuxVisibility: {},
      activeConfigId: null,
      dialogOpen: false,
      sheetView: "root",
      fitViewTick: state.fitViewTick + 1,
    })),
  setAuxNodePosition: (id, position) =>
    set((state) => {
      const current = state.auxNodePositions[id];
      if (current && current.x === position.x && current.y === position.y) {
        return state;
      }
      return {
        auxNodePositions: {
          ...state.auxNodePositions,
          [id]: position,
        },
      };
    }),
  updateConfig: (id, patch) => {
    const applyUpdate = (state: RecipeStudioState) => {
      if (state.executionLocked) {
        return state;
      }
      const current = state.configs[id];
      if (!current) {
        return state;
      }
      const next = { ...current, ...patch } as NodeConfig;
      const oldName = current.name;
      const newName = next.name;
      const nameChanged = oldName !== newName;
      let configs: Record<string, NodeConfig> = {
        ...state.configs,
        [id]: next,
      };
      const nodes = updateNodeData(
        state.nodes,
        id,
        next,
        state.layoutDirection,
      );
      const edges = syncEdgesForConfigPatch(
        current,
        patch,
        configs,
        state.edges,
        state.layoutDirection,
      );
      configs = syncSubcategoryConfigsForCategoryUpdate(
        current,
        next,
        configs,
        oldName,
        newName,
        nameChanged,
      );

      if (nameChanged) {
        configs = applyRenameToConfigs(configs, oldName, newName);
      }

      // When a provider toggles between local and external, keep already
      // linked model_config nodes in sync. applyRenameToConfigs above has
      // already propagated any name change, so providerName here is the
      // post-rename value.
      if (current.kind === "model_provider" && next.kind === "model_provider") {
        const prevIsLocal = current.is_local === true;
        const nextIsLocal = next.is_local === true;
        if (prevIsLocal !== nextIsLocal) {
          const providerName = next.name;
          for (const [cfgId, cfg] of Object.entries(configs)) {
            if (cfg.kind !== "model_config" || cfg.provider !== providerName) {
              continue;
            }
            if (nextIsLocal && !cfg.model.trim()) {
              // external -> local: auto fill the placeholder model id so the
              // config does not fail "model is required" validation.
              configs = { ...configs, [cfgId]: { ...cfg, model: "local" } };
              continue;
            }
            if (!nextIsLocal && cfg.model === "local") {
              // local -> external: clear the placeholder so the user picks a
              // real model id for the new external endpoint.
              configs = { ...configs, [cfgId]: { ...cfg, model: "" } };
            }
          }
        }
      }

      return { configs, nodes, edges };
    };
    set(applyUpdate);
  },
  onNodesChange: (changes) => {
    const applyNodesChange = (state: RecipeStudioState) => {
      if (state.executionLocked) {
        return state;
      }
      const removedIds = changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id);

      const removed = applyNodeRemovals(
        { edges: state.edges, configs: state.configs },
        removedIds,
      );
      const nodes = applyNodeChanges<RecipeNode>(changes, state.nodes);
      const llmAuxVisibility =
        removedIds.length === 0
          ? state.llmAuxVisibility
          : Object.fromEntries(
              Object.entries(state.llmAuxVisibility).filter(
                ([id]) => !removedIds.includes(id),
              ),
            );
      return {
        nodes,
        edges: removed.edges,
        configs: removed.configs,
        llmAuxVisibility,
      };
    };
    set(applyNodesChange);
  },
  onEdgesChange: (changes) => {
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const removedEdges = changes
        .filter((change) => change.type === "remove")
        .map((change) => state.edges.find((edge) => edge.id === change.id))
        .filter((edge): edge is Edge => Boolean(edge));

      const configs = applyEdgeRemovals(state.configs, removedEdges);

      const edges = applyEdgeChanges(changes, state.edges);
      return configs === state.configs ? { edges } : { edges, configs };
    });
  },
  onConnect: (connection) => {
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const result = applyRecipeConnection(
        connection,
        state.configs,
        state.edges,
        state.layoutDirection,
      );
      return result.configs
        ? { edges: result.edges, configs: result.configs }
        : { edges: result.edges };
    });
  },
  isValidConnection: (connection) =>
    isValidRecipeConnection(
      {
        source: connection.source ?? null,
        target: connection.target ?? null,
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null,
      },
      get().configs,
    ),
}));
