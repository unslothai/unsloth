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
  SamplerType,
} from "../types";
import {
  getBlockDefinition,
  type BlockKind,
  type BlockType,
} from "../blocks/registry";
import { isCategoryConfig, isSubcategoryConfig } from "../utils";
import { applyRecipeConnection, isValidRecipeConnection } from "../utils/graph";
import type { RecipeSnapshot } from "../utils/import";
import { getLayoutedElements } from "../utils/layout";
import {
  applyRemovalToConfig,
  applyRemovalToConfigs,
  applyRenameToConfigs,
  applyLayoutDirectionToNodes,
  buildNodeUpdate,
  syncEdgesForConfigPatch,
  syncSubcategoryConfigsForCategoryUpdate,
  updateNodeData,
} from "./recipe-studio-helpers";

type SheetView = "root" | "sampler" | "llm" | "expression" | "processor";

type RecipeStudioState = {
  nodes: RecipeNode[];
  edges: Edge[];
  auxNodePositions: Record<string, XYPosition>;
  auxNodeSizes: Record<string, { width: number; height: number }>;
  configs: Record<string, NodeConfig>;
  processors: RecipeProcessorConfig[];
  sheetView: SheetView;
  activeConfigId: string | null;
  dialogOpen: boolean;
  layoutDirection: LayoutDirection;
  nextId: number;
  nextY: number;
  setSheetView: (view: SheetView) => void;
  setProcessors: (processors: RecipeProcessorConfig[]) => void;
  setDialogOpen: (open: boolean) => void;
  resetRecipe: () => void;
  selectConfig: (id: string) => void;
  openConfig: (id: string) => void;
  setLayoutDirection: (direction: LayoutDirection) => void;
  applyLayout: () => void;
  addSamplerNode: (type: SamplerType) => void;
  addLlmNode: (type: LlmType) => void;
  addModelProviderNode: () => void;
  addModelConfigNode: () => void;
  addExpressionNode: () => void;
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  setAuxNodePosition: (id: string, position: XYPosition) => void;
  setAuxNodeSize: (
    id: string,
    size: { width: number; height: number },
  ) => void;
  syncAuxNodePositions: (
    activeIds: string[],
    defaults: Record<string, XYPosition>,
  ) => void;
  syncAuxNodeSizes: (activeIds: string[]) => void;
  onNodesChange: (changes: NodeChange<RecipeNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (connection: Connection) => void;
  isValidConnection: IsValidConnection;
};

function buildAddedNodeState(
  state: RecipeStudioState,
  kind: BlockKind,
  type: BlockType,
): Partial<RecipeStudioState> | RecipeStudioState {
  const id = `n${state.nextId}`;
  const existing = Object.values(state.configs);
  const definition = getBlockDefinition(kind, type);
  if (!definition) {
    return state;
  }
  const config = definition.createConfig(id, existing);
  return buildNodeUpdate(state, config, state.layoutDirection);
}

export const useRecipeStudioStore = create<RecipeStudioState>((set, get) => ({
  nodes: [],
  edges: [],
  auxNodePositions: {},
  auxNodeSizes: {},
  configs: {},
  processors: [],
  sheetView: "root",
  activeConfigId: null,
  dialogOpen: false,
  layoutDirection: "LR",
  nextId: 3,
  nextY: 280,
  setSheetView: (view) => set({ sheetView: view }),
  setProcessors: (processors) => set({ processors }),
  setDialogOpen: (open) => set({ dialogOpen: open }),
  resetRecipe: () =>
    set({
      nodes: [],
      edges: [],
      auxNodePositions: {},
      auxNodeSizes: {},
      configs: {},
      processors: [],
      sheetView: "root",
      activeConfigId: null,
      dialogOpen: false,
      layoutDirection: "LR",
      nextId: 3,
      nextY: 280,
    }),
  selectConfig: (id) => set({ activeConfigId: id, dialogOpen: false }),
  openConfig: (id) => set({ activeConfigId: id, dialogOpen: true }),
  setLayoutDirection: (direction) =>
    set((state) => ({
      layoutDirection: direction,
      auxNodePositions: {},
      nodes: applyLayoutDirectionToNodes(
        state.nodes,
        state.configs,
        direction,
      ),
    })),
  applyLayout: () =>
    set((state) => {
      const isTopBottom = state.layoutDirection === "TB";
      const { nodes } = getLayoutedElements(state.nodes, state.edges, {
        direction: state.layoutDirection,
        nodesep: isTopBottom ? 120 : 80,
        ranksep: isTopBottom ? 140 : 80,
      });
      return {
        auxNodePositions: {},
        nodes: applyLayoutDirectionToNodes(
          nodes,
          state.configs,
          state.layoutDirection,
        ),
      };
    }),
  addSamplerNode: (type) =>
    set((state) => buildAddedNodeState(state, "sampler", type)),
  addLlmNode: (type) => set((state) => buildAddedNodeState(state, "llm", type)),
  addModelProviderNode: () =>
    set((state) => buildAddedNodeState(state, "llm", "model_provider")),
  addModelConfigNode: () =>
    set((state) => buildAddedNodeState(state, "llm", "model_config")),
  addExpressionNode: () =>
    set((state) => buildAddedNodeState(state, "expression", "expression")),
  loadRecipe: (snapshot) =>
    set((state) => ({
      configs: snapshot.configs,
      nodes: applyLayoutDirectionToNodes(
        snapshot.nodes,
        snapshot.configs,
        state.layoutDirection,
      ),
      edges: snapshot.edges,
      processors: snapshot.processors,
      nextId: snapshot.nextId,
      nextY: snapshot.nextY,
      auxNodePositions: {},
      auxNodeSizes: {},
      activeConfigId: null,
      dialogOpen: false,
      sheetView: "root",
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
  setAuxNodeSize: (id, size) =>
    set((state) => {
      const width = Math.max(1, size.width);
      const height = Math.max(1, size.height);
      const current = state.auxNodeSizes[id];
      if (current && current.width === width && current.height === height) {
        return state;
      }
      return {
        auxNodeSizes: {
          ...state.auxNodeSizes,
          [id]: { width, height },
        },
      };
    }),
  syncAuxNodePositions: (activeIds, defaults) =>
    set((state) => {
      const nextPositions: Record<string, XYPosition> = {};
      for (const id of activeIds) {
        const existing = state.auxNodePositions[id];
        if (existing) {
          nextPositions[id] = existing;
          continue;
        }
        const fallback = defaults[id];
        if (fallback) {
          nextPositions[id] = fallback;
        }
      }
      const prevIds = Object.keys(state.auxNodePositions);
      const nextIds = Object.keys(nextPositions);
      if (prevIds.length !== nextIds.length) {
        return { auxNodePositions: nextPositions };
      }
      for (const id of nextIds) {
        const prev = state.auxNodePositions[id];
        const next = nextPositions[id];
        if (!(prev && prev.x === next.x && prev.y === next.y)) {
          return { auxNodePositions: nextPositions };
        }
      }
      return state;
    }),
  syncAuxNodeSizes: (activeIds) =>
    set((state) => {
      const activeSet = new Set(activeIds);
      const nextSizes: Record<string, { width: number; height: number }> = {};
      for (const [id, size] of Object.entries(state.auxNodeSizes)) {
        if (activeSet.has(id)) {
          nextSizes[id] = size;
        }
      }
      const prevIds = Object.keys(state.auxNodeSizes);
      const nextIds = Object.keys(nextSizes);
      if (prevIds.length !== nextIds.length) {
        return { auxNodeSizes: nextSizes };
      }
      for (const id of nextIds) {
        const prev = state.auxNodeSizes[id];
        const next = nextSizes[id];
        if (!(prev && prev.width === next.width && prev.height === next.height)) {
          return { auxNodeSizes: nextSizes };
        }
      }
      return state;
    }),
  updateConfig: (id, patch) => {
    const applyUpdate = (state: RecipeStudioState) => {
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
      const edges = syncEdgesForConfigPatch(current, patch, configs, state.edges);
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

      return { configs, nodes, edges };
    };
    set(applyUpdate);
  },
  onNodesChange: (changes) => {
    // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: store update
    const applyNodesChange = (state: RecipeStudioState) => {
      const removedIds = changes
        .filter((change) => change.type === "remove")
        .map((change) => change.id);

      let edges = state.edges;
      let configs = state.configs;
      if (removedIds.length > 0) {
        const removedNames: string[] = [];
        edges = edges.filter(
          (edge) =>
            !(
              removedIds.includes(edge.source) ||
              removedIds.includes(edge.target)
            ),
        );
        configs = { ...configs };
        for (const id of removedIds) {
          const removed = configs[id];
          delete configs[id];
          if (removed?.name) {
            removedNames.push(removed.name);
          }
          if (isCategoryConfig(removed)) {
            const removedName = removed.name;
            for (const config of Object.values(configs)) {
              if (!isSubcategoryConfig(config)) {
                continue;
              }
              if (config.subcategory_parent !== removedName) {
                continue;
              }
              configs[config.id] = {
                ...config,
                // biome-ignore lint/style/useNamingConvention: api schema
                subcategory_parent: "",
                // biome-ignore lint/style/useNamingConvention: api schema
                subcategory_mapping: {},
              };
            }
          }
        }
        for (const name of removedNames) {
          configs = applyRemovalToConfigs(configs, name);
        }
      }

      const nodes = applyNodeChanges<RecipeNode>(changes, state.nodes);
      return { nodes, edges, configs };
    };
    set(applyNodesChange);
  },
  onEdgesChange: (changes) => {
    set((state) => {
      const removedEdges = changes
        .filter((change) => change.type === "remove")
        .map((change) => state.edges.find((edge) => edge.id === change.id))
        .filter((edge): edge is Edge => Boolean(edge));

      let configs = state.configs;
      if (removedEdges.length > 0) {
        for (const edge of removedEdges) {
          const source = configs[edge.source];
          const target = configs[edge.target];
          if (!(source && target)) {
            continue;
          }
          const updated = applyRemovalToConfig(target, source.name);
          if (updated !== target) {
            if (configs === state.configs) {
              configs = { ...configs };
            }
            configs[target.id] = updated;
          }
        }
      }

      const edges = applyEdgeChanges(changes, state.edges);
      return configs === state.configs ? { edges } : { edges, configs };
    });
  },
  onConnect: (connection) => {
    set((state) => {
      const result = applyRecipeConnection(
        connection,
        state.configs,
        state.edges,
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
