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
import { deriveDisplayGraph } from "../utils/graph/derive-display-graph";
import { applyRecipeConnection, isValidRecipeConnection } from "../utils/graph";
import { HANDLE_IDS } from "../utils/handles";
import type { RecipeSnapshot } from "../utils/import";
import { getLayoutedElements } from "../utils/layout";
import { syncPositionsRecord, syncSizesRecord } from "./helpers/aux-sync";
import { applyEdgeRemovals, applyNodeRemovals } from "./helpers/removals";
import {
  applyRenameToConfigs,
  applyLayoutDirectionToNodes,
  buildNodeUpdate,
  syncEdgesForConfigPatch,
  syncSubcategoryConfigsForCategoryUpdate,
  updateNodeData,
} from "./recipe-studio-helpers";

type SheetView = "root" | "sampler" | "seed" | "llm" | "expression" | "processor";

type RecipeStudioState = {
  nodes: RecipeNode[];
  edges: Edge[];
  auxNodePositions: Record<string, XYPosition>;
  auxNodeSizes: Record<string, { width: number; height: number }>;
  llmAuxVisibility: Record<string, boolean>;
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
  setLlmAuxVisibility: (id: string, visible: boolean) => void;
  addSamplerNode: (type: SamplerType) => void;
  addSeedNode: () => void;
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

const INITIAL_STATE = {
  nodes: [],
  edges: [],
  auxNodePositions: {},
  auxNodeSizes: {},
  llmAuxVisibility: {},
  configs: {},
  processors: [],
  sheetView: "root",
  activeConfigId: null,
  dialogOpen: false,
  layoutDirection: "LR",
  nextId: 3,
  nextY: 280,
} satisfies Pick<
  RecipeStudioState,
  | "nodes"
  | "edges"
  | "auxNodePositions"
  | "auxNodeSizes"
  | "llmAuxVisibility"
  | "configs"
  | "processors"
  | "sheetView"
  | "activeConfigId"
  | "dialogOpen"
  | "layoutDirection"
  | "nextId"
  | "nextY"
>;

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
  );
  return {
    edges: result.edges,
    configs: result.configs ?? configs,
  };
}

export const useRecipeStudioStore = create<RecipeStudioState>((set, get) => ({
  ...INITIAL_STATE,
  setSheetView: (view) => set({ sheetView: view }),
  setProcessors: (processors) => set({ processors }),
  setDialogOpen: (open) => set({ dialogOpen: open }),
  resetRecipe: () => set(INITIAL_STATE),
  selectConfig: (id) => set({ activeConfigId: id, dialogOpen: false }),
  openConfig: (id) => set({ activeConfigId: id, dialogOpen: true }),
  setLayoutDirection: (direction) =>
    set((state) => ({
      layoutDirection: direction,
      nodes: applyLayoutDirectionToNodes(
        state.nodes,
        state.configs,
        direction,
      ),
    })),
  applyLayout: () =>
    set((state) => {
      const isTopBottom = state.layoutDirection === "TB";
      const displayGraph = deriveDisplayGraph({
        nodes: state.nodes,
        edges: state.edges,
        configs: state.configs,
        layoutDirection: state.layoutDirection,
        auxNodePositions: state.auxNodePositions,
        auxNodeSizes: state.auxNodeSizes,
        llmAuxVisibility: state.llmAuxVisibility,
      });
      const { nodes } = getLayoutedElements(displayGraph.nodes, displayGraph.edges, {
        direction: state.layoutDirection,
        nodesep: isTopBottom ? 120 : 80,
        ranksep: isTopBottom ? 140 : 80,
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
      const nextAuxNodePositions: Record<string, XYPosition> = {};
      for (const auxId of displayGraph.auxNodeIds) {
        const existing = state.auxNodePositions[auxId];
        const layouted = layoutedPositions.get(auxId);
        if (layouted) {
          nextAuxNodePositions[auxId] = layouted;
          continue;
        }
        if (existing) {
          nextAuxNodePositions[auxId] = existing;
          continue;
        }
        const fallback = displayGraph.auxDefaults[auxId];
        if (fallback) {
          nextAuxNodePositions[auxId] = fallback;
        }
      }
      return {
        auxNodePositions: nextAuxNodePositions,
        nodes: applyLayoutDirectionToNodes(
          nextNodes,
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
  addSamplerNode: (type) =>
    set((state) => buildAddedNodeState(state, "sampler", type)),
  addSeedNode: () =>
    set((state) => {
      const existing = Object.values(state.configs).find(
        (config) => config.kind === "seed",
      );
      if (!existing) {
        return buildAddedNodeState(state, "seed", "seed");
      }
      return {
        activeConfigId: existing.id,
        dialogOpen: true,
        nodes: state.nodes.map((node) => ({
          ...node,
          selected: node.id === existing.id,
        })),
      };
    }),
  addLlmNode: (type) => set((state) => buildAddedNodeState(state, "llm", type)),
  addModelProviderNode: () =>
    set((state) => {
      const added = buildAddedNodeState(state, "llm", "model_provider");
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
      if (unboundModelConfigs.length > 0) {
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
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
  addModelConfigNode: () =>
    set((state) => {
      const added = buildAddedNodeState(state, "llm", "model_config");
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
      if (providers.length === 1) {
        nodes = placeNodeNear(
          nodes,
          context.newNodeId,
          providers[0].id,
          state.layoutDirection,
          "after",
        );
      } else if (unboundLlms.length > 0) {
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
        );
        edges = next.edges;
        configs = next.configs;
      }
      return { ...added, nodes, edges, configs };
    }),
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
      llmAuxVisibility: {},
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
      const next = syncPositionsRecord(state.auxNodePositions, activeIds, defaults);
      if (next === state.auxNodePositions) {
        return state;
      }
      return {
        auxNodePositions: next,
      };
    }),
  syncAuxNodeSizes: (activeIds) =>
    set((state) => {
      const next = syncSizesRecord(state.auxNodeSizes, activeIds);
      return next === state.auxNodeSizes ? state : { auxNodeSizes: next };
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
    const applyNodesChange = (state: RecipeStudioState) => {
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
