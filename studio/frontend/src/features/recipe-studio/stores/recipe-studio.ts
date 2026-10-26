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
import {
  type BlockKind,
  type BlockType,
  type SeedBlockType,
  type ValidatorBlockType,
  getBlockDefinition,
} from "../blocks/registry";
import type {
  LayoutDirection,
  LlmType,
  NodeConfig,
  RecipeNode,
  RecipeProcessorConfig,
  SamplerType,
  SeedSourceType,
} from "../types";
import { applyRecipeConnection, isValidRecipeConnection } from "../utils/graph";
import { deriveDisplayGraph } from "../utils/graph/derive-display-graph";
import {
  findRerouteUpstream,
  insertRerouteIntoEdge,
  isRerouteId,
  isRerouteInUse,
  logicalEdgeId,
  removeReroutesFromEdges,
  withEdgeVia,
} from "../utils/graph/reroutes";
import { HANDLE_IDS, migrateRecipeEdgeHandles } from "../utils/handles";
import type { RecipeSnapshot } from "../utils/import";
import { getLayoutedElements } from "../utils/layout";
import { makeUnstructuredUploadUid } from "../utils/config-factories";
import { applyEdgeRemovals, applyNodeRemovals } from "./helpers/removals";
import {
  applyLayoutDirectionToNodes,
  applyRenameToConfigs,
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
  | "model"
  | "validator"
  | "expression"
  | "note"
  | "processor";

type RecipeStudioState = {
  nodes: RecipeNode[];
  edges: Edge[];
  auxNodePositions: Record<string, XYPosition>;
  reroutes: Record<string, XYPosition>;
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
  // Upload-uid directories whose owning block dropped them; server-side deletion is deferred until
  // a save no longer references them, so a reload before autosave cannot leave a saved recipe
  // pointing at deleted files.
  pendingUploadCleanups: string[];
  queueUploadCleanup: (uid: string) => void;
  setSheetOpen: (open: boolean) => void;
  setSheetView: (view: SheetView) => void;
  setProcessors: (processors: RecipeProcessorConfig[]) => void;
  setDialogOpen: (open: boolean) => void;
  setExecutionLocked: (locked: boolean) => void;
  resetRecipe: () => void;
  selectConfig: (id: string) => void;
  openConfig: (id: string) => void;
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
  addLlmNode: (
    type: LlmType,
    position?: XYPosition,
    openDialog?: boolean,
  ) => void;
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
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  setAuxNodePosition: (id: string, position: XYPosition) => void;
  /** Adds a reroute, threaded into the drawn wire `wireId` when one is given. */
  addReroute: (position: XYPosition, wireId?: string | null) => void;
  setReroutePosition: (id: string, position: XYPosition) => void;
  /** Threads a free reroute into a drawn wire; one already in use is left alone. */
  dropRerouteOnWire: (id: string, wireId: string) => void;
  removeReroutes: (ids: string[]) => void;
  onNodesChange: (changes: NodeChange<RecipeNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (connection: Connection) => void;
  isValidConnection: IsValidConnection;
};

const INITIAL_STATE = {
  nodes: [],
  edges: [],
  auxNodePositions: {},
  reroutes: {},
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
  pendingUploadCleanups: [],
} satisfies Pick<
  RecipeStudioState,
  | "nodes"
  | "edges"
  | "auxNodePositions"
  | "reroutes"
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
  | "pendingUploadCleanups"
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

// Upload uid of a seed block whose server-side directory becomes orphaned
// when the block drops it. Only uid directories qualify (single owner);
// legacy node-id directories can be shared by other recipes.
function seedUploadCleanupUid(config: NodeConfig | undefined): string | null {
  if (!config || config.kind !== "seed") {
    return null;
  }
  const uid = config.unstructured_upload_uid?.trim();
  if (!uid || !config.unstructured_file_ids?.length) {
    return null;
  }
  return uid;
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
  queueUploadCleanup: (uid) =>
    set((state) =>
      state.pendingUploadCleanups.includes(uid)
        ? state
        : { pendingUploadCleanups: [...state.pendingUploadCleanups, uid] },
    ),
  selectConfig: (id) => set({ activeConfigId: id, dialogOpen: false }),
  openConfig: (id) => set({ activeConfigId: id, dialogOpen: true }),
  applyLayout: () =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const displayGraph = deriveDisplayGraph({
        nodes: state.nodes,
        edges: state.edges,
        configs: state.configs,
        auxNodePositions: {},
        llmAuxVisibility: state.llmAuxVisibility,
      });
      const { nodes } = getLayoutedElements(
        displayGraph.nodes,
        displayGraph.edges,
        {
          direction: "LR",
          // Wide enough gaps for the router to run a bus and its taps between columns.
          nodesep: 88,
          ranksep: 140,
          configs: state.configs,
        },
      );
      const layoutedPositions = new Map(
        nodes.map((node) => [node.id, node.position] as const),
      );
      const nextNodes = state.nodes.map((node) => {
        const position = layoutedPositions.get(node.id);
        return position ? { ...node, position } : node;
      });
      // Reroutes were placed for the old layout; the wires go straight again.
      return {
        auxNodePositions: {},
        reroutes: {},
        edges: removeReroutesFromEdges(state.edges, new Set(Object.keys(state.reroutes))),
        nodes: applyLayoutDirectionToNodes(nextNodes, state.configs, "LR"),
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
  addSeedNode: (type, position, openDialog = true) => {
    const current = get();
    if (!current.executionLocked) {
      // The reset below clears the block's upload uid and file list; queue
      // its server-side directory for deletion after the next save.
      const uid = seedUploadCleanupUid(
        Object.values(current.configs).find((config) => config.kind === "seed"),
      );
      if (uid) {
        current.queueUploadCleanup(uid);
      }
    }
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const existing = Object.values(state.configs).find(
        (config) => config.kind === "seed",
      );
      if (!existing) {
        return buildAddedNodeState(state, "seed", type, position, openDialog);
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
        // Blank so buildSeedConfig resolves the endpoint at build time.
        hf_endpoint: "",
        local_file_name: "",
        unstructured_upload_uid:
          nextSourceType === "unstructured" ? makeUnstructuredUploadUid() : "",
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
          state.nodes.map((node) => ({
            ...node,
            selected: node.id === existing.id,
          })),
          existing.id,
          nextConfig,
          state.layoutDirection,
        ),
        activeConfigId: existing.id,
        dialogOpen: openDialog,
      };
    });
  },
  addLlmNode: (type, position, openDialog = true) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const added = buildAddedNodeState(
        state,
        "llm",
        type,
        position,
        openDialog,
      );
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
        (config) => config.kind === "model_config" && !config.provider.trim(),
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
        (config) => config.kind === "llm" && !config.tool_alias?.trim(),
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
      nodes: applyLayoutDirectionToNodes(snapshot.nodes, snapshot.configs, "LR"),
      edges: snapshot.edges.map(migrateRecipeEdgeHandles),
      processors: snapshot.processors,
      layoutDirection: "LR",
      reroutes: snapshot.reroutes ?? {},
      nextId: snapshot.nextId,
      nextY: snapshot.nextY,
      auxNodePositions: snapshot.auxNodePositions ?? {},
      llmAuxVisibility: {},
      activeConfigId: null,
      dialogOpen: false,
      sheetView: "root",
      fitViewTick: state.fitViewTick + 1,
      // Queued cleanups belong to the previous recipe; draining them after
      // a save of this one could delete files its saved payload still uses.
      pendingUploadCleanups: [],
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
  addReroute: (position, wireId) =>
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const id = `r${state.nextId}`;
      return {
        nextId: state.nextId + 1,
        reroutes: { ...state.reroutes, [id]: position },
        edges: wireId ? insertRerouteIntoEdge(state.edges, wireId, id) : state.edges,
      };
    }),
  setReroutePosition: (id, position) =>
    set((state) => {
      const current = state.reroutes[id];
      if (!current || (current.x === position.x && current.y === position.y)) {
        return state;
      }
      return { reroutes: { ...state.reroutes, [id]: position } };
    }),
  dropRerouteOnWire: (id, wireId) =>
    set((state) => {
      if (
        state.executionLocked ||
        !isRerouteId(id, state.reroutes) ||
        isRerouteInUse(state.edges, id)
      ) {
        return state;
      }
      return { edges: insertRerouteIntoEdge(state.edges, wireId, id) };
    }),
  removeReroutes: (ids) =>
    set((state) => {
      const removed = new Set(ids.filter((id) => isRerouteId(id, state.reroutes)));
      if (state.executionLocked || removed.size === 0) {
        return state;
      }
      return {
        reroutes: Object.fromEntries(
          Object.entries(state.reroutes).filter(([id]) => !removed.has(id)),
        ),
        edges: removeReroutesFromEdges(state.edges, removed),
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

      // When a provider toggles local/external, keep linked model_config nodes
      // in sync. applyRenameToConfigs above already propagated any name change,
      // so providerName here is the post-rename value.
      if (current.kind === "model_provider" && next.kind === "model_provider") {
        const prevIsLocal = current.is_local === true;
        const nextIsLocal = next.is_local === true;
        if (prevIsLocal !== nextIsLocal) {
          const providerName = next.name;
          for (const [cfgId, cfg] of Object.entries(configs)) {
            if (cfg.kind !== "model_config" || cfg.provider !== providerName) {
              continue;
            }
            configs = {
              ...configs,
              [cfgId]: {
                ...cfg,
                model: "",
                // biome-ignore lint/style/useNamingConvention: api schema
                gguf_variant: undefined,
              },
            };
          }
        }
      }

      return { configs, nodes, edges };
    };
    set(applyUpdate);
  },
  onNodesChange: (changes) => {
    const current = get();
    if (!current.executionLocked) {
      for (const change of changes) {
        if (change.type === "remove") {
          const uid = seedUploadCleanupUid(current.configs[change.id]);
          if (uid) {
            current.queueUploadCleanup(uid);
          }
        }
      }
    }
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
      // A wire through reroutes is drawn as segments; removing any of them removes the edge.
      const logicalChanges = changes.map((change) =>
        "id" in change ? { ...change, id: logicalEdgeId(change.id) } : change,
      ) as EdgeChange<Edge>[];
      const removedEdges = logicalChanges
        .filter((change) => change.type === "remove")
        .map((change) => state.edges.find((edge) => edge.id === change.id))
        .filter((edge): edge is Edge => Boolean(edge));

      const configs = applyEdgeRemovals(state.configs, removedEdges);

      const edges = applyEdgeChanges(logicalChanges, state.edges);
      return configs === state.configs ? { edges } : { edges, configs };
    });
  },
  onConnect: (connection) => {
    set((state) => {
      if (state.executionLocked) {
        return state;
      }
      const upstream = connection.source
        ? findRerouteUpstream(state.edges, connection.source)
        : null;
      if (upstream) {
        // A new output from a reroute: the recipe edge runs from the reroute's own source.
        const result = applyRecipeConnection(
          {
            ...connection,
            source: upstream.edge.source,
            sourceHandle: upstream.edge.sourceHandle ?? null,
          },
          state.configs,
          state.edges,
        );
        const edges = result.edges.map((edge) =>
          edge.source === upstream.edge.source &&
          edge.target === connection.target &&
          !state.edges.includes(edge)
            ? withEdgeVia(edge, upstream.via)
            : edge,
        );
        return result.configs ? { edges, configs: result.configs } : { edges };
      }
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
  isValidConnection: (connection) => {
    const state = get();
    let source = connection.source ?? null;
    let sourceHandle = connection.sourceHandle ?? null;
    if (connection.target && isRerouteId(connection.target, state.reroutes)) {
      // Wires get into a reroute by dropping it on them, not by dragging onto it.
      return false;
    }
    if (source && isRerouteId(source, state.reroutes)) {
      const upstream = findRerouteUpstream(state.edges, source);
      if (!upstream) {
        return false;
      }
      source = upstream.edge.source;
      sourceHandle = upstream.edge.sourceHandle ?? null;
    }
    return isValidRecipeConnection(
      {
        source,
        target: connection.target ?? null,
        sourceHandle,
        targetHandle: connection.targetHandle ?? null,
      },
      state.configs,
    );
  },
}));
