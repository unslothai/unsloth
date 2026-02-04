import {
  type Connection,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import { create } from "zustand";
import type {
  CanvasNode,
  LlmType,
  NodeConfig,
  SamplerConfig,
  SamplerType,
} from "../types";
import { getBlockDefinition } from "../blocks/registry";
import { isCategoryConfig, isSubcategoryConfig } from "../utils";
import { applyCanvasConnection, isValidCanvasConnection } from "../utils/graph";
import type { CanvasSnapshot } from "../utils/import";
import {
  applyRemovalToConfig,
  applyRemovalToConfigs,
  applyRenameToConfigs,
  buildNodeUpdate,
  findNodeIdByName,
  updateNodeData,
} from "./canvas-lab-helpers";

type SheetView = "root" | "sampler" | "llm" | "expression";

type CanvasLabState = {
  nodes: CanvasNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  sheetView: SheetView;
  activeConfigId: string | null;
  dialogOpen: boolean;
  nextId: number;
  nextY: number;
  setSheetView: (view: SheetView) => void;
  setDialogOpen: (open: boolean) => void;
  openConfig: (id: string) => void;
  addSamplerNode: (type: SamplerType) => void;
  addLlmNode: (type: LlmType) => void;
  addExpressionNode: () => void;
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  loadCanvas: (snapshot: CanvasSnapshot) => void;
  onNodesChange: (changes: NodeChange<CanvasNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (connection: Connection) => void;
  isValidConnection: IsValidConnection;
};

export const useCanvasLabStore = create<CanvasLabState>((set, get) => ({
  nodes: [],
  edges: [],
  configs: {},
  sheetView: "root",
  activeConfigId: null,
  dialogOpen: false,
  nextId: 3,
  nextY: 280,
  setSheetView: (view) => set({ sheetView: view }),
  setDialogOpen: (open) => set({ dialogOpen: open }),
  openConfig: (id) => set({ activeConfigId: id, dialogOpen: true }),
  addSamplerNode: (type) => {
    set((state) => {
      const id = `n${state.nextId}`;
      const existing = Object.values(state.configs);
      const definition = getBlockDefinition("sampler", type);
      if (!definition) {
        return state;
      }
      const config = definition.createConfig(id, existing);
      return buildNodeUpdate(state, config);
    });
  },
  addLlmNode: (type) => {
    set((state) => {
      const id = `n${state.nextId}`;
      const existing = Object.values(state.configs);
      const definition = getBlockDefinition("llm", type);
      if (!definition) {
        return state;
      }
      const config = definition.createConfig(id, existing);
      return buildNodeUpdate(state, config);
    });
  },
  addExpressionNode: () => {
    set((state) => {
      const id = `n${state.nextId}`;
      const existing = Object.values(state.configs);
      const definition = getBlockDefinition("expression", "expression");
      if (!definition) {
        return state;
      }
      const config = definition.createConfig(id, existing);
      return buildNodeUpdate(state, config);
    });
  },
  loadCanvas: (snapshot) =>
    set({
      configs: snapshot.configs,
      nodes: snapshot.nodes,
      edges: snapshot.edges,
      nextId: snapshot.nextId,
      nextY: snapshot.nextY,
      activeConfigId: null,
      dialogOpen: false,
      sheetView: "root",
    }),
  updateConfig: (id, patch) => {
    // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: store update
    const applyUpdate = (state: CanvasLabState) => {
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
      const nodes = updateNodeData(state.nodes, id, next);
      let edges = state.edges;

      const hasParentPatch = Object.prototype.hasOwnProperty.call(
        patch,
        "subcategory_parent",
      );
      if (isSubcategoryConfig(current) && hasParentPatch) {
        const nextParent =
          (patch as Partial<SamplerConfig>).subcategory_parent ?? "";
        const parentId = nextParent
          ? findNodeIdByName(configs, nextParent)
          : null;
        edges = edges.filter((edge) => edge.target !== id);
        if (parentId) {
          edges = addEdge(
            {
              source: parentId,
              target: id,
              sourceHandle: null,
              targetHandle: null,
            },
            edges,
          );
        }
      }

      if (isCategoryConfig(current)) {
        const nextCategory = isCategoryConfig(next) ? next : current;
        const oldValues = current.values ?? [];
        const newValues = nextCategory.values ?? [];
        const valuesChanged =
          oldValues.length !== newValues.length ||
          oldValues.some((value, index) => value !== newValues[index]);

        for (const config of Object.values(configs)) {
          if (!isSubcategoryConfig(config)) {
            continue;
          }
          if (config.subcategory_parent !== oldName) {
            continue;
          }
          const mapping = config.subcategory_mapping ?? {};
          const nextMapping: Record<string, string[]> = {};
          for (const value of newValues) {
            nextMapping[value] = mapping[value] ?? [];
          }
          const updated: NodeConfig = {
            ...config,
            // biome-ignore lint/style/useNamingConvention: api schema
            subcategory_parent: nameChanged
              ? newName
              : config.subcategory_parent,
            // biome-ignore lint/style/useNamingConvention: api schema
            subcategory_mapping: valuesChanged ? nextMapping : mapping,
          };
          configs = { ...configs, [config.id]: updated };
        }
      }

      if (nameChanged) {
        configs = applyRenameToConfigs(configs, oldName, newName);
      }

      return { configs, nodes, edges };
    };
    set(applyUpdate);
  },
  onNodesChange: (changes) => {
    // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: store update
    const applyNodesChange = (state: CanvasLabState) => {
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

      const nodes = applyNodeChanges<CanvasNode>(changes, state.nodes);
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
      const result = applyCanvasConnection(
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
    isValidCanvasConnection(connection, get().configs),
}));
