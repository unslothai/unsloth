import {
  type Connection,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
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
import {
  isCategoryConfig,
  isSubcategoryConfig,
  makeExpressionConfig,
  makeLlmConfig,
  makeSamplerConfig,
  nodeDataFromConfig,
} from "../utils";
import { applyCanvasConnection, isValidCanvasConnection } from "../utils/graph";
import type { CanvasSnapshot } from "../utils/import";

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

function updateNodeData(
  nodes: CanvasNode[],
  id: string,
  config: NodeConfig,
): CanvasNode[] {
  return nodes.map((node) =>
    node.id === id ? { ...node, data: nodeDataFromConfig(config) } : node,
  );
}

function findNodeIdByName(
  configs: Record<string, NodeConfig>,
  name: string,
): string | null {
  const entry = Object.entries(configs).find(
    ([, config]) => config.name === name,
  );
  return entry ? entry[0] : null;
}

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
      const config = makeSamplerConfig(id, type, existing);
      const node: CanvasNode = {
        id,
        type: "builder",
        position: { x: 0, y: state.nextY },
        data: nodeDataFromConfig(config),
      };
      return {
        configs: { ...state.configs, [id]: config },
        nodes: [...state.nodes, node],
        nextId: state.nextId + 1,
        nextY: state.nextY + 140,
        activeConfigId: id,
        dialogOpen: true,
      };
    });
  },
  addLlmNode: (type) => {
    set((state) => {
      const id = `n${state.nextId}`;
      const existing = Object.values(state.configs);
      const config = makeLlmConfig(id, type, existing);
      const node: CanvasNode = {
        id,
        type: "builder",
        position: { x: 0, y: state.nextY },
        data: nodeDataFromConfig(config),
      };
      return {
        configs: { ...state.configs, [id]: config },
        nodes: [...state.nodes, node],
        nextId: state.nextId + 1,
        nextY: state.nextY + 140,
        activeConfigId: id,
        dialogOpen: true,
      };
    });
  },
  addExpressionNode: () => {
    set((state) => {
      const id = `n${state.nextId}`;
      const existing = Object.values(state.configs);
      const config = makeExpressionConfig(id, existing);
      const node: CanvasNode = {
        id,
        type: "builder",
        position: { x: 0, y: state.nextY },
        data: nodeDataFromConfig(config),
      };
      return {
        configs: { ...state.configs, [id]: config },
        nodes: [...state.nodes, node],
        nextId: state.nextId + 1,
        nextY: state.nextY + 140,
        activeConfigId: id,
        dialogOpen: true,
      };
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
        const oldName = current.name;
        const nextCategory = isCategoryConfig(next) ? next : current;
        const newName = nextCategory.name;
        const oldValues = current.values ?? [];
        const newValues = nextCategory.values ?? [];
        const nameChanged = oldName !== newName;
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
      }

      const nodes = applyNodeChanges<CanvasNode>(changes, state.nodes);
      return { nodes, edges, configs };
    };
    set(applyNodesChange);
  },
  onEdgesChange: (changes) => {
    set((state) => ({ edges: applyEdgeChanges(changes, state.edges) }));
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
