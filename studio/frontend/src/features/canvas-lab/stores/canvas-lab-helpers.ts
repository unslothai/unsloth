import { type Edge, addEdge } from "@xyflow/react";
import { DEFAULT_NODE_WIDTH } from "../constants";
import type {
  CanvasNode,
  LayoutDirection,
  LlmConfig,
  ModelConfig,
  NodeConfig,
  SamplerConfig,
} from "../types";
import { isCategoryConfig, isSubcategoryConfig, nodeDataFromConfig } from "../utils";
import { HANDLE_IDS } from "../utils/handles";
import { removeRef, replaceRef } from "../utils/refs";
import { getConfigUiMode } from "../components/inline/inline-policy";

type NodeUpdateState = {
  configs: Record<string, NodeConfig>;
  nodes: CanvasNode[];
  nextId: number;
  nextY: number;
};

type NodeUpdateResult = {
  configs: Record<string, NodeConfig>;
  nodes: CanvasNode[];
  nextId: number;
  nextY: number;
  activeConfigId: string;
  dialogOpen: boolean;
};

export function updateNodeData(
  nodes: CanvasNode[],
  id: string,
  config: NodeConfig,
  layoutDirection: LayoutDirection,
): CanvasNode[] {
  return nodes.map((node) =>
    node.id === id
      ? { ...node, data: nodeDataFromConfig(config, layoutDirection) }
      : node,
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

export function buildNodeUpdate(
  state: NodeUpdateState,
  config: NodeConfig,
  layoutDirection: LayoutDirection,
): NodeUpdateResult {
  const node: CanvasNode = {
    id: config.id,
    type: "builder",
    position: { x: 0, y: state.nextY },
    data: nodeDataFromConfig(config, layoutDirection),
    style: { width: DEFAULT_NODE_WIDTH },
    selected: true,
  };
  const mode = getConfigUiMode(config);
  return {
    configs: { ...state.configs, [config.id]: config },
    nodes: [...state.nodes.map((item) => ({ ...item, selected: false })), node],
    nextId: state.nextId + 1,
    nextY: state.nextY + 140,
    activeConfigId: config.id,
    dialogOpen: mode === "dialog",
  };
}

export function applyLayoutDirectionToNodes(
  nodes: CanvasNode[],
  configs: Record<string, NodeConfig>,
  layoutDirection: LayoutDirection,
): CanvasNode[] {
  return nodes.map((node) => {
    const config = configs[node.id];
    if (config) {
      return { ...node, data: nodeDataFromConfig(config, layoutDirection) };
    }
    return {
      ...node,
      data: { ...node.data, layoutDirection },
    };
  });
}

export function syncEdgesForConfigPatch(
  current: NodeConfig,
  patch: Partial<NodeConfig>,
  configs: Record<string, NodeConfig>,
  edges: Edge[],
): Edge[] {
  let nextEdges = edges;

  const hasParentPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "subcategory_parent",
  );
  if (isSubcategoryConfig(current) && hasParentPatch) {
    const nextParent = (patch as Partial<SamplerConfig>).subcategory_parent ?? "";
    const parentId = nextParent ? findNodeIdByName(configs, nextParent) : null;
    nextEdges = nextEdges.filter((edge) => edge.target !== current.id);
    if (parentId) {
      nextEdges = addEdge(
        {
          source: parentId,
          target: current.id,
          sourceHandle: HANDLE_IDS.dataOut,
          targetHandle: HANDLE_IDS.dataIn,
          type: "canvas",
        },
        nextEdges,
      );
    }
  }

  const hasProviderPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "provider",
  );
  if (current.kind === "model_config" && hasProviderPatch) {
    const nextProvider = (patch as Partial<ModelConfig>).provider ?? "";
    nextEdges = nextEdges.filter((edge) => {
      if (edge.target !== current.id) {
        return true;
      }
      const source = configs[edge.source];
      return !(source && source.kind === "model_provider");
    });
    if (nextProvider) {
      const providerId = findNodeIdByName(configs, nextProvider);
      if (providerId) {
        nextEdges = addEdge(
          {
            source: providerId,
            target: current.id,
            sourceHandle: HANDLE_IDS.semanticOut,
            targetHandle: HANDLE_IDS.semanticIn,
            type: "semantic",
          },
          nextEdges,
        );
      }
    }
  }

  const hasReferencePatch = Object.prototype.hasOwnProperty.call(
    patch,
    "reference_column_name",
  );
  if (
    current.kind === "sampler" &&
    current.sampler_type === "timedelta" &&
    hasReferencePatch
  ) {
    const nextReference =
      (patch as Partial<SamplerConfig>).reference_column_name ?? "";
    nextEdges = nextEdges.filter((edge) => {
      if (edge.target !== current.id) {
        return true;
      }
      const source = configs[edge.source];
      return !(
        source &&
        source.kind === "sampler" &&
        source.sampler_type === "datetime"
      );
    });
    if (nextReference) {
      const referenceId = findNodeIdByName(configs, nextReference);
      const source = referenceId ? configs[referenceId] : null;
      if (
        referenceId &&
        source &&
        source.kind === "sampler" &&
        source.sampler_type === "datetime"
      ) {
        nextEdges = addEdge(
          {
            source: referenceId,
            target: current.id,
            sourceHandle: HANDLE_IDS.dataOut,
            targetHandle: HANDLE_IDS.dataIn,
            type: "canvas",
          },
          nextEdges,
        );
      }
    }
  }

  const hasModelAliasPatch = Object.prototype.hasOwnProperty.call(
    patch,
    "model_alias",
  );
  if (current.kind === "llm" && hasModelAliasPatch) {
    const nextAlias =
      (patch as Partial<NodeConfig> & { model_alias?: string }).model_alias ?? "";
    nextEdges = nextEdges.filter((edge) => {
      if (edge.target !== current.id) {
        return true;
      }
      const source = configs[edge.source];
      return !(source && source.kind === "model_config");
    });
    if (nextAlias) {
      const modelConfigId = findNodeIdByName(configs, nextAlias);
      if (modelConfigId) {
        nextEdges = addEdge(
          {
            source: modelConfigId,
            target: current.id,
            sourceHandle: HANDLE_IDS.semanticOut,
            targetHandle: HANDLE_IDS.semanticIn,
            type: "semantic",
          },
          nextEdges,
        );
      }
    }
  }

  return nextEdges;
}

export function syncSubcategoryConfigsForCategoryUpdate(
  current: NodeConfig,
  next: NodeConfig,
  configs: Record<string, NodeConfig>,
  oldName: string,
  newName: string,
  nameChanged: boolean,
): Record<string, NodeConfig> {
  if (!isCategoryConfig(current)) {
    return configs;
  }
  const nextCategory = isCategoryConfig(next) ? next : current;
  const oldValues = current.values ?? [];
  const newValues = nextCategory.values ?? [];
  const valuesChanged =
    oldValues.length !== newValues.length ||
    oldValues.some((value, index) => value !== newValues[index]);

  let nextConfigs = configs;
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
      subcategory_parent: nameChanged ? newName : config.subcategory_parent,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: valuesChanged ? nextMapping : mapping,
    };
    nextConfigs = { ...nextConfigs, [config.id]: updated };
  }
  return nextConfigs;
}

function updateTemplateFields(
  config: NodeConfig,
  updater: (value: string) => string,
): NodeConfig {
  if (config.kind === "llm") {
    const nextPrompt = updater(config.prompt);
    const nextSystem = updater(config.system_prompt);
    const nextOutput =
      typeof config.output_format === "string"
        ? updater(config.output_format)
        : config.output_format;
    if (
      nextPrompt === config.prompt &&
      nextSystem === config.system_prompt &&
      nextOutput === config.output_format
    ) {
      return config;
    }
    return {
      ...config,
      prompt: nextPrompt,
      // biome-ignore lint/style/useNamingConvention: api schema
      system_prompt: nextSystem,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_format: nextOutput,
    };
  }
  if (config.kind === "expression") {
    const nextExpr = updater(config.expr);
    if (nextExpr === config.expr) {
      return config;
    }
    return { ...config, expr: nextExpr };
  }
  return config;
}

export function applyRenameToConfig(
  config: NodeConfig,
  from: string,
  to: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) =>
    replaceRef(value, from, to),
  );
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: to,
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: to,
    };
  }
  if (config.kind === "model_config" && config.provider === from) {
    const base = next as ModelConfig;
    next = { ...base, provider: to };
  }
  if (config.kind === "llm" && config.model_alias === from) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: to };
  }
  return next;
}

export function applyRemovalToConfig(
  config: NodeConfig,
  ref: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) => removeRef(value, ref));
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: {},
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: "",
    };
  }
  if (config.kind === "model_config" && config.provider === ref) {
    const base = next as ModelConfig;
    next = { ...base, provider: "" };
  }
  if (config.kind === "llm" && config.model_alias === ref) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: "" };
  }
  return next;
}

export function applyRenameToConfigs(
  configs: Record<string, NodeConfig>,
  from: string,
  to: string,
): Record<string, NodeConfig> {
  if (!from || from === to) {
    return configs;
  }
  let next = configs;
  for (const [id, config] of Object.entries(configs)) {
    const updated = applyRenameToConfig(config, from, to);
    if (updated !== config) {
      if (next === configs) {
        next = { ...configs };
      }
      next[id] = updated;
    }
  }
  return next;
}

export function applyRemovalToConfigs(
  configs: Record<string, NodeConfig>,
  ref: string,
): Record<string, NodeConfig> {
  if (!ref) {
    return configs;
  }
  let next = configs;
  for (const [id, config] of Object.entries(configs)) {
    const updated = applyRemovalToConfig(config, ref);
    if (updated !== config) {
      if (next === configs) {
        next = { ...configs };
      }
      next[id] = updated;
    }
  }
  return next;
}
