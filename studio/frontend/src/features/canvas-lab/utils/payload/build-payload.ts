import type { Edge } from "@xyflow/react";
import type {
  CanvasProcessorConfig,
  CanvasNode,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
} from "../../types";
import { getConfigErrors } from "../index";
import {
  buildExpressionColumn,
  buildLlmColumn,
  buildModelConfig,
  buildModelProvider,
  buildProcessors,
  buildSamplerColumn,
} from "./builders";
import type { CanvasPayloadResult } from "./types";
import {
  isSemanticRelation,
  validateModelAliasLinks,
  validateModelConfigProviders,
  validateSubcategoryConfigs,
  validateTimedeltaConfigs,
  validateUsedProviders,
} from "./validate";

function getNodeWidth(node: CanvasNode): number | null {
  if (typeof node.width === "number" && Number.isFinite(node.width)) {
    return node.width;
  }
  if (typeof node.style?.width === "number" && Number.isFinite(node.style.width)) {
    return node.style.width;
  }
  if (typeof node.style?.width === "string") {
    const parsed = Number.parseFloat(node.style.width);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: payload build
export function buildCanvasPayload(
  configs: Record<string, NodeConfig>,
  nodes: CanvasNode[],
  edges: Edge[],
  processors: CanvasProcessorConfig[] = [],
): CanvasPayloadResult {
  const errors: string[] = [];
  const columns: Record<string, unknown>[] = [];
  const modelAliases = new Set<string>();
  const modelProviderNames = new Set<string>();
  const modelProviders: Record<string, unknown>[] = [];
  const modelConfigs: Record<string, unknown>[] = [];
  const modelProviderConfigs: ModelProviderConfig[] = [];
  const modelConfigConfigs: ModelConfig[] = [];
  const nameSet = new Set<string>();
  const nameToConfig = new Map<string, NodeConfig>();

  for (const node of nodes) {
    const config = configs[node.id];
    if (!config) {
      continue;
    }
    for (const error of getConfigErrors(config)) {
      errors.push(`${config.name}: ${error}`);
    }
    if (nameSet.has(config.name)) {
      errors.push(`Duplicate node name: ${config.name}.`);
    }
    nameSet.add(config.name);

    if (config.kind === "sampler") {
      nameToConfig.set(config.name, config);
      columns.push(buildSamplerColumn(config, errors));
      continue;
    }
    if (config.kind === "llm") {
      columns.push(buildLlmColumn(config, errors));
      if (config.model_alias) {
        modelAliases.add(config.model_alias);
      }
      nameToConfig.set(config.name, config);
      continue;
    }
    if (config.kind === "expression") {
      columns.push(buildExpressionColumn(config, errors));
      nameToConfig.set(config.name, config);
      continue;
    }
    if (config.kind === "model_provider") {
      modelProviderNames.add(config.name);
      modelProviders.push(buildModelProvider(config, errors));
      modelProviderConfigs.push(config);
      continue;
    }
    modelConfigs.push(buildModelConfig(config));
    modelConfigConfigs.push(config);
  }

  validateSubcategoryConfigs(configs, nameToConfig, errors);
  validateTimedeltaConfigs(configs, nameToConfig, errors);
  validateModelAliasLinks(modelAliases, modelConfigConfigs, errors);
  validateModelConfigProviders(
    modelConfigConfigs,
    modelAliases,
    modelProviderNames,
    errors,
  );
  validateUsedProviders(modelProviderConfigs, modelConfigConfigs, errors);

  const uiNodes = nodes.flatMap((node) => {
    const config = configs[node.id];
    if (!config) {
      return [];
    }
    const width = getNodeWidth(node);
    return [
      {
        id: config.name,
        x: node.position.x,
        y: node.position.y,
        ...(width !== null ? { width } : {}),
      },
    ];
  });

  const uiEdges = edges.flatMap((edge) => {
    const source = edge.source ? configs[edge.source] : null;
    const target = edge.target ? configs[edge.target] : null;
    if (!(source && target)) {
      return [];
    }
    return [
      {
        from: source.name,
        to: target.name,
        type:
          edge.type === "semantic" || isSemanticRelation(source, target)
            ? "semantic"
            : "canvas",
      },
    ];
  });
  const recipeProcessors = buildProcessors(processors, errors);

  return {
    errors,
    payload: {
      recipe: {
        // biome-ignore lint/style/useNamingConvention: api schema
        model_providers: modelProviders,
        // biome-ignore lint/style/useNamingConvention: api schema
        model_configs: modelConfigs,
        columns,
        processors: recipeProcessors,
      },
      run: {
        rows: 5,
        preview: true,
        // biome-ignore lint/style/useNamingConvention: api schema
        output_formats: ["jsonl"],
      },
      ui: {
        nodes: uiNodes,
        edges: uiEdges,
      },
    },
  };
}
