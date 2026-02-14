import type { Edge } from "@xyflow/react";
import type {
  RecipeProcessorConfig,
  RecipeNode,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
} from "../../types";
import { getConfigErrors } from "../index";
import { isSemanticRelation } from "../graph/relations";
import { readNodeWidth } from "../rf-node-dimensions";
import {
  buildExpressionColumn,
  buildLlmMcpProvider,
  buildLlmToolConfig,
  buildLlmColumn,
  buildModelConfig,
  buildModelProvider,
  buildProcessors,
  buildSamplerColumn,
  buildSeedConfig,
  buildSeedDropProcessor,
  pickFirstSeedConfig,
} from "./builders";
import type { RecipePayloadResult } from "./types";
import {
  validateModelAliasLinks,
  validateModelConfigProviders,
  validateSubcategoryConfigs,
  validateTimedeltaConfigs,
  validateUsedProviders,
} from "./validate";

function pushUniqueJson(
  label: string,
  key: string,
  item: Record<string, unknown>,
  seen: Map<string, string>,
  out: Record<string, unknown>[],
  errors: string[],
): void {
  const serialized = JSON.stringify(item);
  const existing = seen.get(key);
  if (existing && existing !== serialized) {
    errors.push(`${label} ${key}: conflicting definitions.`);
    return;
  }
  if (!existing) {
    seen.set(key, serialized);
    out.push(item);
  }
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: payload build
export function buildRecipePayload(
  configs: Record<string, NodeConfig>,
  nodes: RecipeNode[],
  edges: Edge[],
  processors: RecipeProcessorConfig[] = [],
): RecipePayloadResult {
  const errors: string[] = [];
  const columns: Record<string, unknown>[] = [];
  const modelAliases = new Set<string>();
  const modelProviderNames = new Set<string>();
  const modelProviders: Record<string, unknown>[] = [];
  const mcpProviders: Record<string, unknown>[] = [];
  const modelConfigs: Record<string, unknown>[] = [];
  const toolConfigs: Record<string, unknown>[] = [];
  const modelProviderConfigs: ModelProviderConfig[] = [];
  const modelConfigConfigs: ModelConfig[] = [];
  const llmToolAliasesUsed = new Set<string>();
  const mcpProviderJsonByName = new Map<string, string>();
  const toolConfigJsonByAlias = new Map<string, string>();
  const nameSet = new Set<string>();
  const nameToConfig = new Map<string, NodeConfig>();
  const firstSeed = pickFirstSeedConfig(configs);

  for (const node of nodes) {
    const config = configs[node.id];
    if (!config) {
      continue;
    }
    for (const error of getConfigErrors(config)) {
      errors.push(`${config.name}: ${error}`);
    }
    if (config.kind !== "seed") {
      if (nameSet.has(config.name)) {
        errors.push(`Duplicate node name: ${config.name}.`);
      }
      nameSet.add(config.name);
    }

    if (config.kind === "sampler") {
      nameToConfig.set(config.name, config);
      columns.push(buildSamplerColumn(config, errors));
      continue;
    }
    if (config.kind === "llm") {
      columns.push(buildLlmColumn(config, errors));
      for (const provider of config.mcp_providers ?? []) {
        const builtProvider = buildLlmMcpProvider(provider, errors);
        if (!builtProvider) {
          continue;
        }
        pushUniqueJson(
          "MCP provider",
          String(builtProvider.name),
          builtProvider,
          mcpProviderJsonByName,
          mcpProviders,
          errors,
        );
      }
      for (const toolConfig of config.tool_configs ?? []) {
        const builtToolConfig = buildLlmToolConfig(toolConfig, errors);
        if (!builtToolConfig) {
          continue;
        }
        pushUniqueJson(
          "Tool config",
          String(builtToolConfig.tool_alias),
          builtToolConfig,
          toolConfigJsonByAlias,
          toolConfigs,
          errors,
        );
      }
      if (config.model_alias) {
        modelAliases.add(config.model_alias);
      }
      const toolAlias = config.tool_alias?.trim();
      if (toolAlias) {
        llmToolAliasesUsed.add(toolAlias);
      }
      nameToConfig.set(config.name, config);
      continue;
    }
    if (config.kind === "expression") {
      columns.push(buildExpressionColumn(config, errors));
      nameToConfig.set(config.name, config);
      continue;
    }
    if (config.kind === "seed") {
      // SeedConfig is global config (seed_config); seed-dataset columns are added by DataDesigner.
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
  for (const toolAlias of llmToolAliasesUsed) {
    if (!toolConfigJsonByAlias.has(toolAlias)) {
      errors.push(`Tool alias ${toolAlias}: missing tool config.`);
    }
  }

  const uiNodes = nodes.flatMap((node) => {
    const config = configs[node.id];
    if (!config) {
      return [];
    }
    if (config.kind === "seed") {
      return [];
    }
    const width = readNodeWidth(node);
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
  const seedConfig = firstSeed ? buildSeedConfig(firstSeed, errors) : undefined;
  const seedDropProcessor = firstSeed
    ? buildSeedDropProcessor(firstSeed, errors)
    : null;
  if (seedDropProcessor) {
    recipeProcessors.push(seedDropProcessor);
  }

  return {
    errors,
    payload: {
      recipe: {
        // biome-ignore lint/style/useNamingConvention: api schema
        model_providers: modelProviders,
        // biome-ignore lint/style/useNamingConvention: api schema
        mcp_providers: mcpProviders,
        // biome-ignore lint/style/useNamingConvention: api schema
        model_configs: modelConfigs,
        // biome-ignore lint/style/useNamingConvention: api schema
        seed_config: seedConfig,
        // biome-ignore lint/style/useNamingConvention: api schema
        tool_configs: toolConfigs,
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
