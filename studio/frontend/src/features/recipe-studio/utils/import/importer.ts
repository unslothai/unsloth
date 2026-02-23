import type {
  LlmConfig,
  LlmMcpProviderConfig,
  LlmToolConfig,
  NodeConfig,
  RecipeProcessorConfig,
  SeedSourceType,
} from "../../types";
import { buildEdges } from "./edges";
import { isRecord, parseJson, readString } from "./helpers";
import {
  parseColumn,
  parseModelConfig,
  parseModelProvider,
} from "./parsers";
import { parseSeedConfig } from "./parsers/seed-config-parser";
import { buildNodes, parseUi } from "./ui";
import type { ImportResult } from "./types";

type RecipeInput = {
  columns?: unknown;
  model_configs?: unknown;
  model_providers?: unknown;
  mcp_providers?: unknown;
  tool_configs?: unknown;
  processors?: unknown;
  seed_config?: unknown;
};

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
  seed_source_type?: unknown;
  seed_columns?: unknown;
  seed_preview_rows?: unknown;
  local_file_name?: unknown;
  unstructured_file_name?: unknown;
  unstructured_chunk_size?: unknown;
  unstructured_chunk_overlap?: unknown;
};

function readStringNumber(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return undefined;
}

function parseProcessors(input: unknown): RecipeProcessorConfig[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const processors: RecipeProcessorConfig[] = [];
  input.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const type = readString(item.processor_type);
    const templateRaw = item.template;
    const isSchemaTransform =
      type === "schema_transform" || isRecord(templateRaw);
    if (!isSchemaTransform) {
      return;
    }
    const name = readString(item.name) ?? `schema_transform_${index + 1}`;
    const template =
      typeof templateRaw === "string"
        ? templateRaw
        : isRecord(templateRaw)
          ? JSON.stringify(templateRaw, null, 2)
          : "{\n  \"text\": \"{{ column_name }}\"\n}";
    processors.push({
      id: `p${index + 1}`,
      // biome-ignore lint/style/useNamingConvention: api schema
      processor_type: "schema_transform",
      name,
      template,
    });
  });
  return processors;
}

function parseMcpProviders(
  input: unknown,
): Map<string, LlmMcpProviderConfig> {
  const providers = new Map<string, LlmMcpProviderConfig>();
  if (!Array.isArray(input)) {
    return providers;
  }
  input.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const name = readString(item.name)?.trim();
    if (!name) {
      return;
    }
    const providerTypeRaw = readString(item.provider_type);
    const providerType =
      providerTypeRaw === "stdio" ? "stdio" : "streamable_http";
    const args = Array.isArray(item.args)
      ? item.args.map((value) => String(value))
      : [];
    const envPairs =
      isRecord(item.env)
        ? Object.entries(item.env).map(([key, value]) => ({
            key: String(key),
            value: String(value),
          }))
        : [];
    providers.set(name, {
      id: `mcp-${index + 1}`,
      name,
      // biome-ignore lint/style/useNamingConvention: ui schema
      provider_type: providerType,
      command: readString(item.command) ?? "",
      args,
      env: envPairs,
      endpoint: readString(item.endpoint) ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      api_key: readString(item.api_key) ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      api_key_env: readString(item.api_key_env) ?? "",
    });
  });
  return providers;
}

function parseToolConfigs(input: unknown): Map<string, LlmToolConfig> {
  const toolConfigs = new Map<string, LlmToolConfig>();
  if (!Array.isArray(input)) {
    return toolConfigs;
  }
  input.forEach((item, index) => {
    if (!isRecord(item)) {
      return;
    }
    const toolAlias = readString(item.tool_alias)?.trim();
    if (!toolAlias) {
      return;
    }
    const providers = Array.isArray(item.providers)
      ? item.providers.map((value) => String(value).trim()).filter(Boolean)
      : [];
    const allowTools = Array.isArray(item.allow_tools)
      ? item.allow_tools.map((value) => String(value).trim()).filter(Boolean)
      : [];
    toolConfigs.set(toolAlias, {
      id: `tool-${index + 1}`,
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_alias: toolAlias,
      providers,
      // biome-ignore lint/style/useNamingConvention: api schema
      allow_tools: allowTools,
      // biome-ignore lint/style/useNamingConvention: api schema
      max_tool_call_turns:
        item.max_tool_call_turns === null || item.max_tool_call_turns === undefined
          ? "5"
          : String(item.max_tool_call_turns),
      // biome-ignore lint/style/useNamingConvention: api schema
      timeout_sec:
        item.timeout_sec === null || item.timeout_sec === undefined
          ? ""
          : String(item.timeout_sec),
    });
  });
  return toolConfigs;
}

function cloneToolConfig(config: LlmToolConfig): LlmToolConfig {
  return {
    ...config,
    providers: [...config.providers],
    // biome-ignore lint/style/useNamingConvention: api schema
    allow_tools: [...(config.allow_tools ?? [])],
  };
}

function cloneMcpProvider(config: LlmMcpProviderConfig): LlmMcpProviderConfig {
  return {
    ...config,
    args: [...(config.args ?? [])],
    env: [...(config.env ?? [])],
  };
}

function attachLlmTooling(
  config: LlmConfig,
  toolConfigsByAlias: Map<string, LlmToolConfig>,
  mcpProvidersByName: Map<string, LlmMcpProviderConfig>,
): void {
  const toolAlias = config.tool_alias?.trim();
  if (!toolAlias) {
    config.tool_alias = "";
    config.tool_configs = [];
    config.mcp_providers = [];
    return;
  }
  const toolConfig = toolConfigsByAlias.get(toolAlias);
  if (!toolConfig) {
    config.tool_configs = [];
    config.mcp_providers = [];
    return;
  }
  config.tool_configs = [cloneToolConfig(toolConfig)];
  config.mcp_providers = toolConfig.providers
    .map((providerName) => mcpProvidersByName.get(providerName))
    .flatMap((provider) => (provider ? [cloneMcpProvider(provider)] : []));
}

export function importRecipePayload(input: string): ImportResult {
  const parsed = parseJson(input);
  if (!parsed.data || !isRecord(parsed.data)) {
    return {
      errors: [parsed.error ?? "Invalid JSON payload."],
      snapshot: null,
    };
  }

  const recipe = (isRecord(parsed.data.recipe)
    ? parsed.data.recipe
    : parsed.data) as RecipeInput;
  const ui = isRecord(parsed.data.ui) ? (parsed.data.ui as UiInput) : null;

  if (!Array.isArray(recipe.columns)) {
    return { errors: ["Recipe must include columns."], snapshot: null };
  }

  const errors: string[] = [];
  const configs: NodeConfig[] = [];
  const processors = parseProcessors(recipe.processors);
  const mcpProvidersByName = parseMcpProviders(recipe.mcp_providers);
  const toolConfigsByAlias = parseToolConfigs(recipe.tool_configs);
  const nameToId = new Map<string, string>();

  let nextId = 1;
  const uiSeedSourceTypeRaw = readString(ui?.seed_source_type);
  const uiSeedSourceType: SeedSourceType | undefined =
    uiSeedSourceTypeRaw === "hf" ||
    uiSeedSourceTypeRaw === "local" ||
    uiSeedSourceTypeRaw === "unstructured"
      ? uiSeedSourceTypeRaw
      : undefined;
  const uiSeedColumns = Array.isArray(ui?.seed_columns)
    ? ui.seed_columns
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .filter(Boolean)
    : undefined;
  const uiSeedPreviewRows = Array.isArray(ui?.seed_preview_rows)
    ? ui.seed_preview_rows
        .filter((row): row is Record<string, unknown> => isRecord(row))
        .map((row) => ({ ...row }))
    : undefined;
  const uiLocalFileName = readString(ui?.local_file_name) ?? undefined;
  const uiUnstructuredFileName =
    readString(ui?.unstructured_file_name) ?? undefined;
  const uiUnstructuredChunkSize = readStringNumber(ui?.unstructured_chunk_size);
  const uiUnstructuredChunkOverlap = readStringNumber(
    ui?.unstructured_chunk_overlap,
  );

  if (recipe.seed_config) {
    const id = `n${nextId}`;
    nextId += 1;
    const seedConfig = parseSeedConfig(recipe.seed_config, id, {
      preferredSourceType: uiSeedSourceType,
      seed_columns: uiSeedColumns,
      seed_preview_rows: uiSeedPreviewRows,
      local_file_name: uiLocalFileName,
      unstructured_file_name: uiUnstructuredFileName,
      unstructured_chunk_size: uiUnstructuredChunkSize,
      unstructured_chunk_overlap: uiUnstructuredChunkOverlap,
    });
    if (seedConfig) {
      if (nameToId.has(seedConfig.name)) {
        errors.push(`Duplicate column name: ${seedConfig.name}.`);
      } else {
        nameToId.set(seedConfig.name, seedConfig.id);
      }
      configs.push(seedConfig);
    }
  }

  if (Array.isArray(recipe.model_providers)) {
    recipe.model_providers.forEach((provider, index) => {
      if (!isRecord(provider)) {
        errors.push(`Model provider ${index + 1}: invalid object.`);
        return;
      }
      const name = readString(provider.name);
      if (!name) {
        errors.push(`Model provider ${index + 1}: missing name.`);
        return;
      }
      const id = `n${nextId}`;
      nextId += 1;
      const config = parseModelProvider(provider, name, id);
      if (nameToId.has(config.name)) {
        errors.push(`Duplicate column name: ${config.name}.`);
        return;
      }
      nameToId.set(config.name, config.id);
      configs.push(config);
    });
  }

  if (Array.isArray(recipe.model_configs)) {
    recipe.model_configs.forEach((model, index) => {
      if (!isRecord(model)) {
        errors.push(`Model config ${index + 1}: invalid object.`);
        return;
      }
      const name = readString(model.alias) ?? readString(model.name);
      if (!name) {
        errors.push(`Model config ${index + 1}: missing alias.`);
        return;
      }
      const id = `n${nextId}`;
      nextId += 1;
      const config = parseModelConfig(model, name, id);
      if (nameToId.has(config.name)) {
        errors.push(`Duplicate column name: ${config.name}.`);
        return;
      }
      nameToId.set(config.name, config.id);
      configs.push(config);
    });
  }

  recipe.columns.forEach((column, index) => {
    if (!isRecord(column)) {
      errors.push(`Column ${index + 1}: invalid object.`);
      return;
    }
    const id = `n${nextId}`;
    nextId += 1;
    const config = parseColumn(column, id, errors);
    if (!config) {
      return;
    }
    if (config.kind === "llm") {
      attachLlmTooling(config, toolConfigsByAlias, mcpProvidersByName);
    }
    if (nameToId.has(config.name)) {
      errors.push(`Duplicate column name: ${config.name}.`);
      return;
    }
    nameToId.set(config.name, config.id);
    configs.push(config);
  });

  if (errors.length > 0) {
    return { errors, snapshot: null };
  }

  const { layouts, edges: uiEdges } = parseUi(ui);
  const nodes = buildNodes(configs, layouts);
  const edges = buildEdges(configs, nameToId, uiEdges);

  const maxY = nodes.reduce(
    (acc, node) => Math.max(acc, node.position.y),
    0,
  );

  return {
    errors: [],
    snapshot: {
      configs: Object.fromEntries(configs.map((config) => [config.id, config])),
      nodes,
      edges,
      processors,
      nextId,
      nextY: maxY + 140,
    },
  };
}
