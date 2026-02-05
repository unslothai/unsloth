import type { Edge } from "@xyflow/react";
import type {
  CanvasNode,
  ExpressionConfig,
  LlmConfig,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
  SamplerConfig,
} from "../types";
import { getConfigErrors } from "./index";

type CanvasPayload = {
  recipe: {
    // biome-ignore lint/style/useNamingConvention: api schema
    model_providers: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    model_configs: Record<string, unknown>[];
    columns: Record<string, unknown>[];
    processors: Record<string, unknown>[];
  };
  run: {
    rows: number;
    preview: boolean;
    // biome-ignore lint/style/useNamingConvention: api schema
    output_formats: string[];
  };
  ui: {
    nodes: { id: string; x: number; y: number }[];
    edges: { from: string; to: string; type?: string }[];
  };
};

export type CanvasPayloadResult = {
  errors: string[];
  payload: CanvasPayload;
};

function isSemanticRelation(
  source: NodeConfig,
  target: NodeConfig,
): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") {
    return true;
  }
  if (source.kind === "model_config" && target.kind === "llm") {
    return true;
  }
  return (
    source.kind === "sampler" &&
    source.sampler_type === "category" &&
    target.kind === "sampler" &&
    target.sampler_type === "subcategory"
  );
}

function parseNumber(value?: string): number | null {
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function parseAgeRange(value?: string): [number, number] | null {
  if (!value) {
    return null;
  }
  const parts = value.split(/[^0-9.]+/).filter(Boolean);
  if (parts.length !== 2) {
    return null;
  }
  const min = Number(parts[0]);
  const max = Number(parts[1]);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }
  return [min, max];
}

function parseJsonObject(
  value: string | undefined,
  label: string,
  errors: string[],
): Record<string, unknown> | undefined {
  if (!value || !value.trim()) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(value);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    errors.push(`${label}: invalid JSON.`);
    return undefined;
  }
  errors.push(`${label}: must be a JSON object.`);
  return undefined;
}

function buildModelProvider(
  config: ModelProviderConfig,
  errors: string[],
): Record<string, unknown> {
  const extraHeaders = parseJsonObject(
    config.extra_headers,
    `Provider ${config.name} extra_headers`,
    errors,
  );
  const extraBody = parseJsonObject(
    config.extra_body,
    `Provider ${config.name} extra_body`,
    errors,
  );
  return {
    name: config.name,
    endpoint: config.endpoint,
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: config.provider_type,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: config.api_key_env?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: config.api_key?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_headers: extraHeaders ?? {},
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_body: extraBody ?? {},
  };
}

function buildModelConfig(config: ModelConfig): Record<string, unknown> {
  const inference: Record<string, unknown> = {};
  const temp = config.inference_temperature?.trim();
  const topP = config.inference_top_p?.trim();
  const maxTokens = config.inference_max_tokens?.trim();
  if (temp) {
    const parsed = Number(temp);
    if (Number.isFinite(parsed)) {
      inference.temperature = parsed;
    }
  }
  if (topP) {
    const parsed = Number(topP);
    if (Number.isFinite(parsed)) {
      // biome-ignore lint/style/useNamingConvention: api schema
      inference.top_p = parsed;
    }
  }
  if (maxTokens) {
    const parsed = Number(maxTokens);
    if (Number.isFinite(parsed)) {
      // biome-ignore lint/style/useNamingConvention: api schema
      inference.max_tokens = parsed;
    }
  }
  return {
    alias: config.name,
    model: config.model,
    provider: config.provider || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters:
      Object.keys(inference).length > 0 ? inference : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check: config.skip_health_check || undefined,
  };
}

function isValidSex(value?: string): value is "Male" | "Female" {
  if (!value) {
    return false;
  }
  return value === "Male" || value === "Female";
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: per type logic
function buildSamplerParams(
  config: SamplerConfig,
  errors: string[],
): Record<string, unknown> {
  if (config.sampler_type === "category") {
    const values = config.values ?? [];
    const params: Record<string, unknown> = { values };
    const weights = config.weights ?? [];
    const hasWeights = weights.some((weight) => weight !== null);
    if (hasWeights && weights.some((weight) => weight === null)) {
      errors.push(`Sampler ${config.name}: weights missing values.`);
    } else if (hasWeights) {
      params.weights = weights.filter((weight) => weight !== null);
    }
    return params;
  }
  if (config.sampler_type === "subcategory") {
    const mapping = config.subcategory_mapping ?? {};
    for (const [key, values] of Object.entries(mapping)) {
      if (!values || values.length === 0) {
        errors.push(
          `Subcategory ${config.name}: '${key}' needs at least 1 subcategory.`,
        );
      }
    }
    return {
      category: config.subcategory_parent,
      values: mapping,
    };
  }
  if (config.sampler_type === "uniform") {
    return {
      low: parseNumber(config.low),
      high: parseNumber(config.high),
    };
  }
  if (config.sampler_type === "gaussian") {
    return {
      mean: parseNumber(config.mean),
      std: parseNumber(config.std),
    };
  }
  if (config.sampler_type === "datetime") {
    return {
      start: config.datetime_start ?? undefined,
      end: config.datetime_end ?? undefined,
      unit: config.datetime_unit ?? undefined,
    };
  }
  if (config.sampler_type === "uuid") {
    return {
      format: config.uuid_format ?? undefined,
    };
  }
  const params: Record<string, unknown> = {};
  if (config.person_locale?.trim()) {
    params.locale = config.person_locale.trim();
  }
  if (config.sampler_type === "person") {
    if (isValidSex(config.person_sex?.trim())) {
      params.sex = config.person_sex?.trim();
    } else if (config.person_sex?.trim()) {
      errors.push(`Person ${config.name}: sex must be Male or Female.`);
    }
  } else if (config.person_sex?.trim()) {
    params.sex = config.person_sex.trim();
  }
  if (config.person_city?.trim()) {
    params.city = config.person_city.trim();
  }
  if (config.person_age_range?.trim()) {
    const parsed = parseAgeRange(config.person_age_range);
    if (parsed) {
      // biome-ignore lint/style/useNamingConvention: api schema
      params.age_range = parsed;
    } else {
      errors.push(`Person ${config.name}: age range must be like 18-70.`);
    }
  }
  if (config.sampler_type === "person") {
    // biome-ignore lint/style/useNamingConvention: api schema
    params.with_synthetic_personas =
      config.person_with_synthetic_personas ?? undefined;
  }
  return params;
}

function buildLlmColumn(
  config: LlmConfig,
  errors: string[],
): Record<string, unknown> {
  const base = {
    name: config.name,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: config.model_alias,
    prompt: config.prompt,
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: config.system_prompt || undefined,
  };

  if (config.llm_type === "code") {
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-code",
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      code_lang: config.code_lang || "python",
    };
  }
  if (config.llm_type === "structured") {
    let outputFormat: unknown = config.output_format || undefined;
    if (typeof outputFormat === "string" && outputFormat.trim()) {
      try {
        outputFormat = JSON.parse(outputFormat);
      } catch {
        errors.push(`LLM ${config.name}: output_format is not valid JSON.`);
      }
    }
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-structured",
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_format: outputFormat,
    };
  }
  if (config.llm_type === "judge") {
    const scores = (config.scores ?? [])
      .map((score) => {
        const options: Record<string, string> = {};
        for (const option of score.options ?? []) {
          const key = option.value.trim();
          const value = option.description.trim();
          if (!key || !value) {
            continue;
          }
          options[key] = value;
        }
        return {
          name: score.name.trim(),
          description: score.description.trim(),
          options,
        };
      })
      .filter(
        (score) =>
          score.name && score.description && Object.keys(score.options).length > 0,
      );
    if (scores.length === 0) {
      errors.push(`LLM ${config.name}: scores required for LLM Judge.`);
    }
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-judge",
      ...base,
      scores,
    };
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "llm-text",
    ...base,
    // biome-ignore lint/style/useNamingConvention: api schema
    with_trace: false,
  };
}

function buildExpressionColumn(
  config: ExpressionConfig,
  errors: string[],
): Record<string, unknown> {
  if (!config.expr.trim()) {
    errors.push(`Expression ${config.name}: expr required.`);
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "expression",
    name: config.name,
    expr: config.expr,
    dtype: config.dtype,
  };
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: payload build
export function buildCanvasPayload(
  configs: Record<string, NodeConfig>,
  nodes: CanvasNode[],
  edges: Edge[],
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
      columns.push({
        // biome-ignore lint/style/useNamingConvention: api schema
        column_type: "sampler",
        name: config.name,
        // biome-ignore lint/style/useNamingConvention: api schema
        sampler_type: config.sampler_type,
        params: buildSamplerParams(config, errors),
        // biome-ignore lint/style/useNamingConvention: api schema
        convert_to: config.convert_to ?? undefined,
      });
    } else if (config.kind === "llm") {
      columns.push(buildLlmColumn(config, errors));
      if (config.model_alias) {
        modelAliases.add(config.model_alias);
      }
      nameToConfig.set(config.name, config);
    } else if (config.kind === "expression") {
      columns.push(buildExpressionColumn(config, errors));
      nameToConfig.set(config.name, config);
    } else if (config.kind === "model_provider") {
      modelProviderNames.add(config.name);
      modelProviders.push(buildModelProvider(config, errors));
      modelProviderConfigs.push(config);
    } else if (config.kind === "model_config") {
      modelConfigs.push(buildModelConfig(config));
      modelConfigConfigs.push(config);
    }
  }

  for (const config of Object.values(configs)) {
    if (config.kind !== "sampler" || config.sampler_type !== "subcategory") {
      continue;
    }
    const parentName = config.subcategory_parent;
    if (!parentName) {
      errors.push(`Subcategory ${config.name}: parent category required.`);
      continue;
    }
    const parent = nameToConfig.get(parentName);
    const parentValues =
      parent && parent.kind === "sampler" && parent.sampler_type === "category"
        ? (parent.values ?? [])
        : [];
    const mapping = config.subcategory_mapping ?? {};
    for (const value of parentValues) {
      const list = mapping[value];
      if (!list || list.length === 0) {
        errors.push(
          `Subcategory ${config.name}: '${value}' needs at least 1 subcategory.`,
        );
      }
    }
  }

  for (const alias of modelAliases) {
    if (
      !modelConfigs.some(
        (config) => (config.alias as string | undefined) === alias,
      )
    ) {
      errors.push(`LLM model_alias ${alias}: missing model config.`);
    }
  }

  for (const config of modelConfigConfigs) {
    const provider = config.provider.trim();
    const alias = config.name;
    if (modelAliases.has(alias) && !config.model.trim()) {
      errors.push(`Model config ${alias}: model is required.`);
    }
    if (provider && !modelProviderNames.has(provider)) {
      errors.push(`Model config ${alias}: provider ${provider} not found.`);
    }
  }

  const usedProviders = new Set(
    modelConfigConfigs.map((config) => config.provider.trim()).filter(Boolean),
  );
  for (const provider of modelProviderConfigs) {
    if (!usedProviders.has(provider.name)) {
      continue;
    }
    if (!provider.endpoint.trim()) {
      errors.push(`Model provider ${provider.name}: endpoint is required.`);
    }
    if (!provider.provider_type.trim()) {
      errors.push(`Model provider ${provider.name}: provider_type is required.`);
    }
  }

  const uiNodes = nodes.flatMap((node) => {
    const config = configs[node.id];
    if (!config) {
      return [];
    }
    return [
      {
        id: config.name,
        x: node.position.x,
        y: node.position.y,
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

  return {
    errors,
    payload: {
      recipe: {
        // biome-ignore lint/style/useNamingConvention: api schema
        model_providers: modelProviders,
        // biome-ignore lint/style/useNamingConvention: api schema
        model_configs: modelConfigs,
        columns,
        processors: [],
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
