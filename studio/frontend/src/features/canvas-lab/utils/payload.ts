import type { Edge } from "@xyflow/react";
import type {
  CanvasNode,
  ExpressionConfig,
  LlmConfig,
  NodeConfig,
  SamplerConfig,
} from "../types";
import { getConfigErrors } from "./index";

const DEFAULT_PROVIDER = {
  name: "openrouter",
  // biome-ignore lint/style/useNamingConvention: api schema
  provider_type: "openai",
  endpoint: "https://openrouter.ai/api/v1",
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key_env: "OPENROUTER_API_KEY",
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_headers: {},
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_body: {},
};

const DEFAULT_CONFIG = {
  provider: "openrouter",
  model: "stepfun/step-3.5-flash:free",
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_parameters: {
    temperature: 0.7,
    // biome-ignore lint/style/useNamingConvention: api schema
    max_tokens: 256,
  },
};

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
    edges: { from: string; to: string }[];
  };
};

export type CanvasPayloadResult = {
  errors: string[];
  payload: CanvasPayload;
};

function parseNumber(value?: string): number | null {
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
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
  return {
    locale: config.person_locale ?? undefined,
    sex: config.person_sex ?? undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    age_range: config.person_age_range ?? undefined,
    city: config.person_city ?? undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    with_synthetic_personas: config.person_with_synthetic_personas ?? undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    sample_dataset_when_available:
      config.person_sample_dataset_when_available ?? undefined,
  };
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
      errors.push(`Duplicate column name: ${config.name}.`);
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
      });
    } else if (config.kind === "llm") {
      columns.push(buildLlmColumn(config, errors));
      if (config.model_alias) {
        modelAliases.add(config.model_alias);
      }
      nameToConfig.set(config.name, config);
    } else {
      columns.push(buildExpressionColumn(config, errors));
      nameToConfig.set(config.name, config);
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

  const modelProviders = modelAliases.size > 0 ? [DEFAULT_PROVIDER] : [];
  const modelConfigs =
    modelAliases.size > 0
      ? Array.from(modelAliases).map((alias) => ({
          alias,
          ...DEFAULT_CONFIG,
        }))
      : [];

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
