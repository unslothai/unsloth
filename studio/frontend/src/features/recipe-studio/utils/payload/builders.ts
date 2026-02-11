import type {
  RecipeProcessorConfig,
  CategoryConditionalParams,
  ExpressionConfig,
  LlmConfig,
  ModelConfig,
  ModelProviderConfig,
  SamplerConfig,
} from "../../types";
import {
  isValidSex,
  parseAgeRange,
  parseJsonObject,
  parseNumber,
} from "./parse";

function buildCategoryConditionalParams(
  config: SamplerConfig,
  errors: string[],
): Record<string, CategoryConditionalParams> | undefined {
  const conditional = config.conditional_params ?? {};
  const output: Record<string, CategoryConditionalParams> = {};
  for (const [rawCondition, params] of Object.entries(conditional)) {
    const condition = rawCondition.trim();
    if (!condition) {
      errors.push(`Sampler ${config.name}: conditional rule needs condition text.`);
      continue;
    }
    const values = (params.values ?? [])
      .map((value) => value.trim())
      .filter(Boolean);
    if (values.length === 0) {
      errors.push(`Sampler ${config.name}: conditional '${condition}' needs values.`);
      continue;
    }
    const weights = params.weights ?? [];
    const hasWeights = weights.some((weight) => weight !== null);
    if (
      hasWeights &&
      (weights.length !== values.length || weights.some((weight) => weight === null))
    ) {
      errors.push(`Sampler ${config.name}: conditional '${condition}' weights invalid.`);
      continue;
    }
    output[condition] = {
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "category",
      values,
      weights: hasWeights
        ? weights.filter((weight): weight is number => weight !== null)
        : undefined,
    };
  }
  return Object.keys(output).length > 0 ? output : undefined;
}

export function buildModelProvider(
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

export function buildModelConfig(config: ModelConfig): Record<string, unknown> {
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
  if (config.sampler_type === "bernoulli") {
    return {
      p: parseNumber(config.p),
    };
  }
  if (config.sampler_type === "datetime") {
    return {
      start: config.datetime_start ?? undefined,
      end: config.datetime_end ?? undefined,
      unit: config.datetime_unit ?? undefined,
    };
  }
  if (config.sampler_type === "timedelta") {
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      dt_min: parseNumber(config.dt_min),
      // biome-ignore lint/style/useNamingConvention: api schema
      dt_max: parseNumber(config.dt_max),
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: config.reference_column_name || undefined,
      unit: config.timedelta_unit || undefined,
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

export function buildSamplerColumn(
  config: SamplerConfig,
  errors: string[],
): Record<string, unknown> {
  const samplerColumn: Record<string, unknown> = {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "sampler",
    name: config.name,
    drop: config.drop ?? false,
    // biome-ignore lint/style/useNamingConvention: api schema
    sampler_type: config.sampler_type,
    params: buildSamplerParams(config, errors),
    // biome-ignore lint/style/useNamingConvention: api schema
    convert_to: config.convert_to ?? undefined,
  };
  if (config.sampler_type === "category") {
    const conditionalParams = buildCategoryConditionalParams(config, errors);
    if (conditionalParams) {
      // biome-ignore lint/style/useNamingConvention: api schema
      samplerColumn.conditional_params = conditionalParams;
    }
  }
  return samplerColumn;
}

export function buildLlmColumn(
  config: LlmConfig,
  errors: string[],
): Record<string, unknown> {
  const base = {
    name: config.name,
    drop: config.drop ?? false,
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

export function buildExpressionColumn(
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
    drop: config.drop ?? false,
    expr: config.expr,
    dtype: config.dtype,
  };
}

export function buildProcessors(
  processors: RecipeProcessorConfig[],
  errors: string[],
): Record<string, unknown>[] {
  const output: Record<string, unknown>[] = [];
  for (const processor of processors) {
    if (processor.processor_type !== "schema_transform") {
      continue;
    }
    const name = processor.name.trim();
    if (!name) {
      errors.push("Schema transform: name is required.");
      continue;
    }
    const template = parseJsonObject(
      processor.template,
      `Schema transform ${name} template`,
      errors,
    );
    if (!template) {
      continue;
    }
    output.push({
      // biome-ignore lint/style/useNamingConvention: api schema
      processor_type: "schema_transform",
      name,
      // biome-ignore lint/style/useNamingConvention: api schema
      build_stage: "post_batch",
      template,
    });
  }
  return output;
}
