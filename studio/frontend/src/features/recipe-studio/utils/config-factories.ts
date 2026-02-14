import type {
  ExpressionConfig,
  LlmConfig,
  LlmType,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
  SeedConfig,
  SamplerConfig,
  SamplerType,
} from "../types";
import { nextName } from "./naming";

export function makeSamplerConfig(
  id: string,
  samplerType: SamplerType,
  existing: NodeConfig[],
): SamplerConfig {
  const namePrefix =
    samplerType === "subcategory" ? "subcategory" : samplerType;
  const name = nextName(existing, namePrefix);
  if (samplerType === "category") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "category",
      name,
      drop: false,
      values: ["A", "B", "C"],
      weights: [null, null, null],
    };
  }
  if (samplerType === "subcategory") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "subcategory",
      name,
      drop: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: {
        // biome-ignore lint/style/useNamingConvention: sample values
        A: ["A1", "A2"],
        // biome-ignore lint/style/useNamingConvention: sample values
        B: ["B1", "B2"],
      },
    };
  }
  if (samplerType === "uniform") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "uniform",
      name,
      drop: false,
      low: "0",
      high: "1",
    };
  }
  if (samplerType === "gaussian") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "gaussian",
      name,
      drop: false,
      mean: "0",
      std: "1",
    };
  }
  if (samplerType === "bernoulli") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "bernoulli",
      name,
      drop: false,
      p: "0.5",
    };
  }
  if (samplerType === "datetime") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "datetime",
      name,
      drop: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_start: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_end: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_unit: "day",
    };
  }
  if (samplerType === "timedelta") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "timedelta",
      name,
      drop: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      dt_min: "0",
      // biome-ignore lint/style/useNamingConvention: api schema
      dt_max: "1",
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      timedelta_unit: "D",
    };
  }
  if (samplerType === "uuid") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "uuid",
      name,
      drop: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      uuid_format: "",
    };
  }
  if (samplerType === "person_from_faker") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "person_from_faker",
      name,
      drop: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      person_locale: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      person_sex: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      person_age_range: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      person_city: "",
    };
  }
  return {
    id,
    kind: "sampler",
    // biome-ignore lint/style/useNamingConvention: api schema
    sampler_type: "person",
    name,
    drop: false,
    // biome-ignore lint/style/useNamingConvention: api schema
    person_locale: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_sex: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_age_range: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_city: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_with_synthetic_personas: false,
  };
}

export function makeLlmConfig(
  id: string,
  llmType: LlmType,
  existing: NodeConfig[],
): LlmConfig {
  let namePrefix = "llm_text";
  if (llmType === "structured") {
    namePrefix = "llm_structured";
  } else if (llmType === "code") {
    namePrefix = "llm_code";
  } else if (llmType === "judge") {
    namePrefix = "llm_judge";
  }
  const name = nextName(existing, namePrefix);
  return {
    id,
    kind: "llm",
    // biome-ignore lint/style/useNamingConvention: api schema
    llm_type: llmType,
    name,
    drop: false,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: "allenai/olmo-3.1-32b-instruct",
    prompt:
      llmType === "judge"
        ? "Evaluate the content using the scoring criteria below."
        : "Write a response.",
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: llmType === "code" ? "python" : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    output_format:
      llmType === "structured" ? '{\n  "field": "string"\n}' : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_alias: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_configs: [],
    // biome-ignore lint/style/useNamingConvention: ui schema
    mcp_providers: [],
    scores:
      llmType === "judge"
        ? [
            {
              name: "Quality",
              description: "Overall quality based on the criteria.",
              options: [
                { value: "1", description: "Poor" },
                { value: "3", description: "Acceptable" },
                { value: "5", description: "Excellent" },
              ],
            },
          ]
        : undefined,
  };
}

export function makeModelProviderConfig(
  id: string,
  existing: NodeConfig[],
): ModelProviderConfig {
  return {
    id,
    kind: "model_provider",
    name: nextName(existing, "provider"),
    endpoint: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: "openai",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_headers: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_body: "",
  };
}

export function makeModelConfig(
  id: string,
  existing: NodeConfig[],
): ModelConfig {
  return {
    id,
    kind: "model_config",
    name: nextName(existing, "model"),
    model: "",
    provider: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_temperature: "0.7",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_max_tokens: "256",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_top_p: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check: false,
  };
}

export function makeExpressionConfig(
  id: string,
  existing: NodeConfig[],
): ExpressionConfig {
  return {
    id,
    kind: "expression",
    name: nextName(existing, "expr"),
    drop: false,
    expr: "",
    dtype: "str",
  };
}

export function makeSeedConfig(
  id: string,
  existing: NodeConfig[],
): SeedConfig {
  return {
    id,
    kind: "seed",
    name: nextName(existing, "seed"),
    drop: false,
    hf_url: "",
    hf_repo_id: "",
    hf_split: "",
    hf_path: "",
    hf_token: "",
    hf_endpoint: "https://huggingface.co",
    seed_splits: [],
    seed_globs_by_split: {},
    seed_columns: [],
    sampling_strategy: "ordered",
    selection_type: "none",
    selection_start: "0",
    selection_end: "10",
    selection_index: "0",
    selection_num_partitions: "1",
  };
}
