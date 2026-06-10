// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ExpressionConfig,
  LlmConfig,
  LlmType,
  MarkdownNoteConfig,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
  SeedConfig,
  SeedSourceType,
  SamplerConfig,
  SamplerType,
  ToolProfileConfig,
  ValidatorCodeLang,
  ValidatorType,
  ValidatorConfig,
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
      values: [],
      weights: [],
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
      subcategory_mapping: {},
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
  if (samplerType === "person" || samplerType === "person_from_faker") {
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
    model_alias: "",
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
    // biome-ignore lint/style/useNamingConvention: ui schema
    image_context: {
      enabled: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      column_name: "",
    },
    // biome-ignore lint/style/useNamingConvention: api schema
    with_trace: "none",
    // biome-ignore lint/style/useNamingConvention: api schema
    extract_reasoning_content: false,
    scores: llmType === "judge" ? [] : undefined,
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
    // biome-ignore lint/style/useNamingConvention: api schema
    is_local: false,
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
    inference_max_tokens: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_top_p: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_timeout: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_extra_body: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check: false,
  };
}

export function makeToolProfileConfig(
  id: string,
  existing: NodeConfig[],
): ToolProfileConfig {
  return {
    id,
    kind: "tool_config",
    name: nextName(existing, "tools"),
    // biome-ignore lint/style/useNamingConvention: ui schema
    mcp_providers: [],
    // biome-ignore lint/style/useNamingConvention: ui schema
    fetched_tools_by_provider: {},
    // biome-ignore lint/style/useNamingConvention: api schema
    allow_tools: [],
    // biome-ignore lint/style/useNamingConvention: api schema
    max_tool_call_turns: "5",
    // biome-ignore lint/style/useNamingConvention: api schema
    timeout_sec: "",
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

export function makeValidatorConfig(
  id: string,
  validatorType: ValidatorType,
  codeLang: ValidatorCodeLang,
  existing: NodeConfig[],
): ValidatorConfig {
  const isSql = validatorType === "code" && codeLang.startsWith("sql:");
  const isOxc = validatorType === "oxc";
  let namePrefix = "validator_python";
  if (isSql) {
    namePrefix = "validator_sql";
  } else if (isOxc) {
    namePrefix = "validator_oxc";
  }
  return {
    id,
    kind: "validator",
    name: nextName(existing, namePrefix),
    drop: false,
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: [],
    validator_type: validatorType,
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: codeLang,
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    batch_size: "10",
  };
}

export function makeMarkdownNoteConfig(
  id: string,
  existing: NodeConfig[],
): MarkdownNoteConfig {
  return {
    id,
    kind: "markdown_note",
    name: nextName(existing, "note"),
    markdown: "## Note\n\nAdd markdown here.",
    note_color: "#FDE68A",
    note_opacity: "35",
  };
}

export function makeSeedConfig(
  id: string,
  existing: NodeConfig[],
  seedSourceType: SeedSourceType = "hf",
): SeedConfig {
  return {
    id,
    kind: "seed",
    name: nextName(existing, "seed"),
    drop: false,
    seed_drop_columns: [],
    seed_source_type: seedSourceType,
    hf_repo_id: "",
    hf_subset: "",
    hf_split: "",
    hf_path: "",
    hf_token: "",
    hf_endpoint: "https://huggingface.co",
    local_file_name: "",
    unstructured_file_ids: [],
    unstructured_file_names: [],
    unstructured_file_sizes: [],
    seed_preview_rows: [],
    unstructured_chunk_size: "1200",
    unstructured_chunk_overlap: "200",
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
