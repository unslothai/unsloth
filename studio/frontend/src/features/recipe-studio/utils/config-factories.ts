// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  EvaluationDocumentScoreConfig,
  ExpressionConfig,
  LlmConfig,
  LlmType,
  MarkdownNoteConfig,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
  SamplerConfig,
  SamplerType,
  SeedConfig,
  SeedSourceType,
  ToolProfileConfig,
  ValidatorCodeLang,
  ValidatorConfig,
  ValidatorType,
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
      sampler_type: "subcategory",
      name,
      drop: false,
      subcategory_parent: "",
      subcategory_mapping: {},
    };
  }
  if (samplerType === "uniform") {
    return {
      id,
      kind: "sampler",
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
      sampler_type: "datetime",
      name,
      drop: false,
      datetime_start: "",
      datetime_end: "",
      datetime_unit: "day",
    };
  }
  if (samplerType === "timedelta") {
    return {
      id,
      kind: "sampler",
      sampler_type: "timedelta",
      name,
      drop: false,
      dt_min: "0",
      dt_max: "1",
      reference_column_name: "",
      timedelta_unit: "D",
    };
  }
  if (samplerType === "uuid") {
    return {
      id,
      kind: "sampler",
      sampler_type: "uuid",
      name,
      drop: false,
      uuid_format: "",
    };
  }
  if (samplerType === "person" || samplerType === "person_from_faker") {
    return {
      id,
      kind: "sampler",
      sampler_type: "person_from_faker",
      name,
      drop: false,
      person_locale: "",
      person_sex: "",
      person_age_range: "",
      person_city: "",
    };
  }
  return {
    id,
    kind: "sampler",
    sampler_type: "person_from_faker",
    name,
    drop: false,
    person_locale: "",
    person_sex: "",
    person_age_range: "",
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
    llm_type: llmType,
    name,
    drop: false,
    model_alias: "",
    prompt:
      llmType === "judge"
        ? "Evaluate the content using the scoring criteria below."
        : "Write a response.",
    system_prompt: "",
    code_lang: llmType === "code" ? "python" : undefined,
    output_format:
      llmType === "structured" ? '{\n  "field": "string"\n}' : undefined,
    tool_alias: "",
    image_context: {
      enabled: false,
      column_name: "",
    },
    with_trace: "none",
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
    provider_type: "openai",
    api_key_env: "",
    api_key: "",
    extra_headers: "",
    extra_body: "",
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
    inference_temperature: "0.7",
    inference_max_tokens: "",
    inference_top_p: "",
    inference_timeout: "",
    inference_extra_body: "",
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
    mcp_providers: [],
    fetched_tools_by_provider: {},
    allow_tools: [],
    max_tool_call_turns: "5",
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
    target_columns: [],
    validator_type: validatorType,
    code_lang: codeLang,
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    batch_size: "10",
  };
}

export function makeEvaluationDocumentScoreConfig(
  id: string,
  existing: NodeConfig[],
): EvaluationDocumentScoreConfig {
  return {
    id,
    kind: "evaluation",
    evaluation_type: "json_document_score",
    name: nextName(existing, "doc_score"),
    prediction_column: "",
    reference_column: "",
    schema: "",
    default_comparator: "string",
    score_column: "doc_score",
    breakdown_column: "",
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
