// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LlmConfig,
  Score,
  ScoreOption,
} from "../../../types";
import {
  isRecord,
  normalizeOutputFormat,
  readString,
} from "../helpers";

function parseTraceMode(value: unknown): LlmConfig["with_trace"] {
  const traceRaw = readString(value) ?? "none";
  if (traceRaw === "last_message" || traceRaw === "all_messages") {
    return traceRaw;
  }
  return "none";
}

export function parseLlm(
  column: Record<string, unknown>,
  name: string,
  id: string,
): LlmConfig {
  const columnType = readString(column.column_type) ?? "llm-text";
  let llmType: LlmConfig["llm_type"] = "text";
  if (columnType === "llm-structured") {
    llmType = "structured";
  } else if (columnType === "llm-code") {
    llmType = "code";
  } else if (columnType === "llm-judge") {
    llmType = "judge";
  }

  const scores: Score[] =
    columnType === "llm-judge" && Array.isArray(column.scores)
      ? column.scores
          .filter((score) => isRecord(score))
          .map((score) => {
            const options: ScoreOption[] = [];
            const rawOptions = isRecord(score.options) ? score.options : {};
            for (const [key, value] of Object.entries(rawOptions)) {
              const description =
                typeof value === "string" ? value : JSON.stringify(value);
              options.push({ value: String(key), description });
            }
            return {
              name: readString(score.name) ?? "",
              description: readString(score.description) ?? "",
              options,
            };
          })
      : [];

  let imageContext: LlmConfig["image_context"] = {
    enabled: false,
    // biome-ignore lint/style/useNamingConvention: api schema
    column_name: "",
  };
  if (Array.isArray(column.multi_modal_context)) {
    const first = column.multi_modal_context.find((entry) => isRecord(entry));
    if (first && isRecord(first)) {
      const modality = readString(first.modality);
      const columnName = readString(first.column_name) ?? "";
      if (modality === "image" && columnName) {
        imageContext = {
          enabled: true,
          // biome-ignore lint/style/useNamingConvention: api schema
          column_name: columnName,
        };
      }
    }
  }

  const withTrace = parseTraceMode(column.with_trace);
  const extractReasoningContent = column.extract_reasoning_content === true;

  return {
    id,
    kind: "llm",
    // biome-ignore lint/style/useNamingConvention: api schema
    llm_type: llmType,
    name,
    drop: column.drop === true,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: readString(column.model_alias) ?? "",
    prompt: readString(column.prompt) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: readString(column.system_prompt) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: readString(column.code_lang) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    output_format: normalizeOutputFormat(column.output_format),
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_alias: readString(column.tool_alias) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    with_trace: withTrace,
    // biome-ignore lint/style/useNamingConvention: api schema
    extract_reasoning_content: extractReasoningContent,
    scores: llmType === "judge" ? scores : undefined,
    // biome-ignore lint/style/useNamingConvention: ui schema
    image_context: imageContext,
  };
}
