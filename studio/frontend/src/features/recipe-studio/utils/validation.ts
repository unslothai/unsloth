// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../types";
import { isValidSex, parseAgeRange, parseIntNumber, parseNumber } from "./parse";
import { VALIDATOR_OXC_CODE_LANGS, VALIDATOR_SQL_CODE_LANGS } from "./validators/code-lang";
import { isOxcCodeShape } from "./validators/oxc-code-shape";
import { isOxcValidationMode } from "./validators/oxc-mode";

const TRACE_MODES = new Set(["none", "last_message", "all_messages"]);
const GITHUB_ITEM_TYPES = new Set(["issues", "pulls", "commits"]);

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: validation rules
export function getConfigErrors(config: NodeConfig | null): string[] {
  if (!config) {
    return [];
  }
  const errors: string[] = [];
  if (!config.name.trim()) {
    errors.push("Name is required.");
  }
  if (config.kind === "sampler") {
    if (config.sampler_type === "category") {
      const values = config.values ?? [];
      if (values.length < 2) {
        errors.push("Category needs at least 2 values.");
      }
      const weights = config.weights ?? [];
      const hasWeights = weights.some((weight) => weight !== null);
      if (hasWeights && weights.some((weight) => weight === null)) {
        errors.push("Weights must be set for all values.");
      }
      for (const [condition, params] of Object.entries(
        config.conditional_params ?? {},
      )) {
        if (!condition.trim()) {
          errors.push("Category conditional rule needs condition text.");
          continue;
        }
        const conditionalValues = (params.values ?? [])
          .map((value) => value.trim())
          .filter(Boolean);
        if (conditionalValues.length === 0) {
          errors.push(`Category conditional '${condition}' needs values.`);
          continue;
        }
        const conditionalWeights = params.weights ?? [];
        const hasConditionalWeights = conditionalWeights.some(
          (weight) => weight !== null,
        );
        if (
          hasConditionalWeights &&
          (conditionalWeights.length !== conditionalValues.length ||
            conditionalWeights.some((weight) => weight === null))
        ) {
          errors.push(
            `Category conditional '${condition}' weights must be set for all values.`,
          );
        }
      }
    }
    if (config.sampler_type === "uniform") {
      const low = parseNumber(config.low);
      const high = parseNumber(config.high);
      if (low === null || high === null) {
        errors.push("Uniform low/high must be numbers.");
      } else if (low >= high) {
        errors.push("Uniform low must be < high.");
      }
    }
    if (config.sampler_type === "gaussian") {
      const mean = parseNumber(config.mean);
      const std = parseNumber(config.std);
      if (mean === null || std === null) {
        errors.push("Gaussian mean/std must be numbers.");
      } else if (std <= 0) {
        errors.push("Gaussian std must be > 0.");
      }
    }
    if (config.sampler_type === "bernoulli") {
      const p = parseNumber(config.p);
      if (p === null) {
        errors.push("Bernoulli p must be a number.");
      } else if (p < 0 || p > 1) {
        errors.push("Bernoulli p must be between 0 and 1.");
      }
    }
    if (config.sampler_type === "datetime") {
      if (!config.datetime_unit) {
        errors.push("Datetime unit required.");
      }
      if (config.datetime_start && config.datetime_end) {
        const start = new Date(config.datetime_start).getTime();
        const end = new Date(config.datetime_end).getTime();
        if (!(Number.isFinite(start) && Number.isFinite(end))) {
          errors.push("Datetime start/end must be valid.");
        } else if (start >= end) {
          errors.push("Datetime start must be before end.");
        }
      }
    }
    if (config.sampler_type === "timedelta") {
      const min = parseNumber(config.dt_min);
      const max = parseNumber(config.dt_max);
      if (min === null || max === null) {
        errors.push("Timedelta dt_min/dt_max must be numbers.");
      } else if (min >= max) {
        errors.push("Timedelta dt_min must be < dt_max.");
      }
      if (!config.reference_column_name?.trim()) {
        errors.push("Timedelta reference datetime column required.");
      }
      if (!config.timedelta_unit) {
        errors.push("Timedelta unit required.");
      }
    }
    if (config.sampler_type === "subcategory" && !config.subcategory_parent) {
      errors.push("Subcategory needs a parent category column.");
    }
    if (
      config.sampler_type === "person" ||
      config.sampler_type === "person_from_faker"
    ) {
      if (config.person_sex?.trim()) {
        const normalized = config.person_sex.trim();
        if (!isValidSex(normalized)) {
          errors.push("Person sex must be Male or Female.");
        }
      }
      if (config.person_age_range?.trim()) {
        const parsed = parseAgeRange(config.person_age_range);
        if (!parsed) {
          errors.push("Person age range must be like 18-70.");
        }
      }
    }
  }
  if (config.kind === "llm") {
    if (!config.model_alias.trim()) {
      errors.push("Choose a saved model.");
    }
    if (!config.prompt.trim()) {
      errors.push("Prompt is required.");
    }
    if (config.llm_type === "code" && !config.code_lang) {
      errors.push("Code language is required.");
    }
    if (config.llm_type === "structured") {
      if (!config.output_format?.trim()) {
        errors.push("Output format is required.");
      } else {
        try {
          JSON.parse(config.output_format);
        } catch {
          errors.push("Output format must be valid JSON.");
        }
      }
    }
    if (config.llm_type === "judge") {
      const scores = config.scores ?? [];
      if (scores.length === 0) {
        errors.push("Add at least one scoring rule.");
      }
      for (const score of scores) {
        if (!score.name.trim()) {
          errors.push("Each scoring rule needs a name.");
        }
        if (!score.description.trim()) {
          errors.push("Each scoring rule needs a description.");
        }
        const options = score.options ?? [];
        if (options.length === 0) {
          errors.push(`Scoring rule ${score.name || "Untitled"} needs options.`);
        }
        for (const option of options) {
          if (!option.value.trim() || !option.description.trim()) {
            errors.push(
              `Scoring rule ${score.name || "Untitled"} needs both a value and a description for each option.`,
            );
            break;
          }
        }
      }
    }
    if (config.image_context?.enabled) {
      if (!config.image_context.column_name.trim()) {
        errors.push("Image context column is required.");
      }
    }
    if (
      config.with_trace &&
      !TRACE_MODES.has(config.with_trace)
    ) {
      errors.push("Trace mode must be none, last_message, or all_messages.");
    }
  }
  if (config.kind === "expression") {
    if (!config.expr.trim()) {
      errors.push("Expression is required.");
    }
  }
  if (config.kind === "tool_config") {
    if (config.mcp_providers.length === 0) {
      errors.push("Add at least one tool server.");
    }
    const serverNames = new Set<string>();
    for (const provider of config.mcp_providers) {
      const name = provider.name.trim();
      if (!name) {
        errors.push("Each tool server needs a name.");
        continue;
      }
      if (serverNames.has(name)) {
        errors.push(`Tool server names must be unique: ${name}.`);
      }
      serverNames.add(name);
      if (provider.provider_type === "stdio") {
        if (!provider.command?.trim()) {
          errors.push(`Tool server ${name}: add a command.`);
        }
      } else if (!provider.endpoint?.trim()) {
        errors.push(`Tool server ${name}: add an endpoint.`);
      }
    }
    const maxTurnsRaw = config.max_tool_call_turns?.trim();
    if (
      maxTurnsRaw &&
      (!Number.isFinite(Number(maxTurnsRaw)) || Number(maxTurnsRaw) < 1)
    ) {
      errors.push("Max tool-use turns must be 1 or more.");
    }
    const timeoutRaw = config.timeout_sec?.trim();
    if (
      timeoutRaw &&
      (!Number.isFinite(Number(timeoutRaw)) || Number(timeoutRaw) <= 0)
    ) {
      errors.push("Timeout must be > 0.");
    }
  }
  if (config.kind === "validator") {
    const targets = (config.target_columns ?? [])
      .map((value) => value.trim())
      .filter(Boolean);
    if (targets.length === 0) {
      errors.push("Choose the code step to check.");
    }
    const batch = parseIntNumber(config.batch_size);
    if (batch === null || batch < 1) {
      errors.push("Batch size must be an integer >= 1.");
    }
    if (!config.code_lang.trim()) {
      errors.push("Choose a code language for this check.");
    } else if (config.validator_type === "oxc") {
      if (!VALIDATOR_OXC_CODE_LANGS.includes(config.code_lang)) {
        errors.push("This JS/TS check only supports JavaScript or TypeScript.");
      }
      if (!isOxcValidationMode(config.oxc_validation_mode)) {
        errors.push("Choose whether to check syntax, lint rules, or both.");
      }
      if (!isOxcCodeShape(config.oxc_code_shape)) {
        errors.push("Choose whether this code is a full file or a snippet.");
      }
    } else if (
      config.code_lang !== "python" &&
      !VALIDATOR_SQL_CODE_LANGS.includes(config.code_lang)
    ) {
      errors.push("This check supports Python or SQL.");
    }
  }
  if (config.kind === "seed") {
    const seedSourceType = config.seed_source_type ?? "hf";
    if (seedSourceType === "github_repo") {
      const repos = (config.github_repo_slug ?? "")
        .split(/[\n,]/)
        .map((repo) => repo.trim())
        .filter(Boolean);
      if (repos.length === 0) {
        errors.push("Add at least one GitHub repository.");
      }
      if (
        repos.some((repo) => {
          const parts = repo.split("/");
          return parts.length !== 2 || parts.some((part) => !part);
        })
      ) {
        errors.push("GitHub repositories must use owner/name format.");
      }
      const itemTypes = config.github_item_types?.length
        ? config.github_item_types
        : ["issues", "pulls"];
      if (itemTypes.length === 0) {
        errors.push("Choose at least one GitHub item type.");
      } else if (itemTypes.some((itemType) => !GITHUB_ITEM_TYPES.has(itemType))) {
        errors.push("GitHub item types must be issues, pulls, or commits.");
      }
      const limit = parseIntNumber(config.github_limit ?? "100");
      if (limit === null || limit < 1 || limit > 5000) {
        errors.push("Items per repo must be an integer from 1 to 5000.");
      }
      const maxComments = parseIntNumber(config.github_max_comments_per_item ?? "30");
      if (maxComments === null || maxComments < 0 || maxComments > 200) {
        errors.push("Max comments per item must be an integer from 0 to 200.");
      }
    } else if (seedSourceType === "hf" && !config.hf_repo_id.trim()) {
      errors.push("Choose a Hugging Face dataset.");
    }
    if (seedSourceType !== "github_repo") {
      const hasPath =
        seedSourceType === "unstructured"
          ? (config.resolved_paths?.length ?? 0) > 0
          : Boolean(config.hf_path.trim());
      if (!hasPath) {
        errors.push("Load the source-data preview first.");
      }
    }
    if (
      seedSourceType === "hf" &&
      config.hf_endpoint?.trim() &&
      !config.hf_endpoint.trim().startsWith("http")
    ) {
      errors.push("HF endpoint must start with http.");
    }
    if (seedSourceType === "unstructured") {
      if (config.drop && (config.seed_columns?.length ?? 0) === 0) {
        errors.push("Load the available fields before hiding any from the final dataset.");
      }
      const chunkSizeRaw = Number(config.unstructured_chunk_size);
      const chunkOverlapRaw = Number(config.unstructured_chunk_overlap);
      if (!Number.isFinite(chunkSizeRaw) || Math.floor(chunkSizeRaw) < 1) {
        errors.push("Chunk size must be an integer >= 1.");
      }
      if (!Number.isFinite(chunkOverlapRaw) || Math.floor(chunkOverlapRaw) < 0) {
        errors.push("Chunk overlap must be an integer >= 0.");
      }
      if (
        Number.isFinite(chunkSizeRaw) &&
        Number.isFinite(chunkOverlapRaw) &&
        Math.floor(chunkOverlapRaw) >= Math.floor(chunkSizeRaw)
      ) {
        errors.push("Chunk overlap must be less than chunk size.");
      }
    } else if (seedSourceType !== "github_repo") {
      const selectedDropColumns = (config.seed_drop_columns ?? [])
        .map((value) => value.trim())
        .filter(Boolean);
      if (selectedDropColumns.length > 0 && (config.seed_columns?.length ?? 0) === 0) {
        errors.push("Load the available fields before hiding any from the final dataset.");
      }
    }

    if (config.selection_type === "index_range") {
      const start = parseIntNumber(config.selection_start);
      const end = parseIntNumber(config.selection_end);
      if (start === null || end === null) {
        errors.push("Index range start/end must be integers.");
      } else {
        if (start < 0 || end < 0) {
          errors.push("Index range start/end must be >= 0.");
        }
        if (end < start) {
          errors.push("Index range end must be >= start.");
        }
      }
    }
    if (config.selection_type === "partition_block") {
      const index = parseIntNumber(config.selection_index);
      const parts = parseIntNumber(config.selection_num_partitions);
      if (index === null || parts === null) {
        errors.push("Partition index/num_partitions must be integers.");
      } else {
        if (index < 0) errors.push("Partition index must be >= 0.");
        if (parts < 1) errors.push("Partition num_partitions must be >= 1.");
        if (parts >= 1 && index >= parts) {
          errors.push("Partition index must be < num_partitions.");
        }
      }
    }
  }
  return errors;
}
