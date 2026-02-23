import type { NodeConfig } from "../types";
import { isValidSex, parseAgeRange, parseIntNumber, parseNumber } from "./parse";

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
      errors.push("Model alias is required.");
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
        errors.push("LLM Judge needs at least one score.");
      }
      for (const score of scores) {
        if (!score.name.trim()) {
          errors.push("LLM Judge score name is required.");
        }
        if (!score.description.trim()) {
          errors.push("LLM Judge score description is required.");
        }
        const options = score.options ?? [];
        if (options.length === 0) {
          errors.push(`LLM Judge score ${score.name || "Unnamed"} needs options.`);
        }
        for (const option of options) {
          if (!option.value.trim() || !option.description.trim()) {
            errors.push(
              `LLM Judge score ${score.name || "Unnamed"} options need value + description.`,
            );
            break;
          }
        }
      }
    }
  }
  if (config.kind === "expression") {
    if (!config.expr.trim()) {
      errors.push("Expression is required.");
    }
  }
  if (config.kind === "seed") {
    const seedSourceType = config.seed_source_type ?? "hf";
    if (seedSourceType === "hf" && !config.hf_repo_id.trim()) {
      errors.push("Seed dataset repo is required.");
    }
    if (!config.hf_path.trim()) {
      errors.push("Seed metadata not loaded. Click 'Load columns + 10 rows'.");
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
        errors.push("Seed drop needs loaded columns.");
      }
    } else {
      const selectedDropColumns = (config.seed_drop_columns ?? [])
        .map((value) => value.trim())
        .filter(Boolean);
      if (selectedDropColumns.length > 0 && (config.seed_columns?.length ?? 0) === 0) {
        errors.push("Seed drop columns need loaded columns.");
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
