import type { NodeConfig } from "../types";

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
    if (config.sampler_type === "subcategory" && !config.subcategory_parent) {
      errors.push("Subcategory needs a parent category column.");
    }
    if (config.sampler_type === "person") {
      if (config.person_sex?.trim()) {
        const normalized = config.person_sex.trim();
        if (!(normalized === "Male" || normalized === "Female")) {
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
    if (config.sampler_type === "person_from_faker") {
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
  }
  if (config.kind === "expression") {
    if (!config.expr.trim()) {
      errors.push("Expression is required.");
    }
  }
  return errors;
}
