// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CategoryConditionalParams, SamplerConfig } from "../../types";
import { isValidSex, parseAgeRange, parseNumber } from "./parse";

const DATETIME_UNIT_MAP: Record<string, "Y" | "M" | "D" | "h" | "m" | "s"> = {
  year: "Y",
  month: "M",
  day: "D",
  hour: "h",
  minute: "m",
  second: "s",
};

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
      // data_designer expects `stddev`
      stddev: parseNumber(config.std),
    };
  }
  if (config.sampler_type === "bernoulli") {
    return {
      p: parseNumber(config.p),
    };
  }
  if (config.sampler_type === "datetime") {
    const rawUnit = config.datetime_unit?.trim();
    let unit: string | undefined = rawUnit || undefined;
    if (rawUnit && DATETIME_UNIT_MAP[rawUnit]) {
      unit = DATETIME_UNIT_MAP[rawUnit];
    }
    if (rawUnit === "week") {
      errors.push(`Datetime ${config.name}: unit 'week' not supported.`);
      unit = undefined;
    }
    return {
      start: config.datetime_start ?? undefined,
      end: config.datetime_end ?? undefined,
      unit,
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
    const raw = config.uuid_format?.trim();
    if (!raw) {
      return {};
    }
    // UI historically used "uuid4" as a "format". data_designer uuid sampler is always uuid4.
    if (raw.toLowerCase() === "uuid4") {
      return {};
    }
    if (raw.toLowerCase() === "short" || raw.toLowerCase() === "short_form") {
      // biome-ignore lint/style/useNamingConvention: api schema
      return { short_form: true };
    }
    if (raw.toLowerCase() === "upper" || raw.toLowerCase() === "uppercase") {
      return { uppercase: true };
    }
    if (raw.toLowerCase().startsWith("prefix:")) {
      return { prefix: raw.slice("prefix:".length).trim() || undefined };
    }
    return {
      prefix: raw,
    };
  }
  const params: Record<string, unknown> = {};
  if (config.person_locale?.trim()) {
    params.locale = config.person_locale.trim();
  }
  if (config.person_sex?.trim()) {
    if (isValidSex(config.person_sex.trim())) {
      params.sex = config.person_sex.trim();
    } else {
      errors.push(`Person ${config.name}: sex must be Male or Female.`);
    }
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
