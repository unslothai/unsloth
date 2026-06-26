// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  SamplerConfig,
  SamplerType,
} from "../../../types";
import {
  isRecord,
  readNumberString,
  readString,
} from "../helpers";

const SAMPLER_TYPES: SamplerType[] = [
  "category",
  "subcategory",
  "uniform",
  "gaussian",
  "bernoulli",
  "datetime",
  "timedelta",
  "uuid",
  "person",
  "person_from_faker",
];

const TIMEDELTA_UNITS = new Set(["D", "h", "m", "s"]);

function parseCategoryConditionalParams(
  column: Record<string, unknown>,
): SamplerConfig["conditional_params"] {
  if (!isRecord(column.conditional_params)) {
    return undefined;
  }
  const conditional: NonNullable<SamplerConfig["conditional_params"]> = {};
  for (const [condition, rawParams] of Object.entries(column.conditional_params)) {
    if (!isRecord(rawParams)) {
      continue;
    }
    if (readString(rawParams.sampler_type) !== "category") {
      continue;
    }
    const values = Array.isArray(rawParams.values)
      ? rawParams.values.filter((item) => typeof item === "string")
      : [];
    if (values.length === 0) {
      continue;
    }
    const weights = Array.isArray(rawParams.weights)
      ? rawParams.weights.map((item) => (typeof item === "number" ? item : null))
      : undefined;
    conditional[condition] = {
      sampler_type: "category",
      values,
      weights,
    };
  }
  return Object.keys(conditional).length > 0 ? conditional : undefined;
}

export function parseSampler(
  column: Record<string, unknown>,
  name: string,
  id: string,
  errors: string[],
): SamplerConfig | null {
  const drop = column.drop === true;
  const samplerType = readString(column.sampler_type);
  if (!samplerType || !SAMPLER_TYPES.includes(samplerType as SamplerType)) {
    errors.push(`Sampler ${name}: unsupported sampler_type.`);
    return null;
  }
  const convertTo = readString(column.convert_to);
  const normalizedConvertTo =
    convertTo && ["float", "int", "str"].includes(convertTo)
      ? (convertTo as "float" | "int" | "str")
      : undefined;
  const params =
    typeof column.params === "object" && column.params
      ? (column.params as Record<string, unknown>)
      : {};

  if (samplerType === "category") {
    const values = Array.isArray(params.values)
      ? params.values.filter((item) => typeof item === "string")
      : [];
    const weights = Array.isArray(params.weights)
      ? params.weights.map((item) => (typeof item === "number" ? item : null))
      : [];
    return {
      id,
      kind: "sampler",
      sampler_type: "category",
      name,
      drop,
      convert_to: normalizedConvertTo,
      values,
      weights,
      conditional_params: parseCategoryConditionalParams(column),
    };
  }

  if (samplerType === "subcategory") {
    const mapping: Record<string, string[]> = {};
    if (params.values && typeof params.values === "object") {
      for (const [key, value] of Object.entries(params.values)) {
        if (Array.isArray(value)) {
          mapping[key] = value.filter((item) => typeof item === "string");
        }
      }
    }
    return {
      id,
      kind: "sampler",
      sampler_type: "subcategory",
      name,
      drop,
      convert_to: normalizedConvertTo,
      subcategory_parent: readString(params.category) ?? "",
      subcategory_mapping: mapping,
    };
  }

  if (samplerType === "uniform") {
    return {
      id,
      kind: "sampler",
      sampler_type: "uniform",
      name,
      drop,
      convert_to: normalizedConvertTo,
      low: readNumberString(params.low),
      high: readNumberString(params.high),
    };
  }

  if (samplerType === "gaussian") {
    return {
      id,
      kind: "sampler",
      sampler_type: "gaussian",
      name,
      drop,
      convert_to: normalizedConvertTo,
      mean: readNumberString(params.mean),
      std: readNumberString(params.std),
    };
  }

  if (samplerType === "bernoulli") {
    return {
      id,
      kind: "sampler",
      sampler_type: "bernoulli",
      name,
      drop,
      convert_to: normalizedConvertTo,
      p: readNumberString(params.p),
    };
  }

  if (samplerType === "datetime") {
    return {
      id,
      kind: "sampler",
      sampler_type: "datetime",
      name,
      drop,
      convert_to: normalizedConvertTo,
      datetime_start: readString(params.start) ?? "",
      datetime_end: readString(params.end) ?? "",
      datetime_unit: readString(params.unit) ?? "",
    };
  }

  if (samplerType === "timedelta") {
    const rawUnit = readString(params.unit);
    const unit =
      rawUnit && TIMEDELTA_UNITS.has(rawUnit)
        ? (rawUnit as "D" | "h" | "m" | "s")
        : "D";
    return {
      id,
      kind: "sampler",
      sampler_type: "timedelta",
      name,
      drop,
      convert_to: normalizedConvertTo,
      dt_min: readNumberString(params.dt_min),
      dt_max: readNumberString(params.dt_max),
      reference_column_name: readString(params.reference_column_name) ?? "",
      timedelta_unit: unit,
    };
  }

  if (samplerType === "uuid") {
    return {
      id,
      kind: "sampler",
      sampler_type: "uuid",
      name,
      drop,
      convert_to: normalizedConvertTo,
      uuid_format: readString(params.format) ?? "",
    };
  }

  const ageRange =
    Array.isArray(params.age_range) &&
    params.age_range.length === 2 &&
    params.age_range.every((item) => typeof item === "number")
      ? `${params.age_range[0]}-${params.age_range[1]}`
      : readString(params.age_range) ?? "";

  const base: SamplerConfig = {
    id,
    kind: "sampler",
    name,
    drop,
    sampler_type: samplerType as SamplerType,
    convert_to: normalizedConvertTo,
    person_locale: readString(params.locale) ?? "",
    person_sex: readString(params.sex) ?? "",
    person_age_range: ageRange,
    person_city: readString(params.city) ?? "",
  };

  if (samplerType === "person") {
    return {
      ...base,
      person_with_synthetic_personas:
        typeof params.with_synthetic_personas === "boolean"
          ? params.with_synthetic_personas
          : false,
    };
  }

  return base;
}
