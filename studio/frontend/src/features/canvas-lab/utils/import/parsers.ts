import type {
  ExpressionConfig,
  ExpressionDtype,
  LlmConfig,
  NodeConfig,
  SamplerConfig,
  SamplerType,
} from "../../types";
import { normalizeOutputFormat, readNumberString, readString } from "./helpers";

const SAMPLER_TYPES: SamplerType[] = [
  "category",
  "subcategory",
  "uniform",
  "gaussian",
  "datetime",
  "uuid",
  "person",
];

const EXPRESSION_DTYPES: ExpressionDtype[] = ["str", "int", "float", "bool"];

function parseSampler(
  column: Record<string, unknown>,
  name: string,
  id: string,
  errors: string[],
): SamplerConfig | null {
  const samplerType = readString(column.sampler_type);
  if (!samplerType || !SAMPLER_TYPES.includes(samplerType as SamplerType)) {
    errors.push(`Sampler ${name}: unsupported sampler_type.`);
    return null;
  }
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
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "category",
      name,
      values,
      weights,
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
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "subcategory",
      name,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: readString(params.category) ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: mapping,
    };
  }
  if (samplerType === "uniform") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "uniform",
      name,
      low: readNumberString(params.low),
      high: readNumberString(params.high),
    };
  }
  if (samplerType === "gaussian") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "gaussian",
      name,
      mean: readNumberString(params.mean),
      std: readNumberString(params.std),
    };
  }
  if (samplerType === "datetime") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "datetime",
      name,
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_start: readString(params.start) ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_end: readString(params.end) ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_unit: readString(params.unit) ?? "",
    };
  }
  if (samplerType === "uuid") {
    return {
      id,
      kind: "sampler",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "uuid",
      name,
      // biome-ignore lint/style/useNamingConvention: api schema
      uuid_format: readString(params.format) ?? "",
    };
  }
  return {
    id,
    kind: "sampler",
    // biome-ignore lint/style/useNamingConvention: api schema
    sampler_type: "person",
    name,
    // biome-ignore lint/style/useNamingConvention: api schema
    person_locale: readString(params.locale) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_sex: readString(params.sex) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_age_range: readString(params.age_range) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_city: readString(params.city) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    person_with_synthetic_personas:
      typeof params.with_synthetic_personas === "boolean"
        ? params.with_synthetic_personas
        : false,
    // biome-ignore lint/style/useNamingConvention: api schema
    person_sample_dataset_when_available:
      typeof params.sample_dataset_when_available === "boolean"
        ? params.sample_dataset_when_available
        : false,
  };
}

function parseLlm(
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
  }
  return {
    id,
    kind: "llm",
    // biome-ignore lint/style/useNamingConvention: api schema
    llm_type: llmType,
    name,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: readString(column.model_alias) ?? "",
    prompt: readString(column.prompt) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: readString(column.system_prompt) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: readString(column.code_lang) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    output_format: normalizeOutputFormat(column.output_format),
  };
}

function parseExpression(
  column: Record<string, unknown>,
  name: string,
  id: string,
): ExpressionConfig {
  const dtype = readString(column.dtype);
  const normalized = EXPRESSION_DTYPES.includes(dtype as ExpressionDtype)
    ? (dtype as ExpressionDtype)
    : "str";
  return {
    id,
    kind: "expression",
    name,
    expr: readString(column.expr) ?? "",
    dtype: normalized,
  };
}

type ColumnParser = (
  column: Record<string, unknown>,
  name: string,
  id: string,
  errors: string[],
) => NodeConfig | null;

const COLUMN_PARSERS: Record<string, ColumnParser> = {
  sampler: (column, name, id, errors) => parseSampler(column, name, id, errors),
  expression: (column, name, id) => parseExpression(column, name, id),
  "llm-text": (column, name, id) => parseLlm(column, name, id),
  "llm-structured": (column, name, id) => parseLlm(column, name, id),
  "llm-code": (column, name, id) => parseLlm(column, name, id),
};

export function parseColumn(
  column: Record<string, unknown>,
  id: string,
  errors: string[],
): NodeConfig | null {
  const name = readString(column.name);
  if (!name) {
    errors.push("Column missing name.");
    return null;
  }
  const columnType = readString(column.column_type);
  const parser = columnType ? COLUMN_PARSERS[columnType] : null;
  if (parser) {
    return parser(column, name, id, errors);
  }
  errors.push(`Column ${name}: unsupported column_type.`);
  return null;
}
