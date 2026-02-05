import type {
  CanvasNodeData,
  ExpressionConfig,
  ExpressionDtype,
  LayoutDirection,
  LlmConfig,
  LlmType,
  NodeConfig,
  SamplerConfig,
  SamplerType,
} from "../types";
export { getConfigErrors } from "./validation";

const SAMPLER_LABELS: Record<SamplerType, string> = {
  category: "Category",
  subcategory: "Subcategory",
  uniform: "Uniform",
  gaussian: "Gaussian",
  datetime: "Datetime",
  uuid: "UUID",
  person: "Person",
  person_from_faker: "Person (Faker)",
};

const LLM_LABELS: Record<LlmType, string> = {
  text: "LLM Text",
  structured: "LLM Structured",
  code: "LLM Code",
  judge: "LLM Judge",
};

const EXPRESSION_LABELS: Record<ExpressionDtype, string> = {
  str: "Text",
  int: "Int",
  float: "Float",
  bool: "Bool",
};

export function nextName(existing: NodeConfig[], prefix: string): string {
  const counts = existing
    .map((item) => item.name)
    .filter((name) => name.startsWith(prefix))
    .map((name) => {
      const suffix = name.slice(prefix.length);
      const num = Number.parseInt(suffix.replace("_", ""), 10);
      return Number.isNaN(num) ? 0 : num;
    });
  const next = counts.length > 0 ? Math.max(...counts) + 1 : 1;
  return `${prefix}_${next}`;
}

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
      mean: "0",
      std: "1",
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
      datetime_start: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_end: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      datetime_unit: "day",
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

export function makeExpressionConfig(
  id: string,
  existing: NodeConfig[],
): ExpressionConfig {
  return {
    id,
    kind: "expression",
    name: nextName(existing, "expr"),
    expr: "",
    dtype: "str",
  };
}

export function labelForSampler(type: SamplerType): string {
  return SAMPLER_LABELS[type] ?? "Sampler";
}

export function labelForLlm(type: LlmType): string {
  return LLM_LABELS[type] ?? "LLM";
}

export function labelForExpression(type: ExpressionDtype): string {
  return EXPRESSION_LABELS[type] ?? "Expression";
}

export function nodeDataFromConfig(
  config: NodeConfig,
  layoutDirection: LayoutDirection = "LR",
): CanvasNodeData {
  if (config.kind === "sampler") {
    return {
      title: "Sampler",
      kind: "sampler",
      subtype: labelForSampler(config.sampler_type),
      blockType: config.sampler_type,
      name: config.name,
      layoutDirection,
    };
  }
  if (config.kind === "expression") {
    return {
      title: "Expression",
      kind: "expression",
      subtype: labelForExpression(config.dtype),
      blockType: "expression",
      name: config.name,
      layoutDirection,
    };
  }
  return {
    title: "LLM",
    kind: "llm",
    subtype: labelForLlm(config.llm_type),
    blockType: config.llm_type,
    name: config.name,
    layoutDirection,
  };
}

export function isSamplerConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(config && config.kind === "sampler");
}

export function isCategoryConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(
    config && config.kind === "sampler" && config.sampler_type === "category",
  );
}

export function isSubcategoryConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(
    config &&
      config.kind === "sampler" &&
      config.sampler_type === "subcategory",
  );
}

export function isLlmConfig(
  config: NodeConfig | null | undefined,
): config is LlmConfig {
  return Boolean(config && config.kind === "llm");
}

export function isExpressionConfig(
  config: NodeConfig | null | undefined,
): config is ExpressionConfig {
  return Boolean(config && config.kind === "expression");
}
