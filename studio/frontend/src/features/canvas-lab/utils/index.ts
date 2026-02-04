import type {
  CanvasNodeData,
  LlmConfig,
  LlmType,
  NodeConfig,
  SamplerConfig,
  SamplerType,
} from "../types";

const SAMPLER_LABELS: Record<SamplerType, string> = {
  category: "Category",
  subcategory: "Subcategory",
  uniform: "Uniform",
  gaussian: "Gaussian",
  datetime: "Datetime",
  uuid: "UUID",
  person: "Person",
};

const LLM_LABELS: Record<LlmType, string> = {
  text: "LLM Text",
  structured: "LLM Structured",
  code: "LLM Code",
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
    // biome-ignore lint/style/useNamingConvention: api schema
    person_sample_dataset_when_available: false,
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
  }
  const name = nextName(existing, namePrefix);
  return {
    id,
    kind: "llm",
    // biome-ignore lint/style/useNamingConvention: api schema
    llm_type: llmType,
    name,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: "local-text",
    prompt: "Write a response about {{ sampler_1 }}.",
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: "",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: llmType === "code" ? "python" : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    output_format:
      llmType === "structured" ? '{\n  "field": "string"\n}' : undefined,
  };
}

export function labelForSampler(type: SamplerType): string {
  return SAMPLER_LABELS[type] ?? "Sampler";
}

export function labelForLlm(type: LlmType): string {
  return LLM_LABELS[type] ?? "LLM";
}

export function nodeDataFromConfig(config: NodeConfig): CanvasNodeData {
  if (config.kind === "sampler") {
    return {
      title: "Sampler",
      kind: "sampler",
      subtype: labelForSampler(config.sampler_type),
      name: config.name,
    };
  }
  return {
    title: "LLM",
    kind: "llm",
    subtype: labelForLlm(config.llm_type),
    name: config.name,
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

function parseNumber(value?: string): number | null {
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
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
  }
  if (config.kind === "llm" && !config.prompt.trim()) {
    errors.push("Prompt is required.");
  }
  if (
    config.kind === "llm" &&
    config.llm_type === "structured" &&
    typeof config.output_format === "string" &&
    config.output_format.trim()
  ) {
    try {
      JSON.parse(config.output_format);
    } catch {
      errors.push("Output format must be valid JSON.");
    }
  }
  return errors;
}
