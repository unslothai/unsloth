import type { Edge } from "@xyflow/react";
import type {
  CanvasNode,
  ExpressionConfig,
  ExpressionDtype,
  LlmConfig,
  NodeConfig,
  SamplerConfig,
  SamplerType,
} from "../types";
import { nodeDataFromConfig } from "./index";

export type CanvasSnapshot = {
  configs: Record<string, NodeConfig>;
  nodes: CanvasNode[];
  edges: Edge[];
  nextId: number;
  nextY: number;
};

export type ImportResult = {
  errors: string[];
  snapshot: CanvasSnapshot | null;
};

type RecipeInput = {
  columns?: unknown;
  model_configs?: unknown;
  model_providers?: unknown;
  processors?: unknown;
};

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
};

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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readNumberString(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  return "";
}

function parseJson(
  input: string,
): { data: unknown | null; error?: string } {
  try {
    return { data: JSON.parse(input) };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Invalid JSON.",
    };
  }
}

function normalizeOutputFormat(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (isRecord(value)) {
    return JSON.stringify(value, null, 2);
  }
  return "";
}

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
  const params = isRecord(column.params) ? column.params : {};
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
    if (isRecord(params.values)) {
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

function parseColumn(
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

function extractRefs(template: string): string[] {
  const matches = template.matchAll(/{{\s*([a-zA-Z0-9_]+)\s*}}/g);
  const refs = new Set<string>();
  for (const match of matches) {
    if (match[1]) {
      refs.add(match[1]);
    }
  }
  return Array.from(refs);
}

function buildEdges(
  configs: NodeConfig[],
  nameToId: Map<string, string>,
  uiEdges: Array<{ from: string; to: string }> | null,
): Edge[] {
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const addEdgeByName = (from: string, to: string) => {
    const sourceId = nameToId.get(from);
    const targetId = nameToId.get(to);
    if (!(sourceId && targetId)) {
      return;
    }
    const key = `${sourceId}-${targetId}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    edges.push({ id: `e-${key}`, source: sourceId, target: targetId });
  };

  if (uiEdges && uiEdges.length > 0) {
    for (const edge of uiEdges) {
      addEdgeByName(edge.from, edge.to);
    }
  } else {
    for (const config of configs) {
      if (config.kind === "llm") {
        for (const ref of extractRefs(config.prompt ?? "")) {
          addEdgeByName(ref, config.name);
        }
        for (const ref of extractRefs(config.system_prompt ?? "")) {
          addEdgeByName(ref, config.name);
        }
      }
      if (config.kind === "expression") {
        for (const ref of extractRefs(config.expr)) {
          addEdgeByName(ref, config.name);
        }
      }
      if (
        config.kind === "sampler" &&
        config.sampler_type === "subcategory" &&
        config.subcategory_parent
      ) {
        addEdgeByName(config.subcategory_parent, config.name);
      }
    }
  }

  return edges;
}

function buildNodes(
  configs: NodeConfig[],
  positions: Map<string, { x: number; y: number }>,
): CanvasNode[] {
  return configs.map((config, index) => {
    const position =
      positions.get(config.name) ?? ({ x: 0, y: index * 140 } as const);
    return {
      id: config.id,
      type: "builder",
      position,
      data: nodeDataFromConfig(config),
    };
  });
}

function parseUi(
  ui: UiInput | null,
): {
  positions: Map<string, { x: number; y: number }>;
  edges: Array<{ from: string; to: string }> | null;
} {
  const positions = new Map<string, { x: number; y: number }>();
  const edges: Array<{ from: string; to: string }> = [];
  if (ui && Array.isArray(ui.nodes)) {
    for (const node of ui.nodes) {
      if (isRecord(node)) {
        const id = readString(node.id);
        const x = typeof node.x === "number" ? node.x : null;
        const y = typeof node.y === "number" ? node.y : null;
        if (id && x !== null && y !== null) {
          positions.set(id, { x, y });
        }
      }
    }
  }
  if (ui && Array.isArray(ui.edges)) {
    for (const edge of ui.edges) {
      if (isRecord(edge)) {
        const from = readString(edge.from);
        const to = readString(edge.to);
        if (from && to) {
          edges.push({ from, to });
        }
      }
    }
  }
  return { positions, edges: edges.length > 0 ? edges : null };
}

export function importCanvasPayload(input: string): ImportResult {
  const parsed = parseJson(input);
  if (!parsed.data || !isRecord(parsed.data)) {
    return {
      errors: [parsed.error ?? "Invalid JSON payload."],
      snapshot: null,
    };
  }

  const recipe = (isRecord(parsed.data.recipe)
    ? parsed.data.recipe
    : parsed.data) as RecipeInput;
  const ui = isRecord(parsed.data.ui) ? (parsed.data.ui as UiInput) : null;

  if (!Array.isArray(recipe.columns)) {
    return { errors: ["Recipe must include columns."], snapshot: null };
  }

  const errors: string[] = [];
  const configs: NodeConfig[] = [];
  const nameToId = new Map<string, string>();

  recipe.columns.forEach((column, index) => {
    if (!isRecord(column)) {
      errors.push(`Column ${index + 1}: invalid object.`);
      return;
    }
    const id = `n${index + 1}`;
    const config = parseColumn(column, id, errors);
    if (!config) {
      return;
    }
    if (nameToId.has(config.name)) {
      errors.push(`Duplicate column name: ${config.name}.`);
      return;
    }
    nameToId.set(config.name, config.id);
    configs.push(config);
  });

  if (errors.length > 0) {
    return { errors, snapshot: null };
  }

  const { positions, edges: uiEdges } = parseUi(ui);
  const nodes = buildNodes(configs, positions);
  const edges = buildEdges(configs, nameToId, uiEdges);

  const maxY = nodes.reduce(
    (acc, node) => Math.max(acc, node.position.y),
    0,
  );

  return {
    errors: [],
    snapshot: {
      configs: Object.fromEntries(configs.map((config) => [config.id, config])),
      nodes,
      edges,
      nextId: configs.length + 1,
      nextY: maxY + 140,
    },
  };
}
