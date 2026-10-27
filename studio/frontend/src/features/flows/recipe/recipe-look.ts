// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a recipe column reads on the canvas: Recipe Studio's identity per block type, the column's own
// ports (one per column it reads), and the model / tool config cards a group carries. Plain data only.

import type { DocGraph } from "../stores/doc-store";
import type { FlowEdge, FlowGroup, FlowNode, PortType } from "../types";
import {
  COLUMN_TYPE,
  type RecipeColumn,
  columnName,
  columnOf,
  columnReferences,
  groupColumns,
} from "./recipe-columns";

export type ColumnKind = "sampler" | "llm" | "expression" | "validator";

/** Recipe Studio's accent per block kind (`ui-tones.ts`), as hex for Flows' header gradient. */
export const RECIPE_ACCENT = {
  sampler: "#10b981",
  person: "#f59e0b",
  llm: "#0ea5e9",
  expression: "#6366f1",
  validator: "#f43f5e",
  model_provider: "#f59e0b",
  model_config: "#f97316",
  tool_config: "#06b6d4",
} as const;

const SAMPLER_LABEL: Record<string, string> = {
  category: "Category",
  subcategory: "Subcategory",
  uniform: "Random number",
  gaussian: "Bell-curve number",
  bernoulli: "Yes/no value",
  datetime: "Date and time",
  timedelta: "Time offset",
  uuid: "Unique ID",
  person: "Synthetic person",
  // biome-ignore lint/style/useNamingConvention: api schema
  person_from_faker: "Synthetic person",
};

const LLM_LABEL: Record<string, string> = {
  text: "AI text",
  structured: "AI structured data",
  code: "AI code",
  judge: "AI scorer",
};

const DTYPE_LABEL: Record<string, string> = {
  str: "Text",
  int: "Int",
  float: "Float",
  bool: "Bool",
};

export interface ColumnLook {
  kind: ColumnKind;
  /** sampler type, llm type, "expression" or the validator language: picks the icon */
  block: string;
  accent: string;
  /** "Category · Generated field", "AI text · AI step" */
  subtitle: string;
  chips: string[];
  fields: { label: string; value: string }[];
  /** collapsed text: prompts, formula, scores */
  details: { label: string; text: string }[];
  drop: boolean;
  /** the Flows port type its value is, for the port and wire colour */
  valueType: PortType;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function samplerLook(column: RecipeColumn): Partial<ColumnLook> {
  const type = text(column.sampler_type) || "category";
  const params = record(column.params);
  const chips: string[] = [];
  const fields: { label: string; value: string }[] = [];
  let valueType: PortType = "text";
  if (type === "category" && Array.isArray(params.values))
    chips.push(...params.values.map(String));
  else if (type === "subcategory") {
    if (text(params.category))
      fields.push({ label: "Based on", value: text(params.category) });
    const values = record(params.values);
    chips.push(...new Set(Object.values(values).flat().map(String)));
  } else if (type === "uniform") {
    valueType = "number";
    chips.push(`${params.low ?? "?"} – ${params.high ?? "?"}`);
  } else if (type === "gaussian") {
    valueType = "number";
    chips.push(`mean ${params.mean ?? "?"}`, `sd ${params.stddev ?? "?"}`);
  } else if (type === "bernoulli") {
    valueType = "number";
    chips.push(`p ${params.p ?? "?"}`);
  } else if (type === "datetime") {
    chips.push(`${params.start ?? "?"} → ${params.end ?? "?"}`);
  } else if (type === "timedelta") {
    if (text(params.reference_column_name))
      fields.push({ label: "From", value: text(params.reference_column_name) });
    chips.push(`${params.dt_min ?? "?"} – ${params.dt_max ?? "?"} ${params.unit ?? ""}`.trim());
  } else if (type === "person" || type === "person_from_faker") {
    valueType = "json";
    chips.push(text(params.locale) || "any locale");
    if (text(params.city)) chips.push(text(params.city));
  }
  return {
    block: type,
    accent:
      type === "person" || type === "person_from_faker"
        ? RECIPE_ACCENT.person
        : RECIPE_ACCENT.sampler,
    subtitle: `${SAMPLER_LABEL[type] ?? "Generated field"} · Generated field`,
    chips,
    fields,
    valueType,
  };
}

function llmLook(column: RecipeColumn): Partial<ColumnLook> {
  const type = String(column.column_type).slice(4) || "text";
  const fields: { label: string; value: string }[] = [];
  const details: { label: string; text: string }[] = [];
  const chips: string[] = [];
  if (text(column.model_alias))
    fields.push({ label: "Model", value: text(column.model_alias) });
  if (type === "code" && text(column.code_lang))
    fields.push({ label: "Language", value: text(column.code_lang) });
  if (type === "judge" && Array.isArray(column.scores)) {
    for (const score of column.scores) {
      const name = text(record(score).name);
      if (name) chips.push(name);
    }
  }
  if (text(column.prompt)) details.push({ label: "Prompt", text: text(column.prompt) });
  if (text(column.system_prompt))
    details.push({ label: "System", text: text(column.system_prompt) });
  if (type === "structured" && column.output_format) {
    const format = column.output_format;
    details.push({
      label: "Format",
      text: typeof format === "string" ? format : JSON.stringify(format, null, 2),
    });
  }
  return {
    block: type,
    accent: RECIPE_ACCENT.llm,
    subtitle: `${LLM_LABEL[type] ?? "AI"} · AI step`,
    chips,
    fields,
    details,
    valueType: type === "structured" || type === "judge" ? "json" : "text",
  };
}

/** Recipe Studio's identity and key fields for one column. */
export function columnLook(column: RecipeColumn): ColumnLook {
  const type = String(column.column_type ?? "");
  let part: Partial<ColumnLook>;
  let kind: ColumnKind;
  if (type === "sampler") {
    kind = "sampler";
    part = samplerLook(column);
  } else if (type.startsWith("llm-")) {
    kind = "llm";
    part = llmLook(column);
  } else if (type === "expression") {
    kind = "expression";
    const dtype = text(column.dtype) || "str";
    part = {
      block: "expression",
      accent: RECIPE_ACCENT.expression,
      subtitle: `${DTYPE_LABEL[dtype] ?? "Formula"} · Formula`,
      details: text(column.expr) ? [{ label: "Formula", text: text(column.expr) }] : [],
      valueType:
        dtype === "int" || dtype === "float"
          ? "number"
          : dtype === "bool"
            ? "bool"
            : "text",
    };
  } else {
    kind = "validator";
    const params = record(column.validator_params);
    const lang = text(params.code_lang) || text(column.code_lang);
    const validator = text(column.validator_type);
    const language = lang.startsWith("sql")
      ? "SQL"
      : validator === "json" || lang === "json"
        ? "JSON"
        : "Python";
    const targets = Array.isArray(column.target_columns)
      ? column.target_columns.map(String)
      : [];
    part = {
      block: language.toLowerCase(),
      accent: RECIPE_ACCENT.validator,
      subtitle: `${language} · Check`,
      fields: [
        ...(targets.length ? [{ label: "Checks", value: targets.join(", ") }] : []),
        ...(lang ? [{ label: "Language", value: lang }] : []),
      ],
      valueType: "json",
    };
  }
  return {
    kind,
    block: part.block ?? type,
    accent: part.accent ?? RECIPE_ACCENT.sampler,
    subtitle: part.subtitle ?? type,
    chips: part.chips ?? [],
    fields: part.fields ?? [],
    details: part.details ?? [],
    drop: column.drop === true,
    valueType: part.valueType ?? "text",
  };
}

/** Separates a column's input port from the column it carries in a canvas handle id. */
export const REF_HANDLE_SEP = "@";

export interface ColumnInputPort {
  /** handle port id: `in@<source node id>` */
  port: string;
  label: string;
  sourceId: string;
  type: PortType;
}

/** One input port per column a column reads, in the order its text first mentions them. */
export function columnInputPorts(
  graph: DocGraph,
  nodeId: string,
): ColumnInputPort[] {
  const node = graph.nodes[nodeId];
  if (node?.type !== COLUMN_TYPE) return [];
  const sources = new Map<string, FlowNode>();
  for (const id of graph.edgeOrder) {
    const edge = graph.edges[id];
    const source = edge?.target === nodeId ? graph.nodes[edge.source] : undefined;
    if (source?.type === COLUMN_TYPE) sources.set(columnName(source), source);
  }
  const order = columnReferences(columnOf(node), new Set(sources.keys()));
  for (const name of sources.keys()) if (!order.includes(name)) order.push(name);
  return order.map((name) => {
    const source = sources.get(name) as FlowNode;
    return {
      port: `in${REF_HANDLE_SEP}${source.id}`,
      label: name,
      sourceId: source.id,
      type: columnLook(columnOf(source)).valueType,
    };
  });
}

/** The handle a reference wire lands on: the target column's port for that source column. */
export function columnTargetPort(edge: FlowEdge, graph: DocGraph): string {
  return graph.nodes[edge.target]?.type === COLUMN_TYPE &&
    graph.nodes[edge.source]?.type === COLUMN_TYPE
    ? `${edge.target_port}${REF_HANDLE_SEP}${edge.source}`
    : edge.target_port;
}

/** Whether the column picks its inputs by wire ({{ ref }} in a prompt, formula or check). */
export function columnTakesWires(column: RecipeColumn): boolean {
  const type = String(column.column_type ?? "");
  return type.startsWith("llm-") || type === "expression" || type === "validation";
}

export type AuxKind = "model_provider" | "model_config" | "tool_config";

export interface AuxCard {
  /** canvas node id */
  id: string;
  groupId: string;
  kind: AuxKind;
  name: string;
  subtitle: string;
  fields: { label: string; value: string }[];
}

export interface AuxLink {
  id: string;
  source: string;
  /** a canvas node id: another card or a column */
  target: string;
}

export const AUX_PREFIX = "recipe-aux:";

export function auxId(groupId: string, kind: AuxKind, name: string): string {
  return `${AUX_PREFIX}${groupId}:${kind}:${name}`;
}

function list(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record) : [];
}

/** A group's providers, model presets and tool profiles as cards, wired to what uses them. */
export function recipeAux(
  graph: DocGraph,
  group: FlowGroup,
): { cards: AuxCard[]; links: AuxLink[] } {
  const config = group.config ?? {};
  const cards: AuxCard[] = [];
  const links: AuxLink[] = [];
  const card = (
    kind: AuxKind,
    name: string,
    subtitle: string,
    fields: { label: string; value: string }[],
  ) => {
    const id = auxId(group.id, kind, name);
    cards.push({ id, groupId: group.id, kind, name, subtitle, fields });
    return id;
  };
  const providers = new Map<string, string>();
  for (const provider of list(config.model_providers)) {
    const name = text(provider.name);
    if (!name) continue;
    const local = provider.is_local === true;
    providers.set(
      name,
      card(
        "model_provider",
        name,
        `${local ? "Chat model" : text(provider.provider_type) || "Connection"} · Provider connection`,
        local ? [] : [{ label: "Endpoint", value: text(provider.endpoint) || "not set" }],
      ),
    );
  }
  const models = new Map<string, string>();
  for (const model of list(config.model_configs)) {
    const alias = text(model.alias);
    if (!alias) continue;
    const inference = record(model.inference_parameters);
    const fields = [
      { label: "Model", value: text(model.model) || "not set" },
      ...(inference.temperature !== undefined
        ? [{ label: "Temperature", value: String(inference.temperature) }]
        : []),
    ];
    const id = card("model_config", alias, "Model preset", fields);
    models.set(alias, id);
    const provider = providers.get(text(model.provider));
    if (provider) links.push({ id: `${provider}>${id}`, source: provider, target: id });
  }
  const tools = new Map<string, string>();
  for (const tool of list(config.tool_configs)) {
    const alias = text(tool.tool_alias);
    if (!alias) continue;
    const servers = Array.isArray(tool.providers) ? tool.providers.length : 0;
    tools.set(
      alias,
      card("tool_config", alias, `${servers === 1 ? "1 server" : `${servers} servers`} · Tool access`, []),
    );
  }
  for (const node of groupColumns(graph, group.id)) {
    const column = columnOf(node);
    const model = models.get(text(column.model_alias));
    if (model) links.push({ id: `${model}>${node.id}`, source: model, target: node.id });
    const tool = tools.get(text(column.tool_alias));
    if (tool) links.push({ id: `${tool}>${node.id}`, source: tool, target: node.id });
  }
  return { cards, links };
}
