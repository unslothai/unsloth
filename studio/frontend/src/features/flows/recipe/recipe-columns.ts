// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe group columns as plain JSON (studio/backend/core/flows/recipe_import.py): references, wires, summaries.
// Kept free of Recipe Studio's editor code so the canvas can load it eagerly.

import { removeRef } from "@/features/recipe-studio/utils/refs";
import type { DocGraph } from "../stores/doc-store";
import type { FlowEdge, FlowGroup, FlowNode } from "../types";
import { columnBlock } from "./recipe-kinds";

export const COLUMN_TYPE = "recipe.column";
export const GENERATE_TYPE = "recipe.generate";

export type RecipeColumn = Record<string, unknown>;

export function isRecipeGroup(
  group: FlowGroup | undefined,
): group is FlowGroup {
  return group?.kind === "recipe";
}

export function columnOf(node: FlowNode): RecipeColumn {
  const column = node.params.column;
  return column && typeof column === "object" && !Array.isArray(column)
    ? (column as RecipeColumn)
    : {};
}

export function columnName(node: FlowNode): string {
  const name = columnOf(node).name;
  return typeof name === "string" ? name : "";
}

export function groupColumns(graph: DocGraph, groupId: string): FlowNode[] {
  return graph.nodeOrder
    .map((id) => graph.nodes[id])
    .filter((n) => n?.type === COLUMN_TYPE && n.group === groupId);
}

export function groupGenerate(
  graph: DocGraph,
  groupId: string,
): FlowNode | undefined {
  return graph.nodeOrder
    .map((id) => graph.nodes[id])
    .find((n) => n?.type === GENERATE_TYPE && n.group === groupId);
}

const NAME = /[A-Za-z_][A-Za-z0-9_-]*/g;
const BLOCK = /\{\{([\s\S]*?)\}\}|\{%([\s\S]*?)%\}/g;

function templateNames(text: string): string[] {
  const names: string[] = [];
  for (const block of text.matchAll(BLOCK)) {
    const body = block[1] ?? block[2] ?? "";
    const loopVars = new Set(
      [...body.matchAll(/\bfor\s+([A-Za-z_][A-Za-z0-9_]*)/g)].map((m) => m[1]),
    );
    for (const match of body.matchAll(NAME)) {
      const before = body.slice(0, match.index).trimEnd();
      if (before.endsWith(".") || before.endsWith("|")) continue;
      if (loopVars.has(match[0])) continue;
      names.push(match[0]);
    }
  }
  return names;
}

/** The other columns a column reads, in first-mention order; mirrors `recipe_import.column_references`. */
export function columnReferences(
  column: RecipeColumn,
  names: ReadonlySet<string>,
): string[] {
  const seen: string[] = [];
  const add = (name: unknown) => {
    if (
      typeof name === "string" &&
      names.has(name) &&
      name !== column.name &&
      !seen.includes(name)
    )
      seen.push(name);
  };
  const texts = [column.prompt, column.system_prompt, column.expr];
  const conditional = column.conditional_params;
  const nested: unknown[] = [column.params];
  if (conditional && typeof conditional === "object") {
    for (const [rule, value] of Object.entries(conditional)) {
      texts.push(
        rule.includes("{{") || rule.includes("{%") ? rule : `{{ ${rule} }}`,
      );
      nested.push(value);
    }
  }
  for (const text of texts)
    if (typeof text === "string")
      for (const name of templateNames(text)) add(name);
  if (Array.isArray(column.target_columns))
    for (const name of column.target_columns) add(name);
  for (const sub of nested) {
    if (sub && typeof sub === "object") {
      const record = sub as Record<string, unknown>;
      add(record.category);
      add(record.reference_column_name);
    }
  }
  return seen;
}

/** The reference wires a group should have: one `out -> in` edge per column it reads. */
export function referenceEdges(columns: FlowNode[]): FlowEdge[] {
  const byName = new Map(columns.map((n) => [columnName(n), n.id]));
  const names = new Set(byName.keys());
  const edges: FlowEdge[] = [];
  for (const node of columns) {
    for (const ref of columnReferences(columnOf(node), names)) {
      const source = byName.get(ref) as string;
      edges.push({
        id: `${source}__${node.id}`,
        source,
        source_port: "out",
        target: node.id,
        target_port: "in",
        kind: "data",
      });
    }
  }
  return edges;
}

/** Edges to add and remove so a group's wires match its columns' references. */
export function referenceEdgeDiff(
  graph: DocGraph,
  columns: FlowNode[],
): { add: FlowEdge[]; remove: string[] } {
  const ids = new Set(columns.map((n) => n.id));
  const want = referenceEdges(columns);
  const key = (e: FlowEdge) => `${e.source}>${e.target}`;
  const wanted = new Set(want.map(key));
  const have = new Set<string>();
  const remove: string[] = [];
  for (const id of graph.edgeOrder) {
    const edge = graph.edges[id];
    if (!(ids.has(edge.source) && ids.has(edge.target))) continue;
    if (wanted.has(key(edge)) && !have.has(key(edge))) have.add(key(edge));
    else remove.push(id);
  }
  return { add: want.filter((e) => !have.has(key(e))), remove };
}

/** The recipe JSON a group stands for: its config plus its columns in canvas order. */
export function groupPayload(group: FlowGroup, columns: FlowNode[]) {
  return {
    recipe: { ...(group.config ?? {}), columns: columns.map(columnOf) },
    run: group.run ?? {},
    ui: group.ui ?? {},
  };
}

/** The column reading `{{ ref }}` too; null when it picks its inputs another way (samplers). */
export function withReference(
  column: RecipeColumn,
  ref: string,
): RecipeColumn | null {
  const tag = `{{ ${ref} }}`;
  const type = String(column.column_type ?? "");
  if (type.startsWith("llm-")) {
    const prompt = String(column.prompt ?? "");
    if (prompt.includes(tag)) return column;
    return { ...column, prompt: prompt.trim() ? `${prompt}\n${tag}` : tag };
  }
  if (type === "expression") {
    const expr = String(column.expr ?? "");
    if (expr.includes(tag)) return column;
    return { ...column, expr: expr.trim() ? `${expr} ${tag}` : tag };
  }
  if (type === "validation") {
    const targets = Array.isArray(column.target_columns)
      ? column.target_columns
      : [];
    // biome-ignore lint/style/useNamingConvention: api schema
    return targets.includes(ref)
      ? column
      : { ...column, target_columns: [...targets, ref] };
  }
  return null;
}

/** The column without `{{ ref }}` in its prompts or targets; samplers keep their parent. */
export function withoutReference(
  column: RecipeColumn,
  ref: string,
): RecipeColumn {
  const next: RecipeColumn = { ...column };
  for (const key of ["prompt", "system_prompt", "expr"]) {
    if (typeof next[key] === "string")
      next[key] = removeRef(next[key] as string, ref);
  }
  if (Array.isArray(next.target_columns))
    // biome-ignore lint/style/useNamingConvention: api schema
    next.target_columns = next.target_columns.filter((name) => name !== ref);
  return next;
}

/** A short badge and line for a column in a list. */
export function columnSummary(column: RecipeColumn): {
  badge: string;
  line: string;
} {
  const params = (column.params ?? {}) as Record<string, unknown>;
  let line = "";
  if (typeof column.prompt === "string") line = column.prompt;
  else if (typeof column.expr === "string") line = column.expr;
  else if (Array.isArray(params.values)) line = params.values.join(", ");
  else if (Array.isArray(column.target_columns))
    line = `checks ${column.target_columns.join(", ")}`;
  line = line.replace(/\s+/g, " ").trim();
  return {
    badge: columnBlock(column).label,
    line: line.length > 90 ? `${line.slice(0, 90)}…` : line,
  };
}

/** A starter recipe: a topic sampler, one LLM column and the Chat model as its provider. */
export function starterRecipe(): {
  config: Record<string, unknown>;
  columns: RecipeColumn[];
} {
  return {
    config: {
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: [
        {
          name: "local",
          // biome-ignore lint/style/useNamingConvention: api schema
          provider_type: "openai",
          endpoint: "",
          // biome-ignore lint/style/useNamingConvention: api schema
          is_local: true,
        },
      ],
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: [{ alias: "writer", model: "", provider: "local" }],
      processors: [],
    },
    columns: [
      {
        // biome-ignore lint/style/useNamingConvention: api schema
        column_type: "sampler",
        name: "topic",
        drop: false,
        // biome-ignore lint/style/useNamingConvention: api schema
        sampler_type: "category",
        params: { values: ["science", "history", "cooking"] },
      },
      {
        // biome-ignore lint/style/useNamingConvention: api schema
        column_type: "llm-text",
        name: "question",
        drop: false,
        // biome-ignore lint/style/useNamingConvention: api schema
        model_alias: "writer",
        prompt: "Ask one short question about {{ topic }}.",
      },
    ],
  };
}
