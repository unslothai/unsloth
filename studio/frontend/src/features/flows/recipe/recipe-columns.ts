// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe group columns as plain JSON (studio/backend/core/flows/recipe_import.py): references, wires, summaries.
// Kept free of Recipe Studio's editor code so the canvas can load it eagerly.

import {
  type JinjaScope,
  boundAt,
  inRaw,
  jinjaScopes,
  rawSpans,
  tagReadPart,
} from "@/features/recipe-studio/utils/refs";
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

/** The group's Recipe Dataset step: the enabled one, as the compiler runs it, else the first. */
export function groupGenerate(
  graph: DocGraph,
  groupId: string,
): FlowNode | undefined {
  const steps = graph.nodeOrder
    .map((id) => graph.nodes[id])
    .filter((n) => n?.type === GENERATE_TYPE && n.group === groupId);
  return steps.find((n) => !n.disabled) ?? steps[0];
}

const NAME = /[A-Za-z_][A-Za-z0-9_-]*/g;
const BLOCK = /\{\{([\s\S]*?)\}\}|\{%([\s\S]*?)%\}/g;
const QUOTED = /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g;

/** The names a tag at ``at`` reads, leaving out the template's own variables there. */
function tagNames(
  output: string | undefined,
  statement: string | undefined,
  at: number,
  scopes: JinjaScope[],
): string[] {
  // A quoted string is text, not a column: `{{ "topic" }}` reads nothing.
  const part = statement === undefined ? null : tagReadPart(statement);
  const body = (part ? `${part.read} ${part.filter}` : (output ?? "")).replace(
    QUOTED,
    '""',
  );
  const filterAt = part ? part.read.replace(QUOTED, '""').length : Infinity;
  const names: string[] = [];
  for (const match of body.matchAll(NAME)) {
    const before = body.slice(0, match.index).trimEnd();
    if (before.endsWith(".") || before.endsWith("|")) continue;
    // A test (`value is string`, `x is not number`) is Jinja's, not a column.
    if (/\bis(\s+not)?$/.test(before)) continue;
    // A keyword argument's label (`sort(reverse=true)`) is not a column; its value still is.
    const after = body.slice((match.index ?? 0) + match[0].length);
    if (/^\s*=(?!=)/.test(after)) continue;
    if (boundAt(scopes, match[0], at)) continue;
    // A loop's filter reads its own variables: `for x in items if x > 0`.
    if (part?.bound.includes(match[0]) && (match.index ?? 0) >= filterAt)
      continue;
    // `{{gross-discount}}` subtracts two columns unless one column has that hyphenated name: both
    // readings count, as `recipe_import._template_names` reads them.
    names.push(
      match[0],
      ...(match[0].includes("-") ? match[0].split("-").filter(Boolean) : []),
    );
  }
  return names;
}

function templateNames(text: string): string[] {
  const scopes = jinjaScopes(text);
  // A raw block's insides are literal text.
  const raw = rawSpans(text);
  return [...text.matchAll(BLOCK)].flatMap((block) =>
    inRaw(raw, block.index ?? 0)
      ? []
      : tagNames(block[1], block[2], block.index ?? 0, scopes),
  );
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

/** Tool profiles kept only as ui nodes: added but with no tool server yet. */
export function pendingToolProfiles(group: FlowGroup): string[] {
  const nodes = Array.isArray(group.ui?.nodes) ? group.ui.nodes : [];
  const saved = new Set(
    (Array.isArray(group.config?.tool_configs)
      ? group.config.tool_configs
      : []
    ).map((t) => (t as { tool_alias?: string }).tool_alias),
  );
  return nodes.flatMap((node) => {
    const n = node as { node_type?: string; id?: string };
    return n.node_type === "tool_config" && n.id && !saved.has(n.id)
      ? [n.id]
      : [];
  });
}

/** The recipe JSON a group stands for: its config plus its columns in canvas order. */
export function groupPayload(group: FlowGroup, columns: FlowNode[]) {
  return {
    recipe: { ...(group.config ?? {}), columns: columns.map(columnOf) },
    run: group.run ?? {},
    ui: group.ui ?? {},
  };
}

/** Everything the group's recipe is made of, to notice it changing under an open form. */
export function recipeSignature(graph: DocGraph, groupId: string): string {
  const group = graph.groups[groupId];
  return group
    ? JSON.stringify(groupPayload(group, groupColumns(graph, groupId)))
    : "";
}

/** The first loop of columns reading each other (`a -> b -> a`), else null; mirrors recipe_compile's check. */
export function columnLoop(columns: RecipeColumn[]): string[] | null {
  const names = new Set(
    columns.flatMap((c) => (typeof c.name === "string" ? [c.name] : [])),
  );
  const reads = new Map(
    columns.flatMap((c) =>
      typeof c.name === "string"
        ? [[c.name, columnReferences(c, names)] as const]
        : [],
    ),
  );
  const state = new Map<string, 1 | 2>();
  const visit = (name: string, path: string[]): string[] | null => {
    if (state.get(name) === 2) return null;
    if (state.get(name) === 1) return [...path.slice(path.indexOf(name)), name];
    state.set(name, 1);
    for (const ref of reads.get(name) ?? []) {
      const loop = visit(ref, [...path, name]);
      if (loop) return loop;
    }
    state.set(name, 2);
    return null;
  };
  for (const name of reads.keys()) {
    const loop = visit(name, []);
    if (loop) return loop;
  }
  return null;
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

const TEXT_FIELDS = ["prompt", "system_prompt", "expr"];

/** The template without the `{{ ... }}` tags that read `ref`, plain (`{{ ref }}`) or not (`{{ ref.key }}`,
 * `{{ ref | upper }}`); `{% ... %}` tags are left, since dropping one would break the template. */
function withoutOutputTags(template: string, ref: string): string {
  // Read against the whole template, so a loop's own variable of that name is kept. A tag is removed
  // line by line, so a line left empty by it goes too (removeRef's rule).
  const scopes = jinjaScopes(template);
  const raw = rawSpans(template);
  let offset = 0;
  return template
    .split("\n")
    .flatMap((line) => {
      const start = offset;
      offset += line.length + 1;
      const next = line.replace(
        /\{\{([\s\S]*?)\}\}/g,
        (tag, output: string, at: number) =>
          !inRaw(raw, start + at) &&
          tagNames(output, undefined, start + at, scopes).includes(ref)
            ? ""
            : tag,
      );
      if (next === line) return [line];
      return next.trim() ? [next.replace(/\s+$/g, "")] : [];
    })
    .join("\n");
}

/** Whether a column's prompts still read `ref` (in a `{% ... %}` tag) after its output tags were removed. */
export function templateStillReads(column: RecipeColumn, ref: string): boolean {
  return TEXT_FIELDS.some(
    (key) =>
      typeof column[key] === "string" &&
      templateNames(column[key] as string).includes(ref),
  );
}

/** The column without `{{ ref }}` in its prompts or targets; samplers keep their parent. */
export function withoutReference(
  column: RecipeColumn,
  ref: string,
): RecipeColumn {
  const next: RecipeColumn = { ...column };
  for (const key of TEXT_FIELDS) {
    if (typeof next[key] === "string")
      next[key] = withoutOutputTags(next[key] as string, ref);
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
      // "local": whatever model Chat has loaded, which the client check and the run both accept.
      model_configs: [{ alias: "writer", model: "local", provider: "local" }],
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
