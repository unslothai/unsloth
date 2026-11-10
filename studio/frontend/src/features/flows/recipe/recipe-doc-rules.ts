// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The recipe rules the document store applies to framing, pasting and rerouting. Plain functions over
// flow records: no React and none of the recipe editor, so the store stays light.

import { renameRoot, replaceRef } from "@/features/recipe-studio/utils/refs";
import type { FlowEdge, FlowGroup, FlowNode } from "../types";
import { COLUMN_TYPE, GENERATE_TYPE, referenceEdges } from "./recipe-columns";

/** A column or Recipe Dataset step that runs only in its recipe: framing or grouping never moves it out. */
export function recipeOwned(
  node: FlowNode,
  groups: Record<string, FlowGroup>,
): boolean {
  return (
    (node.type === COLUMN_TYPE || node.type === GENERATE_TYPE) &&
    !!node.group &&
    groups[node.group]?.kind === "recipe"
  );
}

/** A recipe column's name, from its params. */
export function columnNameOf(node: FlowNode): string | null {
  const column = (node.params as { column?: { name?: unknown } } | undefined)
    ?.column;
  return typeof column?.name === "string" ? column.name : null;
}

/** A column pasted or duplicated into a recipe that already has its name gets the next free one,
 * since a recipe's column names are unique. */
export function uniqueColumn(
  node: FlowNode,
  group: string | null,
  nodes: Record<string, FlowNode>,
  order: string[],
  config?: Record<string, unknown>,
): FlowNode {
  const name = columnNameOf(node);
  if (node.type !== COLUMN_TYPE || !group || !name) return node;
  const taken = new Set<unknown>([
    ...order
      .filter(
        (id) => nodes[id].type === COLUMN_TYPE && nodes[id].group === group,
      )
      .map((id) => columnNameOf(nodes[id])),
    // Columns share one namespace with the recipe's other blocks: two of one name never open again.
    ...setupNames(config),
  ]);
  if (!taken.has(name)) return node;
  let n = 2;
  while (taken.has(`${name}_${n}`)) n += 1;
  const next = `${name}_${n}`;
  const params = node.params as { column: Record<string, unknown> };
  return {
    ...node,
    label: next,
    params: { ...node.params, column: { ...params.column, name: next } },
  };
}

/** The names a recipe's providers, model presets, tool profiles and seed take. */
function setupNames(config: Record<string, unknown> | undefined): unknown[] {
  if (!config) return [];
  const pick = (key: string, field: string) =>
    Array.isArray(config[key])
      ? (config[key] as Record<string, unknown>[]).map((item) => item?.[field])
      : [];
  return [
    ...pick("model_providers", "name"),
    ...pick("mcp_providers", "name"),
    ...pick("model_configs", "alias"),
    ...pick("tool_configs", "tool_alias"),
    ...(config.seed_config ? ["seed"] : []),
  ];
}

/** Fields that name another column outright, not through a Jinja reference (`column_name` is an
 * AI column's image context). */
const COLUMN_NAME_FIELDS = new Set([
  "category",
  "reference_column_name",
  "subcategory_parent",
  "column_name",
]);

/** A column's definition with column ``from`` renamed to ``to`` wherever it reads it: Jinja
 * references, the fields that name a column, a validator's target list and sampler conditions. */
function renameInColumn(
  value: unknown,
  from: string,
  to: string,
  key = "",
): unknown {
  if (typeof value === "string")
    return COLUMN_NAME_FIELDS.has(key) && value === from
      ? to
      : replaceRef(value, from, to);
  if (Array.isArray(value))
    return key === "target_columns"
      ? value.map((v) => (v === from ? to : v))
      : value.map((v) => renameInColumn(v, from, to));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [
        key === "conditional_params" ? renameRoot(k, from, to) : k,
        renameInColumn(v, from, to, k),
      ]),
    );
  return value;
}

/** A recipe column's wire is a reference in its reader's prompt, not a plain wire. */
export function isRecipeWire(
  nodes: Record<string, FlowNode>,
  wire: { source: string; target: string },
): boolean {
  return (
    nodes[wire.source]?.type === COLUMN_TYPE ||
    nodes[wire.target]?.type === COLUMN_TYPE
  );
}

/** A column or Recipe Dataset step landing outside a recipe frame. */
export function outsideRecipe(
  node: FlowNode,
  group: string | null,
  groups: Record<string, FlowGroup>,
): boolean {
  return (
    (node.type === COLUMN_TYPE || node.type === GENERATE_TYPE) &&
    groups[group ?? ""]?.kind !== "recipe"
  );
}

/** A recipe runs from one Recipe Dataset step: a pasted or duplicated second one comes in off. */
export function extraDataset(
  node: FlowNode,
  group: string | null,
  nodes: Record<string, FlowNode>,
  order: string[],
): boolean {
  return (
    node.type === GENERATE_TYPE &&
    !node.disabled &&
    !!group &&
    order.some(
      (id) =>
        nodes[id].type === GENERATE_TYPE &&
        nodes[id].group === group &&
        !nodes[id].disabled,
    )
  );
}

/** A pasted column with the renames its fellow copies took (`[from, to]`) applied to what it reads. */
export function renamePastedColumn(
  node: FlowNode,
  pairs: [string, string][],
): FlowNode {
  if (node.type !== COLUMN_TYPE) return node;
  const own = columnNameOf(node);
  let column = (node.params as { column: Record<string, unknown> }).column;
  // Through placeholders, so a new name that is another copy's old name (topic -> topic_2 while
  // topic_2 -> topic_2_2) is not renamed a second time.
  const held = pairs.map((_, i) => `__flows_paste_${i}__`);
  pairs.forEach(([from], i) => {
    column = renameInColumn(column, from, held[i]) as Record<string, unknown>;
  });
  pairs.forEach(([, to], i) => {
    column = renameInColumn(column, held[i], to) as Record<string, unknown>;
  });
  column = { ...column, name: own };
  return { ...node, params: { ...node.params, column } };
}

/** A pasted column reading one that was not copied with it reads the recipe's own: it gets that
 * wire too, so deleting the column it reads reaches it. The edges come without ids. */
export function pastedReferenceEdges(
  nodes: Record<string, FlowNode>,
  nodeOrder: string[],
  edges: Record<string, FlowEdge>,
  edgeOrder: string[],
  added: string[],
): FlowEdge[] {
  const pasted = new Set(added.filter((id) => nodes[id].type === COLUMN_TYPE));
  const wired = new Set(
    edgeOrder.map((id) => `${edges[id].source}>${edges[id].target}`),
  );
  const result: FlowEdge[] = [];
  for (const group of new Set([...pasted].map((id) => nodes[id].group))) {
    if (!group) continue;
    const columns = nodeOrder
      .map((id) => nodes[id])
      .filter((n) => n.type === COLUMN_TYPE && n.group === group);
    for (const edge of referenceEdges(columns)) {
      if (
        !pasted.has(edge.target) ||
        wired.has(`${edge.source}>${edge.target}`)
      )
        continue;
      result.push(edge);
    }
  }
  return result;
}
