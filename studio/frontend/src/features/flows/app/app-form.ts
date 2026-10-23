// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FlowDocument, FlowNode, RunEvent } from "../types";

export type AppFieldKind =
  | "text"
  | "textarea"
  | "number"
  | "bool"
  | "select"
  | "json"
  | "image"
  | "audio"
  | "video"
  | "document"
  | "list";

/** One form field of a flow opened as an app: a Flow Input (by name) or an input block (by node id). */
export interface AppField {
  key: string;
  target: "param" | "node";
  kind: AppFieldKind;
  label: string;
  initial: unknown;
  options?: string[];
}

/** The param that holds each input block's value; mirrors INPUT_VALUE_PARAM in core/flows/nodes/inputs.py. */
const NODE_FIELDS: Record<string, { param: string; kind: AppFieldKind }> = {
  "input.text": { param: "value", kind: "textarea" },
  "input.number": { param: "value", kind: "number" },
  "input.toggle": { param: "value", kind: "bool" },
  "input.select": { param: "value", kind: "select" },
  "input.seed": { param: "value", kind: "number" },
  "input.image": { param: "asset", kind: "image" },
  "input.audio": { param: "asset", kind: "audio" },
  "input.video": { param: "asset", kind: "video" },
  "input.file": { param: "asset", kind: "document" },
  "input.list": { param: "items", kind: "list" },
};

const PARAM_KINDS: Record<string, AppFieldKind> = {
  text: "textarea",
  number: "number",
  bool: "bool",
  json: "json",
  image: "image",
  audio: "audio",
};

/** Read order on the canvas: top to bottom, then left to right. */
function byCanvas(a: FlowNode, b: FlowNode): number {
  return a.pos[1] - b.pos[1] || a.pos[0] - b.pos[0];
}

/** Nodes inside a For Each body; mirrors _members in core/flows/compiler.py (a plain frame is not a loop). */
export function loopMembers(nodes: FlowNode[]): Set<string> {
  const loops = new Set<string>();
  for (const node of nodes) {
    if (node.type !== "logic.for_each") continue;
    loops.add(node.id);
    if (node.group) loops.add(node.group);
  }
  const members = new Set<string>();
  for (const node of nodes) {
    if (!node.group || !loops.has(node.group)) continue;
    const ownFrame =
      node.type === "logic.for_each" &&
      !nodes.some(
        (n) =>
          n.id !== node.id &&
          n.type === "logic.for_each" &&
          (n.id === node.group || n.group === node.group),
      );
    if (!ownFrame) members.add(node.id);
  }
  return members;
}

export function appFields(doc: FlowDocument): AppField[] {
  const fields: AppField[] = [];
  const seen = new Set<string>();
  for (const param of doc.params ?? []) {
    seen.add(param.id);
    fields.push({
      key: param.id,
      target: "param",
      kind: PARAM_KINDS[param.type] ?? "textarea",
      label: param.label || param.id,
      initial: param.default ?? "",
    });
  }
  const inLoop = loopMembers(doc.nodes);
  const nodes = doc.nodes
    .filter((n) => !n.disabled && !inLoop.has(n.id))
    .sort(byCanvas);
  for (const node of nodes) {
    if (node.type === "input.param") {
      const name = String(node.params.name ?? "").trim();
      if (!name || seen.has(name)) continue;
      seen.add(name);
      fields.push({
        key: name,
        target: "param",
        kind: PARAM_KINDS[String(node.params.type ?? "text")] ?? "textarea",
        label: node.label || name,
        initial: node.params.default ?? "",
      });
      continue;
    }
    const known = NODE_FIELDS[node.type];
    if (!known) continue;
    fields.push({
      key: node.id,
      target: "node",
      kind: known.kind,
      label:
        node.label ||
        (known.kind === "textarea" ? "Text" : node.type.split(".")[1]),
      initial: node.params[known.param] ?? null,
      options:
        known.kind === "select"
          ? ((node.params.options as unknown[]) ?? []).map(String)
          : undefined,
    });
  }
  return fields;
}

/** The document and run params to start with: input blocks hold the form's values, Flow Inputs go in params. */
export function applyAppValues(
  doc: FlowDocument,
  fields: AppField[],
  values: Record<string, unknown>,
): { doc: FlowDocument; params: Record<string, unknown> } {
  const params: Record<string, unknown> = {};
  const nodeValues = new Map<string, unknown>();
  for (const field of fields) {
    if (!(field.key in values)) continue;
    // An untouched block field sends nothing: the block keeps its own value (and a seed its randomize setting). A
    // Flow Input always sends its value, since a run does not fill in declared defaults itself.
    if (field.target !== "param" && Object.is(values[field.key], field.initial))
      continue;
    let value = values[field.key];
    // The JSON box edits text; a Flow Input of type json is not coerced server-side, so send the parsed value.
    if (field.kind === "json" && typeof value === "string") {
      try {
        value = JSON.parse(value);
      } catch {}
    }
    if (field.target === "param") params[field.key] = value;
    else nodeValues.set(field.key, value);
  }
  const nodes = doc.nodes.map((node) => {
    if (!nodeValues.has(node.id)) return node;
    const known = NODE_FIELDS[node.type];
    if (!known) return node;
    const next = { ...node.params, [known.param]: nodeValues.get(node.id) };
    // A typed-in seed is the seed to use; the untouched default keeps the block's randomize setting.
    if (
      node.type === "input.seed" &&
      Number(next.value) !== Number(node.params.value)
    )
      next.randomize = false;
    return { ...node, params: next };
  });
  return { doc: { ...doc, nodes }, params };
}

/** Output blocks shown as the app's results, in canvas order. */
export function appOutputs(doc: FlowDocument): FlowNode[] {
  const inLoop = loopMembers(doc.nodes);
  return doc.nodes
    .filter(
      (n) =>
        !n.disabled &&
        !inLoop.has(n.id) &&
        (n.type === "output.preview" || n.type === "output.result"),
    )
    .sort(byCanvas);
}

export interface AppRunState {
  status: string;
  done: number;
  total: number;
  results: Record<string, unknown>;
  errors: string[];
}

/** Fold run events into what the app shows: progress over the run's steps, output values, failures. */
export function foldAppEvents(
  state: AppRunState,
  events: RunEvent[],
  outputIds: Set<string>,
): AppRunState {
  let next = state;
  for (const event of events) {
    const data = event.data ?? {};
    if (event.type === "run.started") {
      const order = Array.isArray(data.order) ? data.order : [];
      next = { ...next, status: "running", total: order.length };
    } else if (event.type === "node.status" && event.node_id) {
      const status = String(data.status ?? "");
      if (["done", "cached", "skipped", "failed", "cancelled"].includes(status))
        next = { ...next, done: next.done + 1 };
      const reason = data.error ?? data.message;
      if (status === "failed" && typeof reason === "string")
        next = { ...next, errors: [...next.errors, reason] };
    } else if (
      event.type === "node.output" &&
      event.node_id &&
      outputIds.has(event.node_id)
    ) {
      const outputs = (data.outputs ?? {}) as Record<string, unknown>;
      next = {
        ...next,
        results: { ...next.results, [event.node_id]: outputs.value },
      };
    } else if (event.type === "snapshot") {
      // A run reconciled after a restart ends with a terminal snapshot and no run.* event.
      const run = (data.run ?? {}) as { status?: string; error?: unknown };
      const nodes = (data.nodes ?? {}) as Record<
        string,
        { outputs?: Record<string, unknown> | null }
      >;
      const results = { ...next.results };
      for (const id of outputIds) {
        const outputs = nodes[id]?.outputs;
        if (outputs && "value" in outputs) results[id] = outputs.value;
      }
      next = { ...next, results };
      if (
        run.status &&
        ["completed", "failed", "cancelled", "interrupted"].includes(run.status)
      ) {
        const error = typeof run.error === "string" ? run.error : null;
        next = {
          ...next,
          status: run.status,
          errors:
            error && !next.errors.includes(error)
              ? [...next.errors, error]
              : next.errors,
        };
      }
    } else if (event.type.startsWith("run.") && event.type !== "run.queued") {
      const ended = event.type.slice(4);
      if (["completed", "failed", "cancelled", "interrupted"].includes(ended)) {
        const error = typeof data.error === "string" ? data.error : null;
        next = {
          ...next,
          status: ended,
          errors: error ? [...next.errors, error] : next.errors,
        };
      }
    }
  }
  return next;
}
