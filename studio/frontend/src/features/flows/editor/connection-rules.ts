// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  EdgeKind,
  FlowEdge,
  NodeSpec,
  PortSpec,
  PortType,
} from "../types";

export const FLOW_TARGET_PORT = "flow";

export function inHandle(port: string): string {
  return `i:${port}`;
}
export function outHandle(port: string): string {
  return `o:${port}`;
}
export function flowOutHandle(name: string): string {
  return `fo:${name}`;
}
export const FLOW_IN_HANDLE = "fi";

export interface ParsedHandle {
  kind: EdgeKind;
  side: "in" | "out";
  port: string;
}

export function parseHandle(
  handle: string | null | undefined,
): ParsedHandle | null {
  if (!handle) return null;
  if (handle === FLOW_IN_HANDLE)
    return { kind: "flow", side: "in", port: FLOW_TARGET_PORT };
  if (handle.startsWith("fo:"))
    return { kind: "flow", side: "out", port: handle.slice(3) };
  // A recipe column's port per column it reads ("i:in@<source>") is still its "in" port.
  if (handle.startsWith("i:"))
    return { kind: "data", side: "in", port: handle.slice(2).split("@")[0] };
  if (handle.startsWith("o:"))
    return { kind: "data", side: "out", port: handle.slice(2) };
  return null;
}

export function edgeHandles(edge: FlowEdge): {
  sourceHandle: string;
  targetHandle: string;
} {
  if (edge.kind === "flow") {
    return {
      sourceHandle: flowOutHandle(edge.source_port),
      targetHandle: FLOW_IN_HANDLE,
    };
  }
  return {
    sourceHandle: outHandle(edge.source_port),
    targetHandle: inHandle(edge.target_port),
  };
}

const JSON_LIKE: ReadonlySet<PortType> = new Set<PortType>([
  "text",
  "number",
  "bool",
  "json",
  "list",
  "story",
  "world",
  "table",
  "chart",
  "heatmap",
  "tensor",
]);

/** Whether a value of `source` type may flow into a `target` port; mirrors the compiler's `port_compatible`. */
export function typesCompatible(source: PortSpec, target: PortSpec): boolean {
  const s: PortType = source.type;
  const t: PortType = target.type;
  if (s === "flow" || t === "flow") return false;
  if (s === "any" || t === "any") return true;
  if (s === t) {
    if (s === "list" && source.item && target.item) {
      return (
        source.item === target.item ||
        source.item === "any" ||
        target.item === "any"
      );
    }
    return true;
  }
  if (t === "json" && JSON_LIKE.has(s)) return true;
  if ((s === "image" && t === "mask") || (s === "mask" && t === "image"))
    return true;
  return (
    t === "list" && (!target.item || target.item === "any" || target.item === s)
  );
}

export interface ConnectionCheck {
  ok: boolean;
  reason?: string;
  kind?: EdgeKind;
  sourcePort?: string;
  targetPort?: string;
}

export function checkConnection(
  sourceId: string,
  sourceSpec: NodeSpec | undefined,
  sourceHandle: string | null | undefined,
  targetId: string,
  targetSpec: NodeSpec | undefined,
  targetHandle: string | null | undefined,
): ConnectionCheck {
  if (!sourceSpec || !targetSpec)
    return { ok: false, reason: "Unknown node type" };
  const src = parseHandle(sourceHandle);
  const dst = parseHandle(targetHandle);
  if (!src || !dst || src.side !== "out" || dst.side !== "in") {
    return { ok: false, reason: "Connect an output to an input" };
  }
  if (src.kind !== dst.kind) {
    return { ok: false, reason: "Story paths only connect to story nodes" };
  }
  if (src.kind === "flow") {
    if (!sourceSpec.flow_out.includes(src.port) || !targetSpec.flow_in) {
      return { ok: false, reason: "Not a story path" };
    }
    return {
      ok: true,
      kind: "flow",
      sourcePort: src.port,
      targetPort: FLOW_TARGET_PORT,
    };
  }
  if (sourceId === targetId)
    return { ok: false, reason: "A node cannot feed itself" };
  const out = sourceSpec.outputs.find((p) => p.id === src.port);
  const inp = targetSpec.inputs.find((p) => p.id === dst.port);
  if (!out || !inp) return { ok: false, reason: "Unknown port" };
  if (!typesCompatible(out, inp)) {
    return { ok: false, reason: `${out.type} cannot connect to ${inp.type}` };
  }
  return { ok: true, kind: "data", sourcePort: out.id, targetPort: inp.id };
}

const COLLECT_PORT = "collect";

export function wouldCreateCycle(
  edges: Iterable<FlowEdge>,
  sourceId: string,
  targetId: string,
  targetPort?: string,
): boolean {
  if (sourceId === targetId) return true;
  if (targetPort === COLLECT_PORT) return false;
  const out = new Map<string, string[]>();
  for (const edge of edges) {
    if (edge.kind !== "data" || edge.target_port === COLLECT_PORT) continue;
    const list = out.get(edge.source);
    if (list) list.push(edge.target);
    else out.set(edge.source, [edge.target]);
  }
  const stack = [targetId];
  const seen = new Set<string>();
  while (stack.length) {
    const id = stack.pop() as string;
    if (id === sourceId) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const next of out.get(id) ?? []) stack.push(next);
  }
  return false;
}

export function compatibleSpecs(
  specs: NodeSpec[],
  from: { side: "in" | "out"; kind: EdgeKind; port: PortSpec | null },
): { spec: NodeSpec; port: string }[] {
  const result: { spec: NodeSpec; port: string }[] = [];
  for (const spec of specs) {
    if (from.kind === "flow") {
      if (from.side === "out" && spec.flow_in)
        result.push({ spec, port: FLOW_TARGET_PORT });
      else if (from.side === "in" && spec.flow_out.length)
        result.push({ spec, port: spec.flow_out[0] });
      continue;
    }
    if (!from.port) continue;
    const candidates = from.side === "out" ? spec.inputs : spec.outputs;
    const match = candidates.find((p) =>
      from.side === "out"
        ? typesCompatible(from.port as PortSpec, p)
        : typesCompatible(p, from.port as PortSpec),
    );
    if (match) result.push({ spec, port: match.id });
  }
  return result;
}
