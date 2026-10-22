// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type {
  FlowDocument,
  FlowEdge,
  FlowNode,
  NodeCatalog,
} from "../src/features/flows/types.ts";

export const catalog = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL("../src/features/flows/catalog.snapshot.json", import.meta.url),
    ),
    "utf8",
  ),
) as NodeCatalog;

export const specs = Object.fromEntries(catalog.nodes.map((s) => [s.type, s]));

export function node(
  id: string,
  type: string,
  pos: [number, number] = [0, 0],
  params: Record<string, unknown> = {},
): FlowNode {
  return { id, type, v: 1, params, pos };
}

export function edge(
  id: string,
  source: string,
  sp: string,
  target: string,
  tp: string,
  kind: "data" | "flow" = "data",
): FlowEdge {
  return { id, source, source_port: sp, target, target_port: tp, kind };
}

export function doc(nodes: FlowNode[], edges: FlowEdge[] = []): FlowDocument {
  return {
    format: "unsloth-flow",
    version: 1,
    id: "f1",
    name: "Test",
    description: "",
    kind: "pipeline",
    seed: 1,
    params: [],
    requirements: { models: [] },
    nodes,
    edges,
    groups: [],
    notes: [],
    assets: [],
    baked: {},
    editor: {},
    meta: {},
  };
}
