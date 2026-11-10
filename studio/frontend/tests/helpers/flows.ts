// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type {
  FlowDocument,
  FlowEdge,
  FlowNode,
  NodeCatalog,
} from "../../src/features/flows/types.ts";

export const catalog = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL("../fixtures/flows-catalog.json", import.meta.url),
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

/** A `src/features/flows` file read as text, for the few source guards with no behavioural hook. */
export function readFlowsSrc(relative: string): string {
  return readFileSync(
    new URL(`../../src/features/flows/${relative}`, import.meta.url),
    "utf8",
  );
}

/** A story manifest fixture from tests/fixtures (`flows-<name>.story.json`). */
export function storyFixture<T = import("../../src/features/flows/types.ts").StoryManifest>(
  name: string,
): T {
  return JSON.parse(
    readFileSync(
      new URL(`../fixtures/flows-${name}.story.json`, import.meta.url),
      "utf8",
    ),
  ) as T;
}

/** Drains pending promise callbacks without a timer: enough turns for a few chained `.then`s. */
export async function flushMicrotasks(turns = 20): Promise<void> {
  for (let i = 0; i < turns; i++) await Promise.resolve();
}

/** A bundled Data Recipes learning recipe (`src/features/data-recipes/learning-recipes/<name>`). */
export type LearningRecipe = {
  recipe: Record<string, unknown> & { columns: Record<string, unknown>[] };
  ui?: Record<string, unknown>;
};

export function learningRecipe(name: string): LearningRecipe {
  return JSON.parse(
    readFileSync(
      new URL(`../../src/features/data-recipes/learning-recipes/${name}`, import.meta.url),
      "utf8",
    ),
  ) as LearningRecipe;
}

/** A recipe payload as a Flows recipe group (id "recipe") and its `recipe.column` nodes. */
export function recipeGroup(payload: LearningRecipe, label = "Recipe") {
  const { columns, ...config } = structuredClone(payload.recipe);
  const nodes: FlowNode[] = columns.map((column, i) => ({
    id: `recipe_col_${i + 1}`,
    type: "recipe.column",
    v: 1,
    params: { column },
    pos: [i * 320, 0],
    label: String(column.name),
    group: "recipe",
  }));
  const group = {
    id: "recipe",
    label,
    rect: [0, 0, 100, 100] as [number, number, number, number],
    kind: "recipe" as const,
    config,
    run: { rows: 5 },
  };
  return { group, nodes };
}

/** The three-card chain Text -> Generate -> Preview (ids a, b, c; edges e1, e2). */
export function chainDoc(): FlowDocument {
  return doc(
    [
      node("a", "input.text"),
      node("b", "llm.generate", [300, 0]),
      node("c", "output.preview", [600, 0]),
    ],
    [
      edge("e1", "a", "text", "b", "prompt"),
      edge("e2", "b", "text", "c", "value"),
    ],
  );
}

/** A promise with its resolve/reject exposed, so a test decides when an API call answers. */
export function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
}

/** The first stub element carrying `data-testid={id}`. */
export function byTestId<E extends { props: Record<string, unknown> }>(els: E[], id: string): E | undefined {
  return els.find((e) => e.props["data-testid"] === id);
}

/** A story manifest: no vars, no characters, fixed mode unless `extra` says otherwise. */
export function storyManifest(
  scenes: import("../../src/features/flows/types.ts").StoryScene[],
  extra: Partial<import("../../src/features/flows/types.ts").StoryManifest> = {},
): import("../../src/features/flows/types.ts").StoryManifest {
  return { title: "T", start: "a", vars: {}, characters: [], mode: "fixed", ...extra, scenes };
}
