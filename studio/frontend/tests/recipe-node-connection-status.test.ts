// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { edgeAdjacency, localProviderNamesKey, nodeConnectionStatus } =
  await import("../src/features/recipe-studio/utils/node-connection-status.ts");
const { INFRA_NODE_KINDS } = await import(
  "../src/features/recipe-studio/types/index.ts"
);

type Edge = { id: string; source: string; target: string };
type Config = { id: string; kind: string; name: string; is_local?: boolean };
type Status = { isDisconnected: boolean; missingDataInput: boolean };

// The per-node edge scan this replaces, kept verbatim as the oracle.
function scanEdges(
  nodeId: string,
  config: Config | undefined,
  edges: Edge[],
  configs: Record<string, Config>,
): Status {
  if (!config || config.kind === "markdown_note") {
    return { isDisconnected: false, missingDataInput: false };
  }
  const nodeEdges = edges.filter(
    (e) => e.source === nodeId || e.target === nodeId,
  );
  const isDisconnected = nodeEdges.length === 0;
  let missingDataInput = false;
  if (config.kind === "llm" && !isDisconnected) {
    const hasDataEdge = nodeEdges.some((e) => {
      const otherId = e.source === nodeId ? e.target : e.source;
      const otherConfig = configs[otherId];
      return otherConfig && !INFRA_NODE_KINDS.has(otherConfig.kind);
    });
    missingDataInput = !hasDataEdge;
  }
  return { isDisconnected, missingDataInput };
}

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0;
    return s / 2 ** 32;
  };
}

const KINDS = [
  "llm",
  "sampler",
  "expression",
  "seed",
  "validator",
  "markdown_note",
  "model_provider",
  "model_config",
  "tool_config",
];

function randomGraph(seed: number) {
  const r = rng(seed);
  const pick = <T>(xs: T[]): T => xs[Math.floor(r() * xs.length)] as T;
  const ids = Array.from(
    { length: 2 + Math.floor(r() * 30) },
    (_, i) => `n${i}`,
  );
  ids.push("__proto__", "constructor", "toString");
  const configs: Record<string, Config> = {};
  for (const id of ids) {
    if (r() < 0.85 && id !== "__proto__")
      configs[id] = { id, kind: pick(KINDS), name: id };
  }
  const edges: Edge[] = [];
  const count = Math.floor(r() * ids.length * 2);
  for (let i = 0; i < count; i++) {
    const source = pick(ids);
    const target = r() < 0.08 ? source : pick(ids);
    edges.push({ id: `e${i}`, source, target });
  }
  return { ids: [...ids, "missing"], configs, edges };
}

test("every node's status matches the per-node edge scan on random graphs", () => {
  for (let seed = 1; seed <= 400; seed++) {
    const { ids, configs, edges } = randomGraph(seed);
    for (const id of ids) {
      for (const config of [
        configs[id],
        undefined,
        { id, kind: "llm", name: id },
      ]) {
        assert.deepEqual(
          {
            ...nodeConnectionStatus(
              id,
              config as never,
              edges as never,
              configs as never,
            ),
          },
          scanEdges(id, config, edges, configs),
          `seed ${seed}, node ${id}, kind ${config?.kind}`,
        );
      }
    }
  }
});

test("the adjacency index is built once per edges array and a status is a shared value", () => {
  const edges: Edge[] = [{ id: "a", source: "x", target: "y" }];
  assert.equal(edgeAdjacency(edges as never), edgeAdjacency(edges as never));
  const configs = {
    x: { id: "x", kind: "llm", name: "x" },
    y: { id: "y", kind: "sampler", name: "y" },
  };
  // Configs change per keystroke; an unchanged status must keep its identity or every node re-renders.
  const before = nodeConnectionStatus(
    "x",
    configs.x as never,
    edges as never,
    configs as never,
  );
  const typed = { ...configs, y: { ...configs.y } };
  assert.equal(
    nodeConnectionStatus("x", typed.x as never, edges as never, typed as never),
    before,
  );
});

test("the local provider key changes only with the local provider names", () => {
  const base = {
    a: { id: "a", kind: "model_provider", name: "remote" },
    b: { id: "b", kind: "model_provider", name: "lp", is_local: true },
    c: { id: "c", kind: "llm", name: "q", prompt: "hi" },
  };
  const key = localProviderNamesKey(base as never);
  assert.equal(key, JSON.stringify(["lp"]));
  assert.equal(
    localProviderNamesKey({
      ...base,
      c: { ...base.c, prompt: "hi there" },
    } as never),
    key,
  );
  assert.equal(
    localProviderNamesKey({
      ...base,
      a: { ...base.a, is_local: true },
    } as never),
    JSON.stringify(["remote", "lp"]),
  );
  assert.equal(
    localProviderNamesKey({ ...base, b: { ...base.b, name: "lp2" } } as never),
    JSON.stringify(["lp2"]),
  );
});
