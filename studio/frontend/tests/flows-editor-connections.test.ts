// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, edge, specs } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const rules = await import("../src/features/flows/editor/connection-rules.ts");
const {
  checkConnection,
  typesCompatible,
  wouldCreateCycle,
  compatibleSpecs,
  inHandle,
  outHandle,
  flowOutHandle,
  FLOW_IN_HANDLE,
} = rules;

test("matching types connect; mismatched ones are refused", () => {
  const ok = checkConnection(
    "a",
    specs["input.text"],
    outHandle("text"),
    "b",
    specs["llm.generate"],
    inHandle("prompt"),
  );
  assert.equal(ok.ok, true);
  assert.equal(ok.kind, "data");
  const bad = checkConnection(
    "a",
    specs["input.number"],
    outHandle("value"),
    "b",
    specs["llm.generate"],
    inHandle("prompt"),
  );
  assert.equal(bad.ok, false);
  assert.match(bad.reason ?? "", /number cannot connect to text/);
});

// [why, source id, source type, source handle, target id, target type, target handle, ok]
const cases: [string, string, string, string, string, string, string, boolean][] = [
  ["`any` accepts an image", "a", "image.generate", outHandle("image"), "b", "output.preview", inHandle("value"), true],
  ["a list of images feeds a list of images", "a", "image.generate", outHandle("images"), "b", "image.grid", inHandle("images"), true],
  ["a list of text is not a list of images", "a", "input.list", outHandle("items"), "b", "image.grid", inHandle("images"), false],
  ["a story path joins a story input", "s", "story.scene", flowOutHandle("next"), "c", "story.choice", FLOW_IN_HANDLE, true],
  ["a story path does not join a data input", "s", "story.scene", flowOutHandle("next"), "c", "llm.generate", inHandle("prompt"), false],
  ["Story has no incoming path", "s", "story.start", flowOutHandle("next"), "c", "story.start", FLOW_IN_HANDLE, false],
  ["an input does not connect to an input", "a", "llm.generate", inHandle("prompt"), "b", "llm.generate", inHandle("prompt"), false],
  ["a node cannot feed itself", "a", "llm.generate", outHandle("text"), "a", "llm.generate", inHandle("prompt"), false],
];
for (const [why, sid, stype, sh, tid, ttype, th, ok] of cases)
  test(`connection: ${why}`, () => {
    assert.equal(checkConnection(sid, specs[stype], sh, tid, specs[ttype], th).ok, ok);
  });

test("a story path connects as a flow edge into the flow port", () => {
  const ok = checkConnection(
    "s",
    specs["story.scene"],
    flowOutHandle("next"),
    "c",
    specs["story.choice"],
    FLOW_IN_HANDLE,
  );
  assert.equal(ok.kind, "flow");
  assert.equal(ok.targetPort, "flow");
});

test("data cycles are detected; story loops are allowed", () => {
  const edges = [
    edge("1", "a", "text", "b", "prompt"),
    edge("2", "b", "text", "c", "value"),
    edge("f", "c", "next", "a", "flow", "flow"),
  ];
  assert.equal(wouldCreateCycle(edges, "c", "a"), true);
  assert.equal(wouldCreateCycle(edges, "a", "c"), false);
  assert.equal(
    wouldCreateCycle(
      [edge("f", "s2", "next", "s1", "flow", "flow")],
      "s1",
      "s2",
    ),
    false,
  );
});

test("drag-to-empty suggests nodes that can take the value", () => {
  const port =
    specs["image.generate"].outputs.find((p) => p.id === "image") ?? null;
  const found = compatibleSpecs(catalog.nodes, {
    side: "out",
    kind: "data",
    port,
  }).map((m) => m.spec.type);
  assert.ok(found.includes("image.remove_bg"));
  assert.ok(found.includes("output.preview"));
  assert.ok(!found.includes("input.text"));
  const flows = compatibleSpecs(catalog.nodes, {
    side: "out",
    kind: "flow",
    port: null,
  }).map((m) => m.spec.type);
  assert.ok(flows.includes("story.scene") && flows.includes("story.ending"));
  assert.ok(!flows.includes("story.start"));
});

test("a For Each body can feed its result back to collect", () => {
  const edges = [
    edge("1", "fe", "item", "t", "values"),
    edge("2", "t", "text", "fe", "collect"),
  ];
  assert.equal(
    wouldCreateCycle(edges.slice(0, 1), "t", "fe", "collect"),
    false,
  );
  assert.equal(wouldCreateCycle(edges.slice(1), "fe", "t", "values"), false);
  assert.equal(wouldCreateCycle(edges.slice(0, 1), "t", "fe", "items"), true);
});

const port = (type: string, item?: string) => ({ id: "p", type, ...(item ? { item } : {}) }) as never;
for (const [from, to, ok] of [
  [port("list", "text"), port("json"), true],
  [port("text"), port("list"), true],
  [port("text"), port("list", "text"), true],
  [port("image"), port("list", "text"), false],
  [port("image"), port("json"), false],
  [port("mask"), port("image"), true],
  [port("image"), port("mask"), true],
  [port("mask"), port("audio"), false],
] as const)
  test(`the editor accepts the compiler's coercions: ${JSON.stringify(from)} -> ${JSON.stringify(to)} is ${ok}`, () => {
    assert.equal(typesCompatible(from, to), ok);
  });
