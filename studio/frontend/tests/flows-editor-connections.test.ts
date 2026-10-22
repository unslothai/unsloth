// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, edge, specs } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const rules = await import("../src/features/flows/editor/connection-rules.ts");
const {
  checkConnection,
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

test("`any` accepts everything and list items must agree", () => {
  assert.equal(
    checkConnection(
      "a",
      specs["image.generate"],
      outHandle("image"),
      "b",
      specs["output.preview"],
      inHandle("value"),
    ).ok,
    true,
  );
  assert.equal(
    checkConnection(
      "a",
      specs["image.generate"],
      outHandle("images"),
      "b",
      specs["image.grid"],
      inHandle("images"),
    ).ok,
    true,
  );
  assert.equal(
    checkConnection(
      "a",
      specs["input.list"],
      outHandle("items"),
      "b",
      specs["image.grid"],
      inHandle("images"),
    ).ok,
    false,
    "a list of text is not a list of images",
  );
});

test("story paths only join story handles", () => {
  const ok = checkConnection(
    "s",
    specs["story.scene"],
    flowOutHandle("next"),
    "c",
    specs["story.choice"],
    FLOW_IN_HANDLE,
  );
  assert.equal(ok.ok, true);
  assert.equal(ok.kind, "flow");
  assert.equal(ok.targetPort, "flow");
  assert.equal(
    checkConnection(
      "s",
      specs["story.scene"],
      flowOutHandle("next"),
      "c",
      specs["llm.generate"],
      inHandle("prompt"),
    ).ok,
    false,
  );
  assert.equal(
    checkConnection(
      "s",
      specs["story.start"],
      flowOutHandle("next"),
      "c",
      specs["story.start"],
      FLOW_IN_HANDLE,
    ).ok,
    false,
    "Story has no incoming path",
  );
});

test("inputs connect to outputs only, and a node cannot feed itself", () => {
  assert.equal(
    checkConnection(
      "a",
      specs["llm.generate"],
      inHandle("prompt"),
      "b",
      specs["llm.generate"],
      inHandle("prompt"),
    ).ok,
    false,
  );
  assert.equal(
    checkConnection(
      "a",
      specs["llm.generate"],
      outHandle("text"),
      "a",
      specs["llm.generate"],
      inHandle("prompt"),
    ).ok,
    false,
  );
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
