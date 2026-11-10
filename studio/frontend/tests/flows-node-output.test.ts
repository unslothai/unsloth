// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a card and the inspector show of a block's output, and when a rewired card drops it.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import {
  elementsOf,
  loadWithStubs,
  stubEveryImport,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { useRuntimeStore } = await import("../src/features/flows/stores/runtime-store.ts");
const { rewiredCards, watchRewiredOutputs, withDownstream } = await import("../src/features/flows/editor/stale-outputs.ts");

const flow = () =>
  doc(
    [node("a", "input.text"), node("b", "input.text", [0, 200]), node("p", "output.preview", [500, 0])],
    [edge("e1", "a", "text", "p", "value")],
  );

test("rewiring a card's input drops the output it showed; a card mid-run keeps its own", () => {
  useDocStore.getState().load(flow());
  const stop = watchRewiredOutputs();
  useRuntimeStore.getState().seedOutputs({ p: { status: "done", outputs: { value: "old" } }, a: { status: "done" } });
  useDocStore.getState().removeElements({ edges: ["e1"] });
  useDocStore.getState().connect(edge("e2", "b", "text", "p", "value"), true);
  assert.equal(useRuntimeStore.getState().nodes.p, undefined, "p no longer shows the old run");
  assert.ok(useRuntimeStore.getState().nodes.a, "a was not rewired");
  useRuntimeStore.getState().seedOutputs({ p: { status: "running" } });
  useDocStore.getState().removeElements({ edges: ["e2"] });
  assert.equal(useRuntimeStore.getState().nodes.p?.status, "running");
  stop();
});

test("only cards whose incoming wires changed count as rewired", () => {
  useDocStore.getState().load(flow());
  const before = useDocStore.getState().graph;
  useDocStore.getState().updateNode("a", { pos: [10, 10] });
  assert.deepEqual(rewiredCards(before, useDocStore.getState().graph), []);
});

test("a rewired card's downstream cards drop the outputs they computed from it", () => {
  useDocStore.getState().load(
    doc(
      [node("a", "input.text"), node("b", "input.text", [0, 200]), node("p", "output.preview", [500, 0]), node("q", "output.preview", [900, 0])],
      [edge("e1", "a", "text", "p", "value"), edge("e2", "p", "value", "q", "value")],
    ),
  );
  assert.deepEqual(withDownstream(useDocStore.getState().graph, ["a"]).sort(), ["a", "p", "q"]);
  assert.deepEqual(withDownstream(useDocStore.getState().graph, ["b"]), ["b"]);
});



test("a card shows its long text (system prompt, template) above Show inputs, in full once opened", () => {
  let open = false;
  const Button = function Button() { return null; };
  const card = loadWithStubs<{ NodeDisclosure: (p: { sections: { label: string; text: string }[] }) => unknown }>(
    new URL("../src/features/flows/editor/node-card.tsx", import.meta.url),
    stubEveryImport({
      react: { useState: () => [open, () => undefined] },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/button": { Button },
    }),
  );
  const sections = [{ label: "System prompt", text: "Be brief." }];
  const texts = (all: StubElement[]) => all.filter((e) => e.props.children === "Be brief.");
  const closed = elementsOf(card.NodeDisclosure({ sections }));
  const toggle = closed.findIndex((e) => e.type === Button);
  assert.equal(closed[toggle].props.children, "Show inputs");
  const preview = texts(closed);
  assert.equal(preview.length, 1, "the closed card previews the text");
  assert.ok(closed.indexOf(preview[0]) < toggle, "the preview sits above the toggle");
  open = true;
  const opened = elementsOf(card.NodeDisclosure({ sections }));
  assert.ok(opened.some((e) => e.type === Button && e.props.children === "Hide inputs"));
  assert.equal(card.NodeDisclosure({ sections: [] }), null);
});

test("a resized card writes lists and objects out in full", async () => {
  const Badge = function Badge() { return null; };
  const view = loadWithStubs<{
    OutputValues: (p: { outputs: Record<string, unknown>; spec: { outputs: { id: string }[] }; expanded?: boolean }) => unknown;
  }>(
    new URL("../src/features/flows/editor/flow-node.tsx", import.meta.url),
    stubEveryImport({
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/badge": { Badge },
      "./value-kinds": await import("../src/features/flows/editor/value-kinds.ts"),
      "./media": { MediaThumb: () => null, isMediaValue: () => false },
      "./value-viewers": { StructuredBadge: () => null, isStructuredValue: () => false },
      "../dataset/dataset-value": { isDatasetValue: () => false },
    }),
  );
  const spec = { outputs: [{ id: "rows" }] };
  const outputs = { rows: ["one", { two: 2 }] };
  const big = view.OutputValues({ outputs, spec, expanded: true }) as StubElement;
  assert.equal(big.type, "pre");
  assert.equal(big.props.children, 'one\n{\n  "two": 2\n}');
  const small = view.OutputValues({ outputs, spec }) as StubElement;
  assert.equal(small.type, Badge, "an unsized card shows a count");
});

test("the inspector shows a Preview block's value, not an empty Output section", () => {
  const runtime: Record<string, unknown> = {
    preview: { status: "done", outputs: { value: "hello" } },
    typed: { status: "done", outputs: { text: "port value", hidden: "x" } },
  };
  const OutputValue = function OutputValue() { return null; };
  const inspector = loadWithStubs<{
    NodeOutputs: (p: { nodeId: string; spec: { outputs: { id: string; label: string }[] } }) => unknown;
  }>(
    new URL("../src/features/flows/editor/inspector.tsx", import.meta.url),
    stubEveryImport({
      "react/jsx-runtime": stubJsxRuntime(),
      "../stores/runtime-store": {
        IDLE_RUNTIME: { status: "idle" },
        useRuntimeStore: (pick: (s: unknown) => unknown) => pick({ nodes: runtime }),
      },
      "./output-value": { OutputValue },
    }),
  );
  const shown = (nodeId: string, outputs: { id: string; label: string }[]) =>
    elementsOf(inspector.NodeOutputs({ nodeId, spec: { outputs } }))
      .filter((e) => e.type === OutputValue)
      .map((e) => e.props.value);
  assert.deepEqual(shown("preview", []), ["hello"], "a block with no output ports shows its raw outputs");
  assert.deepEqual(shown("typed", [{ id: "text", label: "Text" }]), ["port value"], "declared ports only");
  assert.equal(inspector.NodeOutputs({ nodeId: "idle", spec: { outputs: [] } }), null);
});
