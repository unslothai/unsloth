// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { chainDoc } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { useRuntimeStore, reduceEvents, IDLE_RUNTIME } = await import(
  "../src/features/flows/stores/runtime-store.ts"
);
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");

const ev = (seq: number, type: string, nodeId: string | null, data: Record<string, unknown> = {}) =>
  ({ seq, run_id: "r1", type, node_id: nodeId, ts: seq, data }) as never;

type RuntimeBase = Parameters<typeof reduceEvents>[0];
const runtime = (over: Partial<RuntimeBase> = {}): RuntimeBase =>
  ({ runId: "r1", runStatus: "running", runError: null, lastSeq: 0, nodes: {}, logs: [], ...over }) as RuntimeBase;

const docGraph = () => useDocStore.getState().graph;

test("a runtime event for one node leaves every other node's entry identical", () => {
  const store = useRuntimeStore.getState();
  store.clear();
  store.startRun("r1");
  store.applyEvents([
    ev(1, "node.status", "a", { status: "running" }),
    ev(2, "node.status", "b", { status: "queued" }),
  ]);
  const before = useRuntimeStore.getState().nodes;
  useRuntimeStore.getState().applyEvents([ev(3, "node.progress", "a", { fraction: 0.5 })]);
  const after = useRuntimeStore.getState().nodes;
  assert.notEqual(after.a, before.a, "the touched node gets a new entry");
  assert.equal(after.a.progress, 0.5);
  assert.equal(after.b, before.b, "an untouched node keeps its entry, so its selector does not fire");
});

test("a batch folds into one update and drops replayed sequence numbers", () => {
  const next = reduceEvents(runtime({ lastSeq: 5 }), [
    ev(4, "node.progress", "a", { fraction: 0.9 }),
    ev(6, "node.progress", "a", { fraction: 0.2 }),
    ev(7, "node.text_delta", "a", { delta: "hi" }),
    ev(8, "run.completed", null),
  ]);
  assert.equal(next.lastSeq, 8);
  assert.equal(next.nodes.a.progress, 0.2, "seq 4 is a replay and must not apply");
  assert.equal(next.nodes.a.text, "hi");
  assert.equal(next.runStatus, "completed");
});

test("a snapshot rebuilds node state from the server", () => {
  const base = runtime({ runStatus: null, nodes: { x: { ...IDLE_RUNTIME, status: "running" } } });
  const next = reduceEvents(base, [
    ev(9, "snapshot", null, {
      run: { status: "running" },
      nodes: { a: { status: "done", outputs: { text: "ok" } } },
    }),
  ]);
  assert.equal(next.nodes.a.status, "done");
  assert.deepEqual(next.nodes.a.outputs, { text: "ok" });
  assert.equal(next.nodes.x, undefined);
});

test("editing one node's params changes only that node record", () => {
  useDocStore.getState().load(chainDoc());
  const before = docGraph();
  useDocStore.getState().updateParams("b", { prompt: "hello" });
  const after = docGraph();
  assert.notEqual(after.nodes.b, before.nodes.b);
  assert.equal(after.nodes.a, before.nodes.a);
  assert.equal(after.nodes.c, before.nodes.c);
  assert.equal(after.edges, before.edges, "edges are untouched by a param edit");
});

test("the React Flow arrays reuse every unchanged element", () => {
  useDocStore.getState().load(chainDoc());
  const deriver = new RfDeriver();
  const none = new Set<string>();
  const portType = () => "text";
  const n1 = deriver.nodes(docGraph(), none, null);
  const e1 = deriver.edges(docGraph(), none, portType);
  assert.equal(deriver.nodes(docGraph(), none, null), n1, "same graph, same array");
  assert.equal(deriver.edges(docGraph(), none, portType), e1);

  useDocStore.getState().updateParams("b", { prompt: "changed" });
  const n2 = deriver.nodes(docGraph(), none, null);
  assert.notEqual(n2, n1);
  assert.equal(n2[0], n1[0], "node a keeps its object");
  assert.notEqual(n2[1], n1[1], "node b is rebuilt");
  assert.equal(n2[1].data, n1[1].data, "node data stays stable so memo'd components can bail out");
  assert.equal(n2[2], n1[2]);

  const dragged = deriver.nodes(docGraph(), none, { c: { x: 10, y: 20 } });
  assert.equal(dragged[0], n2[0], "dragging c does not rebuild a");
  assert.deepEqual(dragged[2].position, { x: 10, y: 20 });

  const selected = deriver.nodes(docGraph(), new Set(["a"]), null);
  assert.equal(selected[1], n2[1], "selecting a does not rebuild b");
  assert.equal(selected[0].selected, true);
});

test("positions reach the document only when a drag is committed", () => {
  useDocStore.getState().load(chainDoc());
  const version = useDocStore.getState().version;
  new RfDeriver().nodes(docGraph(), new Set(), { a: { x: 99, y: 99 } });
  assert.equal(useDocStore.getState().version, version, "a drag frame is view state, not a document change");
  useDocStore.getState().moveNodes({ a: [99, 99] });
  assert.equal(useDocStore.getState().version, version + 1);
  assert.deepEqual(docGraph().nodes.a.pos, [99, 99]);
});

test("a failed node keeps the backend's error text", () => {
  const next = reduceEvents(runtime(), [
    ev(1, "node.status", "a", { status: "failed", error: "No chat model is loaded" }),
  ]);
  assert.equal(next.nodes.a.error, "No chat model is loaded");
});
