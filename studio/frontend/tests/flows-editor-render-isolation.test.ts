// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const { useRuntimeStore, reduceEvents, IDLE_RUNTIME } = await import(
  "../src/features/flows/stores/runtime-store.ts"
);
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");

const ev = (
  seq: number,
  type: string,
  nodeId: string | null,
  data: Record<string, unknown> = {},
) => ({ seq, run_id: "r1", type, node_id: nodeId, ts: seq, data }) as never;

function loadGraph() {
  useDocStore
    .getState()
    .load(
      doc(
        [
          node("a", "input.text"),
          node("b", "llm.generate", [300, 0]),
          node("c", "output.preview", [600, 0]),
        ],
        [
          edge("e1", "a", "text", "b", "prompt"),
          edge("e2", "b", "text", "c", "value"),
        ],
      ),
    );
}

test("a runtime event for one node leaves every other node's entry identical", () => {
  useRuntimeStore.getState().clear();
  useRuntimeStore.getState().startRun("r1");
  useRuntimeStore
    .getState()
    .applyEvents([
      ev(1, "node.status", "a", { status: "running" }),
      ev(2, "node.status", "b", { status: "queued" }),
    ]);
  const before = useRuntimeStore.getState().nodes;
  useRuntimeStore
    .getState()
    .applyEvents([ev(3, "node.progress", "a", { fraction: 0.5 })]);
  const after = useRuntimeStore.getState().nodes;
  assert.notEqual(after.a, before.a, "the touched node gets a new entry");
  assert.equal(after.a.progress, 0.5);
  assert.equal(
    after.b,
    before.b,
    "an untouched node keeps its entry, so its selector does not fire",
  );
});

test("a batch folds into one update and drops replayed sequence numbers", () => {
  const base = {
    runId: "r1",
    runStatus: "running" as const,
    runError: null,
    lastSeq: 5,
    nodes: {},
    logs: [],
  };
  const next = reduceEvents(base, [
    ev(4, "node.progress", "a", { fraction: 0.9 }),
    ev(6, "node.progress", "a", { fraction: 0.2 }),
    ev(7, "node.text_delta", "a", { delta: "hi" }),
    ev(8, "run.completed", null),
  ]);
  assert.equal(next.lastSeq, 8);
  assert.equal(
    next.nodes.a.progress,
    0.2,
    "seq 4 is a replay and must not apply",
  );
  assert.equal(next.nodes.a.text, "hi");
  assert.equal(next.runStatus, "completed");
});

test("a snapshot rebuilds node state from the server", () => {
  const base = {
    runId: "r1",
    runStatus: null,
    runError: null,
    lastSeq: 0,
    nodes: { x: { ...IDLE_RUNTIME, status: "running" as const } },
    logs: [],
  };
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
  loadGraph();
  const before = useDocStore.getState().graph;
  useDocStore.getState().updateParams("b", { prompt: "hello" });
  const after = useDocStore.getState().graph;
  assert.notEqual(after.nodes.b, before.nodes.b);
  assert.equal(after.nodes.a, before.nodes.a);
  assert.equal(after.nodes.c, before.nodes.c);
  assert.equal(
    after.edges,
    before.edges,
    "edges are untouched by a param edit",
  );
});

test("the React Flow arrays reuse every unchanged element", () => {
  loadGraph();
  const deriver = new RfDeriver();
  const none = new Set<string>();
  const portType = () => "text";
  const n1 = deriver.nodes(useDocStore.getState().graph, none, null);
  const e1 = deriver.edges(useDocStore.getState().graph, none, portType);
  assert.equal(
    deriver.nodes(useDocStore.getState().graph, none, null),
    n1,
    "same graph, same array",
  );
  assert.equal(deriver.edges(useDocStore.getState().graph, none, portType), e1);

  useDocStore.getState().updateParams("b", { prompt: "changed" });
  const n2 = deriver.nodes(useDocStore.getState().graph, none, null);
  assert.notEqual(n2, n1);
  assert.equal(n2[0], n1[0], "node a keeps its object");
  assert.notEqual(n2[1], n1[1], "node b is rebuilt");
  assert.equal(
    n2[1].data,
    n1[1].data,
    "node data stays stable so memo'd components can bail out",
  );
  assert.equal(n2[2], n1[2]);

  const dragged = deriver.nodes(useDocStore.getState().graph, none, {
    c: { x: 10, y: 20 },
  });
  assert.equal(dragged[0], n2[0], "dragging c does not rebuild a");
  assert.deepEqual(dragged[2].position, { x: 10, y: 20 });

  const selected = deriver.nodes(
    useDocStore.getState().graph,
    new Set(["a"]),
    null,
  );
  assert.equal(selected[1], n2[1], "selecting a does not rebuild b");
  assert.equal(selected[0].selected, true);
});

test("positions reach the document only when a drag is committed", () => {
  loadGraph();
  const version = useDocStore.getState().version;
  const deriver = new RfDeriver();
  deriver.nodes(useDocStore.getState().graph, new Set(), {
    a: { x: 99, y: 99 },
  });
  assert.equal(
    useDocStore.getState().version,
    version,
    "a drag frame is view state, not a document change",
  );
  useDocStore.getState().moveNodes({ a: [99, 99] });
  assert.equal(useDocStore.getState().version, version + 1);
  assert.deepEqual(useDocStore.getState().graph.nodes.a.pos, [99, 99]);
});
