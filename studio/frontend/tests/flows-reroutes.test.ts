// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useCatalogStore } = await import(
  "../src/features/flows/stores/catalog-store.ts"
);
const { useDocStore, documentFromState, documentSignature, graphFromDocument } =
  await import("../src/features/flows/stores/doc-store.ts");
const {
  REROUTE_IN_HANDLE,
  REROUTE_OUT_HANDLE,
  checkRerouteConnection,
  reroutePort,
} = await import("../src/features/flows/editor/reroutes.ts");
const { tryConnect } = await import(
  "../src/features/flows/editor/node-factory.ts"
);
const { inHandle, outHandle } = await import(
  "../src/features/flows/editor/connection-rules.ts"
);
const { layoutRequest } = await import("../src/features/flows/editor/layout.ts");
const { computeLayout } = await import(
  "../src/features/flows/editor/layout-core.ts"
);

useCatalogStore.getState().setCatalog(catalog);

function fresh(edges = [edge("e1", "src", "text", "llm", "prompt")]) {
  useDocStore
    .getState()
    .load(
      doc(
        [
          node("src", "input.text", [0, 0]),
          node("num", "input.number", [0, 200]),
          node("llm", "llm.generate", [600, 0]),
          node("img", "image.generate", [600, 300]),
          node("out", "output.preview", [1000, 0]),
        ],
        edges,
      ),
    );
}

const graph = () => useDocStore.getState().graph;
const plain = () =>
  documentFromState(useDocStore.getState()).edges.map(
    (e) => `${e.id}:${e.source}.${e.source_port}->${e.target}.${e.target_port}`,
  );

test("splitting a wire with a reroute keeps the plain edge the compiler sees", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
  assert.equal(graph().wireOrder.length, 2);
  const saved = documentFromState(useDocStore.getState());
  assert.deepEqual(
    (saved.editor.reroutes as { id: string }[]).map((r) => r.id),
    ["r1"],
  );
  assert.equal(saved.nodes.length, 5, "reroutes never become nodes");
});

test("an empty reroute takes a wire in or out, and types itself from what is wired in", () => {
  fresh([]);
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] });
  assert.equal(reroutePort(graph(), "r1").type, "any");
  // Output first: the empty reroute feeds an image input...
  assert.equal(
    tryConnect({
      source: "r1",
      sourceHandle: REROUTE_OUT_HANDLE,
      target: "img",
      targetHandle: inHandle("init_image"),
    }),
    null,
  );
  assert.deepEqual(plain(), [], "nothing reaches the compiler yet");
  assert.equal(reroutePort(graph(), "r1").type, "image");
  // ...so text cannot come in, but an image can.
  assert.match(
    tryConnect({
      source: "src",
      sourceHandle: outHandle("text"),
      target: "r1",
      targetHandle: REROUTE_IN_HANDLE,
    }) ?? "",
    /text cannot connect to image/,
  );
  useDocStore.getState().load({
    ...documentFromState(useDocStore.getState()),
    nodes: [
      ...documentFromState(useDocStore.getState()).nodes,
      node("gen", "image.generate", [0, 400]),
    ],
  });
  assert.equal(
    tryConnect({
      source: "gen",
      sourceHandle: outHandle("image"),
      target: "r1",
      targetHandle: REROUTE_IN_HANDLE,
    }),
    null,
  );
  assert.equal(plain().length, 1);
  assert.match(plain()[0], /gen\.image->img\.init_image$/);
  assert.equal(reroutePort(graph(), "r1").type, "image");
});

test("an empty reroute feeds any mix of inputs; the source wired in later must fit them all", () => {
  fresh([]);
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] });
  const to = (target: string, port: string) =>
    tryConnect({ source: "r1", sourceHandle: REROUTE_OUT_HANDLE, target, targetHandle: inHandle(port) });
  assert.equal(to("img", "init_image"), null);
  // An `any` output (Get JSON's value, a file) feeds an image and a text input alike.
  assert.equal(to("llm", "prompt"), null);
  assert.equal(to("out", "value"), null);
  assert.match(
    tryConnect({ source: "src", sourceHandle: outHandle("text"), target: "r1", targetHandle: REROUTE_IN_HANDLE }) ?? "",
    /text cannot connect to image/,
  );
});

test("reroute connection rules: story paths, self, loops and single inputs", () => {
  fresh();
  const store = useDocStore.getState();
  store.addReroute({ id: "r1", pos: [300, 10] }, "e1");
  store.addReroute({ id: "r2", pos: [400, 10] });
  const g = graph();
  assert.equal(
    checkRerouteConnection(g, {
      source: "r1",
      sourceHandle: REROUTE_OUT_HANDLE,
      target: "r1",
      targetHandle: REROUTE_IN_HANDLE,
    }).ok,
    false,
  );
  assert.equal(
    checkRerouteConnection(g, {
      source: "llm",
      sourceHandle: outHandle("text"),
      target: "r2",
      targetHandle: REROUTE_IN_HANDLE,
    }).ok,
    true,
  );
  tryConnect({
    source: "r1",
    sourceHandle: REROUTE_OUT_HANDLE,
    target: "r2",
    targetHandle: REROUTE_IN_HANDLE,
  });
  const loop = checkRerouteConnection(graph(), {
    source: "r2",
    sourceHandle: REROUTE_OUT_HANDLE,
    target: "r1",
    targetHandle: REROUTE_IN_HANDLE,
  });
  assert.equal(loop.ok, false);
  // llm -> r3 -> llm.prompt would feed llm its own output.
  store.addReroute({ id: "r3", pos: [800, 10] });
  tryConnect({
    source: "llm",
    sourceHandle: outHandle("text"),
    target: "r3",
    targetHandle: REROUTE_IN_HANDLE,
  });
  const cycle = checkRerouteConnection(graph(), {
    source: "r3",
    sourceHandle: REROUTE_OUT_HANDLE,
    target: "llm",
    targetHandle: inHandle("prompt"),
  });
  assert.equal(cycle.ok, false);
  assert.match(cycle.ok ? "" : cycle.reason, /loop/);
  assert.equal(
    checkRerouteConnection(graph(), {
      source: "llm",
      sourceHandle: "fo:next",
      target: "r2",
      targetHandle: REROUTE_IN_HANDLE,
    }).ok,
    false,
  );
  // A reroute input takes one wire: a new one replaces the old.
  tryConnect({
    source: "num",
    sourceHandle: outHandle("value"),
    target: "r2",
    targetHandle: REROUTE_IN_HANDLE,
  });
  const into = graph().wireOrder.filter((id) => graph().wires[id].target === "r2");
  assert.equal(into.length, 1);
  assert.equal(graph().wires[into[0]].source, "num");
});

test("deleting a reroute heals the wire; deleting its source drops the plain edge", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  tryConnect({
    source: "r1",
    sourceHandle: REROUTE_OUT_HANDLE,
    target: "out",
    targetHandle: inHandle("value"),
  });
  assert.equal(plain().length, 2);
  const toOut = graph().wireOrder.find((id) => graph().wires[id].target === "out");
  useDocStore.getState().removeElements({ reroutes: ["r1"] });
  assert.deepEqual(graph().rerouteOrder, []);
  assert.deepEqual(graph().wireOrder, []);
  assert.deepEqual(plain().sort(), [
    "e1:src.text->llm.prompt",
    `${toOut}:src.text->out.value`,
  ].sort());

  // The canvas deletes a node together with its wires; the reroute still heals.
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  useDocStore
    .getState()
    .removeElements({ reroutes: ["r1"], edges: [...graph().wireOrder] });
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);

  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  useDocStore.getState().removeElements({ nodes: ["src"] });
  assert.deepEqual(plain(), []);
  assert.deepEqual(graph().rerouteOrder, ["r1"], "the reroute stays, empty");
});

test("a plain wire into an input fed through a reroute replaces it", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  useDocStore.getState().load({
    ...documentFromState(useDocStore.getState()),
    nodes: [
      ...documentFromState(useDocStore.getState()).nodes,
      node("src2", "input.text", [0, 100]),
    ],
  });
  assert.equal(
    tryConnect({
      source: "src2",
      sourceHandle: outHandle("text"),
      target: "llm",
      targetHandle: inHandle("prompt"),
    }),
    null,
  );
  assert.equal(plain().length, 1);
  assert.match(plain()[0], /src2\.text->llm\.prompt$/);
  assert.equal(
    graph().wireOrder.some((id) => graph().wires[id].target === "llm"),
    false,
  );
});

test("reroutes round-trip through the saved document, and moving one is an edit", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  const saved = documentFromState(useDocStore.getState());
  const reloaded = graphFromDocument(saved);
  assert.deepEqual(reloaded.rerouteOrder, ["r1"]);
  assert.deepEqual(reloaded.wireOrder, graph().wireOrder);
  assert.deepEqual(reloaded.edgeOrder, graph().edgeOrder);
  const before = documentSignature(saved);
  assert.equal(
    documentSignature({ ...saved, editor: { ...saved.editor, viewport: { x: 1, y: 2, zoom: 3 } } }),
    before,
  );
  useDocStore.getState().moveNodes({ r1: [320, 40] });
  assert.notEqual(
    documentSignature(documentFromState(useDocStore.getState())),
    before,
  );
  // A flow edited elsewhere that dropped the edge loses the wire too.
  const edited = { ...saved, edges: [] };
  assert.deepEqual(graphFromDocument(edited).edgeOrder, []);
  assert.equal(
    graphFromDocument(edited).wireOrder.some(
      (id) => graphFromDocument(edited).wires[id].target === "llm",
    ),
    false,
  );
});

test("a document without reroutes saves exactly as before", () => {
  fresh();
  const saved = documentFromState(useDocStore.getState());
  assert.deepEqual(saved.editor, {});
  assert.equal(
    documentSignature(saved),
    JSON.stringify((({ editor: _e, ...rest }) => rest)(saved)),
  );
});

test("auto layout places reroutes in line and keeps them", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [-900, -900] }, "e1");
  const request = layoutRequest(graph(), () => undefined, "LR");
  assert.ok(request.nodes.some((n) => n.id === "r1"));
  assert.ok(request.edges.some((e) => e.source === "src" && e.target === "r1"));
  assert.ok(request.edges.some((e) => e.source === "r1" && e.target === "llm"));
  assert.ok(
    !request.edges.some((e) => e.source === "src" && e.target === "llm"),
    "the derived edge is not laid out twice",
  );
  const result = computeLayout(request);
  useDocStore.getState().moveNodes(result);
  const r1 = graph().reroutes.r1;
  assert.ok(r1, "the reroute survives");
  assert.ok(
    r1.pos[0] > graph().nodes.src.pos[0] && r1.pos[0] < graph().nodes.llm.pos[0],
    "and sits between its ends",
  );
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
});

test("rerouting a wire into a multi input keeps its place in the edge order", () => {
  fresh([
    edge("a", "src", "text", "out", "value"),
    edge("b", "num", "value", "out", "value"),
    edge("c", "llm", "text", "out", "value"),
  ]);
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "b");
  assert.deepEqual(
    documentFromState(useDocStore.getState()).edges.map((e) => e.id),
    ["a", "b", "c"],
  );
  useDocStore.getState().removeElements({ reroutes: ["r1"] });
  assert.deepEqual(
    documentFromState(useDocStore.getState()).edges.map((e) => e.id),
    ["a", "b", "c"],
  );
});

test("undo takes a reroute split back in one step", () => {
  fresh();
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] }, "e1");
  useDocStore.getState().undo();
  assert.deepEqual(graph().rerouteOrder, []);
  assert.deepEqual(graph().wireOrder, []);
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
});

test("a typed source into a reroute is checked against every input the chain feeds", () => {
  useDocStore.getState().load(
    doc(
      [
        node("src", "input.text", [0, 0]),
        node("num", "input.number", [0, 200]),
        node("jg", "json.get", [600, 0]),
        node("llm", "llm.generate", [600, 200]),
      ],
      [],
    ),
  );
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] });
  for (const [target, port] of [["jg", "json"], ["llm", "prompt"]] as const)
    tryConnect({ source: "r1", sourceHandle: REROUTE_OUT_HANDLE, target, targetHandle: inHandle(port) });
  // A text output feeds json and text alike; a number does not feed text.
  assert.match(
    tryConnect({ source: "num", sourceHandle: outHandle("value"), target: "r1", targetHandle: REROUTE_IN_HANDLE }) ?? "",
    /cannot connect/,
  );
  assert.equal(
    tryConnect({ source: "src", sourceHandle: outHandle("text"), target: "r1", targetHandle: REROUTE_IN_HANDLE }),
    null,
  );
});

test("a reroute never adds a second copy of a wire a multi input already has", () => {
  fresh([edge("a", "src", "text", "out", "value")]);
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] });
  assert.equal(
    tryConnect({
      source: "src",
      sourceHandle: outHandle("text"),
      target: "r1",
      targetHandle: REROUTE_IN_HANDLE,
    }),
    null,
  );
  assert.match(
    tryConnect({
      source: "r1",
      sourceHandle: REROUTE_OUT_HANDLE,
      target: "out",
      targetHandle: inHandle("value"),
    }) ?? "",
    /already/,
  );
  assert.deepEqual(plain(), ["a:src.text->out.value"]);
  // The store refuses it too, whoever calls it.

  useDocStore.getState().connectWire(
    { id: "wx", source: "r1", source_port: "out", target: "out", target_port: "value" },
    false,
  );
  assert.deepEqual(plain(), ["a:src.text->out.value"]);
  // Rerouting a single input's own wire is still allowed: it replaces the direct edge.
  fresh();
  useDocStore.getState().addReroute({ id: "r2", pos: [300, 10] });
  tryConnect({ source: "src", sourceHandle: outHandle("text"), target: "r2", targetHandle: REROUTE_IN_HANDLE });
  assert.equal(
    tryConnect({ source: "r2", sourceHandle: REROUTE_OUT_HANDLE, target: "llm", targetHandle: inHandle("prompt") }),
    null,
  );
  assert.equal(plain().length, 1);
});

test("an empty reroute into an `any` input connects, and Quick Add offers what the chain could feed", async () => {
  fresh([]);
  useDocStore.getState().addReroute({ id: "r1", pos: [300, 10] });
  assert.equal(
    tryConnect({ source: "r1", sourceHandle: REROUTE_OUT_HANDLE, target: "out", targetHandle: inHandle("value") }),
    null,
  );
  const { rerouteCanFeed } = await import("../src/features/flows/editor/reroutes.ts");
  const text = { id: "t", label: "t", type: "text" } as never;
  const json = { id: "j", label: "j", type: "json" } as never;
  const image = { id: "i", label: "i", type: "image" } as never;
  assert.equal(rerouteCanFeed(graph(), "r1", text), true);
  useDocStore.getState().load(
    doc([node("jg", "json.get", [600, 0]), node("src", "input.text", [0, 0])], []),
  );
  useDocStore.getState().addReroute({ id: "r2", pos: [300, 10] });
  tryConnect({ source: "r2", sourceHandle: REROUTE_OUT_HANDLE, target: "jg", targetHandle: inHandle("json") });
  assert.equal(rerouteCanFeed(graph(), "r2", text), true);
  assert.equal(rerouteCanFeed(graph(), "r2", image), true, "an empty chain takes any input");
  tryConnect({ source: "src", sourceHandle: outHandle("text"), target: "r2", targetHandle: REROUTE_IN_HANDLE });
  assert.equal(rerouteCanFeed(graph(), "r2", json), true, "text feeds json");
  assert.equal(rerouteCanFeed(graph(), "r2", image), false, "text does not feed an image");
  // Into a reroute: an output has to fit every input the chain feeds.
  const { rerouteTakes } = await import("../src/features/flows/editor/reroutes.ts");
  fresh([]);
  useDocStore.getState().addReroute({ id: "r3", pos: [300, 10] });
  for (const [target, port] of [["img", "init_image"], ["llm", "prompt"]] as const)
    tryConnect({ source: "r3", sourceHandle: REROUTE_OUT_HANDLE, target, targetHandle: inHandle(port) });
  const anyOut = { id: "v", label: "v", type: "any" } as never;
  assert.equal(rerouteTakes(graph(), "r3", image), false);
  assert.equal(rerouteTakes(graph(), "r3", text), false);
  assert.equal(rerouteTakes(graph(), "r3", anyOut), true);
});
