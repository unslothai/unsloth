// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { FlowEdge, FlowNode } from "../src/features/flows/types.ts";
import { catalog, doc, edge, node } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
const { useDocStore, documentFromState, documentSignature, graphFromDocument } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const reroutes = await import("../src/features/flows/editor/reroutes.ts");
const { REROUTE_IN_HANDLE, REROUTE_OUT_HANDLE, checkRerouteConnection, reroutePort } = reroutes;
const { tryConnect } = await import("../src/features/flows/editor/node-factory.ts");
const { inHandle, outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
const { computeLayout } = await import("../src/features/flows/editor/layout-core.ts");
const { tidyRequest, tidyPatch } = await import("../src/features/flows/editor/tidy.ts");

useCatalogStore.getState().setCatalog(catalog);

const store = () => useDocStore.getState();
const graph = () => store().graph;
const saved = () => documentFromState(store());
const load = (nodes: FlowNode[], edges: FlowEdge[] = []) => store().load(doc(nodes, edges));

function fresh(edges = [edge("e1", "src", "text", "llm", "prompt")]) {
  load(
    [
      node("src", "input.text", [0, 0]),
      node("num", "input.number", [0, 200]),
      node("llm", "llm.generate", [600, 0]),
      node("img", "image.generate", [600, 300]),
      node("out", "output.preview", [1000, 0]),
    ],
    edges,
  );
}

/** Drops reroute `id` at (300, 10), splitting `edgeId` when given. */
const reroute = (id = "r1", edgeId?: string) => store().addReroute({ id, pos: [300, 10] }, edgeId);
/** A connection from a card output into a reroute's input. */
const into = (source: string, port: string, r = "r1") => ({
  source, sourceHandle: outHandle(port), target: r, targetHandle: REROUTE_IN_HANDLE,
});
/** A connection from a reroute's output into a card input (or another reroute's input). */
const from = (target: string, port: string | null, r = "r1") => ({
  source: r, sourceHandle: REROUTE_OUT_HANDLE, target, targetHandle: port === null ? REROUTE_IN_HANDLE : inHandle(port),
});
const addNode = (n: FlowNode) => store().load({ ...saved(), nodes: [...saved().nodes, n] });
const wiresInto = (target: string) => graph().wireOrder.filter((id) => graph().wires[id].target === target);
const plain = () => saved().edges.map((e) => `${e.id}:${e.source}.${e.source_port}->${e.target}.${e.target_port}`);

test("splitting a wire with a reroute keeps the plain edge the compiler sees", () => {
  fresh();
  reroute("r1", "e1");
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
  assert.equal(graph().wireOrder.length, 2);
  assert.deepEqual((saved().editor.reroutes as { id: string }[]).map((r) => r.id), ["r1"]);
  assert.equal(saved().nodes.length, 5, "reroutes never become nodes");
});

test("an empty reroute takes a wire in or out, and types itself from what is wired in", () => {
  fresh([]);
  reroute();
  assert.equal(reroutePort(graph(), "r1").type, "any");
  // Output first: the empty reroute feeds an image input...
  assert.equal(tryConnect(from("img", "init_image")), null);
  assert.deepEqual(plain(), [], "nothing reaches the compiler yet");
  assert.equal(reroutePort(graph(), "r1").type, "image");
  // ...so text cannot come in, but an image can.
  assert.match(tryConnect(into("src", "text")) ?? "", /text cannot connect to image/);
  addNode(node("gen", "image.generate", [0, 400]));
  assert.equal(tryConnect(into("gen", "image")), null);
  assert.equal(plain().length, 1);
  assert.match(plain()[0], /gen\.image->img\.init_image$/);
  assert.equal(reroutePort(graph(), "r1").type, "image");
});

test("an empty reroute feeds any mix of inputs; the source wired in later must fit them all", () => {
  fresh([]);
  reroute();
  assert.equal(tryConnect(from("img", "init_image")), null);
  // An `any` output (Get JSON's value, a file) feeds an image and a text input alike.
  assert.equal(tryConnect(from("llm", "prompt")), null);
  assert.equal(tryConnect(from("out", "value")), null);
  assert.match(tryConnect(into("src", "text")) ?? "", /text cannot connect to image/);
});

test("reroute connection rules: story paths, self, loops and single inputs", () => {
  fresh();
  reroute("r1", "e1");
  store().addReroute({ id: "r2", pos: [400, 10] });
  assert.equal(checkRerouteConnection(graph(), from("r1", null)).ok, false);
  assert.equal(checkRerouteConnection(graph(), into("llm", "text", "r2")).ok, true);
  tryConnect(from("r2", null));
  assert.equal(checkRerouteConnection(graph(), from("r1", null, "r2")).ok, false);
  // llm -> r3 -> llm.prompt would feed llm its own output.
  store().addReroute({ id: "r3", pos: [800, 10] });
  tryConnect(into("llm", "text", "r3"));
  const cycle = checkRerouteConnection(graph(), from("llm", "prompt", "r3"));
  assert.equal(cycle.ok, false);
  assert.match(cycle.ok ? "" : cycle.reason, /loop/);
  assert.equal(checkRerouteConnection(graph(), { ...into("llm", "text", "r2"), sourceHandle: "fo:next" }).ok, false);
  // A reroute input takes one wire: a new one replaces the old.
  tryConnect(into("num", "value", "r2"));
  const r2In = wiresInto("r2");
  assert.equal(r2In.length, 1);
  assert.equal(graph().wires[r2In[0]].source, "num");
});

test("deleting a reroute heals the wire; deleting its source drops the plain edge", () => {
  fresh();
  reroute("r1", "e1");
  tryConnect(from("out", "value"));
  assert.equal(plain().length, 2);
  const [toOut] = wiresInto("out");
  store().removeElements({ reroutes: ["r1"] });
  assert.deepEqual(graph().rerouteOrder, []);
  assert.deepEqual(graph().wireOrder, []);
  assert.deepEqual(plain().sort(), ["e1:src.text->llm.prompt", `${toOut}:src.text->out.value`].sort());

  // The canvas deletes a node together with its wires; the reroute still heals.
  fresh();
  reroute("r1", "e1");
  store().removeElements({ reroutes: ["r1"], edges: [...graph().wireOrder] });
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);

  fresh();
  reroute("r1", "e1");
  store().removeElements({ nodes: ["src"] });
  assert.deepEqual(plain(), []);
  assert.deepEqual(graph().rerouteOrder, ["r1"], "the reroute stays, empty");
});

test("a plain wire into an input fed through a reroute replaces it", () => {
  fresh();
  reroute("r1", "e1");
  addNode(node("src2", "input.text", [0, 100]));
  assert.equal(tryConnect({ ...into("src2", "text"), target: "llm", targetHandle: inHandle("prompt") }), null);
  assert.equal(plain().length, 1);
  assert.match(plain()[0], /src2\.text->llm\.prompt$/);
  assert.deepEqual(wiresInto("llm"), []);
});

test("reroutes round-trip through the saved document, and moving one is an edit", () => {
  fresh();
  reroute("r1", "e1");
  const doc0 = saved();
  const reloaded = graphFromDocument(doc0);
  assert.deepEqual(reloaded.rerouteOrder, ["r1"]);
  assert.deepEqual(reloaded.wireOrder, graph().wireOrder);
  assert.deepEqual(reloaded.edgeOrder, graph().edgeOrder);
  const before = documentSignature(doc0);
  const panned = { ...doc0, editor: { ...doc0.editor, viewport: { x: 1, y: 2, zoom: 3 } } };
  assert.equal(documentSignature(panned), before);
  store().moveNodes({ r1: [320, 40] });
  assert.notEqual(documentSignature(saved()), before);
  // A flow edited elsewhere that dropped the edge loses the wire too.
  const edited = graphFromDocument({ ...doc0, edges: [] });
  assert.deepEqual(edited.edgeOrder, []);
  assert.equal(edited.wireOrder.some((id) => edited.wires[id].target === "llm"), false);
});

test("a document without reroutes saves exactly as before", () => {
  fresh();
  const doc0 = saved();
  assert.deepEqual(doc0.editor, {});
  assert.equal(documentSignature(doc0), JSON.stringify((({ editor: _e, ...rest }) => rest)(doc0)));
});

const tidyCtx = (extra: { wireStyle: "bus" | "curved"; direction?: "TB" }) => ({
  graph: graph(), size: () => ({ width: 260, height: 140 }), portY: () => 40, ...extra,
});

test("tidy up lays out cards only and puts reroutes back between their ends", () => {
  fresh();
  store().addReroute({ id: "r1", pos: [-900, -900] }, "e1");
  for (const wireStyle of ["bus", "curved"] as const) {
    const ctx = tidyCtx({ wireStyle });
    const request = tidyRequest(ctx);
    assert.ok(!request.nodes.some((n) => n.id === "r1"), "reroutes are not laid out as cards");
    assert.ok(request.edges.some((e) => e.source === "src" && e.target === "llm"));
    store().applyTidy(tidyPatch(ctx, computeLayout(request)));
    const r1 = graph().reroutes.r1;
    assert.ok(r1, "the reroute survives");
    const [src, llm] = [graph().nodes.src.pos, graph().nodes.llm.pos];
    assert.ok(r1.pos[0] > src[0] + 260 && r1.pos[0] + 28 < llm[0], `${wireStyle}: between its ends`);
    if (wireStyle === "bus") assert.equal(r1.pos[1] + 8, src[1] + 40, "bus: level with its port");
  }
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
});

test("top to bottom, upright reroutes on one wire spread along it instead of sharing a row", () => {
  fresh();
  store().addReroute({ id: "r1", pos: [-900, -900], dir: "down" }, "e1");
  const [toLlm] = wiresInto("llm");
  assert.ok(toLlm);
  store().addReroute({ id: "r2", pos: [-900, -600], dir: "down" }, toLlm);
  const ctx = tidyCtx({ wireStyle: "bus", direction: "TB" });
  store().applyTidy(tidyPatch(ctx, computeLayout(tidyRequest(ctx))));
  const { r1, r2 } = graph().reroutes;
  assert.ok(r1.pos[1] < r2.pos[1], `r1 above r2: ${r1.pos[1]} vs ${r2.pos[1]}`);
});

test("rerouting a wire into a multi input keeps its place in the edge order", () => {
  fresh([
    edge("a", "src", "text", "out", "value"),
    edge("b", "num", "value", "out", "value"),
    edge("c", "llm", "text", "out", "value"),
  ]);
  reroute("r1", "b");
  assert.deepEqual(saved().edges.map((e) => e.id), ["a", "b", "c"]);
  store().removeElements({ reroutes: ["r1"] });
  assert.deepEqual(saved().edges.map((e) => e.id), ["a", "b", "c"]);
});

test("undo takes a reroute split back in one step", () => {
  fresh();
  reroute("r1", "e1");
  store().undo();
  assert.deepEqual(graph().rerouteOrder, []);
  assert.deepEqual(graph().wireOrder, []);
  assert.deepEqual(plain(), ["e1:src.text->llm.prompt"]);
});

test("a typed source into a reroute is checked against every input the chain feeds", () => {
  load([
    node("src", "input.text", [0, 0]),
    node("num", "input.number", [0, 200]),
    node("jg", "json.get", [600, 0]),
    node("llm", "llm.generate", [600, 200]),
  ]);
  reroute();
  tryConnect(from("jg", "json"));
  tryConnect(from("llm", "prompt"));
  // A text output feeds json and text alike; a number does not feed text.
  assert.match(tryConnect(into("num", "value")) ?? "", /cannot connect/);
  assert.equal(tryConnect(into("src", "text")), null);
});

test("a reroute never adds a second copy of a wire a multi input already has", () => {
  fresh([edge("a", "src", "text", "out", "value")]);
  reroute();
  assert.equal(tryConnect(into("src", "text")), null);
  assert.match(tryConnect(from("out", "value")) ?? "", /already/);
  assert.deepEqual(plain(), ["a:src.text->out.value"]);
  // The store refuses it too, whoever calls it.
  store().connectWire({ id: "wx", source: "r1", source_port: "out", target: "out", target_port: "value" }, false);
  assert.deepEqual(plain(), ["a:src.text->out.value"]);
  // Rerouting a single input's own wire is still allowed: it replaces the direct edge.
  fresh();
  reroute("r2");
  tryConnect(into("src", "text", "r2"));
  assert.equal(tryConnect(from("llm", "prompt", "r2")), null);
  assert.equal(plain().length, 1);
});

test("an empty reroute into an `any` input connects, and Quick Add offers what the chain could feed", () => {
  const { rerouteCanFeed, rerouteTakes } = reroutes;
  fresh([]);
  reroute();
  assert.equal(tryConnect(from("out", "value")), null);
  const port = (type: string) => ({ id: type[0], label: type[0], type }) as never;
  const [text, json, image, anyOut] = ["text", "json", "image", "any"].map(port);
  assert.equal(rerouteCanFeed(graph(), "r1", text), true);
  load([node("jg", "json.get", [600, 0]), node("src", "input.text", [0, 0])]);
  reroute("r2");
  tryConnect(from("jg", "json", "r2"));
  assert.equal(rerouteCanFeed(graph(), "r2", text), true);
  assert.equal(rerouteCanFeed(graph(), "r2", image), true, "an empty chain takes any input");
  tryConnect(into("src", "text", "r2"));
  assert.equal(rerouteCanFeed(graph(), "r2", json), true, "text feeds json");
  assert.equal(rerouteCanFeed(graph(), "r2", image), false, "text does not feed an image");
  // Into a reroute: an output has to fit every input the chain feeds.
  fresh([]);
  reroute("r3");
  tryConnect(from("img", "init_image", "r3"));
  tryConnect(from("llm", "prompt", "r3"));
  assert.equal(rerouteTakes(graph(), "r3", image), false);
  assert.equal(rerouteTakes(graph(), "r3", text), false);
  assert.equal(rerouteTakes(graph(), "r3", anyOut), true);
});

test("Shift-cutting a wire pins both halves to the other style, and it saves", () => {
  const { flippedStyle } = reroutes;
  fresh();
  const style = flippedStyle(graph(), "e1", "bus");
  assert.equal(style, "curved");
  store().addReroute({ id: "r1", pos: [300, 10] }, "e1", style);
  const halves = Object.values(graph().wires);
  assert.equal(halves.length, 2);
  assert.ok(halves.every((w) => w.style === "curved"));
  // A later cut of a pinned half keeps the pin unless Shift flips it back.
  store().addReroute({ id: "r2", pos: [200, 10] }, halves[0].id);
  assert.ok(Object.values(graph().wires).every((w) => w.style === "curved"));
  assert.equal(flippedStyle(graph(), halves[0].id, "bus"), "bus");
  assert.ok(Object.values(graphFromDocument(saved()).wires).every((w) => w.style === "curved"));
});

test("reroutes face along the wire they land on, and R turns them about their centre", () => {
  const { dirAlong, rerouteSize, turnReroutes } = reroutes;
  fresh();
  assert.equal(dirAlong({ x: 8, y: 0.5 }), "right");
  assert.equal(dirAlong({ x: 0, y: 8 }), "down");
  assert.equal(dirAlong({ x: 1, y: -8 }), "up");
  assert.equal(dirAlong({ x: 6, y: 6 }), "right", "a diagonal stays horizontal");
  assert.deepEqual(rerouteSize("down"), { width: 16, height: 28 });
  reroute("r1", "e1");
  const centre = (r: { pos: [number, number]; dir?: "right" | "down" | "up" }) => {
    const size = rerouteSize(r.dir);
    return [r.pos[0] + size.width / 2, r.pos[1] + size.height / 2];
  };
  const before = centre(graph().reroutes.r1);
  assert.equal(turnReroutes(["r1"]), true);
  assert.equal(graph().reroutes.r1.dir, "down");
  assert.deepEqual(centre(graph().reroutes.r1), before);
  turnReroutes(["r1"]);
  assert.equal(graph().reroutes.r1.dir, "up");
  turnReroutes(["r1"]);
  assert.equal(graph().reroutes.r1.dir, undefined, "back to right, stored as the default");
  assert.equal(turnReroutes(["llm"]), false, "cards do not turn");
  turnReroutes(["r1"]);
  assert.equal(graphFromDocument(saved()).reroutes.r1.dir, "down");
});

test("deleting a style-pinned reroute heals to a plain edge the backend accepts", () => {
  fresh();
  store().addReroute({ id: "r1", pos: [300, 10] }, "e1", "bus");
  store().removeElements({ reroutes: ["r1"] });
  const edges = Object.values(graph().edges);
  assert.ok(edges.length && edges.every((e) => !("style" in e)));
});

test("a direct wire into a single input clears an empty reroute's segment there, so a later feed cannot double it", () => {
  load([
    node("a", "input.text", [0, 0]),
    node("b", "input.text", [0, 200]),
    node("llm", "llm.generate", [600, 0]),
  ]);
  reroute();
  assert.equal(tryConnect(from("llm", "prompt")), null);
  assert.equal(tryConnect({ ...into("a", "text"), target: "llm", targetHandle: inHandle("prompt") }), null);
  assert.deepEqual(graph().wireOrder, []);
  tryConnect(into("b", "text"));
  const toPrompt = plain().filter((e) => e.endsWith("->llm.prompt"));
  assert.equal(toPrompt.length, 1);
  assert.match(toPrompt[0], /:a\.text->/);
});

test("recipe columns never wire through a reroute: only a direct wire writes the reference", () => {
  load([
    node("topic", "recipe.column", [0, 0]),
    node("expr", "recipe.column", [600, 0]),
    node("llm", "llm.generate", [600, 300]),
  ]);
  reroute();
  assert.equal(checkRerouteConnection(graph(), into("topic", "out")).ok, false);
  assert.notEqual(tryConnect(into("topic", "out")), null);
  assert.equal(checkRerouteConnection(graph(), from("expr", "in")).ok, false);
  assert.deepEqual(Object.keys(graph().edges), []);
});

test("a recipe column's wire is never split by a reroute", () => {
  load([node("a", "recipe.column", [0, 0]), node("b", "recipe.column", [600, 0])], [edge("ref", "a", "out", "b", "in")]);
  store().addReroute({ id: "r9", pos: [300, 10] }, "ref");
  assert.equal(graph().reroutes.r9, undefined);
  assert.ok(graph().edges.ref, "the reference wire stays whole");
});
