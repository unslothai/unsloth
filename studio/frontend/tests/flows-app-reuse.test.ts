// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { byTestId, deferred, doc, flushMicrotasks, node } from "./helpers/flows.ts";
import { fakeReact } from "./helpers/fake-react.ts";
import { elementsOf, loadWithStubs, type StubElement as El, stubEveryImport, stubJsxRuntime } from "./helpers/module-stubs.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const appForm = await import("../src/features/flows/app/app-form.ts");
const pinsStore = await import("../src/features/flows/stores/pins-store.ts");
const { appFields, applyAppValues, appOutputs, foldAppEvents, textFieldValue } = appForm;
const { historyAgo, historyCard, historyPreview, pinsForRun, sameOutputs, usePinsStore } = pinsStore;

type Doc = ReturnType<typeof doc>;

function sample(): Doc {
  return doc([
    node("out", "output.preview", [0, 400]),
    node("topic", "input.param", [0, 0], { name: "topic", type: "text", default: "otters" }),
    { ...node("pic", "input.image", [0, 200]), label: "Photo" },
    node("n", "input.number", [300, 100], { value: 3 }),
    node("sel", "input.select", [0, 100], { options: ["a", "b"], value: "b" }),
    node("loop", "logic.for_each", [0, 500]),
    { ...node("inner", "input.text", [0, 300]), group: "loop" },
    { ...node("lout", "output.preview", [0, 460]), group: "loop" },
    { ...node("framed", "input.text", [0, 250]), group: "frame1" },
    { ...node("fout", "output.result", [0, 450]), group: "frame1" },
    { ...node("off", "input.text", [0, 350]), disabled: true },
  ]);
}

/** A flow with no nodes and the given Flow Inputs. */
const withFlowParams = (params: unknown[]) => Object.assign(doc([]), { params }) as Doc;
/** What the form sends when the user touches nothing. */
const untouched = (fields: { key: string; initial: unknown }[]) =>
  Object.fromEntries(fields.map((f) => [f.key, f.initial]));
const ev = (type: string, node_id: string | null, data: object, seq = 1) =>
  ({ seq, run_id: "r", type, node_id, ts: 0, data });

test("an app form lists the flow's inputs in canvas order", () => {
  const fields = appFields(sample() as never);
  assert.deepEqual(
    fields.map((f) => [f.key, f.target, f.kind]),
    [
      ["topic", "param", "textarea"],
      ["sel", "node", "select"],
      ["n", "node", "number"],
      ["pic", "node", "image"],
      ["framed", "node", "textarea"],
    ],
  );
  assert.equal(fields[0].initial, "otters");
  assert.equal(fields[3].label, "Photo");
  assert.deepEqual(fields[1].options, ["a", "b"]);
});

test("form values fill input blocks and Flow Inputs without touching the flow", () => {
  const d = sample();
  const media = { kind: "image", ref: "image:1" };
  const values = { topic: "cats", pic: media, n: 7 };
  const { doc: filled, params } = applyAppValues(d as never, appFields(d as never), values);
  assert.deepEqual(params, { topic: "cats" });
  const byId = Object.fromEntries(filled.nodes.map((n) => [n.id, n]));
  assert.deepEqual(byId.pic.params.asset, media);
  assert.equal(byId.n.params.value, 7);
  assert.equal(byId.sel.params.value, "b");
  assert.equal(d.nodes.find((n) => n.id === "n")?.params.value, 3, "the loaded flow is unchanged");
  assert.deepEqual(appOutputs(d as never).map((n) => n.id), ["out", "fout"]);
});

test("run events fold into progress, results and errors", () => {
  const s = foldAppEvents(
    { status: "idle", done: 0, total: 0, results: {}, errors: [] as string[] },
    [
      ev("run.started", null, { order: ["a", "b", "out"] }),
      ev("node.status", "a", { status: "done" }),
      ev("node.status", "b", { status: "failed", error: "boom" }),
      ev("node.output", "out", { outputs: { value: "hi" } }),
      ev("node.output", "a", { outputs: { value: "ignored" } }),
      ev("run.failed", null, { error: "1 step(s) failed" }),
    ] as never,
    new Set(["out"]),
  );
  assert.equal(s.status, "failed");
  assert.equal(s.total, 3);
  assert.equal(s.done, 2);
  assert.deepEqual(s.results, { out: "hi" });
  assert.deepEqual(s.errors, ["boom", "1 step(s) failed"]);
});

test("a JSON field sends the parsed value, and a terminal snapshot ends the run", () => {
  const d = withFlowParams([{ id: "config", type: "json", default: { name: "a" } }]);
  const fields = appFields(d as never);
  const sent = (config: string) => applyAppValues(d as never, fields, { config }).params;
  assert.deepEqual(sent('{"name": "b"}'), { config: { name: "b" } });
  assert.deepEqual(sent("{not json"), { config: "{not json" });
  const snapshot = ev(
    "snapshot",
    null,
    { run: { status: "interrupted", error: "Studio restarted" }, nodes: { out: { outputs: { value: "kept" } } } },
    9,
  );
  const s = foldAppEvents(
    { status: "running", done: 1, total: 3, results: {}, errors: [] },
    [snapshot] as never,
    new Set(["out"]),
  );
  assert.equal(s.status, "interrupted");
  assert.deepEqual(s.results, { out: "kept" });
  assert.deepEqual(s.errors, ["Studio restarted"]);
});

test("history cells preview media first, and pins only go with nodes still in the flow", () => {
  const media = { kind: "image", ref: "image:9" };
  assert.deepEqual(historyPreview({ images: [media], text: "x" }), { media });
  assert.deepEqual(historyPreview({ text: "hello" }), { text: "hello" });
  assert.ok(sameOutputs({ a: 1 }, { a: 1 }));
  usePinsStore.getState().load("flow-x");
  usePinsStore.getState().setPin("a", { text: "kept" });
  usePinsStore.getState().setPin("gone", { text: "old" });
  assert.deepEqual(pinsForRun(["a", "b"]), { a: { text: "kept" } });
  usePinsStore.getState().setPin("a", null);
  assert.deepEqual(pinsForRun(["a"]), {});
});

test("an untouched seed keeps randomizing, a typed-in seed is used as is", () => {
  const d = doc([node("seed", "input.seed", [0, 0], { value: 42, randomize: true })]);
  const fields = appFields(d as never);
  const seedOf = (value: unknown) => applyAppValues(d as never, fields, { seed: value }).doc.nodes[0].params;
  assert.equal(seedOf(42).randomize, true);
  assert.deepEqual(seedOf(7), { value: 7, randomize: false });
});

// An app text field shows an empty value as an empty box (never `""` or `null`), and objects as JSON.
for (const [value, shown] of [
  [undefined, ""],
  [null, ""],
  ["", ""],
  ["hi", "hi"],
  [{ a: 1 }, '{"a":1}'],
  [3, "3"],
] as const)
  test(`an app text field shows ${JSON.stringify(value) ?? "undefined"} as ${JSON.stringify(shown)}`, () => {
    assert.equal(textFieldValue(value), shown);
  });

test("an untouched seed with no stored value keeps randomizing", () => {
  const d = doc([node("s", "input.seed", [0, 0], { randomize: true })]);
  const fields = appFields(d as never);
  const { doc: filled } = applyAppValues(d as never, fields, untouched(fields));
  assert.deepEqual((filled as { nodes: { params: unknown }[] }).nodes[0].params, { randomize: true });
});

test("an untouched Flow Input still sends its declared default", () => {
  const d = withFlowParams([{ id: "topic", type: "text", default: "cats" }]);
  const fields = appFields(d as never);
  assert.deepEqual(applyAppValues(d as never, fields, untouched(fields)).params, { topic: "cats" });
});

test("a Document input is an app field that takes an upload", () => {
  const d = doc([{ ...node("doc", "input.file", [0, 0]), label: "Report" }]);
  const [field] = appFields(d as never);
  assert.equal(field.kind, "document");
  const media = { kind: "document", ref: "sha256:ab", name: "q3.pdf" };
  const filled = applyAppValues(d as never, [field], { [field.key]: media });
  assert.deepEqual(filled.doc.nodes[0].params.asset, media);
});

test("history entries read like small Library cards: type, name and detail", () => {
  const media = { kind: "image", ref: "image:1", width: 1024, height: 768, prompt: "a sloth" };
  assert.deepEqual(historyCard({ image: media }), { kind: "image", title: "a sloth", detail: "1024×768", media });
  assert.deepEqual(historyCard({ text: "  hello there " }), {
    kind: "text",
    title: "hello there",
    detail: "14 chars",
  });
  assert.equal(historyCard({ items: ["a", "b"] }).title, "2 items");
  assert.equal(historyCard({ n: 3 }).kind, "number");
  assert.deepEqual(
    historyCard({ out: { kind: "dataset", source: "hf", ref: "org/rows", rows: 50 } }),
    { kind: "dataset", title: "rows", detail: "50 rows" },
  );
  assert.deepEqual(historyCard({ obj: { a: 1, b: 2 } }), { kind: "json", title: "JSON", detail: "2 keys" });
});

test("history captions say how long ago", () => {
  const now = Date.parse("2026-10-30T12:00:00Z");
  assert.equal(historyAgo(Date.parse("2026-10-30T11:59:30Z"), now), "just now");
  assert.equal(historyAgo(Date.parse("2026-10-30T11:55:00Z"), now), "5m ago");
  assert.equal(historyAgo(Date.parse("2026-10-30T09:00:00Z"), now), "3h ago");
});

// ---- The app page and history strip, rendered with deterministic hooks (no DOM). ----

let active = fakeReact();
const reactProxy = new Proxy({}, { get: (_t, key) => active.react[key as keyof typeof active.react] });

/** An API stub whose every call returns a fresh deferred, collected in `calls` for the test to resolve. */
function pendingCalls<T>(calls: ReturnType<typeof deferred<T>>[]) {
  return () => {
    const d = deferred<T>();
    calls.push(d);
    return d.promise;
  };
}

function appPage() {
  const flows: Record<string, ReturnType<typeof deferred<{ doc: Doc }>>[]> = {};
  const starts: ReturnType<typeof deferred<{ run_id: string }>>[] = [];
  const uploads: ReturnType<typeof deferred<unknown>>[] = [];
  const followed: string[] = [];
  const api = {
    getFlow: (id: string) => pendingCalls((flows[id] ??= []))(),
    startRun: pendingCalls(starts),
    followRunEvents: (id: string) => {
      followed.push(id);
      return new Promise(() => undefined);
    },
    cancelRun: async () => undefined,
    uploadAsset: pendingCalls(uploads),
  };
  const mod = loadWithStubs<{ FlowAppPage: (p: { flowId: string }) => unknown }>(
    new URL("../src/features/flows/app/flow-app-page.tsx", import.meta.url),
    stubEveryImport({
      react: reactProxy,
      "react/jsx-runtime": stubJsxRuntime(),
      "../api": api,
      "./app-form": appForm,
      "../editor/param-fields": { DOCUMENT_ACCEPT: ".pdf", documentName: () => null },
    }),
  );
  const page = fakeReact();
  const render = (flowId: string) => {
    active = page;
    return elementsOf(page.render(mod.FlowAppPage, { flowId }));
  };
  /** Renders a nested component element (e.g. FieldInput) with its own hook state. */
  const child = (el: El, hooks = fakeReact()) => {
    active = hooks;
    const out = elementsOf(hooks.render(el.type as (p: unknown) => unknown, el.props));
    active = page;
    return out;
  };
  return { api, flows, starts, uploads, followed, render, child };
}

const isNamed = (e: El, name: string) => (e.type as { name?: string }).name === name;
const fieldOf = (els: El[], key: string) =>
  els.find((e) => isNamed(e, "FieldInput") && (e.props.field as { key: string }).key === key) as El;
const appDoc = (name: string) =>
  Object.assign(
    doc([
      { ...node("pic", "input.image", [0, 0]), label: "Photo" },
      node("sel", "input.select", [0, 100], { options: ["", "a"], value: "a" }),
      node("t", "input.text", [0, 200], { value: "hi" }),
      node("out", "output.preview", [0, 400]),
    ]),
    { name },
  );

/** The app page with flow f1 ("one") loaded. */
async function openApp() {
  const app = appPage();
  app.render("f1");
  app.flows.f1[0].resolve({ doc: appDoc("one") });
  await flushMicrotasks();
  return app;
}
const fileInputOf = (app: ReturnType<typeof appPage>) =>
  app.child(fieldOf(app.render("f1"), "pic")).find((e) => e.type === "input") as El;

test("the app's file input is cleared after each pick, so a retry with the same file uploads again", async () => {
  const app = await openApp();
  const pick = { target: { files: [{ name: "a.png" }], value: "C:\\fakepath\\a.png" } };
  (fileInputOf(app).props.onChange as (e: unknown) => void)(pick);
  assert.equal(pick.target.value, "", "the input forgets the file");
  assert.equal(app.uploads.length, 1);
});

test("the app's Run waits for every media upload to finish", async () => {
  const app = await openApp();
  (fileInputOf(app).props.onChange as (e: unknown) => void)({ target: { files: [{ name: "a.png" }], value: "" } });
  const runDisabled = () => byTestId(app.render("f1"), "flows-app-run")?.props.disabled;
  assert.equal(runDisabled(), true, "Run is off while the upload is in flight");
  app.uploads[0].resolve({ kind: "image", ref: "image:1" });
  await flushMicrotasks();
  assert.equal(runDisabled(), false);
  assert.deepEqual((fieldOf(app.render("f1"), "pic").props.value as { ref: string }).ref, "image:1");
});

test("the app page resets on another flow and ignores the old flow's late load, edits and run", async () => {
  const app = await openApp();
  const els = app.render("f1");
  const oldText = fieldOf(els, "t");
  (byTestId(els, "flows-app-run")?.props.onClick as () => void)();
  await flushMicrotasks();
  assert.equal(app.starts.length, 1);
  // Switch flows: the page drops the old doc at once and shows Loading.
  const loading = app.render("f2");
  assert.equal(byTestId(loading, "flows-app"), undefined);
  assert.ok(loading.some((e) => isNamed(e, "Spinner")));
  app.flows.f2[0].resolve({ doc: appDoc("two") });
  await flushMicrotasks();
  app.render("f2");
  // The old flow's field and its run come back late: neither touches the new flow.
  (oldText.props.onChange as (v: unknown) => void)("stale");
  app.starts[0].resolve({ run_id: "old-run" });
  await flushMicrotasks();
  const now = app.render("f2");
  assert.equal(fieldOf(now, "t").props.value, "hi");
  assert.deepEqual(app.followed, [], "the old run is not followed");
  assert.ok(now.some((e) => e.props.children === "two"));
});

test("a load that finishes after the user moved to another flow is dropped", async () => {
  const app = appPage();
  app.render("f1");
  app.render("f2");
  app.flows.f1[0].resolve({ doc: appDoc("one") });
  await flushMicrotasks();
  assert.equal(byTestId(app.render("f2"), "flows-app"), undefined, "still loading f2");
  app.flows.f2[0].resolve({ doc: appDoc("two") });
  await flushMicrotasks();
  assert.ok(app.render("f2").some((e) => e.props.children === "two"));
});

test("an app's Select keeps a configured empty option", async () => {
  const app = await openApp();
  const field = fieldOf(app.render("f1"), "sel");
  const picked: unknown[] = [];
  const els = app.child({ ...field, props: { ...field.props, onChange: (v: unknown) => picked.push(v) } });
  const items = els.filter((e) => isNamed(e, "SelectItem"));
  assert.equal(items.length, 2, "the empty option is listed");
  assert.equal(items[0].props.children, "(none)");
  assert.notEqual(items[0].props.value, "", "Radix gets a non-empty stand-in");
  const select = els.find((e) => isNamed(e, "Select")) as El;
  (select.props.onValueChange as (v: unknown) => void)(items[0].props.value);
  assert.deepEqual(picked, [""], "picking it stores an empty string");
});

test("history never shows another node's strip", async () => {
  const pending: Record<string, ReturnType<typeof deferred<{ entries: unknown[] }>>> = {};
  const viewed: unknown[] = [];
  const pins = { pins: {} as Record<string, unknown> };
  const mod = loadWithStubs<{ NodeHistory: (p: unknown) => unknown }>(
    new URL("../src/features/flows/editor/node-history.tsx", import.meta.url),
    stubEveryImport({
      react: reactProxy,
      "react/jsx-runtime": stubJsxRuntime(),
      "../api": {
        nodeHistory: (_flow: string, id: string) => {
          pending[id] = deferred();
          return pending[id].promise;
        },
      },
      "../stores/doc-store": { useDocStore: (sel: (s: unknown) => unknown) => sel({ flowId: "f" }) },
      "../stores/runtime-store": { useRuntimeStore: (sel: (s: unknown) => unknown) => sel({ runStatus: null }) },
      "../stores/pins-store": {
        ...pinsStore,
        usePinsStore: Object.assign((sel: (s: unknown) => unknown) => sel(pins), { getState: () => pins }),
      },
    }),
  );
  const hooks = fakeReact();
  active = hooks;
  const onView = (v: unknown) => viewed.push(v);
  const render = (nodeId: string) => elementsOf(hooks.render(mod.NodeHistory, { nodeId, onView }));
  (byTestId(render("a"), "flows-history-toggle")?.props.onClick as () => void)();
  render("a");
  pending.a.resolve({ entries: [{ run_id: "r1", updated_at: 0, outputs: { text: "from a" } }] });
  await flushMicrotasks();
  assert.ok(byTestId(render("a"), "flows-history-strip"), "a's strip shows");
  viewed.length = 0;
  assert.equal(byTestId(render("b"), "flows-history-strip"), undefined, "b starts with nothing");
  assert.deepEqual(viewed, [null], "and nothing viewed");
});
