// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { appFields, applyAppValues, appOutputs, foldAppEvents } = await import(
  "../src/features/flows/app/app-form.ts"
);
const { historyAgo, historyCard, historyPreview, pinsForRun, sameOutputs, usePinsStore } = await import(
  "../src/features/flows/stores/pins-store.ts"
);

type Doc = ReturnType<typeof doc>;

function sample(): Doc {
  const d = doc([
    { ...node("out", "output.preview", [0, 400]) },
    {
      ...node("topic", "input.param", [0, 0]),
      params: { name: "topic", type: "text", default: "otters" },
    },
    { ...node("pic", "input.image", [0, 200]), label: "Photo" },
    { ...node("n", "input.number", [300, 100]), params: { value: 3 } },
    {
      ...node("sel", "input.select", [0, 100]),
      params: { options: ["a", "b"], value: "b" },
    },
    node("loop", "logic.for_each", [0, 500]),
    { ...node("inner", "input.text", [0, 300]), group: "loop" },
    { ...node("lout", "output.preview", [0, 460]), group: "loop" },
    { ...node("framed", "input.text", [0, 250]), group: "frame1" },
    { ...node("fout", "output.result", [0, 450]), group: "frame1" },
    { ...node("off", "input.text", [0, 350]), disabled: true },
  ]);
  return d;
}

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
  const fields = appFields(d as never);
  const media = { kind: "image", ref: "image:1" };
  const { doc: filled, params } = applyAppValues(d as never, fields, {
    topic: "cats",
    pic: media,
    n: 7,
  });
  assert.deepEqual(params, { topic: "cats" });
  const byId = Object.fromEntries(filled.nodes.map((n) => [n.id, n]));
  assert.deepEqual(byId.pic.params.asset, media);
  assert.equal(byId.n.params.value, 7);
  assert.equal(byId.sel.params.value, "b");
  assert.equal(
    d.nodes.find((n) => n.id === "n")?.params.value,
    3,
    "the loaded flow is unchanged",
  );
  assert.deepEqual(
    appOutputs(d as never).map((n) => n.id),
    ["out", "fout"],
  );
});

test("run events fold into progress, results and errors", () => {
  const ev = (type: string, node_id: string | null, data: object) => ({
    seq: 1,
    run_id: "r",
    type,
    node_id,
    ts: 0,
    data,
  });
  let s = {
    status: "idle",
    done: 0,
    total: 0,
    results: {},
    errors: [] as string[],
  };
  s = foldAppEvents(
    s,
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
  const d = doc([]);
  (d as { params?: unknown[] }).params = [
    { id: "config", type: "json", default: { name: "a" } },
  ];
  const fields = appFields(d as never);
  assert.deepEqual(
    applyAppValues(d as never, fields, { config: '{"name": "b"}' }).params,
    { config: { name: "b" } },
  );
  assert.deepEqual(
    applyAppValues(d as never, fields, { config: "{not json" }).params,
    { config: "{not json" },
  );
  const s = foldAppEvents(
    { status: "running", done: 1, total: 3, results: {}, errors: [] },
    [
      {
        seq: 9,
        run_id: "r",
        type: "snapshot",
        node_id: null,
        ts: 0,
        data: {
          run: { status: "interrupted", error: "Studio restarted" },
          nodes: { out: { outputs: { value: "kept" } } },
        },
      },
    ] as never,
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

test("the Run Flow picker refetches the flow list and history never shows another node's strip", () => {
  const fields = readSrc("features/flows/editor/param-fields.tsx");
  assert.match(fields, /source === "\/api\/flows"\) remoteOptions\.delete\(key\)/);
  const history = readSrc("features/flows/editor/node-history.tsx");
  const reset = history.slice(history.indexOf("a different node starts with nothing viewed"));
  assert.ok(reset.indexOf("setEntries(null)") < reset.indexOf("}, [nodeId, onView]);"));
});

test("an untouched seed keeps randomizing, a typed-in seed is used as is", () => {
  const d = doc([
    {
      ...node("seed", "input.seed", [0, 0]),
      params: { value: 42, randomize: true },
    },
  ]);
  const fields = appFields(d as never);
  const seedOf = (value: unknown) =>
    applyAppValues(d as never, fields, { seed: value }).doc.nodes[0].params;
  assert.equal(seedOf(42).randomize, true);
  assert.deepEqual(seedOf(7), { value: 7, randomize: false });
});

test("the app's file input is cleared after each pick, so a retry with the same file uploads again", async () => {
  const { readFileSync } = await import("node:fs");
  const source = readFileSync(
    new URL("../src/features/flows/app/flow-app-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /const file = e\.target\.files\?\.\[0\];[\s\S]{0,120}e\.target\.value = "";/);
});

test("the app's Run waits for every media upload to finish", async () => {
  const { readFileSync } = await import("node:fs");
  const source = readFileSync(
    new URL("../src/features/flows/app/flow-app-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /onUploading\?\.\(true\)/);
  assert.match(source, /onUploading\?\.\(false\)/);
  assert.match(source, /data-testid="flows-app-run"/);
  assert.match(source, /disabled=\{uploadingAny\}/);
});

test("the app page resets on another flow, shows an empty text as empty, and history shows output blocks", async () => {
  const { readFileSync } = await import("node:fs");
  const read = (path: string) =>
    readFileSync(new URL(path, import.meta.url), "utf8");
  const page = read("../src/features/flows/app/flow-app-page.tsx");
  assert.match(page, /let alive = true;[\s\S]{0,200}setDoc\(null\);[\s\S]{0,80}setRun\(IDLE\);\s*setRunId\(null\);/);
  assert.doesNotMatch(page, /JSON\.stringify\(value \?\? ""\)/);
  assert.match(page, /value == null\s*\?\s*""/);
  const inspector = read("../src/features/flows/editor/inspector.tsx");
  assert.match(inspector, /Object\.keys\(viewed\)\.map\(\(id\) => \(\{ id, label: id \}\)\)/);
});

test("an untouched seed with no stored value keeps randomizing, and stale flow loads are ignored", async () => {
  const d = doc([node("s", "input.seed", [0, 0], { randomize: true })]);
  const fields = appFields(d as never);
  const untouched = Object.fromEntries(fields.map((f) => [f.key, f.initial]));
  const { doc: filled } = applyAppValues(d as never, fields, untouched);
  assert.deepEqual(
    (filled as { nodes: { params: unknown }[] }).nodes[0].params,
    { randomize: true },
  );
  const { readFileSync } = await import("node:fs");
  const page = readFileSync(
    new URL("../src/features/flows/app/flow-app-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(page, /if \(started !== generation\.current\) return;\s*setRunId\(run_id\);/);
  assert.match(page, /loaded === generation\.current &&\s*setValues/);
});

test("an untouched Flow Input still sends its declared default", () => {
  const d = doc([]);
  (d as { params?: unknown[] }).params = [
    { id: "topic", type: "text", default: "cats" },
  ];
  const fields = appFields(d as never);
  const untouched = Object.fromEntries(fields.map((f) => [f.key, f.initial]));
  assert.deepEqual(applyAppValues(d as never, fields, untouched).params, {
    topic: "cats",
  });
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
  assert.deepEqual(historyCard({ image: media }), {
    kind: "image",
    title: "a sloth",
    detail: "1024×768",
    media,
  });
  assert.deepEqual(historyCard({ text: "  hello there " }), {
    kind: "text",
    title: "hello there",
    detail: "14 chars",
  });
  assert.equal(historyCard({ items: ["a", "b"] }).title, "2 items");
  assert.equal(historyCard({ n: 3 }).kind, "number");
  assert.deepEqual(historyCard({ out: { kind: "dataset", source: "hf", ref: "org/rows", rows: 50 } }), {
    kind: "dataset",
    title: "rows",
    detail: "50 rows",
  });
  assert.deepEqual(historyCard({ obj: { a: 1, b: 2 } }), { kind: "json", title: "JSON", detail: "2 keys" });
});

test("history captions say how long ago", () => {
  const now = Date.parse("2026-10-30T12:00:00Z");
  assert.equal(historyAgo(Date.parse("2026-10-30T11:59:30Z"), now), "just now");
  assert.equal(historyAgo(Date.parse("2026-10-30T11:55:00Z"), now), "5m ago");
  assert.equal(historyAgo(Date.parse("2026-10-30T09:00:00Z"), now), "3h ago");
});
