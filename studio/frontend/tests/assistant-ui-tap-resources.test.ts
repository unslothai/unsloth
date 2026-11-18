// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

// tap schedules flushes on a MessageChannel created at import, which keeps Node's event loop alive after
// the tests; without one it falls back to setTimeout.
delete (globalThis as { MessageChannel?: unknown }).MessageChannel;
const {
  createResourceRoot,
  flushResourcesSync,
  resource,
  tapEffect,
  tapResources: upstreamTapResources,
  tapState,
  withKey,
} = await import("../node_modules/@assistant-ui/tap/dist/index.js");
const { propsEqual, sameContextValue, tapResources, withContextProvider } = await import("../src/lib/assistant-ui-tap-resources.ts");
const { createResourceContext, tap } = await import("../node_modules/@assistant-ui/tap/dist/index.js");

type Probe = { renders: Map<string, number>; setters: Map<string, (v: number) => void>; unmounted: string[] };

const harness = (impl: typeof tapResources) => {
  const probe: Probe = { renders: new Map(), setters: new Map(), unmounted: [] };
  const Child = resource(({ id, label }: { id: string; label: string }) => {
    probe.renders.set(id, (probe.renders.get(id) ?? 0) + 1);
    const [count, setCount] = tapState(0);
    probe.setters.set(id, setCount);
    tapEffect(() => () => void probe.unmounted.push(id), []);
    return `${id}:${label}:${count}`;
  });
  // The shape tapClientLookup hands tapResources: a wrapper element whose props are a fresh inner element.
  const Wrapper = resource((inner: { type: unknown; props: unknown; key: string }) => {
    const [out] = upstreamTapResources(() => [inner as never], [inner]);
    return out;
  });
  const Parent = resource(({ items }: { items: { id: string; label: string }[] }) =>
    impl(
      () => items.map((it) => withKey(it.id, Wrapper(withKey(it.id, Child({ id: it.id, label: it.label })) as never)) as never),
      [items],
    ),
  );
  const root = createResourceRoot();
  type Sub = { getValue: () => unknown; subscribe: (f: () => void) => () => void };
  root.render(Parent({ items: [] }) as never);
  const render = (items: { id: string; label: string }[]) =>
    (flushResourcesSync(() => root.render(Parent({ items }) as never)) as Sub).getValue() as string[];
  return { probe, render, root };
};

const items = (labels: Record<string, string>) => Object.entries(labels).map(([id, label]) => ({ id, label }));
const snapshot = (p: Probe) => Object.fromEntries(p.renders);

test("an equal element list re-renders no child; upstream re-renders all of them", () => {
  for (const [impl, expectSkip] of [[tapResources, true], [upstreamTapResources as typeof tapResources, false]] as const) {
    const h = harness(impl);
    h.render(items({ a: "x", b: "y", c: "z" }));
    const before = snapshot(h.probe);
    const out = h.render(items({ a: "x", b: "y", c: "z" }));
    const after = snapshot(h.probe);
    assert.deepEqual(out, ["a:x:0", "b:y:0", "c:z:0"]);
    if (expectSkip) assert.deepEqual(after, before, "no child re-rendered");
    else assert.ok(Object.keys(before).every((k) => after[k] > before[k]), "upstream re-renders every child");
  }
});

test("a changed child re-renders alone and its new output shows", () => {
  const h = harness(tapResources);
  h.render(items({ a: "x", b: "y", c: "z" }));
  const before = snapshot(h.probe);
  const out = h.render(items({ a: "x", b: "Y", c: "z" }));
  const after = snapshot(h.probe);
  assert.deepEqual(out, ["a:x:0", "b:Y:0", "c:z:0"]);
  assert.equal(after.a, before.a);
  assert.equal(after.c, before.c);
  assert.ok(after.b > before.b);
});

test("a child's own update re-renders only that child", () => {
  const h = harness(tapResources);
  h.render(items({ a: "x", b: "y", c: "z" }));
  const before = snapshot(h.probe);
  flushResourcesSync(() => h.probe.setters.get("c")!(5));
  const after = snapshot(h.probe);
  assert.equal(after.a, before.a);
  assert.equal(after.b, before.b);
  assert.ok(after.c > before.c);
  // A later equal render still shows the updated state (the cached output is the committed one).
  assert.deepEqual(h.render(items({ a: "x", b: "y", c: "z" })), ["a:x:0", "b:y:0", "c:z:5"]);
});

test("a removed key unmounts and an added key mounts, in order", () => {
  const h = harness(tapResources);
  h.render(items({ a: "x", b: "y", c: "z" }));
  const out = h.render(items({ c: "z", a: "x", d: "w" }));
  assert.deepEqual(out, ["c:z:0", "a:x:0", "d:w:0"]);
  assert.ok(h.probe.unmounted.includes("b"));
  assert.ok(!h.probe.unmounted.includes("a"));
});

test("props compare key by key and nested resource elements by type, key and props", () => {
  const T = () => null;
  const el = (props: object, key = "k") => ({ type: T, props, key });
  const runtime = {};
  assert.ok(propsEqual({ a: 1, r: runtime }, { a: 1, r: runtime }));
  assert.ok(!propsEqual({ a: 1 }, { a: 2 }));
  assert.ok(!propsEqual({ a: 1 }, { a: 1, b: 2 }));
  assert.ok(propsEqual(el({ id: "m1", runtime }), el({ id: "m1", runtime })));
  assert.ok(!propsEqual(el({ id: "m1", runtime }), el({ id: "m2", runtime })));
  assert.ok(!propsEqual(el({ id: "m1" }, "k1"), el({ id: "m1" }, "k2")));
  assert.ok(!propsEqual(el({ r: {} }), el({ r: {} })), "a new object prop is a change");
  assert.ok(!propsEqual([1], [1]), "arrays are compared by identity");
});

test("the app and assistant-ui load tap through the stand-in, and only tapResources and withContextProvider differ", () => {
  const vite = readFileSync(new URL("../vite.config.ts", import.meta.url), "utf8");
  assert.match(vite, /\["@assistant-ui\/tap", path\.resolve\(__dirname, "\.\/src\/lib\/assistant-ui-tap\.ts"\)\]/);
  assert.match(vite, /\["@assistant-ui\/tap\/react", path\.resolve\(__dirname, "\.\/src\/lib\/assistant-ui-tap-react\.ts"\)\]/);
  // The build resolves the stand-in with the plugin, the dev server's optimizer with the same plugin; a
  // resolve alias would send the app's own import to tap's raw files on the dev server, a second copy.
  assert.match(vite, /plugins: \[[^\]]*\{ \.\.\.assistantUiTapStandIn\(\), apply: "build", enforce: "pre" \}/);
  assert.match(vite, /rolldownOptions: \{ plugins: \[assistantUiTapStandIn\(\)\] \}/);
  assert.doesNotMatch(vite, /find: \/\^@assistant-ui\\\/tap/);
  const shim = readFileSync(new URL("../src/lib/assistant-ui-tap.ts", import.meta.url), "utf8");
  assert.match(shim, /export \* from "\.\.\/\.\.\/node_modules\/@assistant-ui\/tap\/dist\/index\.js";\nexport \{ tapResources, withContextProvider \} from "\.\/assistant-ui-tap-resources";/);
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.equal(pkg.dependencies["@assistant-ui/tap"], "0.5.10", "the stand-in follows tap 0.5.10's tapResources; re-check it on upgrade");
});

test("context values compare by identity or one level deep, so a fresh { clientRef, emit } literal is equal", () => {
  const clientRef = { current: null };
  const emit = () => undefined;
  assert.ok(sameContextValue({ clientRef, emit }, { clientRef, emit }));
  assert.ok(!sameContextValue({ clientRef, emit }, { clientRef, emit: () => undefined }));
  const stack = [{}];
  assert.ok(sameContextValue(stack, [stack[0]]));
  assert.ok(!sameContextValue([{}], [{}]));
});

test("a changed context value above the list re-renders every child; an equal fresh one re-renders none", () => {
  const Ctx = createResourceContext({ v: 0 });
  const renders: number[] = [];
  const Child = resource(({ i }: { i: number }) => {
    renders[i] = (renders[i] ?? 0) + 1;
    return `${i}:${(tap(Ctx) as { v: number }).v}`;
  });
  const Parent = resource(({ v }: { v: number }) =>
    // No deps: the list is rebuilt on every render, as the thread's message list is on every update.
    withContextProvider(Ctx as never, { v }, () => tapResources(() => [0, 1].map((i) => withKey(String(i), Child({ i })) as never))),
  );
  const root = createResourceRoot();
  const render = (v: number) => (flushResourcesSync(() => root.render(Parent({ v }) as never)) as { getValue: () => string[] }).getValue();
  assert.deepEqual(render(1), ["0:1", "1:1"]);
  const before = [...renders];
  assert.deepEqual(render(1), ["0:1", "1:1"]);
  assert.deepEqual(renders, before, "a fresh but equal context value re-renders no child");
  assert.deepEqual(render(2), ["0:2", "1:2"]);
  assert.ok(renders.every((n, i) => n > before[i]), "a changed context value re-renders every child");
});
