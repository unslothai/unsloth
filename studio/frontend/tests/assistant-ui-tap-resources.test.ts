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
const { propsEqual, tapResources } = await import("../src/lib/assistant-ui-tap-resources.ts");

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
  const sub = root.render(Parent({ items: [] }) as never) as { getValue: () => unknown; subscribe: (f: () => void) => () => void };
  const render = (items: { id: string; label: string }[]) =>
    (flushResourcesSync(() => root.render(Parent({ items }) as never)) as typeof sub).getValue() as string[];
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
  assert.match(vite, /find: \/\^@assistant-ui\\\/tap\$\/,\s*replacement: path\.resolve\(__dirname, "\.\/src\/lib\/assistant-ui-tap\.ts"\)/);
  assert.match(vite, /find: \/\^@assistant-ui\\\/tap\\\/react\$\/,\s*replacement: path\.resolve\(__dirname, "\.\/src\/lib\/assistant-ui-tap-react\.ts"\)/);
  const shim = readFileSync(new URL("../src/lib/assistant-ui-tap.ts", import.meta.url), "utf8");
  assert.match(shim, /export \* from "\.\.\/\.\.\/node_modules\/@assistant-ui\/tap\/dist\/index\.js";\nexport \{ tapResources, withContextProvider \} from "\.\/assistant-ui-tap-resources";/);
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.equal(pkg.dependencies["@assistant-ui/tap"], "0.5.10", "the stand-in follows tap 0.5.10's tapResources; re-check it on upgrade");
});
