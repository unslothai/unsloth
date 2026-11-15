// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// tap reads NODE_ENV at import: in development a root mounts in strict mode, running its effects,
// cleaning them up (which unmounts every tapResources child) and running them again, as the dev
// server does under React's StrictMode.
process.env.NODE_ENV = "development";
delete (globalThis as { MessageChannel?: unknown }).MessageChannel;
const { createResourceRoot, flushResourcesSync, resource, tapEffect, tapResources: upstreamTapResources, tapState, withKey } = await import(
  "../node_modules/@assistant-ui/tap/dist/index.js"
);
const { tapResources } = await import("../src/lib/assistant-ui-tap-resources.ts");

const mount = (impl: typeof tapResources) => {
  const setters = new Map<string, (v: number) => void>();
  const live = new Set<string>();
  const Child = resource(({ id }: { id: string }) => {
    const [count, setCount] = tapState(0);
    setters.set(id, setCount);
    tapEffect(() => {
      live.add(id);
      return () => void live.delete(id);
    }, []);
    return `${id}:${count}`;
  });
  const Parent = resource(({ ids }: { ids: string[] }) => impl(() => ids.map((id) => withKey(id, Child({ id })) as never), [ids]));
  const root = createResourceRoot();
  const ids = ["a", "b"];
  const sub = root.render(Parent({ ids }) as never) as { getValue: () => string[] };
  return { sub, setters, live, render: () => (flushResourcesSync(() => root.render(Parent({ ids }) as never)) as typeof sub).getValue() };
};

test("children stay mounted after tap's strict-mode effect cleanup, as upstream", () => {
  for (const impl of [upstreamTapResources as typeof tapResources, tapResources]) {
    const m = mount(impl);
    assert.deepEqual([...m.live].sort(), ["a", "b"], "every child's effects are live after the strict remount");
    flushResourcesSync(() => m.setters.get("b")!(3));
    assert.deepEqual(m.sub.getValue(), ["a:0", "b:3"], "a child's own update still renders");
    assert.deepEqual(m.render(), ["a:0", "b:3"]);
  }
});
