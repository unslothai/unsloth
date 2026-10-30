// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";
import type { StubElement } from "./helpers/module-stubs.ts";

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("type" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}

test("only verified model-copy rows expose Clear and retain individual locations", async () => {
  const state: unknown[] = [];
  let cursor = 0;
  let effect: (() => void) | undefined;
  let deps: unknown[] = [];
  const good = {
    name: "models--Org--Model",
    sizeBytes: 100,
    repoId: "Org/Model",
    repoType: "model",
    deleteAllowed: true,
  };
  const copies = [
    {
      ...good,
      path: "/one/models--Org--Model",
      cachePath: "/one/models--Org--Model",
    },
    {
      ...good,
      path: "/two/models--Org--Model",
      cachePath: "/two/models--Org--Model",
    },
    {
      ...good,
      path: "/forbidden",
      cachePath: "/forbidden",
      deleteAllowed: false,
    },
    { ...good, path: "/dataset", cachePath: "/dataset", repoType: "dataset" },
    { ...good, path: "/missing-copy" },
    { name: ".locks", path: "/locks", sizeBytes: 1 },
    { name: ".locks", path: "/empty-locks", sizeBytes: 0 },
  ];
  const component = loadWithStubs<{
    HuggingFaceCacheFolders: (props: unknown) => unknown;
  }>(
    new URL(
      "../src/features/settings/components/cache-storage-rows.tsx",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          const index = cursor++;
          if (!(index in state)) state[index] = initial;
          return [
            state[index],
            (next: unknown) => {
              state[index] =
                typeof next === "function" ? next(state[index]) : next;
            },
          ];
        },
        useEffect: (next: () => void, values: unknown[]) => {
          if (values.some((value, index) => value !== deps[index])) {
            deps = values;
            effect = next;
          }
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dialog": {},
      "@/features/hub/stores/inventory-events": {},
      "@/i18n": { useT: () => (key: string) => key },
      "@/lib/toast": {},
      "@hugeicons/core-free-icons": {},
      "@/lib/folder-icons": {},
      "@hugeicons/react": {},
      "../api/caches": { loadCacheBreakdown: async () => copies },
      "./hf-cache-model-clear": { HfCacheModelClear: "HfCacheModelClear" },
      "./storage-overview": {},
    },
  );
  const version = {};
  const render = () => {
    cursor = 0;
    const tree = component.HuggingFaceCacheFolders({
      entry: { key: "hf_hub" },
      version,
      disabled: true,
    });
    const pending = effect;
    effect = undefined;
    pending?.();
    return tree;
  };
  const trigger = elements(render()).find(
    (element) => element.type === "Button",
  )!;
  (trigger.props.onClick as () => void)();
  render();
  await Promise.resolve();
  const actions = elements(render()).filter(
    (element) => element.type === "HfCacheModelClear",
  );
  const listed = elements(render())
    .map((element) => element.props.title)
    .filter((title) => typeof title === "string");
  assert.ok(listed.includes("/locks"));
  assert.ok(!listed.includes("/empty-locks"));
  assert.deepEqual(
    actions.map((action) => action.props),
    [
      {
        repoId: "Org/Model",
        cachePath: "/one/models--Org--Model",
        disabled: true,
      },
      {
        repoId: "Org/Model",
        cachePath: "/two/models--Org--Model",
        disabled: true,
      },
    ],
  );
});

test("HF folders are requested only on expansion and stale walks cannot replace a refreshed result", async () => {
  const state: unknown[] = [];
  let cursor = 0;
  let effect: (() => (() => void) | void) | undefined;
  let previousDeps: unknown[] = [];
  let cleanup: (() => void) | void;
  const requests: ((rows: unknown[]) => void)[] = [];
  const rows = loadWithStubs<{
    HuggingFaceCacheFolders: (props: unknown) => unknown;
  }>(
    new URL(
      "../src/features/settings/components/cache-storage-rows.tsx",
      import.meta.url,
    ),
    {
      react: {
        useState: (initial: unknown) => {
          const index = cursor++;
          if (!(index in state)) state[index] = initial;
          return [
            state[index],
            (next: unknown) => {
              state[index] =
                typeof next === "function" ? next(state[index]) : next;
            },
          ];
        },
        useEffect: (next: typeof effect, deps: unknown[]) => {
          if (deps.some((value, index) => value !== previousDeps[index])) {
            cleanup?.();
            previousDeps = deps;
            effect = next;
          }
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dialog": {},
      "@/features/hub/stores/inventory-events": {},
      "@/i18n": { useT: () => (key: string) => key },
      "@/lib/toast": {},
      "@hugeicons/core-free-icons": {},
      "@/lib/folder-icons": {},
      "@hugeicons/react": {},
      "../api/caches": {
        loadCacheBreakdown: () =>
          new Promise((resolve) => requests.push(resolve)),
      },
      "./hf-cache-model-clear": { HfCacheModelClear: "HfCacheModelClear" },
      "./storage-overview": {},
    },
  );
  function render(version: unknown) {
    cursor = 0;
    const tree = rows.HuggingFaceCacheFolders({
      entry: { key: "hf_hub" },
      version,
    }) as { props: { children: { props: { onClick: () => void } }[] } };
    const pending = effect;
    effect = undefined;
    if (pending) cleanup = pending();
    return tree;
  }
  const first = {};
  const tree = render(first);
  assert.equal(requests.length, 0);
  tree.props.children[0].props.onClick();
  render(first);
  assert.equal(requests.length, 1);
  const refreshed = {};
  render(refreshed);
  assert.equal(requests.length, 2);
  requests[1]([{ name: "models--new--repo", path: "/new", sizeBytes: 100 }]);
  await Promise.resolve();
  requests[0]([{ name: "stale", path: "/stale", sizeBytes: 999 }]);
  await Promise.resolve();
  assert.match(JSON.stringify(render(refreshed)), /new\/repo/);
  assert.doesNotMatch(JSON.stringify(render(refreshed)), /stale/);
});
