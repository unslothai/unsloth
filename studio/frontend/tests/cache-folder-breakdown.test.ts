// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

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
      "@hugeicons/react": {},
      "../api/caches": {
        loadCacheBreakdown: () =>
          new Promise((resolve) => requests.push(resolve)),
      },
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
