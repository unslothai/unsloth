// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";
import type { StubElement } from "./helpers/module-stubs.ts";

const ROWS_URL = new URL(
  "../src/features/settings/components/cache-storage-rows.tsx",
  import.meta.url,
);

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("type" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}

function loadRows(breakdown: unknown[] = []) {
  const state: unknown[] = [];
  let cursor = 0;
  let effect: (() => void) | undefined;
  let deps: unknown[] = [];
  const noop = () => undefined;
  const rows = loadWithStubs<{
    HuggingFaceCacheFolders: (props: unknown) => unknown;
    singleClearDescriptionKeys: (key: string) => string[];
  }>(ROWS_URL, {
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
      useCallback: noop,
      useRef: () => ({ current: 0 }),
    },
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/dialog": {},
    "@/features/hub/stores/inventory-events": {},
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/toast": {},
    "@hugeicons/core-free-icons": {},
    "@/lib/hugeicons-derived": {},
    "@hugeicons/react": {},
    "../api/caches": { loadCacheBreakdown: async () => breakdown },
    "./hf-cache-model-clear": { HfCacheModelClear: "HfCacheModelClear" },
    "./npu-model-clear": {
      NpuModelClear: "NpuModelClear",
      refreshNpuCatalogIfListed: noop,
    },
    "./storage-overview": {},
  });
  const render = (props: unknown) => {
    cursor = 0;
    const tree = rows.HuggingFaceCacheFolders(props);
    const pending = effect;
    effect = undefined;
    pending?.();
    return tree;
  };
  return { rows, render };
}

test("NPU model folders list by folder name and delete through the NPU catalog", async () => {
  const { render } = loadRows([
    {
      name: "Qwen3-0.6B-NPU2",
      path: "/flm/models/Qwen3-0.6B-NPU2",
      sizeBytes: 600,
      checkpoint: "qwen3:0.6b",
      deleteAllowed: true,
    },
    {
      name: "unknown--folder",
      path: "/flm/models/unknown--folder",
      sizeBytes: 20,
      deleteAllowed: false,
    },
  ]);
  const props = { entry: { key: "npu_models" }, version: {}, disabled: false };
  const trigger = elements(render(props)).find(
    (element) => element.type === "Button",
  )!;
  (trigger.props.onClick as () => void)();
  render(props);
  await Promise.resolve();
  const tree = elements(render(props));
  assert.deepEqual(
    tree
      .filter((element) => element.type === "NpuModelClear")
      .map((element) => element.props),
    [
      {
        name: "Qwen3-0.6B-NPU2",
        checkpoint: "qwen3:0.6b",
        path: "/flm/models/Qwen3-0.6B-NPU2",
        disabled: false,
      },
    ],
  );
  assert.equal(
    tree.filter((element) => element.type === "HfCacheModelClear").length,
    0,
  );
  const labels = tree
    .filter((element) => element.type === "span")
    .map((element) => element.props.children);
  assert.ok(labels.includes("unknown--folder"));
});

test("clearing NPU models never promises downloaded models are untouched", () => {
  const { rows } = loadRows();
  assert.deepEqual(rows.singleClearDescriptionKeys("npu_models"), [
    "settings.resources.storage.caches.npuCost",
  ]);
});

test("NPU models count toward the Models segment of the storage bar", () => {
  const tab = readFileSync(
    new URL("../src/features/settings/tabs/resources-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    tab,
    /entry\.key === "hf_hub" \|\| entry\.key === "npu_models"/,
  );
});
