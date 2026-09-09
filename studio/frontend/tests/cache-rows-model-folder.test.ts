// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The cache rows sit directly under Models Folder in one section, and one of them
// IS the model cache. So the two things this covers are what the rows say when
// that folder moves, and what the confirmation promises before the models go.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

const ROWS_URL = new URL(
  "../src/features/settings/components/cache-storage-rows.tsx",
  import.meta.url,
);

type Rows = {
  singleClearDescriptionKeys: (key: string) => string[];
};

function loadRows(): Rows {
  const noop = () => undefined;
  return loadWithStubs<Rows>(ROWS_URL, {
    react: {
      useCallback: noop,
      useEffect: noop,
      useRef: () => ({ current: 0 }),
      useState: () => [null, noop],
    },
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/button": { Button: noop },
    "@/components/ui/dialog": {
      Dialog: noop,
      DialogContent: noop,
      DialogDescription: noop,
      DialogFooter: noop,
      DialogHeader: noop,
      DialogTitle: noop,
    },
    "@/features/hub/stores/inventory-events": { useInventoryVersion: () => 0 },
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/toast": { toast: {} },
    "../api/caches": {
      bulkPurgeKeys: () => [],
      loadCacheInventory: async () => null,
      purgeCaches: async () => null,
    },
    "./settings-row": { SettingsRow: noop },
  });
}

const SAFETY = "settings.resources.storage.caches.safety";

test("the model cache clear does not promise that models are untouched", () => {
  // The generic assurance is the contradiction: it names downloaded models,
  // which is exactly what an hf_hub clear deletes.
  assert.ok(
    en.settings.resources.storage.caches.safety.includes("Downloaded models"),
  );
  const { singleClearDescriptionKeys } = loadRows();

  assert.deepEqual(singleClearDescriptionKeys("hf_hub"), [
    "settings.resources.storage.caches.hubCost",
  ]);
  // It still holds for the caches that do spare the models.
  assert.deepEqual(singleClearDescriptionKeys("hf_datasets"), [
    "settings.resources.storage.caches.datasetsCost",
    SAFETY,
  ]);
  assert.deepEqual(singleClearDescriptionKeys("unsloth_compiled"), [SAFETY]);
});

test("the bulk clear keeps the assurance, because it excludes the model cache", () => {
  const source = readFileSync(ROWS_URL, "utf8");
  // bulkPurgeKeys drops every opt-in cache, so "downloaded models are not
  // touched" is true of the bulk dialog and only of the bulk dialog.
  const bulk = source.slice(source.indexOf("confirmDescription"));
  assert.match(bulk, /caches\.safety/);
});

test("the cache rows follow the model folder when it moves", () => {
  const source = readFileSync(ROWS_URL, "utf8");
  // Saving Models Folder bumps the inventory version, and the field sits in the
  // same section as these rows: without this they keep showing the path and the
  // size of the folder the user just moved off, next to a Clear button that
  // acts on the new one.
  assert.match(source, /useInventoryVersion\(\)/);
  assert.match(source, /\}, \[refresh, inventoryVersion\]\);/);
  // A move has to force the re-measure. The backend memoises a size for a
  // minute, so an unforced read returns the very figures being replaced.
  assert.match(source, /refresh\(moved \? \{ refresh: true \} : \{\}\)/);

  const api = readFileSync(
    new URL(
      "../src/features/settings/api/hugging-face-cache.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(api, /bumpInventoryVersion\(\)/);
});

/** Drive the real component with a React the test steps by hand. */
function driveRows(options: {
  load: (options: { refresh?: boolean }) => Promise<unknown>;
  purge?: (keys: readonly string[]) => Promise<unknown>;
  version: () => number;
}) {
  const states: unknown[] = [];
  const refs: { current: unknown }[] = [];
  const callbacks: unknown[] = [];
  let stateCursor = 0;
  let refCursor = 0;
  let callbackCursor = 0;
  let previousDeps: unknown[] | null = null;
  let pending: (() => void) | null = null;

  const react = {
    useState: (initial: unknown) => {
      const index = stateCursor++;
      if (states.length <= index) states[index] = initial;
      return [
        states[index],
        (next: unknown) => {
          states[index] =
            typeof next === "function"
              ? (next as (value: unknown) => unknown)(states[index])
              : next;
        },
      ];
    },
    useRef: (initial: unknown) => (refs[refCursor++] ??= { current: initial }),
    useCallback: (fn: unknown) => (callbacks[callbackCursor++] ??= fn),
    useEffect: (effect: () => void, deps: unknown[]) => {
      if (
        previousDeps === null ||
        deps.some((d, i) => d !== previousDeps?.[i])
      ) {
        previousDeps = deps;
        pending = effect;
      }
    },
  };

  const rows = loadWithStubs<{ CacheStorageRows: () => unknown }>(ROWS_URL, {
    react,
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/dialog": {
      Dialog: "Dialog",
      DialogContent: "DialogContent",
      DialogDescription: "DialogDescription",
      DialogFooter: "DialogFooter",
      DialogHeader: "DialogHeader",
      DialogTitle: "DialogTitle",
    },
    "@/features/hub/stores/inventory-events": {
      useInventoryVersion: options.version,
      bumpInventoryVersion: () => undefined,
      getInventoryVersion: options.version,
    },
    "@/i18n": { useT: () => (key: string) => key },
    "@/lib/toast": {
      toast: {
        success: () => undefined,
        warning: () => undefined,
        error: () => undefined,
      },
    },
    "../api/caches": {
      bulkPurgeKeys: () => ["uv"],
      loadCacheInventory: options.load,
      purgeCaches: options.purge ?? (async () => ({})),
    },
    "./settings-row": { SettingsRow: "SettingsRow" },
  });

  return () => {
    stateCursor = 0;
    refCursor = 0;
    callbackCursor = 0;
    const tree = rows.CacheStorageRows();
    const effect = pending;
    pending = null;
    effect?.();
    return tree;
  };
}

type Element = { type: unknown; props?: Record<string, unknown> };

function* walk(node: unknown): Generator<Element> {
  if (Array.isArray(node)) {
    for (const child of node) yield* walk(child);
    return;
  }
  if (!node || typeof node !== "object") return;
  const element = node as Element;
  yield element;
  yield* walk(element.props?.children);
}

const buttons = (tree: unknown, label: string): Element[] =>
  [...walk(tree)].filter(
    (element) => element.type === "Button" && element.props?.children === label,
  );

const click = (element: Element | undefined) =>
  (element?.props?.onClick as (() => void) | undefined)?.();

const CLEAR_ONE = "settings.resources.storage.caches.clearOneAction";
const DETAILS = "settings.resources.storage.caches.detailsAction";

const inventory = (path: string) => ({
  caches: [
    {
      key: "hf_hub",
      group: "models",
      optIn: true,
      paths: [path],
      sizeBytes: 1000,
      entryCount: 1,
      present: true,
      purgeable: true,
      blockedReason: null,
    },
  ],
  totalBytes: 1000,
  reclaimableBytes: 0,
  freeBytes: 1,
  totalDiskBytes: 2,
});

const settle = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};

test("the newest measurement wins, whatever order the walks finish in", async () => {
  // A cold walk of a large hub takes tens of seconds, so the load a mount starts
  // can still be running when a folder save starts a forced one. Without a
  // generation guard the slower FIRST request installs last and the rows settle
  // on the folder the user moved off, while a clear resolves the new one.
  const pending: ((value: unknown) => void)[] = [];
  let version = 0;
  const render = driveRows({
    load: () => new Promise((resolve) => pending.push(resolve)),
    version: () => version,
  });

  render();
  version = 1;
  render();
  assert.equal(pending.length, 2);

  pending[1](inventory("/new/hub"));
  await settle();
  render();
  pending[0](inventory("/old/hub"));
  await settle();

  const tree = render();
  click(buttons(tree, DETAILS)[0]);
  const paths = [...walk(render())]
    .map((element) => element.props?.title)
    .filter((title) => typeof title === "string");
  assert.deepEqual(paths, ["/new/hub"]);
});

test("a clear waits for the measurement that is replacing the rows", async () => {
  // The three section buttons already honour loading. While a forced re-measure
  // is running the rows still show the PREVIOUS inventory, and purgeCaches
  // resolves its key against the current one on the backend, so a per-row clear
  // left enabled there is the destructive mismatch again.
  const pending: ((value: unknown) => void)[] = [];
  let version = 0;
  const render = driveRows({
    load: () => new Promise((resolve) => pending.push(resolve)),
    version: () => version,
  });

  render();
  pending[0](inventory("/old/hub"));
  await settle();
  click(buttons(render(), DETAILS)[0]);

  const settled = render();
  assert.equal(buttons(settled, CLEAR_ONE).length, 1);
  assert.equal(buttons(settled, CLEAR_ONE)[0].props?.disabled, false);

  version = 1;
  render(); // the version change is seen here; the effect runs after the tree
  const reloading = render();
  assert.equal(buttons(reloading, CLEAR_ONE)[0].props?.disabled, true);
  // ...and so does the confirmation, which is the button that actually deletes.
  const confirm = [...walk(reloading)].filter(
    (element) =>
      element.type === "Button" &&
      typeof element.props?.className === "string" &&
      element.props.className.includes("bg-destructive"),
  );
  assert.equal(confirm.length, 1);
  assert.equal(confirm[0].props?.disabled, true);
});
