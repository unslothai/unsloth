// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import {
  pinKey,
  pinnedQuantEntries,
} from "../src/features/model-picker/components/model-selector/pinned-models.ts";
import {
  missingPinnedQuants,
  resolvePinnedQuantSources,
} from "../src/features/model-picker/components/model-selector/pinned-quant-sources.ts";

test("bare GGUF pins survive another copy and disappear after the last copy", async () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/reconcile-gguf-pins.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const declaration = source
    .slice(source.indexOf("export async function"))
    .replace("export ", "");
  const compile = new Function(
    "listCachedGguf",
    "listGgufVariants",
    "pinKey",
    "pinnedQuantEntries",
    "usePinnedModelsStore",
    "missingPinnedQuants",
    `${ts.transpileModule(declaration, { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText}; return reconcileGgufPinsAfterDelete;`,
  );
  const repoId = "Org/Model";
  for (const copies of [[{ repo_id: repoId, cache_path: "/other" }], []]) {
    let pinned = [repoId, "Other/Model"];
    const reconcile = compile(
      async () => copies,
      async () => ({ variants: [] }),
      pinKey,
      pinnedQuantEntries,
      {
        getState: () => ({
          pinned,
          togglePinned: (id: string, quant?: string) => {
            pinned = pinned.filter((key) => key !== pinKey(id, quant));
          },
        }),
      },
      missingPinnedQuants,
    );
    await reconcile(repoId);
    assert.deepEqual(
      pinned,
      copies.length ? [repoId, "Other/Model"] : ["Other/Model"],
    );
  }
  const reconcile = compile(
    async () => {
      throw new Error("scan unavailable");
    },
    () => assert.fail("unexpected variant lookup"),
    pinKey,
    pinnedQuantEntries,
    {
      getState: () => ({
        pinned: [repoId],
        togglePinned: () => assert.fail("an unavailable scan must retain pins"),
      }),
    },
    missingPinnedQuants,
  );
  await reconcile(repoId);
});

test("the pinned-row delete handler revalidates the surviving copy", async () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const start = source.indexOf(
    "await deleteCachedModel(\n                  entry.repoId,",
  );
  assert.ok(start >= 0);
  const body = source.slice(start, source.indexOf("\n              },", start));
  const pins = [{ repoId: "Org/Model", quant: "Q8_0" }];
  let pinned = true;
  let deleted = false;
  const run = new Function(
    "entry",
    "hfToken",
    "deleteCachedModel",
    "refreshCachedLists",
    "togglePinned",
    "reconcileGgufPinsAfterDelete",
    `return (async () => { ${body} })();`,
  );
  await run(
    { ...pins[0], cachePath: "/active" },
    "",
    async () => {
      deleted = true;
    },
    () => {},
    () => {
      pinned = false;
    },
    async () => {
      assert.equal(deleted, true);
      const missing = await missingPinnedQuants(
        pins,
        [{ repo_id: "Org/Model", cache_path: "/old" }],
        async () => [
          { quant: "Q8_0", filename: "model.gguf", downloaded: true },
        ],
      );
      if (missing.length) pinned = false;
    },
  );
  assert.equal(pinned, true);
});

test("deleting one copy retains its shared pin until the last complete copy is gone", async () => {
  const pins = [{ repoId: "Org/Model", quant: "Q8_0" }];
  const remaining = [{ repo_id: "Org/Model", cache_path: "/old" }];
  const complete = [
    { quant: "Q8_0", filename: "model.gguf", downloaded: true },
  ];
  assert.deepEqual(
    await missingPinnedQuants(pins, remaining, async () => complete),
    [],
  );
  assert.deepEqual(
    await missingPinnedQuants(pins, [], async () => complete),
    pins,
  );
  assert.deepEqual(
    await missingPinnedQuants(pins, remaining, async () => [
      { ...complete[0], partial: true },
    ]),
    pins,
  );
  await assert.rejects(
    missingPinnedQuants(pins, remaining, async () => {
      throw new Error("offline");
    }),
    /offline/,
  );
});

test("the picker inventory adapter preserves the active copy for duplicate pins", async () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const start = source.indexOf("function toCachedGgufRepo(");
  assert.ok(start >= 0);
  const code = source.slice(start, source.indexOf("\n}", start) + 2);
  const convert = new Function(
    "epochMillisecondsToSeconds",
    `${
      ts.transpileModule(code, {
        compilerOptions: { target: ts.ScriptTarget.ES2020 },
      }).outputText
    }; return toCachedGgufRepo;`,
  )((value: number) => value / 1000);
  const copies = [false, true].map((active) =>
    convert({
      repoId: "Org/Model",
      id: active ? "active" : "old",
      loadId: active ? "Org/Model" : "/old/rev",
      cachePath: active ? "/active" : "/old",
      activeCache: active,
      bytes: 100,
      lastModified: 0,
      capabilities: { supportsVision: false },
    }),
  );
  const entries = await resolvePinnedQuantSources(
    [{ repoId: "Org/Model", quant: "Q8_0" }],
    copies,
    async () => [
      { quant: "Q8_0", filename: "Model-Q8_0.gguf", downloaded: true },
    ],
  );
  assert.equal(entries[0].cachePath, "/active");
});

test("a pinned quant resolves its load, reveal, and delete target from the cache that holds it", async () => {
  const copies = [
    {
      repo_id: "Org/Model",
      cache_path: "/default",
      load_id: "Org/Model",
      active_cache: true,
    },
    {
      repo_id: "Org/Model",
      cache_path: "/custom",
      load_id: "/custom/snapshots/rev",
      active_cache: false,
    },
  ];
  const reads: string[] = [];
  const pins = [{ repoId: "Org/Model", quant: "Q8_0" }];
  const entries = await resolvePinnedQuantSources(
    pins,
    copies,
    async (copy) => {
      assert.ok(copy.cache_path);
      reads.push(copy.cache_path);
      const quant = copy.active_cache ? "Q6_K" : "Q8_0";
      return [{ quant, filename: `Model-${quant}.gguf`, downloaded: true }];
    },
  );
  assert.deepEqual(reads, ["/default", "/custom"]);
  assert.deepEqual(entries, [
    {
      ...pins[0],
      loadId: "/custom/snapshots/rev",
      cachePath: "/custom",
      filename: "Model-Q8_0.gguf",
    },
  ]);
  assert.deepEqual(
    await resolvePinnedQuantSources(pins, copies, async () => [
      {
        quant: "Q8_0",
        filename: "Model-Q8_0.gguf",
        downloaded: true,
        partial: true,
      },
    ]),
    [],
  );
});

test("duplicate pinned quants prefer the active copy and survive another copy's read failure", async () => {
  const copies = [
    { repo_id: "Org/Model", cache_path: "/old", active_cache: false },
    { repo_id: "Org/Model", cache_path: "/active", active_cache: true },
  ];
  for (const failOld of [false, true]) {
    const entries = await resolvePinnedQuantSources(
      [{ repoId: "Org/Model", quant: "Q8_0" }],
      copies,
      async (copy) => {
        if (failOld && copy.cache_path === "/old")
          throw new Error("unavailable");
        return [
          { quant: "Q8_0", filename: "Model-Q8_0.gguf", downloaded: true },
        ];
      },
    );
    assert.equal(entries[0].cachePath, "/active");
  }
});
