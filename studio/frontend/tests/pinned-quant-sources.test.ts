// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { resolvePinnedQuantSources } from "../src/features/model-picker/components/model-selector/pinned-quant-sources.ts";

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
