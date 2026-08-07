// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const { loadHfDatasetSplits } = await import(
  "../src/hooks/hf-dataset-split-sources.ts"
);

function args(overrides: Record<string, unknown> = {}) {
  return {
    datasetName: "org/data",
    localPath: "/cache/datasets--org--data",
    online: true,
    preferLocalCache: true,
    signal: new AbortController().signal,
    ...overrides,
  };
}

const localEntry = {
  dataset: "org/data",
  config: "offline",
  split: "validation",
};
const remoteEntry = {
  dataset: "org/data",
  config: "default",
  split: "train",
};

test("cached dataset split resolution uses local metadata without a remote request", async () => {
  let remoteCalls = 0;
  const result = await loadHfDatasetSplits(args(), {
    local: async () => [localEntry],
    remote: async () => {
      remoteCalls += 1;
      return [remoteEntry];
    },
  });

  assert.equal(result.source, "local");
  assert.deepEqual(result.entries, [localEntry]);
  assert.equal(remoteCalls, 0);
});

test("online resolution falls back to datasets-server when local metadata is absent", async () => {
  const result = await loadHfDatasetSplits(args(), {
    local: async () => [],
    remote: async () => [remoteEntry],
  });

  assert.equal(result.source, "remote");
  assert.deepEqual(result.entries, [remoteEntry]);
});

test("offline resolution exposes manual entry instead of attempting the network", async () => {
  let remoteCalls = 0;
  const result = await loadHfDatasetSplits(args({ online: false }), {
    local: async () => [],
    remote: async () => {
      remoteCalls += 1;
      return [remoteEntry];
    },
  });

  assert.equal(result.source, "manual");
  assert.deepEqual(result.entries, []);
  assert.match(result.error ?? "", /Enter the values manually/i);
  assert.equal(remoteCalls, 0);
});

test("an aborted local lookup cannot publish stale dataset options", async () => {
  const controller = new AbortController();
  await assert.rejects(
    loadHfDatasetSplits(args({ signal: controller.signal }), {
      local: async () => {
        controller.abort();
        return [localEntry];
      },
      remote: async () => [remoteEntry],
    }),
    (error: unknown) =>
      error instanceof DOMException && error.name === "AbortError",
  );
});
