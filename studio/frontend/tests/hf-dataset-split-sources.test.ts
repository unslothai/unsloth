// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);
const { loadHfDatasetSplits } = await import(
  "../src/hooks/hf-dataset-split-sources.ts"
);
const { resetHfEndpoints, setHfEndpoints } = await import("../src/lib/hf-endpoint.ts");
const hub = async (): Promise<{ dataset: string; config: string; split: string }[]> => [];

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
    hub,
  });

  assert.equal(result.source, "local");
  assert.deepEqual(result.entries, [localEntry]);
  assert.equal(remoteCalls, 0);
});

test("online resolution falls back to datasets-server when local metadata is absent", async () => {
  const result = await loadHfDatasetSplits(args(), {
    local: async () => [],
    remote: async () => [remoteEntry],
    hub,
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
    hub,
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
      hub,
    }),
    (error: unknown) =>
      error instanceof DOMException && error.name === "AbortError",
  );
});

test("a datasets-server that fails or knows nothing falls back to the repo files", async (t) => {
  const load = (remote: typeof hub, hubFetch: typeof hub) =>
    loadHfDatasetSplits(args({ preferLocalCache: false }), { local: hub, remote, hub: hubFetch });
  const fail = (message: string) => () => Promise.reject(new Error(message));
  for (const remote of [fail("Unexpected token '<'"), hub]) {
    const result = await load(remote, async () => [remoteEntry]);
    assert.deepEqual([result.source, result.entries], ["hub", [remoteEntry]]);
  }
  const failed = await load(fail("Failed to fetch splits (502)"), fail("Dataset scripts are no longer supported"));
  assert.deepEqual([failed.source, /legacy custom script/.test(failed.error ?? "")], ["manual", true]);
  setHfEndpoints("http://127.0.0.1:8888/api/hub/modelscope", undefined, "modelscope");
  t.after(resetHfEndpoints);
  assert.equal((await load(async () => [remoteEntry], async () => [remoteEntry])).source, "hub");
});
