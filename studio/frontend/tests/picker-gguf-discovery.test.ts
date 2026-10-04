// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { GgufVariantsResponse } from "../src/features/chat/types/api.ts";
import {
  loadPickerGgufVariants,
  readSoleQuantLocalFirst,
} from "../src/features/model-picker/components/model-selector/gguf-discovery.ts";

const local: GgufVariantsResponse = {
  repo_id: "Org/Cached",
  variants: [
    {
      quant: "Q4_K_M",
      filename: "old-Q4_K_M.gguf",
      size_bytes: 256,
      downloaded: true,
      cache_path: "/cache/old",
    },
  ],
  has_vision: false,
  default_variant: "Q4_K_M",
  context_length: 4096,
  dependencies_resolved: true,
};
const options = {
  onDevice: true,
  canDiscoverRemote: () => true,
};

test("On Device answers from disk alone once the Hub is known unreachable", async () => {
  const calls: boolean[] = [];
  const shown: (typeof local)[] = [];
  const result = await loadPickerGgufVariants(
    async (localOnly) => {
      calls.push(localOnly);
      return local;
    },
    { ...options, canDiscoverRemote: () => false },
    (response) => shown.push(response),
  );
  assert.deepEqual(calls, [true]);
  assert.deepEqual(shown, [local]);
  assert.equal(result, local);
});

test("On Device publishes cached quants before a stalled remote request finishes", async () => {
  const calls: boolean[] = [];
  let displayed: GgufVariantsResponse | null = null;
  let rejectRemote!: (error: Error) => void;
  const remote = new Promise<typeof local>((_resolve, reject) => {
    rejectRemote = reject;
  });
  const pending = loadPickerGgufVariants(
    async (localOnly) => {
      calls.push(localOnly);
      return localOnly ? local : remote;
    },
    options,
    (response) => {
      displayed = response;
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(
    displayed,
    local,
    "cached quant must be usable while remote discovery is pending",
  );
  assert.deepEqual(calls, [true, false]);
  rejectRemote(new Error("Hugging Face unreachable"));
  assert.equal(await pending, local);
});

test("online discovery adds published quants without changing a cached quant's load path or readiness", async () => {
  const remote = {
    ...local,
    context_length: 8192,
    variants: [
      {
        ...local.variants[0],
        filename: "new-Q4_K_M.gguf",
        downloaded: false,
        cache_path: "/cache/new",
        update_available: true,
      },
      {
        quant: "Q8_0",
        filename: "model-Q8_0.gguf",
        size_bytes: 512,
        downloaded: false,
        cache_path: null,
      },
    ],
  };
  const result = await loadPickerGgufVariants(
    async (localOnly) => (localOnly ? local : remote),
    options,
    () => {},
  );
  assert.equal(result.variants.length, 2);
  const cached = result.variants.find((v) => v.quant === "Q4_K_M");
  assert.equal(cached?.filename, "old-Q4_K_M.gguf");
  assert.equal(cached?.cache_path, "/cache/old");
  assert.equal(cached?.downloaded, true);
  assert.equal(cached?.update_available, true);
  assert.equal(result.context_length, 4096);
  assert.equal(
    result.variants.find((v) => v.quant === "Q8_0")?.downloaded,
    false,
  );
});

test("a remote answer cannot promote a locally incomplete quant", async () => {
  const incomplete = {
    ...local,
    variants: [{ ...local.variants[0], downloaded: false, partial: true }],
    dependencies_resolved: false,
  };
  const result = await loadPickerGgufVariants(
    async (localOnly) => (localOnly ? incomplete : local),
    options,
    () => {},
  );
  assert.equal(result.variants[0].downloaded, false);
  assert.equal(result.variants[0].partial, true);
  assert.equal(result.dependencies_resolved, false);
});

test("known Hub backoff keeps On Device local", async () => {
  const calls: boolean[] = [];
  await loadPickerGgufVariants(
    async (localOnly) => {
      calls.push(localOnly);
      return local;
    },
    { ...options, canDiscoverRemote: () => false },
    () => {},
  );
  assert.deepEqual(calls, [true]);
});

test("collapsing the row after its cached answer prevents remote work", async () => {
  const controller = new AbortController();
  const calls: boolean[] = [];
  await loadPickerGgufVariants(
    async (localOnly) => {
      calls.push(localOnly);
      return local;
    },
    { ...options, signal: controller.signal },
    () => controller.abort(),
  );
  assert.deepEqual(calls, [true]);
});

test("catalog rows retain remote discovery", async () => {
  const calls: boolean[] = [];
  await loadPickerGgufVariants(
    async (localOnly) => {
      calls.push(localOnly);
      return local;
    },
    { ...options, onDevice: false },
    () => assert.fail("catalog rows must not publish a cached phase"),
  );
  assert.deepEqual(calls, [false]);
});

test("a Hub update or missing drafter reaches a cached quant's row", async () => {
  const remote = {
    ...local,
    variants: [
      {
        ...local.variants[0],
        update_available: true,
        pending_drafter_filename: "mtp-drafter.gguf",
        pending_drafter_size_bytes: 64,
      },
    ],
  };
  const result = await loadPickerGgufVariants(
    async (localOnly) => (localOnly ? local : remote),
    options,
    () => {},
  );
  assert.equal(result.variants[0].update_available, true);
  assert.equal(result.variants[0].pending_drafter_filename, "mtp-drafter.gguf");
  assert.equal(result.variants[0].pending_drafter_size_bytes, 64);
  assert.equal(result.variants[0].filename, "old-Q4_K_M.gguf");
});

const pickSole = (res: GgufVariantsResponse) => {
  const sole = res.variants.filter((v) => v.downloaded === true);
  return sole.length === 1 ? { variant: sole[0] } : null;
};

test("a sole cached quant collapses from disk when the Hub is unreachable", async () => {
  const calls: boolean[] = [];
  const stalled = readSoleQuantLocalFirst(
    async (localOnly) => {
      calls.push(localOnly);
      if (localOnly) return local;
      throw new Error("Hugging Face unreachable");
    },
    pickSole,
    () => true,
  );
  assert.equal((await stalled)?.variant.filename, "old-Q4_K_M.gguf");
  const offline = await readSoleQuantLocalFirst(
    async (localOnly) => {
      calls.push(localOnly);
      return local;
    },
    pickSole,
    () => false,
  );
  assert.equal(offline?.variant.filename, "old-Q4_K_M.gguf");
  assert.deepEqual(calls, [true, false, true]);
});

test("a Hub update or missing drafter keeps the sole quant's expander", async () => {
  for (const extra of [
    { update_available: true },
    { pending_drafter_filename: "mtp-drafter.gguf" },
  ]) {
    const remote = {
      ...local,
      variants: [{ ...local.variants[0], ...extra }],
    };
    const sole = await readSoleQuantLocalFirst(
      async (localOnly) => (localOnly ? local : remote),
      pickSole,
      () => true,
    );
    assert.equal(sole, null);
  }
  const current = await readSoleQuantLocalFirst(
    async () => local,
    pickSole,
    () => true,
  );
  assert.equal(current?.variant.quant, "Q4_K_M");
});
