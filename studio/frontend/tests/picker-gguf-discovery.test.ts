// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { GgufVariantsResponse } from "../src/features/chat/types/api.ts";
import {
  loadPickerGgufVariants,
  createTaskLimiter,
  hubWithdrawsSoleQuant,
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

test("an update downloads the published revision's size and the Hub default wins", async () => {
  const remote: GgufVariantsResponse = {
    ...local,
    default_variant: "Q8_0",
    variants: [
      {
        ...local.variants[0],
        size_bytes: 300,
        download_size_bytes: 300,
        update_available: true,
      },
      { quant: "Q8_0", filename: "model-Q8_0.gguf", size_bytes: 512 },
    ],
  };
  const result = await loadPickerGgufVariants(
    async (localOnly) =>
      localOnly
        ? {
            ...local,
            variants: [{ ...local.variants[0], download_size_bytes: 0 }],
          }
        : remote,
    options,
    () => {},
  );
  const cached = result.variants.find((v) => v.quant === "Q4_K_M");
  assert.equal(cached?.download_size_bytes, 300);
  assert.equal(cached?.size_bytes, 300);
  assert.equal(cached?.cache_path, "/cache/old");
  assert.equal(result.default_variant, "Q8_0");
  const current = await loadPickerGgufVariants(
    async (localOnly) =>
      localOnly ? local : { ...remote, variants: [local.variants[0]] },
    options,
    () => {},
  );
  assert.equal(current.variants[0].size_bytes, 256);
});

test("a Hub update or missing drafter withdraws a collapsed sole quant", async () => {
  for (const extra of [
    { update_available: true },
    { pending_drafter_filename: "mtp-drafter.gguf" },
  ]) {
    const remote = { ...local, variants: [{ ...local.variants[0], ...extra }] };
    assert.equal(
      await hubWithdrawsSoleQuant(async () => remote, "q4_k_m"),
      true,
    );
  }
  assert.equal(await hubWithdrawsSoleQuant(async () => local, "Q4_K_M"), false);
  assert.equal(
    await hubWithdrawsSoleQuant(async () => {
      throw new Error("Hugging Face unreachable");
    }, "Q4_K_M"),
    false,
  );
});

test("follow-up Hub probes never exceed the limiter's concurrency", async () => {
  const run = createTaskLimiter(2);
  let active = 0;
  let peak = 0;
  const done = await Promise.all(
    Array.from({ length: 7 }, (_, i) =>
      run(async () => {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 5));
        active -= 1;
        if (i === 3) throw new Error("probe failed");
        return i;
      }).catch(() => -1),
    ),
  );
  assert.equal(peak, 2);
  assert.deepEqual(done, [0, 1, 2, -1, 4, 5, 6]);
});

test("a failed disk phase falls back to the Hub only when it is reachable", async () => {
  const calls: boolean[] = [];
  const list = async (localOnly: boolean) => {
    calls.push(localOnly);
    if (localOnly) throw new Error("Model not found");
    return local;
  };
  const shown: GgufVariantsResponse[] = [];
  assert.equal(
    await loadPickerGgufVariants(list, options, (r) => shown.push(r)),
    local,
  );
  assert.deepEqual(shown, []);
  await assert.rejects(
    loadPickerGgufVariants(
      list,
      { ...options, canDiscoverRemote: () => false },
      () => {},
    ),
    /Model not found/,
  );
  assert.deepEqual(calls, [true, false, true]);
});

test("an up-to-date cached quant keeps the Hub's companion-inclusive size", async () => {
  const result = await loadPickerGgufVariants(
    async (localOnly) =>
      localOnly
        ? {
            ...local,
            variants: [{ ...local.variants[0], download_size_bytes: 256 }],
          }
        : {
            ...local,
            variants: [{ ...local.variants[0], download_size_bytes: 900 }],
          },
    options,
    () => {},
  );
  assert.equal(result.variants[0].download_size_bytes, 900);
  assert.equal(result.variants[0].cache_path, "/cache/old");
});
