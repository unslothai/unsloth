// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as HfReadme from "../src/features/hub/lib/hf-readme.ts";
import { LruMap } from "../src/features/hub/lib/lru-map.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function readme(status: (url: string) => number) {
  const urls: string[] = [];
  const api = loadWithStubs<typeof HfReadme>(
    new URL("../src/features/hub/lib/hf-readme.ts", import.meta.url),
    {
      "@/features/hub/lib/lru-map": { LruMap },
      "@/features/hub/lib/network": {
        fetchWithTimeout: async (url: string) => {
          urls.push(url);
          return new Response("# card", { status: status(url) });
        },
      },
      "@/features/hub/lib/token-fingerprint": { fingerprintToken: () => "" },
      "@/lib/hf-endpoint": { getHfEndpoint: () => "https://huggingface.co" },
      streamdown: { defaultUrlTransform: (url: string) => url },
    },
  );
  return { api, urls };
}

// FastFlowLM pins some NPU repos to a branch whose card differs from main's.
test("a README pinned to a revision is read from that revision only", async () => {
  const r = readme(() => 200);
  const fetched = await r.api.fetchReadme(
    "FastFlowLM/Qwen3.5-9B-NPU2",
    "model",
    null,
    "flm_q4k_high_precision",
  );
  assert.deepEqual(fetched, {
    markdown: "# card",
    branch: "flm_q4k_high_precision",
  });
  assert.deepEqual(r.urls, [
    "https://huggingface.co/FastFlowLM/Qwen3.5-9B-NPU2/resolve/flm_q4k_high_precision/README.md",
  ]);
  assert.equal(
    r.api.readmeBaseUrl("FastFlowLM/Qwen3.5-9B-NPU2", "model", fetched!.branch),
    "https://huggingface.co/FastFlowLM/Qwen3.5-9B-NPU2/resolve/flm_q4k_high_precision/",
  );
});

test("without a revision the README falls back from main to master", async () => {
  const r = readme((url) => (url.includes("/main/") ? 404 : 200));
  const fetched = await r.api.fetchReadme("FastFlowLM/Qwen3-0.6B-NPU2");
  assert.equal(fetched?.branch, "master");
  assert.equal(r.urls.length, 2);

  const pinned = readme(() => 404);
  assert.equal(
    await pinned.api.fetchReadme("FastFlowLM/Gone-NPU2", "model", null, "v2"),
    null,
  );
  assert.equal(pinned.urls.length, 1);
});
