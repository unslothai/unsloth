// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

type Pending = {
  url: string;
  body: Record<string, unknown>;
  settle: (status: number, payload: unknown) => void;
};
const requests: Pending[] = [];

globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) =>
  new Promise<Response>((resolve) => {
    requests.push({
      url: String(input),
      body: JSON.parse(String(init?.body ?? "{}")),
      settle: (status, payload) =>
        resolve(
          new Response(JSON.stringify(payload), {
            status,
            headers: { "Content-Type": "application/json" },
          }),
        ),
    });
  })) as typeof fetch;

const { inFlightMediaDownloadPlan } = await import(
  "../src/features/hub/lib/media-download-plan.ts"
);
const { resolveCompanionBytes } = await import(
  "../src/features/hub/hooks/use-media-companion-bytes.ts"
);

const REPO = "org/model-GGUF";
const body = (filename = "m-Q8_0.gguf") => ({
  // biome-ignore lint/style/useNamingConvention: API schema
  model_path: REPO,
  // biome-ignore lint/style/useNamingConvention: API schema
  gguf_filename: filename,
  // biome-ignore lint/style/useNamingConvention: API schema
  model_kind: "gguf" as const,
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_token: "hf_x",
});
const plan = (extra: Record<string, unknown> = {}) => ({
  entries: [
    {
      // biome-ignore lint/style/useNamingConvention: API schema
      repo_id: "org/base",
      files: ["enc.safetensors"],
      bytes: 9,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_filename: null,
      checkpoint: false,
    },
  ],
  // biome-ignore lint/style/useNamingConvention: API schema
  total_bytes: 9,
  // biome-ignore lint/style/useNamingConvention: API schema
  checkpoint_bytes: 4,
  ...extra,
});
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test("identical concurrent plans share one request and one answer", async () => {
  requests.length = 0;
  const first = inFlightMediaDownloadPlan("images", body(), 0);
  const second = inFlightMediaDownloadPlan("images", body(), 0);
  await tick();
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /\/api\/inference\/images\/download-plan$/);
  assert.deepEqual(requests[0].body, body());
  requests[0].settle(200, plan());
  const [a, b] = await Promise.all([first, second]);
  assert.deepEqual(a, plan());
  assert.equal(a, b);
});

test("nothing is kept once a plan settles", async () => {
  requests.length = 0;
  const first = inFlightMediaDownloadPlan("images", body(), 0);
  await tick();
  requests[0].settle(200, plan());
  await first;
  const again = inFlightMediaDownloadPlan("images", body(), 0);
  await tick();
  assert.equal(requests.length, 2);
  requests[1].settle(200, plan());
  await again;
});

test("a different file, page or inventory version asks its own plan", async () => {
  requests.length = 0;
  const asks = [
    inFlightMediaDownloadPlan("images", body(), 0),
    inFlightMediaDownloadPlan("images", body("m-BF16.gguf"), 0),
    inFlightMediaDownloadPlan("video", body(), 0),
    inFlightMediaDownloadPlan("images", body(), 1),
    inFlightMediaDownloadPlan("images", { ...body(), hf_token: undefined }, 0),
  ];
  await tick();
  assert.equal(requests.length, 5);
  assert.match(requests[2].url, /\/api\/inference\/video\/download-plan$/);
  for (const request of requests) request.settle(200, plan());
  await Promise.all(asks);
});

test("a refusal reaches every caller and the next ask requests again", async () => {
  requests.length = 0;
  const first = inFlightMediaDownloadPlan("video", body(), 0);
  const second = inFlightMediaDownloadPlan("video", body(), 0);
  await tick();
  assert.equal(requests.length, 1);
  requests[0].settle(400, { detail: "Not a supported video model" });
  await assert.rejects(first, /Not a supported video model/);
  await assert.rejects(second, /Not a supported video model/);
  const retry = inFlightMediaDownloadPlan("video", body(), 0);
  await tick();
  assert.equal(requests.length, 2);
  requests[1].settle(200, plan());
  assert.deepEqual(await retry, plan());
});

test("a plan still pending past the share window is not joined", async () => {
  requests.length = 0;
  const realNow = Date.now;
  const start = realNow();
  try {
    Date.now = () => start;
    const stalled = inFlightMediaDownloadPlan("images", body(), 0);
    Date.now = () => start + 4_999;
    const joined = inFlightMediaDownloadPlan("images", body(), 0);
    await tick();
    assert.equal(requests.length, 1);
    assert.equal(joined, stalled);
    Date.now = () => start + 5_000;
    const fresh = inFlightMediaDownloadPlan("images", body(), 0);
    const freshJoined = inFlightMediaDownloadPlan("images", body(), 0);
    await tick();
    assert.equal(requests.length, 2);
    assert.notEqual(fresh, stalled);
    assert.equal(freshJoined, fresh);
    // The stalled request settling late must not drop the newer entry.
    requests[0].settle(200, plan());
    await stalled;
    assert.equal(inFlightMediaDownloadPlan("images", body(), 0), fresh);
    await tick();
    assert.equal(requests.length, 2);
    requests[1].settle(200, plan());
    await fresh;
  } finally {
    Date.now = realNow;
  }
});

test("each caller keeps its own handling of a shared answer", async () => {
  requests.length = 0;
  // biome-ignore lint/style/useNamingConvention: API schema
  const failed = plan({ plan_failed: true });
  const companion = resolveCompanionBytes(
    "images",
    REPO,
    "m-Q8_0.gguf",
    4,
    "hf_x",
  );
  const direct = inFlightMediaDownloadPlan("images", body(), 0);
  await tick();
  assert.equal(requests.length, 1);
  requests[0].settle(200, failed);
  await assert.rejects(companion, /Download plan incomplete/);
  assert.deepEqual(await direct, failed);

  const bytes = resolveCompanionBytes("images", REPO, "m-Q8_0.gguf", 4, "hf_x");
  await tick();
  assert.equal(requests.length, 2);
  requests[1].settle(200, plan());
  assert.equal(await bytes, 9);
});

test("both Hub plan callers go through the shared request without new resets", () => {
  const companion = readSrc("features/hub/hooks/use-media-companion-bytes.ts");
  assert.match(
    companion,
    /await inFlightMediaDownloadPlan\(page, request, inventoryVersion\)/,
  );
  assert.match(companion, /hfToken,\s*inventoryVersion,\s*\);/);
  const assets = readSrc(
    "features/hub/catalog/use-required-assets-download.tsx",
  );
  assert.match(assets, /await inFlightMediaDownloadPlan\(/);
  assert.match(assets, /promise: fetchPlan\(inventoryVersion\)/);
  // An inventory bump must not re-key fetchPlan: that would reset an open dialog.
  assert.match(assets, /\}, \[repoId, filename, token, runtime\]\);/);
  assert.doesNotMatch(assets, /getDiffusionDownloadPlan|getVideoDownloadPlan/);
});
