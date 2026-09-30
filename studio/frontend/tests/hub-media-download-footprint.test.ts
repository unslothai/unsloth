// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub sizes must include the companion assets fetched on Run.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

const GGUF_BYTES = 5_390_223_072;
const COMPANION_BYTES = 18_900_000_000;

let planBody: Record<string, unknown> = {};
let planStatus = 200;
const requests: { url: string; body: Record<string, unknown> }[] = [];

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  requests.push({
    url: String(input),
    body: JSON.parse(String(init?.body ?? "{}")),
  });
  return new Response(JSON.stringify(planBody), {
    status: planStatus,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const {
  cachedCompanionBytes,
  companionPlanRequests,
  footprintRequestsKey,
  ggufVariantFootprint,
  resolveCompanionBytes,
  visibleFootprints,
  withResolvedFootprint,
} = await import("../src/features/hub/hooks/use-media-download-footprints.ts");

function variant(
  quant: string,
  overrides: Record<string, unknown> = {},
): Parameters<typeof ggufVariantFootprint>[0] {
  return {
    filename: `qwen-image-2.1-${quant}.gguf`,
    quant,
    // biome-ignore lint/style/useNamingConvention: API schema
    size_bytes: GGUF_BYTES,
    // biome-ignore lint/style/useNamingConvention: API schema
    download_size_bytes: GGUF_BYTES,
    // biome-ignore lint/style/useNamingConvention: API schema
    dependency_key: "qwen-image-2.1",
    ...overrides,
  };
}

test("the plan's companion bytes are what it requires beyond the checkpoint", async () => {
  requests.length = 0;
  planBody = {
    entries: [],
    // biome-ignore lint/style/useNamingConvention: API schema
    total_bytes: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    required_bytes: GGUF_BYTES + COMPANION_BYTES,
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: GGUF_BYTES,
  };
  const companion = await resolveCompanionBytes(
    "images",
    "unsloth/Qwen-Image-2.1-GGUF",
    "qwen-image-2.1-Q5_K_M.gguf",
    GGUF_BYTES,
    "hf_token_value",
  );
  assert.equal(companion, COMPANION_BYTES);
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /\/api\/inference\/images\/download-plan$/);
  assert.deepEqual(requests[0].body, {
    // biome-ignore lint/style/useNamingConvention: API schema
    model_path: "unsloth/Qwen-Image-2.1-GGUF",
    // biome-ignore lint/style/useNamingConvention: API schema
    gguf_filename: "qwen-image-2.1-Q5_K_M.gguf",
    // biome-ignore lint/style/useNamingConvention: API schema
    model_kind: "gguf",
    // biome-ignore lint/style/useNamingConvention: API schema
    hf_token: "hf_token_value",
  });
});

test("a video repo asks the video planner", async () => {
  requests.length = 0;
  await resolveCompanionBytes("video", "org/video-GGUF", "v.gguf", 1, null);
  assert.match(requests[0].url, /\/api\/inference\/video\/download-plan$/);
});

test("the listed size stands in for a checkpoint the planner could not size", async () => {
  // biome-ignore lint/style/useNamingConvention: API schema
  planBody = { entries: [], total_bytes: 0, required_bytes: GGUF_BYTES + 10 };
  assert.equal(
    await resolveCompanionBytes("images", "r", "f.gguf", GGUF_BYTES, null),
    10,
  );
});

test("a plan with nothing beyond the checkpoint adds nothing", async () => {
  planBody = {
    entries: [],
    // biome-ignore lint/style/useNamingConvention: API schema
    total_bytes: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    required_bytes: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: 0,
  };
  assert.equal(
    await resolveCompanionBytes("images", "r", "f.gguf", GGUF_BYTES, null),
    null,
  );
});

test("a remount reuses the plan, and a failed plan is asked again", async () => {
  planBody = {
    entries: [],
    // biome-ignore lint/style/useNamingConvention: API schema
    total_bytes: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    required_bytes: GGUF_BYTES + COMPANION_BYTES,
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: GGUF_BYTES,
  };
  requests.length = 0;
  const ask = () =>
    cachedCompanionBytes("images", "cached/repo", "a.gguf", GGUF_BYTES, null);
  const [first, second] = await Promise.all([ask(), ask()]);
  assert.equal(first, COMPANION_BYTES);
  assert.equal(second, COMPANION_BYTES);
  assert.equal(await ask(), COMPANION_BYTES);
  assert.equal(requests.length, 1);
  // Another token can reach another base, so it gets its own plan.
  await cachedCompanionBytes(
    "images",
    "cached/repo",
    "a.gguf",
    GGUF_BYTES,
    "hf_x",
  );
  assert.equal(requests.length, 2);

  requests.length = 0;
  planStatus = 500;
  const failing = () =>
    cachedCompanionBytes("images", "failing/repo", "a.gguf", GGUF_BYTES, null);
  await assert.rejects(failing());
  planStatus = 200;
  assert.equal(await failing(), COMPANION_BYTES);
  assert.equal(requests.length, 2);
});

test("a plan the backend flags as incomplete is not shown or kept", async () => {
  planBody = {
    // biome-ignore lint/style/useNamingConvention: API schema
    plan_failed: true,
    entries: [],
    // biome-ignore lint/style/useNamingConvention: API schema
    total_bytes: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    required_bytes: GGUF_BYTES + 1,
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: GGUF_BYTES,
  };
  requests.length = 0;
  const ask = () =>
    cachedCompanionBytes("images", "flagged/repo", "a.gguf", GGUF_BYTES, null);
  await assert.rejects(ask());
  await assert.rejects(ask());
  // Incomplete plans are retried.
  assert.equal(requests.length, 2);
});

test("a new repo, page or token never shows the previous total", () => {
  const variants = [variant("Q5_K_M")];
  const keyA = footprintRequestsKey("images", "org/repo", variants, null);
  for (const other of [
    footprintRequestsKey("images", "org/repo", variants, "hf_other"),
    footprintRequestsKey("images", "org/mirror", variants, null),
    footprintRequestsKey("video", "org/repo", variants, null),
  ]) {
    assert.notEqual(other, keyA);
  }
  assert.equal(footprintRequestsKey(undefined, "org/repo", variants, null), "");

  const resolvedA = withResolvedFootprint(
    { requestsKey: "", companionBytes: new Map() },
    keyA,
    "qwen-image-2.1",
    COMPANION_BYTES,
  );
  assert.equal(
    visibleFootprints(resolvedA, keyA).get("qwen-image-2.1"),
    COMPANION_BYTES,
  );
  // Hide the old total even if the new token's plan fails or returns no companions.
  const keyB = footprintRequestsKey("images", "org/repo", variants, "hf_other");
  assert.equal(visibleFootprints(resolvedA, keyB).size, 0);
  // Switching back can reuse the retained result.
  assert.equal(visibleFootprints(resolvedA, keyA).size, 1);
});

test("one plan per companion set, and none outside the media pages", () => {
  const variants = [
    variant("Q5_K_M"),
    variant("Q4_K_M"),
    // biome-ignore lint/style/useNamingConvention: API schema
    variant("Q8_0", { dependency_key: "other-base" }),
    // biome-ignore lint/style/useNamingConvention: API schema
    variant("F16", { dependency_key: null }),
  ];
  // Unkeyed variants share one repo-wide group.
  assert.deepEqual(
    companionPlanRequests("images", variants).map(([key]) => key),
    ["qwen-image-2.1", "other-base", ""],
  );
  assert.deepEqual(companionPlanRequests(undefined, variants), []);
  assert.deepEqual(companionPlanRequests("images", null), []);
});

test("a row adds its set's companions to its own download size", () => {
  const resolved = new Map([["qwen-image-2.1", COMPANION_BYTES]]);
  assert.deepEqual(ggufVariantFootprint(variant("Q5_K_M"), resolved), {
    checkpointBytes: GGUF_BYTES,
    companionBytes: COMPANION_BYTES,
  });
  // An unkeyed row reads the repo-wide group.
  assert.deepEqual(
    ggufVariantFootprint(
      // biome-ignore lint/style/useNamingConvention: API schema
      variant("F16", { dependency_key: null }),
      new Map([["", 7]]),
    ),
    { checkpointBytes: GGUF_BYTES, companionBytes: 7 },
  );
  // Unresolved or partial rows keep their plain size.
  assert.equal(ggufVariantFootprint(variant("Q5_K_M"), new Map()), null);
  assert.equal(
    ggufVariantFootprint(variant("Q4_K_M", { partial: true }), resolved),
    null,
  );
});

test("the Hub card sizes both the menu rows and the selected quant this way", () => {
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.ok(
    card.includes(
      "footprint: ggufVariantFootprint(variant, companionBytesByKey),",
    ),
  );
  assert.ok(
    card.includes("ggufVariantFootprint(selected, companionBytesByKey)"),
  );
  assert.equal(card.split("<GgufVariantSizeLabel").length - 1, 2);
});
