// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub sizes must include the companion assets Run would still download.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

const GGUF_BYTES = 5_390_223_072;
const COMPANION_BYTES = 18_900_000_000;
const GGUF_REPO = "unsloth/Qwen-Image-2.1-GGUF";

let planBody: Record<string, unknown> = {};
const requests: { url: string; body: Record<string, unknown> }[] = [];

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  requests.push({
    url: String(input),
    body: JSON.parse(String(init?.body ?? "{}")),
  });
  return new Response(JSON.stringify(planBody), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { companionPlanRequests, ggufVariantFootprint, resolveCompanionBytes } =
  await import("../src/features/hub/hooks/use-media-companion-bytes.ts");

function entry(repo: string, bytes: number, checkpoint = false) {
  // biome-ignore lint/style/useNamingConvention: API schema
  return { repo_id: repo, files: [], bytes, gguf_filename: null, checkpoint };
}

function plan(
  entries: ({ bytes: number } & Record<string, unknown>)[],
  extra = {},
) {
  return {
    entries,
    // biome-ignore lint/style/useNamingConvention: API schema
    total_bytes: entries.reduce((sum, e) => sum + e.bytes, 0),
    // biome-ignore lint/style/useNamingConvention: API schema
    required_bytes: GGUF_BYTES + COMPANION_BYTES,
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: GGUF_BYTES,
    ...extra,
  };
}

const resolve = () =>
  resolveCompanionBytes("images", GGUF_REPO, "q5.gguf", GGUF_BYTES, "hf_x");

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
    dependency_key: "qwen-image-2.1",
    ...overrides,
  };
}

test("companions are the uncached plan entries beyond the checkpoint", async () => {
  requests.length = 0;
  planBody = plan([
    entry(GGUF_REPO, GGUF_BYTES, true),
    entry("Qwen/Qwen-Image-2.1", COMPANION_BYTES),
  ]);
  assert.equal(await resolve(), COMPANION_BYTES);
  assert.match(requests[0].url, /\/api\/inference\/images\/download-plan$/);
  assert.deepEqual(requests[0].body, {
    // biome-ignore lint/style/useNamingConvention: API schema
    model_path: GGUF_REPO,
    // biome-ignore lint/style/useNamingConvention: API schema
    gguf_filename: "q5.gguf",
    // biome-ignore lint/style/useNamingConvention: API schema
    model_kind: "gguf",
    // biome-ignore lint/style/useNamingConvention: API schema
    hf_token: "hf_x",
  });

  // A checkpoint entry can also carry companions from the same repo.
  planBody = plan([entry(GGUF_REPO, GGUF_BYTES + 7, true)]);
  assert.equal(await resolve(), 7);
  // A cached checkpoint leaves its repo's missing companions unflagged.
  planBody = plan([entry(GGUF_REPO, 7)]);
  assert.equal(await resolve(), 7);
  // Without the flag (older backend) the bytes cannot be attributed: a scoped
  // file list names the GGUF whether or not it is cached.
  planBody = plan([
    {
      ...entry(GGUF_REPO, GGUF_BYTES + 7),
      files: ["q5.gguf"],
      checkpoint: undefined,
    },
  ]);
  assert.equal(await resolve(), null);
  // The listed size stands in for a checkpoint the planner could not size.
  planBody = plan([entry(GGUF_REPO, GGUF_BYTES + 7, true)], {
    // biome-ignore lint/style/useNamingConvention: API schema
    checkpoint_bytes: 0,
  });
  assert.equal(await resolve(), 7);
});

test("cached companions add nothing, whatever required_bytes says", async () => {
  planBody = plan([entry(GGUF_REPO, GGUF_BYTES, true)]);
  assert.equal(await resolve(), null);
});

test("an incomplete plan is rejected so it is not kept", async () => {
  planBody = plan([entry("Qwen/Qwen-Image-2.1", COMPANION_BYTES)], {
    // biome-ignore lint/style/useNamingConvention: API schema
    plan_failed: true,
  });
  await assert.rejects(resolve());
});

test("a video repo asks the video planner", async () => {
  requests.length = 0;
  planBody = plan([]);
  await resolveCompanionBytes("video", "org/video-GGUF", "v.gguf", 1, null);
  assert.match(requests[0].url, /\/api\/inference\/video\/download-plan$/);
});

test("one plan per companion set, and none outside the media pages", () => {
  const unkeyed = (quant: string) =>
    // biome-ignore lint/style/useNamingConvention: API schema
    variant(quant, { dependency_key: null });
  const variants = [
    variant("Q5_K_M"),
    variant("Q4_K_M"),
    // biome-ignore lint/style/useNamingConvention: API schema
    variant("Q8_0", { dependency_key: "other-base" }),
    unkeyed("F16"),
    unkeyed("BF16"),
  ];
  // Unkeyed files (video) can need different companions, so only the selected one is planned.
  assert.deepEqual(
    companionPlanRequests("images", variants, unkeyed("BF16").filename).map(
      ([key, filename]) => [key, filename],
    ),
    [
      ["qwen-image-2.1", variant("Q5_K_M").filename],
      ["other-base", variant("Q8_0").filename],
      [`file:${unkeyed("BF16").filename}`, unkeyed("BF16").filename],
    ],
  );
  assert.deepEqual(
    companionPlanRequests("images", variants, variant("Q4_K_M").filename)
      .length,
    2,
  );
  assert.deepEqual(companionPlanRequests(undefined, variants, null), []);
  const resolved = new Map([[`file:${unkeyed("BF16").filename}`, 7]]);
  assert.equal(
    ggufVariantFootprint(unkeyed("BF16"), resolved)?.companionBytes,
    7,
  );
  assert.equal(ggufVariantFootprint(unkeyed("F16"), resolved), null);
});

test("rows add their own set's companions, except partials", () => {
  const resolved = new Map([["qwen-image-2.1", COMPANION_BYTES]]);
  // Run, offered once the GGUF is on disk, is what fetches companions.
  for (const state of [{}, { downloaded: true }]) {
    assert.deepEqual(ggufVariantFootprint(variant("Q5_K_M", state), resolved), {
      checkpointBytes: GGUF_BYTES,
      companionBytes: COMPANION_BYTES,
    });
  }
  assert.equal(
    ggufVariantFootprint(
      // biome-ignore lint/style/useNamingConvention: API schema
      variant("Q8_0", { dependency_key: "other" }),
      resolved,
    ),
    null,
  );
  assert.equal(
    ggufVariantFootprint(variant("Q4_K_M", { partial: true }), resolved),
    null,
  );
});
