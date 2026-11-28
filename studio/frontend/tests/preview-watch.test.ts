// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Progress polls ask for the live preview only from a visible page, so a hidden tab costs the render nothing.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Poll = (videoId?: string) => Promise<unknown>;

function polls(calls: string[]): { image: Poll; video: Poll } {
  const common = {
    "@/lib/model-lifecycle-events": { withBackgroundLoadNotice: (f: unknown) => f },
    "@/features/auth": {
      authFetch: async (path: string) => {
        calls.push(path);
        return new Response(JSON.stringify({ active: true }), { status: 200 });
      },
    },
    "@/lib/api-base": { apiUrl: (path: string) => path },
    "@/lib/format-fastapi-error": { readFastApiError: async () => "" },
    "@/lib/multi-model-refusal": { parseLoadJson: async () => ({}) },
    "@/lib/unified-memory-load-refusal": { withAllowOversizedLoad: (body: unknown) => body },
    "@/features/settings/api/vram-budget": {
      dropVramBudgetRetry: () => {},
      loadVramBudgetSettings: async () => null,
      settleVramBudgetSave: () => null,
      setVramBudgetLocked: () => {},
    },
  };
  const images = loadWithStubs<{ getGenerateProgress: Poll }>(
    new URL("../src/features/images/api.ts", import.meta.url),
    common,
    { relativePassthrough: true },
  );
  const video = loadWithStubs<{ getVideoGenerateProgress: Poll }>(
    new URL("../src/features/video/api.ts", import.meta.url),
    common,
    { relativePassthrough: true },
  );
  return { image: images.getGenerateProgress, video: video.getVideoGenerateProgress };
}

async function withVisibility(state: string | undefined, run: () => Promise<void>): Promise<void> {
  const g = globalThis as { document?: unknown };
  const had = "document" in g;
  const prev = g.document;
  if (state === undefined) delete g.document;
  else g.document = { visibilityState: state };
  try {
    await run();
  } finally {
    if (had) g.document = prev;
    else delete g.document;
  }
}

test("a visible page asks for the preview; a hidden tab or no document does not", async () => {
  const calls: string[] = [];
  const { image, video } = polls(calls);
  await withVisibility("visible", async () => {
    await image();
    await video();
    await video("a b");
  });
  await withVisibility("hidden", async () => {
    await image();
    await video("a b");
  });
  await withVisibility(undefined, async () => {
    await image();
    await video();
  });
  assert.deepEqual(calls, [
    "/api/inference/images/generate-progress?preview=1",
    "/api/inference/video/generate-progress?preview=1",
    "/api/inference/video/generate-progress?video_id=a%20b&preview=1",
    "/api/inference/images/generate-progress",
    "/api/inference/video/generate-progress?video_id=a%20b",
    "/api/inference/images/generate-progress",
    "/api/inference/video/generate-progress",
  ]);
});

test("the Images page's Live preview switch turned off stops asking, even while visible", async () => {
  const calls: string[] = [];
  const { image } = polls(calls);
  const g = globalThis as { localStorage?: unknown };
  const had = "localStorage" in g;
  const prev = g.localStorage;
  const store: Record<string, string> = { unsloth_images_live_preview_off: "true" };
  g.localStorage = { getItem: (k: string) => store[k] ?? null };
  try {
    await withVisibility("visible", async () => {
      await image();
      delete store.unsloth_images_live_preview_off;
      await image();
    });
  } finally {
    if (had) g.localStorage = prev;
    else delete g.localStorage;
  }
  assert.deepEqual(calls, [
    "/api/inference/images/generate-progress",
    "/api/inference/images/generate-progress?preview=1",
  ]);
});
