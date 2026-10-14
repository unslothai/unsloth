// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The combined update unloads audio.cpp models too, and the Audio page keeps its own status. With its
// own update notice gone, the update card's resync is the only signal it gets: without it Generate
// stays enabled against an unloaded slot and the next run 400s with "No model loaded".

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

test("the post-update resync tells the Audio page to re-read both slots", () => {
  const runtime = readText(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  const start = runtime.indexOf(
    "export async function resyncInferenceStatusAfterServerModelChange",
  );
  assert.ok(start >= 0);
  const body = runtime.slice(start, runtime.indexOf("\n}\n", start));
  assert.match(
    body,
    /notifyModelLifecycle\(\{ runtime: "chat", loading: false/,
  );
  assert.match(body, /notifyModelLifecycle\(\{ runtime: "stt", loading: false/);

  const page = readText("../src/features/audio/audio-page.tsx");
  assert.match(
    page,
    /subscribeModelLifecycle\(\(\{ runtime, loading \}\) => \{\s*if \(loading\) return;\s*if \(runtime === "chat"\) void refreshStatus\(\);\s*if \(runtime === "stt"\) void refreshSttStatus\(\);/,
  );
});

test("the update card resyncs through that function", () => {
  const banner = readText("../src/components/llama-update-banner.tsx");
  assert.match(
    banner,
    /onReloadRequired: resyncInferenceStatusAfterServerModelChange/,
  );
});

test("a settled update job re-reads dictation even without a reload", () => {
  const hook = readText("../src/hooks/use-llama-update-check.ts");
  assert.match(
    hook,
    /if \(presentation\.running\) return;\s*clearPollTimer\(\);\s*(\/\/[^\n]*\n\s*)?notifyModelLifecycle\(\{ runtime: "stt", loading: false/,
  );
});
