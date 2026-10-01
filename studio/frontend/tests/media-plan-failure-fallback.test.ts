// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A download plan needs the Hub, so offline (or on a Hub hiccup) it errors or comes back
 * plan_failed. The backend's contract is that the load then pulls what it needs inline, so a
 * load pick must fall through to it: refusing there stops an already-downloaded model loading.
 * Download-only picks have no load to fall back to and still report the failure.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("images: only a download-only pick refuses a plan_failed plan", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /if \(downloadOnly && plan\.plan_failed\) \{/);
  assert.doesNotMatch(source, /toast\.error\("Could not check required files"/);
});

test("video: a failed plan falls back to the load", () => {
  const source = readSrc("features/video/video-page.tsx");
  assert.doesNotMatch(source, /if \(plan\.plan_failed\) throw/);
  assert.doesNotMatch(source, /toast\.error\("Could not check required files"/);
});

test("audio: a downloaded TTS model still loads when its plan errors", () => {
  const source = readSrc("features/audio/audio-page.tsx");
  const start = source.indexOf("plan = await getAudioDownloadPlan(");
  const fallback = source.slice(start, source.indexOf("const plannedEntries = plan.entries;", start));
  assert.match(fallback, /if \(meta\.isDownloaded === true\) \{\s*void loadTtsModelRef\.current\(/);
});
