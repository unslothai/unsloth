// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Offline the plan fails; a load must fall back to its inline download or a cached model cannot load.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

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
  const source = readAudioWorkspaceSource();
  const start = source.indexOf("plan = await getAudioDownloadPlan(");
  const fallback = source.slice(start, source.indexOf("const plannedEntries = plan.entries;", start));
  assert.match(fallback, /if \(meta\.isDownloaded === true\) \{\s*void loadTtsModelRef\.current\(/);
});
