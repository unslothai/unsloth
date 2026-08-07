// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/settings/lib/stt-download-mirror.ts",
    import.meta.url,
  ),
  "utf8",
);

test("an adopted STT transfer keeps its existing progress and completion owner", () => {
  assert.match(
    source,
    /const key = trackerKey\(model, explicitEngine\);[\s\S]*if \(trackers\.has\(key\)\) return;[\s\S]*warmSelectedVoiceModelOnComplete\.set/,
  );
});

test("tracking-only STT jobs never invoke the Voice-owned completion load", () => {
  assert.match(
    source,
    /const key = trackerKey\(model, engine\);[\s\S]*const shouldWarmVoiceModel =[\s\S]*warmSelectedVoiceModelOnComplete\.get\(key\) \?\? true;[\s\S]*warmSelectedVoiceModelOnComplete\.delete\(key\);[\s\S]*shouldWarmVoiceModel &&[\s\S]*outcome === "complete"/,
  );
});

test("same-key downloads on different STT engines have independent jobs", () => {
  assert.match(source, /return engine \? `\$\{engine\}:\$\{model\}` : model/);
  assert.match(source, /cancelSttDownload\(model, resolvedEngine\)/);
  assert.match(
    source,
    /engine === undefined \|\| engine === "transformers" \? model : undefined/,
  );
  assert.match(source, /sttEngineStatusFor\(status, model, engine\)/);
});
