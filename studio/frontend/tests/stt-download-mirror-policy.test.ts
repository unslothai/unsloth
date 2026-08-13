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

test("an adopted STT transfer keeps progress and merges its Voice owner", () => {
  assert.match(
    source,
    /const resolvedEngine = options\.engine \?\? sttEngineFor\(model\);[\s\S]*const key = trackerKey\(model, resolvedEngine\);[\s\S]*if \(trackers\.has\(key\)\) \{[\s\S]*warmSelectedVoiceModelOnComplete\.set\(key, true\);[\s\S]*return;[\s\S]*\}/,
  );
});

test("tracking-only STT jobs never invoke the Voice-owned completion load", () => {
  assert.match(
    source,
    /const key = trackerKey\(model, engine\);[\s\S]*const shouldWarmVoiceModel =[\s\S]*warmSelectedVoiceModelOnComplete\.get\(key\) \?\? true;[\s\S]*warmSelectedVoiceModelOnComplete\.delete\(key\);[\s\S]*shouldWarmVoiceModel &&[\s\S]*outcome === "complete"/,
  );
});

test("non-default STT engines have independent jobs", () => {
  assert.match(
    source,
    /engine && engine !== "transformers" \? `\$\{engine\}:\$\{model\}` : model/,
  );
  assert.match(source, /cancelSttDownload\(model, resolvedEngine\)/);
  assert.match(
    source,
    /engine === undefined \|\| engine === "transformers" \? model : undefined/,
  );
  assert.match(source, /sttEngineStatusFor\(status, model, engine\)/);
});

test("an explicit Transformers transfer keeps its serving engine", () => {
  assert.match(
    source,
    /const resolvedEngine = options\.engine \?\? sttEngineFor\(model\)/,
  );
  assert.match(source, /cancelSttDownload\(model, resolvedEngine\)/);
  assert.match(source, /poll\(model, startedAt, resolvedEngine\)/);
  assert.doesNotMatch(
    source,
    /options\.engine === "transformers" \? undefined/,
  );
});
