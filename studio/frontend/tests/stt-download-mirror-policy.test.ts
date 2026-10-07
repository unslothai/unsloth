// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/settings/lib/stt-download-mirror.ts");

test("an adopted STT transfer keeps progress and merges its Voice owner", () => {
  assert.match(
    source,
    /trackedDownloadIds\.set\(key, options\.downloadId\);[\s\S]*if \(wasTracking && !changedAttempt\) \{[\s\S]*warmSelectedVoiceModelOnComplete\.set\(key, true\);[\s\S]*return;/,
  );
  assert.match(source, /trackedVariants\.set\(key, options\.ggufVariant\);/);
  assert.match(source, /startExternalJob\([\s\S]*if \(wasTracking\) return;/);
  assert.match(
    source,
    /const requestedDownloadId = trackedDownloadIds\.get\(key\);[\s\S]*if \(trackedDownloadIds\.get\(key\) !== requestedDownloadId\) return;/,
  );
  assert.match(
    source,
    /requestedDownloadId !== download\.download_id[\s\S]*elapsed > START_GRACE_MS[\s\S]*settle\(/,
  );
  assert.match(
    source,
    /completed_download_ids\?\.includes\(requestedDownloadId\)[\s\S]*settle\(model, "complete"/,
  );
  assert.match(
    source,
    /completed_download_ids\?\.includes\(candidate\)[\s\S]*download\?\.download_id === candidate \|\| candidateCompleted[\s\S]*sttReplacementAction\([\s\S]*action === "track"[\s\S]*trackSttDownloadNow\(model, options\)/,
  );
  assert.match(source, /action === "retry"[\s\S]*retry = true/);
  assert.match(
    source,
    /shouldRecheckSttReplacement\(trackedDownloadIds\.get\(key\), candidate\)[\s\S]*confirmSttDownloadReplacement\(/,
  );
  assert.match(
    source,
    /previousDownloadId !== options\.downloadId[\s\S]*confirmSttDownloadReplacement\(/,
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
  assert.match(
    source,
    /cancelSttDownload\([\s\S]*model,[\s\S]*resolvedEngine,[\s\S]*trackedVariants\.get\(key\),[\s\S]*trackedDownloadIds\.get\(key\),[\s\S]*\)/,
  );
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
  assert.match(
    source,
    /cancelSttDownload\([\s\S]*model,[\s\S]*resolvedEngine,[\s\S]*trackedVariants\.get\(key\),[\s\S]*trackedDownloadIds\.get\(key\),[\s\S]*\)/,
  );
  assert.match(source, /poll\(model, startedAt, resolvedEngine\)/);
  assert.doesNotMatch(
    source,
    /options\.engine === "transformers" \? undefined/,
  );
});
