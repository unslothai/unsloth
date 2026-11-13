// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

const source = readAudioWorkspaceSource();
const mirrorSource = readSrc("features/settings/lib/stt-download-mirror.ts");

test("STT download polling uses the available engine fallback", () => {
  assert.match(
    source,
    /const block = sttEngineStatusFor\(stt, sidecarKey, engine\)/,
  );
  assert.doesNotMatch(
    source,
    /engine === "gguf"[\s\S]*\? stt\.gguf[\s\S]*: stt\.transformers/,
  );
});

test("Audio mirrors an STT transfer into Downloads without resetting an adopted job", () => {
  assert.match(
    source,
    /const download = await startSttDownload\(\s*sidecarKey,[\s\S]*engine,\s*ggufVariant,\s*\);[\s\S]*trackSttDownload\(sidecarKey, \{[\s\S]*warmSelectedVoiceModelOnComplete: false,[\s\S]*engine,[\s\S]*repoId,[\s\S]*ggufVariant,[\s\S]*downloadId: download\.download_id,[\s\S]*\}\);[\s\S]*for \(;;\)/,
  );
  assert.doesNotMatch(source, /`Downloading \$\{sidecarKey\}:/);
});

test("cancelling the shared STT download never loads a partial checkpoint", () => {
  assert.match(
    source,
    /if \(download\?\.cancelled\) return;[\s\S]*if \(download\?\.error\)[\s\S]*if \(!download\?\.downloading\) break;[\s\S]*await loadSttModel\(\s*sidecarKey,\s*engine,\s*controller\.signal,/,
  );
});

test("hidden STT preparation resumes once only for the same selected repo", () => {
  assert.match(
    source,
    /deferredSttLoad\.current = repoId[\s\S]*sttLoadGeneration\.current \+= 1/,
  );
  assert.match(
    source,
    /const deferred = deferredSttLoad\.current;[\s\S]*deferredSttLoad\.current = null;[\s\S]*selectedSttRepoRef\.current === deferred\.repoId[\s\S]*ensureSttLoaded\([\s\S]*deferred\.repoId,[\s\S]*deferred\.sidecarKey,[\s\S]*deferred\.engine/,
  );
});

test("resolved Transformers transfers share one tracker identity", () => {
  assert.match(
    mirrorSource,
    /engine && engine !== "transformers" \? `\$\{engine\}:\$\{model\}` : model/,
  );
  assert.match(
    mirrorSource,
    /const resolvedEngine = options\.engine \?\? sttEngineFor\(model\);[\s\S]*trackerKey\(model, resolvedEngine\)/,
  );
  assert.match(
    mirrorSource,
    /if \(wasTracking && !changedAttempt\)[\s\S]*warmSelectedVoiceModelOnComplete\.set\(key, true\)/,
  );
});

test("hidden Audio cancels only an active sidecar load", () => {
  assert.match(
    source,
    /if \(sttLoadingGeneration\.current !== null\)[\s\S]*sttLoadAbort\.current\?\.abort\(\)/,
  );
  assert.doesNotMatch(source, /cancelSttLoad/);
});

test("STT loading stays busy until authoritative residency refresh completes", () => {
  assert.match(
    source,
    /if \(sttLoadingGeneration\.current === generation\) \{[\s\S]*sttLoadingGeneration\.current = null;[\s\S]*await refreshSttStatus\(\);[\s\S]*generation === sttLoadGeneration\.current[\s\S]*sttLoadingGeneration\.current === null[\s\S]*setBusy\(null\)/,
  );
});

test("cancelled deferred downloads remain cancelled on return", () => {
  // `model` is null once the download thread stops, so the id comes from cancelled_model.
  assert.match(
    source,
    /const download = sttEngineStatusFor[\s\S]*download\?\.cancelled[\s\S]*\(download\.model \?\? download\.cancelled_model\) === deferred\.sidecarKey[\s\S]*return/,
  );
});
