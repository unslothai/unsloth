// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const mirrorSource = readFileSync(
  new URL(
    "../src/features/settings/lib/stt-download-mirror.ts",
    import.meta.url,
  ),
  "utf8",
);

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
    /await startSttDownload\(sidecarKey,[\s\S]*engine\);[\s\S]*if \(!isTrackingSttDownload\(sidecarKey, engine\)\) \{[\s\S]*trackSttDownload\(sidecarKey, \{[\s\S]*warmSelectedVoiceModelOnComplete: false,[\s\S]*engine,[\s\S]*repoId,[\s\S]*\}\);[\s\S]*\}[\s\S]*for \(;;\)/,
  );
  assert.doesNotMatch(source, /`Downloading \$\{sidecarKey\}:/);
});

test("cancelling the shared STT download never loads a partial checkpoint", () => {
  assert.match(
    source,
    /if \(download\?\.cancelled\) return;[\s\S]*if \(download\?\.error\)[\s\S]*if \(!download\?\.downloading\) break;[\s\S]*await loadSttModel\(sidecarKey, engine\)/,
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
    /if \(trackers\.has\(key\)\)[\s\S]*warmSelectedVoiceModelOnComplete\.set\(key, true\)/,
  );
});

test("hidden Audio cancels only an active sidecar load", () => {
  assert.match(
    source,
    /if \(sttLoadingGeneration\.current !== null\)[\s\S]*cancelSttLoad\(engine\)/,
  );
});

test("cancelled deferred downloads remain cancelled on return", () => {
  assert.match(
    source,
    /const download = sttEngineStatusFor[\s\S]*download\?\.cancelled[\s\S]*download\.model === deferred\.sidecarKey[\s\S]*return/,
  );
});
