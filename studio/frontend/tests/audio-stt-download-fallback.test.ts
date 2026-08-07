// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
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
    /await startSttDownload\([\s\S]*sidecarKey,[\s\S]*engine,[\s\S]*if \(!isTrackingSttDownload\(sidecarKey, engine\)\) \{[\s\S]*trackSttDownload\(sidecarKey, \{[\s\S]*warmSelectedVoiceModelOnComplete: false,[\s\S]*engine,[\s\S]*repoId,[\s\S]*\}\);[\s\S]*\}[\s\S]*for \(;;\)/,
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
