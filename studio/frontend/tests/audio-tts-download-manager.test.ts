// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("uncached remote TTS GGUFs stage the exact file through the shared manager", () => {
  assert.match(source, /useStagedDownload\(\{\s*scopeId: "audio"/);
  assert.match(
    source,
    /meta\.source === "hub"[\s\S]*meta\.isDownloaded === false[\s\S]*ggufFilename/,
  );
  assert.match(
    source,
    /stageTtsDownload\(\[\s*\{[\s\S]*repoId,[\s\S]*files: \[ggufFilename\],[\s\S]*bytes: meta\.expectedBytes \?\? 0/,
  );
  assert.match(
    source,
    /const exactGguf = exactGgufLoadSelector\(meta\)[\s\S]*loadOrStageTtsModel\(id, exactGguf, meta\)/,
  );
});

test("cached and local TTS picks keep the direct load path and supersede stale staging", () => {
  assert.match(
    source,
    /pendingStagedTtsLoad\.current = null;[\s\S]*stageTtsDownload\(\[\]\);[\s\S]*loadTtsModelRef\.current\(repoId, ggufFilename\)/,
  );
  assert.match(source, /if \(busyRef\.current !== null\) return;/);
  assert.match(source, /if \(ttsLoadInFlight\.current\) return;/);
});

test("managed completion loads the exact GGUF only when Audio is active and idle", () => {
  assert.match(
    source,
    /onReady: \(\) => \{[\s\S]*if \(!active \|\| busyRef\.current !== null\)[\s\S]*stagedTtsLoadDeferred\.current = true/,
  );
  assert.match(
    source,
    /loadTtsModelRef\.current\(pending\.repoId, pending\.ggufFilename\)/,
  );
  assert.match(
    source,
    /!active \|\|[\s\S]*busy !== null \|\|[\s\S]*!stagedTtsLoadDeferred\.current/,
  );
});

test("switching to Transcribe invalidates a pending staged TTS auto-load", () => {
  assert.match(
    source,
    /if \(nextMode === "transcribe"\) invalidatePendingStagedTts\(\)/,
  );
  assert.match(
    source,
    /stagedTtsLoadIsOwned\([\s\S]*pending\.generation[\s\S]*stagedTtsGeneration\.current[\s\S]*modeRef\.current/,
  );
  assert.match(
    source,
    /if \(nextMode === mode\)[\s\S]*return true;[\s\S]*if \(!canTransitionAudioMode\(busyRef\.current\)\)[\s\S]*return false;[\s\S]*if \(nextMode === "transcribe"\) invalidatePendingStagedTts\(\)/,
    "a rejected mode switch must not discard the still-owned staged load",
  );
});

test("a GGUF forwarded from Chat enters the managed staging path", () => {
  assert.match(
    source,
    /ggufFilename: routeSearch\.quant \?\? undefined,[\s\S]*isDownloaded: routeSearch\.quant \? false : undefined/,
  );
});

test("Mac sibling inspection reserves the lifecycle slot before awaiting inventory", () => {
  assert.match(
    source,
    /ttsInspectionGeneration\.current = selectionGeneration;[\s\S]*busyRef\.current = "loading";[\s\S]*setBusy\("loading"\);[\s\S]*await listGgufVariants/,
  );
});
