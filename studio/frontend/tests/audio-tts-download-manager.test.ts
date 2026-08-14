// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const apiSource = readFileSync(
  new URL("../src/features/audio/api.ts", import.meta.url),
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
    // meta.loadId is the load target for a row cached in a non-active HF cache; sending
    // the display repo id instead failed offline or re-downloaded into the active cache.
    /pendingStagedTtsLoad\.current = null;[\s\S]*stageTtsDownload\(\[\]\);[\s\S]*loadTtsModelRef\.current\([\s\S]*repoId,[\s\S]*ggufFilename,[\s\S]*meta\.loadId,[\s\S]*meta\.audioType/,
  );
  assert.match(source, /if \(busyRef\.current !== null\) return;/);
  assert.match(
    source,
    /\/\*\* Start a pick that lost the race[\s\S]*?const replayQueuedTtsPick = useCallback/,
  );
  // Still single-flight, but the loser is queued rather than dropped: a pick arriving
  // while a cancelled load settles used to vanish, and the route effect had already
  // cleared ?model=, so nothing retried it.
  assert.match(
    source,
    /if \(ttsLoadInFlight\.current\) \{\s*pendingRoutedTtsPick\.current = \{[\s\S]*repoId,[\s\S]*ggufFilename,[\s\S]*loadId,[\s\S]*audioType,[\s\S]*\};\s*return;\s*\}/,
  );
  assert.match(
    source,
    // Replayed only while Audio is visible, and again when it becomes visible. Replaying
    // unconditionally started a load with activeRef already false, which the deactivation
    // effect had already stopped watching, so a hidden page could replace Chat's model.
    /if \(activeRef\.current\) replayQueuedTtsPick\(\);/,
  );
});

test("Hub TTS repos and native companion codecs use a cache-aware exact-file plan", () => {
  assert.match(apiSource, /\/api\/inference\/audio\/download-plan/);
  assert.match(
    source,
    /meta\.source === "hub" && !ggufFilename[\s\S]*await getAudioDownloadPlan\([\s\S]*plannedEntries\.length > 0/,
  );
  assert.match(
    source,
    /plannedEntries\.map\(\(entry\) => \(\{[\s\S]*repoId: entry\.repo_id,[\s\S]*files: entry\.files,[\s\S]*bytes: entry\.bytes,[\s\S]*checkpoint: entry\.checkpoint/,
  );
  assert.doesNotMatch(
    source,
    /meta\.source === "hub" && !ggufFilename[\s\S]{0,400}meta\.isDownloaded === false/,
    "a cached main repo can still be missing its companion codec",
  );
});

test("a cached Hub TTS model can still load when the metadata plan is offline", () => {
  assert.match(
    source,
    /catch \(error\) \{[\s\S]*meta\.isDownloaded === true[\s\S]*loadTtsModelRef\.current\([\s\S]*repoId,[\s\S]*ggufFilename,[\s\S]*meta\.loadId,[\s\S]*meta\.audioType/,
  );
});

test("remembered-cache TTS rows stage companions without duplicating the checkpoint", () => {
  assert.match(
    source,
    /const plannedEntries =\s*meta\.isDownloaded === true && meta\.loadId[\s\S]*plan\.entries\.filter\(\(entry\) => !entry\.checkpoint\)/,
  );
  assert.match(
    source,
    /if \(plannedEntries\.length > 0\)[\s\S]*plannedEntries\.map\(\(entry\) => \(\{/,
  );
});

test("managed completion loads the exact GGUF only when Audio is active and idle", () => {
  assert.match(
    source,
    /onReady: \(\) => \{[\s\S]*if \(!active \|\| busyRef\.current !== null\)[\s\S]*stagedTtsLoadDeferred\.current = true/,
  );
  assert.match(
    source,
    /loadTtsModelRef\.current\([\s\S]*pending\.repoId,[\s\S]*pending\.ggufFilename,[\s\S]*pending\.loadId,[\s\S]*pending\.audioType/,
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
