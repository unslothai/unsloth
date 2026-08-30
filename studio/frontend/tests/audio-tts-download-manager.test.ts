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

test("remote code approval precedes native model staging and survives completion", () => {
  const start = source.indexOf("const loadOrStageTtsModel");
  const end = source.indexOf("const ensureSttLoaded", start);
  const stagedFlow = source.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.ok(
    stagedFlow.indexOf("confirmRemoteCodeIfNeeded(") <
      stagedFlow.indexOf("getAudioDownloadPlan("),
  );
  assert.match(
    stagedFlow,
    /pendingStagedTtsLoad\.current = \{[\s\S]*remoteCodeApproval,[\s\S]*generation/,
  );
  assert.match(source, /pending\.audioType,[\s\S]*pending\.remoteCodeApproval/);
  assert.match(
    source,
    /audioModelRequiresRemoteCode\(repoId, audioType\) &&[\s\S]*!remoteCodeApproval/,
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
    /if \(ttsLoadInFlight\.current \|\| busyRef\.current === "generating"\) \{\s*pendingRoutedTtsPick\.current = \{[\s\S]*repoId,[\s\S]*ggufFilename,[\s\S]*loadId,[\s\S]*audioType,[\s\S]*remoteCodeApproval,[\s\S]*\};\s*return;\s*\}/,
  );
  assert.match(
    source,
    // Replayed only while Audio is visible, and again when it becomes visible. Replaying
    // unconditionally started a load with activeRef already false, which the deactivation
    // effect had already stopped watching, so a hidden page could replace Chat's model.
    /if \(activeRef\.current\) replayQueuedTtsPick\(\);/,
  );
});

test("a preflight that loses to generation queues its load until generation ends", () => {
  assert.match(
    source,
    /if \(ttsLoadInFlight\.current \|\| busyRef\.current === "generating"\) \{[\s\S]*pendingRoutedTtsPick\.current/,
  );
  assert.match(
    source,
    /generateAbort\.current = null;\s*busyRef\.current = null;\s*setBusy\(null\);\s*if \(activeRef\.current && modeRef\.current === "speak"\)\s*replayQueuedTtsPick\(\)/,
  );
});

test("a completed load keeps controls disabled until status catches up", () => {
  const start = source.indexOf("const loadTtsModel = useCallback");
  const end = source.indexOf("const loadTtsModelRef", start);
  const loadFlow = source.slice(start, end);
  assert.match(
    loadFlow,
    /if \(activeRef\.current\) await refreshStatus\(\);[\s\S]*busyRef\.current = null;[\s\S]*setBusy\(null\)/,
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
  const invalidateStart = source.indexOf("const invalidatePendingTtsSelection");
  const transitionStart = source.indexOf("const transitionMode", invalidateStart);
  const invalidateSelection = source.slice(invalidateStart, transitionStart);
  assert.match(invalidateSelection, /ttsPickGeneration\.current \+= 1;/);
  assert.match(invalidateSelection, /pendingRoutedTtsPick\.current = null;/);
  assert.match(invalidateSelection, /invalidatePendingStagedTts\(\);/);
  assert.match(
    source,
    /if \(nextMode === "transcribe"\) invalidatePendingTtsSelection\(\)/,
  );
  assert.match(
    source,
    /stagedTtsLoadIsOwned\([\s\S]*pending\.generation[\s\S]*stagedTtsGeneration\.current[\s\S]*modeRef\.current/,
  );
  assert.match(
    source,
    /if \(nextMode === mode\)[\s\S]*return true;[\s\S]*if \(!canTransitionAudioMode\(busyRef\.current\)\)[\s\S]*return false;[\s\S]*if \(nextMode === "transcribe"\) invalidatePendingTtsSelection\(\)/,
    "a rejected mode switch must not discard the still-owned staged load",
  );
});

test("a GGUF forwarded from Chat enters the managed staging path", () => {
  assert.match(
    source,
    /ggufFilename: routeSearch\.quant \?\? undefined,[\s\S]*isDownloaded: routeSearch\.loadId[\s\S]*routeSearch\.quant[\s\S]*\? false/,
  );
});

test("Mac sibling inspection reserves the lifecycle slot before awaiting inventory", () => {
  assert.match(
    source,
    /ttsInspectionGeneration\.current = selectionGeneration;[\s\S]*busyRef\.current = "loading";[\s\S]*setBusy\("loading"\);[\s\S]*await listGgufVariants/,
  );
});
