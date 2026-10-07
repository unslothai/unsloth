// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

const source = readAudioWorkspaceSource();
const adapterSource = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");

test("Audio exposes the shared picker eject action only while idle", () => {
  assert.match(
    source,
    /onEject=\{\s*busy === null && selectorValue && !showLastPageModel\s*\? handleEject\s*: undefined\s*\}/,
  );
  assert.match(source, /const handleEject = useCallback\(\(\) => \{\s*if \(busy !== null\) \{/);
  assert.match(
    source,
    /loaded=\{\s*mode === "transcribe"\s*\? sttReady\s*: showLastPageModel\s*\? false\s*: undefined\s*\}/,
  );
});

test("Speak eject unloads the live main model and cancels stale auto-load", () => {
  assert.match(
    source,
    /const activeModel = status\?\.active_model;[\s\S]*unloadModel\(\{\s*model_path: activeModel,\s*force_cancel_active: stopDecision\.forceCancelActive,\s*\}\)/,
  );
  assert.match(
    source,
    /pendingStagedTtsLoad\.current = null;[\s\S]*stagedTtsLoadDeferred\.current = false;[\s\S]*stageTtsDownload\(\[\]\)/,
  );
  assert.match(source, /await unloadModel[\s\S]*await refreshStatus\(\)/);
});

test("Speak eject asks about running chats before tearing anything down", () => {
  assert.match(
    source,
    /const activeModel = status\?\.active_model;[\s\S]*confirmStopRunningChatsIfNeeded\(\s*"Unloading the model",\s*"unload",\s*\)/,
  );
  assert.match(
    source,
    /if \(!stopDecision\.proceed\) \{\s*setBusy\(null\);\s*return;\s*\}\s*\n\s*\/\/ An old managed completion[\s\S]*invalidatePendingStagedTts\(\);/,
  );
  assert.match(
    source,
    /cancelPreStreamRunReservations\(stopDecision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(stopDecision\.promptQueueThreadIds\);\s*await unloadModel/,
  );
});

test("a Speak load asks the same question and forces from the answer", () => {
  assert.match(
    source,
    /const stopDecision = await confirmStopRunningChatsIfNeeded\(\);/,
  );
  assert.match(
    source,
    /if \(ttsLoadInFlight\.current \|\| busyRef\.current === "generating"\) \{\s*pendingRoutedTtsPick\.current = \{\s*repoId,\s*ggufFilename,\s*loadId,\s*audioType,\s*remoteCodeApproval,\s*isGguf,\s*\};\s*return;\s*\}[\s\S]{0,400}?ttsLoadInFlight\.current = true;/,
  );
  assert.match(
    source,
    /if \(!stopDecision\.proceed\) \{\s*releaseLifecycle\(\);\s*ttsLoadInFlight\.current = false;[\s\S]*?pendingRoutedTtsPick\.current = null;\s*return;\s*\}/,
  );
  assert.match(
    source,
    /load_request_id: loadRequestId,\s*force_cancel_active: stopDecision\.forceCancelActive,/,
  );
});

test("a Speak load stops local queues only once /load is going out", () => {
  // loadModel can return without sending (invalid HF token), so cancelling before it loses work.
  assert.match(
    source,
    /onRequestStart: \(\) => \{\s*pending\.requestStarted = true;[\s\S]{0,700}?cancelPreStreamRunReservations\(stopDecision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(stopDecision\.promptQueueThreadIds\);\s*\},/,
  );
  assert.doesNotMatch(
    source,
    /cancelPreStreamRunReservations\(stopDecision\.preStreamRunTokens\);\s*requestLocalPromptQueueStop\(stopDecision\.promptQueueThreadIds\);\s*const res = await loadModel\(/,
  );
});

test("a model swap holds Chat's lifecycle gate across the question", () => {
  assert.match(
    source,
    /ttsLoadInFlight\.current = true;[\s\S]{0,400}?const lifecycleLease = useChatRuntimeStore\.getState\(\)\.beginModelLoading\(\);\s*if \(lifecycleLease === null\) \{[\s\S]*?return;\s*\}[\s\S]{0,200}?const stopDecision = await confirmStopRunningChatsIfNeeded\(\);/,
  );
  assert.match(
    source,
    /if \(activeRef\.current\) await refreshStatus\(\);\s*ttsLoadInFlight\.current = false;[\s\S]{0,120}?releaseLifecycle\(\);[\s\S]*?replayQueuedTtsPick\(\);/,
  );
  assert.match(
    source,
    /const lifecycleLease = useChatRuntimeStore\.getState\(\)\.beginModelLoading\(\);[\s\S]{0,300}?setBusy\("unloading"\);[\s\S]{0,400}?confirmStopRunningChatsIfNeeded\(\s*"Unloading the model",/,
  );
  assert.match(
    source,
    /\} finally \{\s*useChatRuntimeStore\.getState\(\)\.endModelLoading\(lifecycleLease\);\s*\}/,
  );
});

test("a load confirmed after Audio is hidden is deferred, not sent", () => {
  assert.match(
    source,
    /if \(!activeRef\.current\) \{\s*releaseLifecycle\(\);\s*ttsLoadInFlight\.current = false;\s*pendingRoutedTtsPick\.current = \{\s*repoId,\s*ggufFilename,\s*loadId,\s*audioType,\s*remoteCodeApproval,\s*isGguf,\s*\};\s*return;\s*\}/,
  );
  assert.match(
    source,
    /if \(!active\) \{[\s\S]*?\n    \}\s*\n\s*\/\/[\s\S]*?replayQueuedTtsPick\(\);/,
  );
});

test("Transcribe eject only unloads a sidecar owned by the current selection", () => {
  assert.match(
    source,
    /const handleEject[\s\S]*?if \(mode === "transcribe"\)/,
  );
  assert.match(
    source,
    /const releaseTranscribeSelection = useCallback\([\s\S]*await unloadSttModel\(sttEngineForRepoId\(selected\), claim\);\s*forget\(\);\s*await refreshSttStatus\(\)/,
  );
  assert.match(
    source,
    /const forget = \(\) => \{[\s\S]*setSelectedSttRepo\(null\);\s*\};[\s\S]*if \(!owned\) \{\s*forget\(\);/,
  );
  assert.match(
    source,
    /if \(mode === "transcribe"\)[\s\S]*await releaseTranscribeSelection\(\)/,
  );
  assert.match(
    adapterSource,
    /unloadSttModel\(\s*engine\?: SttEngine,[\s\S]*params\.set\("engine", engine\)/,
  );
  assert.match(
    source,
    /if \(!sttReady\) \{[\s\S]*sttStatusRefreshGeneration\.current \+= 1;[\s\S]*void releaseTranscribeSelection\(\)/,
  );
});

test("leaving Transcribe releases the sidecar it loaded", () => {
  // Anchored inside transitionMode: an unanchored [\s\S]* matched handleEject instead.
  assert.match(
    source,
    /setMode\(nextMode\);[\s\S]*?if \(mode === "transcribe"\) \{[\s\S]*?const release = releaseTranscribeSelection\(\)\.then\(/,
  );
  assert.match(source, /pendingTranscribeRelease\.current = release;/);
  assert.match(
    source,
    /const releaseInFlight = pendingTranscribeRelease\.current;[\s\S]*?if \(releaseInFlight && !\(await releaseInFlight\)\) \{\s*setMode\("transcribe"\);\s*return;/,
  );
});

test("selected and fallback clip actions remain named and downloadable", () => {
  assert.match(source, /aria-label="Download audio clip"/);
  assert.match(
    source,
    /onDelete=\{\(\) => void handleDeleteClip\(clip\.id\)\}/,
  );
  assert.match(source, /menu=\{clipMenu\(selectedClip, "row"\)\}/);
  assert.match(
    source,
    /const handleDownloadFallbackClip[\s\S]*saveAudio\(\s*clipFileName\(fallbackClip\),\s*fallbackClip\.url,/,
  );
  assert.match(source, /onDownload=\{handleDownloadFallbackClip\}/);
});

test("a dictation model this page did not load survives a mode switch", () => {
  // match model identity because gguf can report the Transformers fallback engine
  assert.match(source, /claim !== null &&\s*claim === sttLoadedModel;/);
  // claim only after load succeeds because cancellation can leave the previous model resident
  assert.doesNotMatch(
    source,
    /setBusy\("loading"\);\s*sttLoadedByThisPage\.current = sidecarKey;/,
  );
  assert.match(
    source,
    /await loadSttModel\(\s*sidecarKey,\s*engine,\s*controller\.signal,\s*undefined,\s*ggufVariant,\s*\);\s*sttLoadedByThisPage\.current = sidecarKey;/,
  );
});

test("the eject unload names the model this page claimed", () => {
  assert.match(
    source,
    /await unloadSttModel\(sttEngineForRepoId\(selected\), claim\);/,
  );
});

test("the unload request carries the claimed model to the backend", () => {
  const adapter = adapterSource;
  assert.match(adapter, /export function unloadSttModel\(\s*engine\?: SttEngine,\s*model\?: string,/);
  assert.match(adapter, /if \(model\) params\.set\("model", model\);/);
});
