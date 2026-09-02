// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const adapterSource = readFileSync(
  new URL(
    "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
    import.meta.url,
  ),
  "utf8",
);

test("Audio exposes the shared picker eject action only while idle", () => {
  assert.match(
    source,
    /onEject=\{busy === null && selectorValue \? handleEject : undefined\}/,
  );
  assert.match(source, /if \(busy !== null \|\| isRecording\)/);
  assert.match(
    source,
    /loaded=\{mode === "transcribe" \? sttReady : undefined\}/,
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
  // Unforced, the backend refused with a 409 the page could only print as a toast.
  assert.match(
    source,
    /const activeModel = status\?\.active_model;[\s\S]*confirmStopRunningChatsIfNeeded\(\s*"Unloading the model",\s*"unload",\s*\)/,
  );
  // Declining leaves the page as it was: the staged download dies only past the check.
  assert.match(
    source,
    /if \(!stopDecision\.proceed\) \{\s*setBusy\(null\);\s*return;\s*\}\s*\n\s*\/\/ An old managed completion[\s\S]*invalidatePendingStagedTts\(\);/,
  );
  // Queues would otherwise start a fresh run on the model this eject removes.
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
  // The slot is claimed before the await, so a routed pick arriving mid-dialog queues.
  assert.match(
    source,
    /if \(ttsLoadInFlight\.current \|\| busyRef\.current === "generating"\) \{\s*pendingRoutedTtsPick\.current = \{\s*repoId,\s*ggufFilename,\s*loadId,\s*audioType,\s*remoteCodeApproval,\s*\};\s*return;\s*\}[\s\S]{0,400}?ttsLoadInFlight\.current = true;/,
  );
  // Declining releases the slot and drops the queued pick, which would else re-ask.
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
  // loadModel prepares the stored HF token first and returns without sending when the
  // token is invalid and the user picks replace or dismisses the warning. Cancelling
  // before that call discarded accepted sends and queued prompts for a swap that never
  // happened, leaving the old model resident and the work gone.
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
  // Without the gate a queue can materialize while the dialog is open, so it is missing
  // from the snapshot the answer was given for: the eject's blanket queue stop then hits
  // work nobody confirmed stopping, and a load started in that window 409s again.
  assert.match(
    source,
    /ttsLoadInFlight\.current = true;[\s\S]{0,400}?const lifecycleLease = useChatRuntimeStore\.getState\(\)\.beginModelLoading\(\);\s*if \(lifecycleLease === null\) \{[\s\S]*?return;\s*\}[\s\S]{0,200}?const stopDecision = await confirmStopRunningChatsIfNeeded\(\);/,
  );
  // Released before the queued replay, which needs the gate for its own attempt.
  assert.match(
    source,
    /if \(activeRef\.current\) await refreshStatus\(\);\s*ttsLoadInFlight\.current = false;[\s\S]{0,120}?releaseLifecycle\(\);[\s\S]*?replayQueuedTtsPick\(\);/,
  );
  // Eject takes it before it goes busy, so it is held across its own question too.
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
  // pendingTtsLoad is still null while the dialog is open, so the deactivation effect has
  // nothing to abort. Sending anyway let a hidden page replace the visible page's model.
  assert.match(
    source,
    /if \(!activeRef\.current\) \{\s*releaseLifecycle\(\);\s*ttsLoadInFlight\.current = false;\s*pendingRoutedTtsPick\.current = \{\s*repoId,\s*ggufFilename,\s*loadId,\s*audioType,\s*remoteCodeApproval,\s*\};\s*return;\s*\}/,
  );
  // The activation effect replays exactly that queue, so the pick is not lost.
  assert.match(
    source,
    /if \(!active\) \{[\s\S]*?\n    \}\s*\n\s*\/\/[\s\S]*?replayQueuedTtsPick\(\);/,
  );
});

test("Transcribe eject only unloads a sidecar owned by the current selection", () => {
  assert.match(
    source,
    /const handleEject[\s\S]*stopAndDiscardRecording\(\);[\s\S]*if \(mode === "transcribe"\)/,
  );
  // One release path, shared with the Generate-mode transition, so both stay owned.
  // The selection is forgotten only after the unload lands, so a 500 leaves Eject usable.
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
  // Holding it through Generate doubled VRAM for the whole keep-alive window (PR 7984 report).
  // Anchored inside transitionMode: an unanchored [\s\S]* matched handleEject instead, so
  // deleting the release from the mode switch still passed.
  // The release is now captured rather than fire-and-forget, so a following TTS load can
  // wait behind the teardown instead of allocating alongside it.
  assert.match(
    source,
    /setMode\(nextMode\);[\s\S]*?if \(mode === "transcribe"\) \{[\s\S]*?const release = releaseTranscribeSelection\(\)\.then\(/,
  );
  assert.match(source, /pendingTranscribeRelease\.current = release;/);
  assert.match(
    source,
    // The release resolves to whether the sidecar is gone. A failure must not hand off to
    // a speech load on top of a still-resident dictation model.
    /const releaseInFlight = pendingTranscribeRelease\.current;[\s\S]*?if \(releaseInFlight && !\(await releaseInFlight\)\) \{\s*setMode\("transcribe"\);\s*return;/,
  );
});

test("selected and fallback clip actions remain named and downloadable", () => {
  assert.match(source, /aria-label="Download audio clip"/);
  assert.match(source, /aria-label="Delete audio clip"/);
  assert.match(
    source,
    /const handleDownloadFallbackClip[\s\S]*anchor\.download = "generated-audio\.wav"/,
  );
  assert.match(
    source,
    /onClick=\{handleDownloadFallbackClip\}[\s\S]*Download WAV/,
  );
});

test("a dictation model this page did not load survives a mode switch", () => {
  // The activation resync adopts whatever a sidecar holds, including chat dictation's model.
  // The identity, not a boolean. Another surface can swap the sidecar's model while Audio
  // is inactive; the activation resync then adopts it, and a bare flag claimed it too, so
  // Eject unloaded a model this page never loaded. Model only, not model plus engine: a
  // "gguf" pick without whisper-server is served by the Transformers fallback and reports
  // residency under that engine, so requiring the requested engine leaked the sidecar.
  assert.match(source, /claim !== null &&\s*claim === sttLoadedModel;/);
  // Ownership is claimed after a successful load, not before it: claiming up front left the
  // flag set when a download was cancelled while the backend kept the previous resident
  // model, so leaving Transcribe unloaded another surface's model.
  assert.doesNotMatch(
    source,
    /setBusy\("loading"\);\s*sttLoadedByThisPage\.current = sidecarKey;/,
  );
  assert.match(
    source,
    /await loadSttModel\(sidecarKey, engine, controller\.signal\);\s*sttLoadedByThisPage\.current = sidecarKey;/,
  );
});

test("the eject unload names the model this page claimed", () => {
  // `owned` is decided locally, so another surface can switch the same engine before the
  // request lands; an unscoped unload then tore down a model this page never owned.
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
