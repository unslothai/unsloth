// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

const source = readAudioWorkspaceSource();
const adapterSource = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");

test("Audio exposes the shared picker eject action only while idle", () => {
  // The selector keeps showing the last Transcribe model once it is unloaded; Eject there did nothing.
  assert.match(
    source,
    /onEject=\{\s*audioEjectAvailable\(\{\s*mode,\s*busy,\s*hasValue: Boolean\(selectorValue\),\s*showLastPageModel,\s*sttReady,\s*\}\)\s*\? handleEject\s*: undefined\s*\}/,
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

test("the Generate-mode release only unloads a sidecar owned by the current selection", () => {
  assert.match(
    source,
    /const handleEject[\s\S]*?if \(mode === "transcribe"\)/,
  );
  // The selection is forgotten only after the unload lands, so a 500 leaves Eject usable.
  assert.match(
    source,
    /const releaseTranscribeSelection = useCallback\([\s\S]*?await unloadSttModel\(sttEngineFor\(selected\), claim\);\s*forgetSttSelection\(\);\s*await refreshSttStatus\(\)/,
  );
  assert.match(
    source,
    /const forgetSttSelection = useCallback\(\(\) => \{[\s\S]*?setSelectedSttRepo\(null\);\s*\}, \[\]\);[\s\S]*?if \(!owned\) \{\s*forgetSttSelection\(\);/,
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

test("an explicit Transcribe eject unloads what is resident, claimed or not", () => {
  // A reload, Settings > Voice or dictation leave no page claim; the claim-gated release then
  // skipped the unload while the page still toasted success.
  const eject = source.match(
    /const ejectTranscribeModel = useCallback\([\s\S]*?\n {2}\}, \[/,
  )?.[0];
  assert.ok(eject);
  assert.match(
    eject,
    /const \{ released \} = await unloadSttModel\(\s*sttLoadedEngine \?\? undefined,\s*sttLoadedModel,\s*\);\s*forgetSttSelection\(\);\s*await refreshSttStatus\(\);\s*return sttEjectSucceeded\(released\) \? "unloaded" : "kept";/,
  );
  assert.doesNotMatch(eject, /sttLoadedByThisPage|claim/);
  assert.match(
    source,
    /if \(mode === "transcribe"\)[\s\S]*?const outcome = await ejectTranscribeModel\(\);\s*if \(outcome === "unloaded"\) \{\s*toast\.success\("Transcription model unloaded"/,
  );
  // Only the Generate-mode transition keeps the ownership-gated release.
  assert.doesNotMatch(
    source,
    /const handleEject[\s\S]*?await releaseTranscribeSelection\(\)[\s\S]*?const activeModel = status\?\.active_model;/,
  );
  assert.match(
    adapterSource,
    /const body = \(await response\.json\(\)\.catch\(\(\) => null\)\) as \{\s*released\?: boolean;\s*\} \| null;\s*return \{ released: body\?\.released \};/,
  );
});

test("Transcribe re-reads STT status while a model shows as loaded", () => {
  // The sidecar unloads after its idle keep-alive without telling the page.
  assert.match(
    source,
    /if \(!active \|\| mode !== "transcribe" \|\| busy !== null \|\| !sttLoadedModel\)\s*return;[\s\S]*?window\.setInterval\(refresh, STT_STATUS_POLL_MS\);[\s\S]*?return \(\) => \{\s*window\.clearInterval\(interval\);\s*window\.removeEventListener\("focus", refresh\);\s*document\.removeEventListener\("visibilitychange", refresh\);/,
  );
  assert.match(source, /if \(!document\.hidden\) void refreshSttStatus\(\);/);
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
    /await unloadSttModel\(sttEngineFor\(selected\), claim\);/,
  );
});

test("the unload request carries the claimed model to the backend", () => {
  const adapter = adapterSource;
  assert.match(adapter, /export function unloadSttModel\(\s*engine\?: SttEngine,\s*model\?: string,/);
  assert.match(adapter, /if \(model\) params\.set\("model", model\);/);
});

test("the engine picked for a Transcribe model is remembered and reused", () => {
  // Re-deriving it from the id drops the GGUF flag, so a suffix-less GGUF repo read as Transformers.
  assert.match(
    source,
    /const engine = sttEngineForRepoId\(id, meta\.isGguf\);\s*sttEngines\.current\.set\(id\.toLowerCase\(\), engine\);/,
  );
  assert.match(source, /usePersistedChoice\("unsloth:audio:last-stt-engine", ""\)/);
  assert.match(
    source,
    /setLastSttEngine\(sttEngines\.current\.get\(repo\.toLowerCase\(\)\) \?\? ""\);/,
  );
  assert.match(source, /engineForRepo: sttEngineFor,/);
  assert.match(source, /selectedSttRepo \? sttEngineFor\(selectedSttRepo\) : null,/);
  assert.match(source, /const engine = sttEngineFor\(repo\);[\s\S]{0,400}?await readSttResidency\(engine\)/);
});

test("Transcribe reuses a resident model that a legacy key or other casing names", () => {
  // An exact compare reloaded it on every run and took over dictation's model.
  assert.match(
    source,
    /const resident = await readSttResidency\(engine\)[\s\S]*?if \(\s*!resident \|\|\s*resident\.engine !== engine \|\|\s*!sttSameModel\(\s*repo,\s*resident\.model,\s*sttSidecarKeyFor,\s*engine,\s*sttRepoIdForSidecarKey,\s*\)\s*\) \{\s*if \(!\(await ensureSttLoaded\(repo, model, engine\)\)\) return null;/,
  );
});

test("a status read never overwrites a pick made while it was in flight", () => {
  // The poll widened the window in which a stale read reverted the selection and dropped the pick.
  assert.match(
    source,
    /if \(selectedSttRepoRef\.current !== selectedRepo\) return;[\s\S]{0,400}?selectedSttRepoRef\.current = reconciled;\s*setSelectedSttRepo\(reconciled\);/,
  );
});

test("the resident engine is adopted even when the repo id is unchanged", () => {
  // Reconcile returns the same id when only the engine moved, so a guard on a changed id
  // left a stale Transformers engine and the row never showed ready.
  assert.match(
    source,
    /if \(reconciled && residency\) \{\s*sttEngines\.current\.set\(reconciled\.toLowerCase\(\), residency\.engine\);\s*\}/,
  );
});

test("a pick made while Transcribe reads residency is not overridden by the old run", () => {
  assert.match(
    source,
    /const generation = sttLoadGeneration\.current;[\s\S]*?await readSttResidency\(engine\)[\s\S]*?if \(sttLoadGeneration\.current !== generation \|\| !activeRef\.current\) return null;\s*const current = selectedSttRepoRef\.current;\s*if \(current !== null && current !== repo\) return null;\s*if \(current === null\) \{\s*selectedSttRepoRef\.current = repo;\s*setSelectedSttRepo\(repo\);\s*\}/,
  );
});

test("leaving Audio during the residency read skips the transcription load", () => {
  // The hidden page has no load to abort yet, so ensureSttLoaded would still post /stt/load.
  assert.match(
    source,
    /await readSttResidency\(engine\)[\s\S]{0,400}?\|\| !activeRef\.current\) return null;[\s\S]*?await ensureSttLoaded\(repo, model, engine\)/,
  );
});
