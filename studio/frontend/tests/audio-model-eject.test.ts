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
    /const activeModel = status\?\.active_model;[\s\S]*unloadModel\(\{ model_path: activeModel \}\)/,
  );
  assert.match(
    source,
    /pendingStagedTtsLoad\.current = null;[\s\S]*stagedTtsLoadDeferred\.current = false;[\s\S]*stageTtsDownload\(\[\]\)/,
  );
  assert.match(source, /await unloadModel[\s\S]*await refreshStatus\(\)/);
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
    /const releaseTranscribeSelection = useCallback\([\s\S]*await unloadSttModel\(sttEngineForRepoId\(selected\)\);\s*forget\(\);\s*await refreshSttStatus\(\)/,
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
    /unloadSttModel\(engine\?: SttEngine\)[\s\S]*\?engine=\$\{encodeURIComponent\(engine\)\}/,
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
    /setMode\(nextMode\);[^}]*if \(mode === "transcribe"\) \{\s*const release = releaseTranscribeSelection\(\)/,
  );
  assert.match(source, /pendingTranscribeRelease\.current = release;/);
  assert.match(
    source,
    /const releaseInFlight = pendingTranscribeRelease\.current;\s*if \(releaseInFlight\) await releaseInFlight;/,
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
  assert.match(source, /claim !== null && claim === sttLoadedModel;/);
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
