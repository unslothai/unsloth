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
  assert.match(
    source,
    /if \(mode === "transcribe"\)[\s\S]*const selectedEngine = sttEngineForRepoId\(selected\)[\s\S]*sttLoadedModel !== sttSidecarKeyFor\(selected\)[\s\S]*sttLoadedEngine !== selectedEngine[\s\S]*await unloadSttModel\(selectedEngine\)/,
  );
  assert.match(
    source,
    /await unloadSttModel\(selectedEngine\);[\s\S]*setSelectedSttRepo\(null\);[\s\S]*await refreshSttStatus\(\)/,
  );
  assert.match(
    adapterSource,
    /unloadSttModel\(engine\?: SttEngine\)[\s\S]*\?engine=\$\{encodeURIComponent\(engine\)\}/,
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
