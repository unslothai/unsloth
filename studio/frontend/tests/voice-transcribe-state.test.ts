// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const panel = read("../src/components/assistant-ui/voice-panel.tsx");
const adapter = read("../src/features/chat/adapters/studio-voice-loop-adapter.ts");
const store = read("../src/features/chat/stores/chat-runtime-store.ts");

test("the Transcribe row shows the transcriber's real load state", () => {
  // It was always "loaded", and counted as ready, with no transcriber downloaded.
  const row = panel.slice(panel.indexOf("transcribe: {"), panel.indexOf("decide: {"));
  assert.match(row, /state: sttLoaded \? "loaded" : "not_loaded"/);
  assert.match(panel, /<SttModelSelector[\s\S]*?ready=\{sttLoaded\}/);
});

test("the listen session reports the transcriber load and asks for a missing model once", () => {
  const load = adapter.slice(adapter.indexOf("void loadSttModel(sttPick)"), adapter.indexOf("if (!vadDisabled) {"));
  assert.match(load, /setSttSlotLoaded\(true\)/);
  assert.match(load, /setSttSlotLoaded\(false\)/);
  // Asked as listening begins: waiting for the first utterance lost that utterance to the prompt.
  assert.match(
    load,
    /error instanceof SttModelNotDownloadedError && sttDownloadAskedFor !== sttPick\) \{\s*sttDownloadAskedFor = sttPick;\s*requestSttDownload\(sttPick\);/,
  );
});

test("another transcriber pick starts as not loaded", () => {
  assert.match(store, /return \{ selectedSttModelId, sttSlotLoaded: false \};/);
});
