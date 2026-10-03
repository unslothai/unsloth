// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const host = readSrc("features/audio/audio-page.tsx");
const page = readSrc("features/audio/pages/edit-page.tsx");
const hook = readSrc("features/audio/hooks/use-edit-generation.ts");
const gallery = readSrc("features/audio/hooks/use-audio-gallery.tsx");
const output = readSrc("features/audio/pages/tts-workspace.tsx");

test("Edit runs through its own hook, from the button and from Mod+Enter", () => {
  assert.match(host, /const edit = useEditGeneration\(\{/);
  assert.match(
    host,
    /const handleWorkflowGenerate =\s*ttsWorkflow === "edit" \? edit\.handleGenerate : handlePageGenerate;/,
  );
  assert.match(host, /handleGenerate=\{edit\.handleGenerate\}|edit=\{edit\}/);
  assert.match(page, /handleGenerate=\{edit\.handleGenerate\}/);
  assert.match(host, /: ttsWorkflow === "edit"\s*\? edit\.blocker/);
});

test("the recording takes uploads, recordings and history, not saved voices", () => {
  assert.match(page, /id="edit-recording"[\s\S]*?allowSavedVoice=\{false\}/);
  assert.match(page, /<AudioHistoryProvider value=\{historyClips\}>/);
});

test("Edit lists only models that can edit, DotTTS Edit first", () => {
  assert.match(page, /audioCppWorkflowsFor\(entry\)\.includes\("edit"\)/);
  assert.match(
    page,
    /EDIT_MODEL_ORDER = \[\s*"DotTTS-Edit-GGUF",\s*"Vevo2-GGUF",\s*"FireRedAudio-GGUF",?\s*\]/,
  );
  assert.match(host, /editPageModels\(MODELS_BY_MODE\.speak, isMac\)/);
});

test("history hides an edit's original, and the selected edit plays as Original | Edited", () => {
  assert.match(gallery, /\.filter\(\(clip\) => clip\.role !== "source"\)/);
  assert.match(page, /clip\.source_clip_id \?/);
  assert.match(page, /<ABCompare/);
  // A clip without an original falls back to the plain player.
  assert.match(output, /customPlayer \?/);
});

test("① transcribes itself once per recording and never sends the model's text back", () => {
  assert.match(hook, /autoTranscribed\.current === source\.id/);
  assert.match(hook, /buildEditRun\(adapter,/);
  assert.match(hook, /sourceRefOf\(state\.source\)/);
});

test("the compare draws its bars from the server's file, since the CSP blocks fetching object URLs", () => {
  const compare = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(compare, /if \(src\.startsWith\("blob:"\)\) return null;/);
  assert.match(page, /fileUrl: galleryFileUrl\(clip\.id\)/);
  assert.match(page, /fileUrl: sourceId \? galleryFileUrl\(sourceId\) : null/);
});
