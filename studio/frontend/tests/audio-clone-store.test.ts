// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
registerBundlerResolver();

const SAVED = {
  state: {
    reference: { kind: "voice", id: "v1", name: "Narrator", durationS: 6 },
    referenceText: "Hello there.",
    text: "Read this.",
    language: "English",
    toolValues: { "m:clone:qwen3-timbre": { timbreOnly: true } },
  },
  version: 1,
};
store.set("unsloth_audio_clone_v1", JSON.stringify(SAVED));

const { AUDIO_CLONE_STORAGE_KEY, useAudioCloneStore } = await import(
  "../src/features/audio/stores/audio-clone-store.ts"
);
const { toolValueKey } = await import("../src/features/audio/tools/select.ts");

test("the draft comes back after a reload", () => {
  assert.equal(AUDIO_CLONE_STORAGE_KEY, "unsloth_audio_clone_v1");
  const state = useAudioCloneStore.getState();
  assert.deepEqual(state.reference, SAVED.state.reference);
  assert.equal(state.referenceText, "Hello there.");
  assert.equal(state.text, "Read this.");
  assert.equal(state.language, "English");
  assert.deepEqual(state.toolValues["m:clone:qwen3-timbre"], {
    timbreOnly: true,
  });
});

test("edits are written through, actions are not", () => {
  const state = useAudioCloneStore.getState();
  state.setText("New line.");
  state.setReference({
    kind: "input",
    id: "abc",
    name: "take.webm",
    durationS: 4,
    expiresAt: "2026-10-04T00:00:00Z",
  });
  const saved = JSON.parse(store.get(AUDIO_CLONE_STORAGE_KEY) ?? "{}");
  assert.equal(saved.state.text, "New line.");
  assert.equal(saved.state.reference.id, "abc");
  assert.deepEqual(Object.keys(saved.state).sort(), [
    "language",
    "reference",
    "referenceText",
    "text",
    "toolValues",
  ]);
});

test("tool values are kept per model, page and panel", () => {
  const state = useAudioCloneStore.getState();
  state.setToolValue(toolValueKey("a/one", "clone", "index-emotion"), {
    mode: "mixer",
  });
  state.setToolValue(toolValueKey("b/two", "clone", "index-emotion"), {
    mode: "text",
  });
  state.setToolValue(toolValueKey("a/one", "speak", "speak-voice"), {
    source: "saved",
    voiceId: "v",
  });
  const values = useAudioCloneStore.getState().toolValues;
  assert.deepEqual(values["a/one:clone:index-emotion"], { mode: "mixer" });
  assert.deepEqual(values["b/two:clone:index-emotion"], { mode: "text" });
  assert.deepEqual(values["a/one:speak:speak-voice"], {
    source: "saved",
    voiceId: "v",
  });
});

test("tool values stay bounded, oldest first", () => {
  const state = useAudioCloneStore.getState();
  for (let i = 0; i < 260; i += 1) state.setToolValue(`m${i}:clone:p`, i);
  const keys = Object.keys(useAudioCloneStore.getState().toolValues);
  assert.equal(keys.length, 200);
  assert.equal(keys.at(-1), "m259:clone:p");
  assert.ok(!keys.includes("m0:clone:p"));
});

test("another reference replaces the transcript; a running transcription is cancelled", () => {
  const page = readSrc("features/audio/pages/clone-page.tsx");
  assert.match(
    page,
    /if \(previous\?\.kind !== next\.kind \|\| previous\?\.id !== next\.id\) \{\s*store\.setReferenceText\(next\.transcript \?\? ""\);/,
  );
  const hook = readSrc("features/audio/hooks/use-clone-generation.ts");
  assert.match(
    hook,
    /useEffect\(\(\) => cancelTranscribe\(\), \[referenceKey, cancelTranscribe\]\)/,
  );
});
