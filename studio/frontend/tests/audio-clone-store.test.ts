// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
const KEY = "unsloth_audio_clone_v1";
store.set(KEY, JSON.stringify(SAVED));

const { useAudioCloneStore } = await import(
  "../src/features/audio/stores/audio-clone-store.ts"
);
const { toolValueKey } = await import("../src/features/audio/tools/select.ts");

test("the draft comes back after a reload", () => {
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
  const saved = JSON.parse(store.get(KEY) ?? "{}");
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

test("a transcript lands only on the clip it was made from, and stays with it", () => {
  const state = useAudioCloneStore.getState();
  const a = { kind: "input" as const, id: "a", name: "a.wav", durationS: 3 };
  const b = { kind: "input" as const, id: "b", name: "b.wav", durationS: 3 };
  state.setReference(a);
  state.setReferenceText("");
  useAudioCloneStore.getState().applyTranscript(b, "stale words");
  assert.equal(useAudioCloneStore.getState().referenceText, "");
  useAudioCloneStore.getState().applyTranscript(a, "clip a words");
  const after = useAudioCloneStore.getState();
  assert.equal(after.referenceText, "clip a words");
  assert.equal(after.reference?.transcript, "clip a words");
});

test("removing a clip drops its transcript but keeps typed words", () => {
  const a = {
    kind: "input" as const,
    id: "a2",
    name: "a.wav",
    durationS: 3,
    transcript: "clip a words",
  };
  const b = {
    kind: "input" as const,
    id: "b2",
    name: "b.wav",
    durationS: 3,
    transcript: "clip b words",
  };
  const store = () => useAudioCloneStore.getState();
  store().setReferenceText("");
  store().adoptReference(a);
  assert.equal(store().referenceText, "clip a words");
  store().adoptReference(null);
  store().adoptReference(b);
  assert.equal(store().referenceText, "clip b words");
  store().setReferenceText("my own words");
  store().adoptReference(null);
  store().adoptReference(a);
  assert.equal(store().referenceText, "my own words");
});

test("Add it again under Generate drops the expired clip's transcript too", () => {
  const a = {
    kind: "input" as const,
    id: "a3",
    name: "a.wav",
    durationS: 3,
    transcript: null,
  };
  const b = {
    kind: "input" as const,
    id: "b3",
    name: "b.wav",
    durationS: 3,
    transcript: null,
  };
  const store = () => useAudioCloneStore.getState();
  store().setReferenceText("");
  store().adoptReference(a);
  store().applyTranscript(a, "clip a words");
  // What the reference-expired action does before opening the file picker.
  const generation = readFileSync(
    new URL(
      "../src/features/audio/hooks/use-clone-generation.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const action = generation.slice(generation.indexOf('"reference-expired": ['));
  assert.match(
    action.slice(0, 400),
    /getState\(\)\.adoptReference\(null\);\s*referenceHandle\.current\?\.browse\(\);/,
  );
  store().adoptReference(null);
  store().adoptReference(b);
  assert.equal(store().referenceText, "");
});

test("a failed voice deletion puts back only that voice", () => {
  const src = readSrc("features/audio/stores/audio-voices-store.ts");
  const body = src.slice(src.indexOf("remove: async (id) => {"));
  const handler = body.slice(0, body.indexOf("\n  },"));
  assert.doesNotMatch(handler, /set\(\{ voices: before \}\)/);
  assert.match(handler, /voices\.splice\(Math\.min\(index, voices\.length\), 0, failed\)/);
});

test("a failed voice rename puts back only that voice", () => {
  const src = readSrc("features/audio/stores/audio-voices-store.ts");
  const body = src.slice(src.indexOf("rename: async (id, patch) => {"));
  const handler = body.slice(0, body.indexOf("\n  },"));
  assert.doesNotMatch(handler, /set\(\{ voices: before \}\)/);
  assert.match(handler, /voice\.id === id \? previous : voice/);
});

test("voice previews are bounded, stale ones dropped, and the menu follows disabled", () => {
  const src = readSrc("features/audio/components/voice-picker.tsx");
  assert.match(src, /const voiceUrls = new BlobUrlCache\(/);
  assert.match(src, /voiceUrls\.prune\(\[voice\.id\]\)/);
  assert.match(src, /if \(request !== previewRequest\.current\) return;\s*audio\.src = url;/);
  assert.match(src, /await remove\(voice\.id\);\s*voiceUrls\.delete\(voice\.id\);/);
  assert.match(src, /aria-label=\{`More for \$\{voice\.name\}`\}\s*disabled=\{disabled\}/);
});

test("a voice list fetched across a save, rename or delete is fetched again", () => {
  const src = readSrc("features/audio/stores/audio-voices-store.ts");
  assert.match(src, /const started = mutations;[\s\S]*?if \(started !== mutations\) \{\s*set\(\{ loading: false \}\);\s*return get\(\)\.refresh\(\);/);
  for (const call of ["createVoice\\(request\\)", "updateVoice\\(id, patch\\)", "deleteVoice\\(id\\)"]) {
    assert.match(src, new RegExp(`await ${call};\\s*mutations \\+= 1;`), call);
  }
});

test("a reference longer than the 30 s cut brings no transcript, since only 30 s is sent", () => {
  const store = () => useAudioCloneStore.getState();
  store().adoptReference(null);
  store().setReferenceText("");
  store().adoptReference({
    kind: "clip" as const,
    id: "long",
    name: "A long story.",
    durationS: 45,
    transcript: "A long story.",
  });
  assert.equal(store().referenceText, "");
  store().adoptReference({
    kind: "clip" as const,
    id: "short",
    name: "Hi.",
    durationS: 4,
    transcript: "Hi.",
  });
  assert.equal(store().referenceText, "Hi.");
});
