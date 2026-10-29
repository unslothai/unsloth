// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
registerBundlerResolver();

store.set(
  "unsloth_audio_music_v1",
  JSON.stringify({
    state: {
      mode: "sfx",
      song: { instrumental: true, durationS: 45, variations: 2 },
      sfx: { prompt: "rain", durationS: 8, variations: 1 },
      edit: {
        source: null,
        action: "inpaint",
        ranges: [],
        strength: 0.4,
        prompt: "",
      },
    },
    version: 1,
  }),
);

const { AUDIO_MUSIC_STORAGE_KEY, useAudioMusicStore } = await import(
  "../src/features/audio/stores/audio-music-store.ts"
);

test("drafts come back after a reload, with fields added since filled in", () => {
  assert.equal(AUDIO_MUSIC_STORAGE_KEY, "unsloth_audio_music_v1");
  const state = useAudioMusicStore.getState();
  assert.equal(state.mode, "sfx");
  assert.deepEqual(state.song, {
    instrumental: true,
    durationS: 45,
    variations: 2,
  });
  assert.equal(state.sfx.prompt, "rain");
  assert.equal(state.edit.action, "inpaint");
  assert.equal(state.edit.extendS, 15);
});

test("switching mode keeps every other mode's draft", () => {
  const { setMode, patchDraft } = useAudioMusicStore.getState();
  patchDraft("song", { variations: 3 });
  setMode("edit");
  patchDraft("edit", { prompt: "brighter" });
  setMode("song");
  const state = useAudioMusicStore.getState();
  assert.equal(state.song.variations, 3);
  assert.equal(state.sfx.prompt, "rain");
  assert.equal(state.edit.prompt, "brighter");
  const saved = JSON.parse(store.get("unsloth_audio_music_v1") ?? "{}");
  assert.equal(saved.state.edit.prompt, "brighter");
  assert.equal(saved.state.mode, "song");
});

test("Edit in Music opens Edit on the clip and drops ranges drawn on another clip", () => {
  const { patchDraft, pushClipToEdit } = useAudioMusicStore.getState();
  const first = {
    kind: "clip" as const,
    id: "a",
    name: "first",
    durationS: 20,
  };
  pushClipToEdit(first, "repaint");
  patchDraft("edit", { ranges: [{ start_s: 1, end_s: 2 }] });
  pushClipToEdit(first);
  let state = useAudioMusicStore.getState();
  assert.equal(state.mode, "edit");
  assert.equal(state.edit.action, "repaint");
  assert.equal(state.edit.ranges.length, 1);
  pushClipToEdit(
    { kind: "clip" as const, id: "b", name: "second", durationS: 9 },
    "extend",
  );
  state = useAudioMusicStore.getState();
  assert.equal(state.edit.source?.id, "b");
  assert.equal(state.edit.action, "extend");
  assert.deepEqual(state.edit.ranges, []);
  assert.equal(state.sfx.prompt, "rain");
});

test("an empty change prompt starts from the clip's own description", () => {
  const { patchDraft, pushClipToEdit } = useAudioMusicStore.getState();
  patchDraft("edit", { prompt: "" });
  pushClipToEdit({
    kind: "clip" as const,
    id: "c",
    name: "lofi beat",
    durationS: 30,
  });
  assert.equal(useAudioMusicStore.getState().edit.prompt, "lofi beat");
  patchDraft("edit", { prompt: "brighter" });
  pushClipToEdit({
    kind: "clip" as const,
    id: "d",
    name: "other",
    durationS: 30,
  });
  assert.equal(useAudioMusicStore.getState().edit.prompt, "brighter");
  useAudioMusicStore.getState().setLoadedEditActions(["inpaint"]);
  const saved = JSON.parse(store.get("unsloth_audio_music_v1") ?? "{}");
  assert.equal("loadedEditActions" in saved.state, false);
});
