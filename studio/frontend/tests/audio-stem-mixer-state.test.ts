// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const {
  INITIAL_STEM_MIXER_STATE,
  anySolo,
  effectiveGain,
  stemMix,
  stemMixerReducer,
} = await import("../src/features/audio/components/stem-mixer-state.ts");
const {
  sanitizeFileNamePart,
  stemFileName,
  stemZipName,
  uniqueStemNames,
  zipStems,
} = await import("../src/features/audio/components/stem-zip.ts");
const { unzipSync, strFromU8 } = await import("fflate");

type State = typeof INITIAL_STEM_MIXER_STATE;
type Action = Parameters<typeof stemMixerReducer>[1];
const run = (...actions: Action[]): State =>
  actions.reduce(stemMixerReducer, INITIAL_STEM_MIXER_STATE);
const ROLES = ["vocals", "drums", "bass", "other"];
const gains = (state: State) => ROLES.map((role) => effectiveGain(state, role));

test("every stem starts at full level, unmuted, unsoloed", () => {
  assert.deepEqual(stemMix(INITIAL_STEM_MIXER_STATE, "vocals"), {
    volume: 1,
    muted: false,
    solo: false,
  });
  assert.deepEqual(gains(INITIAL_STEM_MIXER_STATE), [1, 1, 1, 1]);
});

test("one solo silences every other stem; toggling it again restores them", () => {
  const solo = run({ type: "toggleSolo", role: "vocals" });
  assert.equal(anySolo(solo), true);
  assert.deepEqual(gains(solo), [1, 0, 0, 0]);
  const off = stemMixerReducer(solo, { type: "toggleSolo", role: "vocals" });
  assert.equal(anySolo(off), false);
  assert.deepEqual(gains(off), [1, 1, 1, 1]);
});

test("several solos play together", () => {
  const state = run(
    { type: "toggleSolo", role: "vocals" },
    { type: "toggleSolo", role: "bass" },
  );
  assert.deepEqual(gains(state), [1, 0, 1, 0]);
});

test("mute silences only that stem", () => {
  const state = run({ type: "toggleMute", role: "drums" });
  assert.deepEqual(gains(state), [1, 0, 1, 1]);
  assert.deepEqual(
    gains(stemMixerReducer(state, { type: "toggleMute", role: "drums" })),
    [1, 1, 1, 1],
  );
});

test("solo beats mute, and mute comes back when the solo ends", () => {
  const state = run(
    { type: "toggleMute", role: "vocals" },
    { type: "toggleSolo", role: "vocals" },
  );
  assert.equal(effectiveGain(state, "vocals"), 1);
  assert.equal(effectiveGain(state, "drums"), 0);
  const unsolo = stemMixerReducer(state, {
    type: "toggleSolo",
    role: "vocals",
  });
  assert.deepEqual(gains(unsolo), [0, 1, 1, 1]);
});

test("volume scales the gain and is clamped to 0..1", () => {
  const half = run({ type: "setVolume", role: "bass", volume: 0.5 });
  assert.equal(effectiveGain(half, "bass"), 0.5);
  assert.equal(
    stemMix(run({ type: "setVolume", role: "bass", volume: 3 }), "bass").volume,
    1,
  );
  assert.equal(
    stemMix(run({ type: "setVolume", role: "bass", volume: -2 }), "bass")
      .volume,
    0,
  );
  assert.equal(
    stemMix(
      stemMixerReducer(half, {
        type: "setVolume",
        role: "bass",
        volume: Number.NaN,
      }),
      "bass",
    ).volume,
    0.5,
  );
  const soloHalf = stemMixerReducer(half, { type: "toggleSolo", role: "bass" });
  assert.deepEqual(gains(soloHalf), [0, 0, 0.5, 0]);
});

test("reset returns to the initial levels; no-op actions keep the same object", () => {
  const busy = run(
    { type: "toggleSolo", role: "vocals" },
    { type: "toggleMute", role: "drums" },
    { type: "setVolume", role: "bass", volume: 0.2 },
  );
  const reset = stemMixerReducer(busy, { type: "reset" });
  assert.deepEqual(reset, INITIAL_STEM_MIXER_STATE);
  assert.deepEqual(gains(reset), [1, 1, 1, 1]);
  assert.equal(stemMixerReducer(reset, { type: "reset" }), reset);
  const same = stemMixerReducer(busy, {
    type: "setVolume",
    role: "bass",
    volume: 0.2,
  });
  assert.equal(same, busy);
});

test("stem file names read '<title> - <Label>.wav' and are safe on disk", () => {
  assert.equal(stemFileName("My Song.mp3", "Vocals"), "My Song - Vocals.wav");
  assert.equal(stemFileName('a/b:c*?"<>|', "Drums"), "a_b_c______ - Drums.wav");
  assert.equal(stemFileName("  ", "Bass"), "Separated track - Bass.wav");
  assert.equal(stemFileName("trailing...", "Other"), "trailing - Other.wav");
  assert.equal(stemFileName("tab\there", "Piano"), "tab_here - Piano.wav");
  const long = stemFileName("x".repeat(500), "Instrumental");
  assert.ok(long.endsWith(" - Instrumental.wav"));
  assert.ok(long.length <= 120 + " - Instrumental.wav".length);
  assert.equal(stemZipName("My Song.wav"), "My Song - stems.zip");
  assert.equal(sanitizeFileNamePart("", "fallback"), "fallback");
});

test("repeated names get a counter instead of overwriting", () => {
  assert.deepEqual(
    uniqueStemNames(["a - X.wav", "a - X.wav", "A - x.wav", "b.wav"]),
    ["a - X.wav", "a - X (2).wav", "A - x (3).wav", "b.wav"],
  );
});

test("zipStems stores every file under its name", async () => {
  const zip = await zipStems([
    { name: "t - Vocals.wav", blob: new Blob(["vocals-bytes"]) },
    { name: "t - Vocals.wav", blob: new Blob(["second"]) },
    { name: "t - Drums.wav", blob: new Blob([new Uint8Array(0)]) },
  ]);
  assert.equal(zip.type, "application/zip");
  const files = unzipSync(new Uint8Array(await zip.arrayBuffer()));
  assert.deepEqual(Object.keys(files), [
    "t - Vocals.wav",
    "t - Vocals (2).wav",
    "t - Drums.wav",
  ]);
  assert.equal(strFromU8(files["t - Vocals.wav"]), "vocals-bytes");
  assert.equal(strFromU8(files["t - Vocals (2).wav"]), "second");
  assert.equal(files["t - Drums.wav"].length, 0);
});
