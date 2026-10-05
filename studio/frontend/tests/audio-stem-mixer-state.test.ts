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

test("gain follows solo, mute and volume; solo beats mute", () => {
  const solo = (role: string): Action => ({ type: "toggleSolo", role });
  const mute = (role: string): Action => ({ type: "toggleMute", role });
  const vol = (role: string, volume: number): Action => ({
    type: "setVolume",
    role,
    volume,
  });
  const cases: [string, Action[], number[]][] = [
    ["initial", [], [1, 1, 1, 1]],
    ["one solo silences the rest", [solo("vocals")], [1, 0, 0, 0]],
    [
      "solo toggled off restores",
      [solo("vocals"), solo("vocals")],
      [1, 1, 1, 1],
    ],
    [
      "several solos play together",
      [solo("vocals"), solo("bass")],
      [1, 0, 1, 0],
    ],
    ["mute silences only that stem", [mute("drums")], [1, 0, 1, 1]],
    ["mute toggled off restores", [mute("drums"), mute("drums")], [1, 1, 1, 1]],
    ["solo beats mute", [mute("vocals"), solo("vocals")], [1, 0, 0, 0]],
    [
      "mute returns when the solo ends",
      [mute("vocals"), solo("vocals"), solo("vocals")],
      [0, 1, 1, 1],
    ],
    ["volume scales the gain", [vol("bass", 0.5)], [1, 1, 0.5, 1]],
    ["volume clamps high", [vol("bass", 3)], [1, 1, 1, 1]],
    ["volume clamps low", [vol("bass", -2)], [1, 1, 0, 1]],
    [
      "NaN volume is ignored",
      [vol("bass", 0.5), vol("bass", Number.NaN)],
      [1, 1, 0.5, 1],
    ],
    ["solo keeps the volume", [vol("bass", 0.5), solo("bass")], [0, 0, 0.5, 0]],
  ];
  for (const [name, actions, want] of cases) {
    assert.deepEqual(gains(run(...actions)), want, name);
  }
  assert.deepEqual(stemMix(INITIAL_STEM_MIXER_STATE, "vocals"), {
    volume: 1,
    muted: false,
    solo: false,
  });
  assert.equal(anySolo(run(solo("vocals"))), true);
  assert.equal(anySolo(run(solo("vocals"), solo("vocals"))), false);
});

test("reset returns to the initial levels; no-op actions keep the same object", () => {
  const busy = run(
    { type: "toggleSolo", role: "vocals" },
    { type: "toggleMute", role: "drums" },
    { type: "setVolume", role: "bass", volume: 0.2 },
  );
  const reset = stemMixerReducer(busy, { type: "reset" });
  assert.deepEqual(reset, INITIAL_STEM_MIXER_STATE);
  assert.equal(stemMixerReducer(reset, { type: "reset" }), reset);
  assert.equal(
    stemMixerReducer(busy, { type: "setVolume", role: "bass", volume: 0.2 }),
    busy,
  );
});

test("stem file names read '<title> - <Label>.wav' and are safe on disk", () => {
  const cases: [string, string, string][] = [
    ["My Song.mp3", "Vocals", "My Song - Vocals.wav"],
    ['a/b:c*?"<>|', "Drums", "a_b_c______ - Drums.wav"],
    ["  ", "Bass", "Separated track - Bass.wav"],
    ["trailing...", "Other", "trailing - Other.wav"],
    ["tab\there", "Piano", "tab_here - Piano.wav"],
  ];
  for (const [title, label, want] of cases) {
    assert.equal(stemFileName(title, label), want);
  }
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

test("zipStems loads each stem only when the archive reaches it", async () => {
  const order: string[] = [];
  const lazy = (name: string, body: string) => async () => {
    order.push(name);
    return new Blob([body]);
  };
  const zip = await zipStems([
    { name: "a.wav", blob: lazy("a", "first") },
    { name: "b.wav", blob: lazy("b", "second") },
  ]);
  assert.deepEqual(order, ["a", "b"]);
  const files = unzipSync(new Uint8Array(await zip.arrayBuffer()));
  assert.equal(strFromU8(files["a.wav"]), "first");
  assert.equal(strFromU8(files["b.wav"]), "second");
});
