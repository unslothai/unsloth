// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  aceStepMusicalLogic,
  stableAudioSamplerLogic,
  yueCompositionLogic,
  stepsRange,
} = await import("../src/features/audio/tools/music-panel-logic.ts");
const { audioModelContextFor, legacyMusicDescription, panelApplies } =
  await import("../src/features/audio/tools/select.ts");

const registry = readSrc("features/audio/tools/registry.tsx");
const panels = readSrc("features/audio/tools/music-panels.tsx");

const MUSIC_STATUS = { modes: [{ id: "song" }] };

function ctx(family: string, audioMusic: unknown = MUSIC_STATUS) {
  return audioModelContextFor(
    {
      audio_type: "audiocpp_music",
      audio_family: family,
      audio_music: audioMusic,
    },
    {
      musicGeneration: true,
      cudaMusicGeneration: false,
      musicNeedsDescription: false,
    },
  );
}

test("each music panel shows on Music for its own family only", () => {
  for (const [panel, family] of [
    [aceStepMusicalLogic, "ace_step"],
    [yueCompositionLogic, "yue2"],
    [stableAudioSamplerLogic, "stable_audio"],
  ] as const) {
    assert.equal(panelApplies(panel, "music", ctx(family)), true, panel.id);
    assert.equal(panelApplies(panel, "speak", ctx(family)), false, panel.id);
    assert.equal(
      panelApplies(panel, "music", ctx("heartmula")),
      false,
      panel.id,
    );
  }
  assert.match(registry, /\.\.\.MUSIC_TOOL_PANELS/);
  assert.match(
    panels,
    /aceStepMusicalPanel,\s*yueCompositionPanel,\s*stableAudioSamplerPanel/,
  );
});

test("the status's music block reaches the tool context", () => {
  assert.equal(ctx("ace_step").audioMusic, true);
  assert.equal(ctx("ace_step", null).audioMusic, false);
  assert.equal(ctx("ace_step", { modes: [] }).audioMusic, false);
});

test("the old Music description shows only for a loaded music model without studio modes", () => {
  const described = (status: Record<string, unknown> | null) =>
    legacyMusicDescription(
      audioModelContextFor(status, {
        musicGeneration: true,
        cudaMusicGeneration: false,
        musicNeedsDescription: false,
      }),
    );
  assert.equal(described(null), false);
  assert.equal(described({ audio_type: "audiocpp_tts" }), false);
  assert.equal(described({ audio_type: "minimax_music3" }), true);
  assert.equal(
    described({ audio_type: "audiocpp_music", audio_music: MUSIC_STATUS }),
    false,
  );
  assert.match(registry, /kind !== "music" \|\| legacyMusicDescription\(ctx\)/);
});

test("ACE-Step sends only what the user set, in the runtime's spelling", () => {
  const value = aceStepMusicalLogic.initial([]);
  assert.deepEqual(aceStepMusicalLogic.toRequest(value), {});
  assert.deepEqual(
    aceStepMusicalLogic.toRequest({
      bpm: 999,
      keyscale: "A minor",
      timesignature: "3",
      avoid: "  shouting ",
      sampler: "heun",
    }),
    {
      options: {
        bpm: 300,
        keyscale: "A minor",
        timesignature: "3",
        negative_prompt: "shouting",
        sampler_mode: "heun",
      },
    },
  );
  assert.deepEqual(
    aceStepMusicalLogic.toRequest({
      ...value,
      keyscale: "H major",
      timesignature: "4/4",
    }),
    {},
  );
  assert.deepEqual(aceStepMusicalLogic.claims, [
    "bpm",
    "keyscale",
    "timesignature",
    "negative_prompt",
    "sampler_mode",
  ]);
});

test("YuE2 planning starts from the spec default and sends cot", () => {
  assert.deepEqual(yueCompositionLogic.initial([]), { cot: "full" });
  assert.deepEqual(
    yueCompositionLogic.initial([
      { name: "cot", type: "enum", default: "off" },
    ]),
    { cot: "off" },
  );
  assert.deepEqual(yueCompositionLogic.toRequest({ cot: "melody" }), {
    options: { cot: "melody" },
  });
});

test("Stable Audio's sampler sends pingpong or euler and whole steps", () => {
  assert.deepEqual(
    stableAudioSamplerLogic.toRequest({ sampler: "", steps: null }),
    {},
  );
  assert.deepEqual(
    stableAudioSamplerLogic.toRequest({ sampler: "dpm", steps: 0 }),
    {},
  );
  assert.deepEqual(
    stableAudioSamplerLogic.toRequest({ sampler: "euler", steps: 12.4 }),
    {
      options: { sampler: "euler", num_inference_steps: 12 },
    },
  );
  assert.deepEqual(
    stepsRange([
      { name: "num_inference_steps", type: "int", min: 1, max: 50, default: 8 },
    ]),
    { min: 1, max: 50, default: 8 },
  );
});

test("the Music studio's Advanced hides speech sampling it never sends", () => {
  const page = readSrc("features/audio/pages/music-page.tsx");
  assert.match(
    page,
    /musicGeneration=\{false\}\s*samplingControls=\{false\}\s*inputs=\{\s*<MusicStudioInputs/,
  );
});

test("Music's source card releases the mic when Audio is hidden, as Clone's does", () => {
  const host = readSrc("features/audio/audio-page.tsx");
  assert.match(host, /<AudioActiveProvider value=\{active\}>\s*<MusicRail/);
  assert.match(host, /<AudioActiveProvider value=\{active\}>\s*<CloneRail/);
});
