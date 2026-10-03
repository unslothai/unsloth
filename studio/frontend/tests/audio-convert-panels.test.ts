// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AudioModelContext } from "../src/features/audio/tools/types.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { audioModelContextFor, collectToolRequest, panelApplies, panelValue } =
  await import("../src/features/audio/tools/select.ts");
const {
  CONVERT_PANEL_LOGIC,
  SEED_VC_ENGINES,
  chatterboxConvertLogic,
  rvcLogic,
  seedVcLogic,
  vevo2StyleLogic,
} = await import("../src/features/audio/tools/convert-panel-logic.ts");
const { CLONE_PANEL_LOGIC } = await import(
  "../src/features/audio/tools/panel-logic.ts"
);

const ctx = (
  overrides: Partial<AudioModelContext> = {},
): AudioModelContext => ({
  audioType: "audiocpp_tts",
  audioFamily: null,
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
  audioWorkflows: ["convert"],
  requiredInputs: [],
  referenceTextMode: null,
  ...overrides,
});

const shown = (
  family: string,
  workflow: "speak" | "clone" | "convert" = "convert",
) =>
  CONVERT_PANEL_LOGIC.filter((panel) =>
    panelApplies(panel, workflow, ctx({ audioFamily: family })),
  ).map((panel) => panel.id);

test("each converting family gets its one titled panel, only on Convert", () => {
  assert.deepEqual(shown("seed_vc"), ["seed-vc"]);
  assert.deepEqual(shown("rvc"), ["rvc"]);
  assert.deepEqual(shown("chatterbox"), ["chatterbox-convert"]);
  assert.deepEqual(shown("vevo2"), ["vevo2-style"]);
  assert.deepEqual(shown("meanvc2"), []);
  for (const family of ["seed_vc", "rvc", "chatterbox", "vevo2"]) {
    assert.deepEqual(shown(family, "clone"), [], family);
    assert.deepEqual(shown(family, "speak"), [], family);
  }
  const cloneOnConvert = CLONE_PANEL_LOGIC.filter((panel) =>
    panelApplies(panel, "convert", ctx({ audioFamily: "chatterbox" })),
  );
  assert.deepEqual(cloneOnConvert, []);
  assert.deepEqual(
    CONVERT_PANEL_LOGIC.map((panel) => panel.title),
    ["Seed-VC", "RVC", "Chatterbox", "Vevo2"],
  );
  const jsx = readSrc("features/audio/tools/convert-panels.tsx");
  for (const label of [
    "Sound like the target",
    "Keep words clear",
    "Index blend",
    "Protect consonants",
    "Volume envelope mix",
    "Keep source style",
    "Take target style",
    "Engine",
  ]) {
    assert.ok(jsx.includes(label), label);
  }
  assert.doesNotMatch(jsx, /label="[a-z]+_[a-z_]+"/);
});

test("claims name the runtime's own options, so Advanced leaves them out", () => {
  assert.deepEqual([...seedVcLogic.claims].sort(), [
    "auto_f0_adjust",
    "f0_condition",
    "inference_guidance_scale",
    "intelligibility_guidance_scale",
    "length_adjust",
    "num_inference_steps",
    "route",
    "semitone_shift",
    "similarity_guidance_scale",
    "voice_anonymization",
  ]);
  assert.deepEqual([...rvcLogic.claims].sort(), [
    "retrieval_blend",
    "rms_mix_rate",
    "semitone_shift",
    "unvoiced_protection",
    "voice_id",
  ]);
  assert.deepEqual([...chatterboxConvertLogic.claims].sort(), [
    "num_inference_steps",
    "s3gen_cfg_rate",
  ]);
  assert.deepEqual(vevo2StyleLogic.claims, []);
});

test("Seed-VC sends only what the chosen engine reads", () => {
  const value = seedVcLogic.initial([]);
  assert.equal(value.engine, "v2_vc");
  const speech = ctx({ audioFamily: "seed_vc", convertMode: "speech" });
  const v2 = seedVcLogic.toRequest(value, speech);
  assert.deepEqual(v2.convert, { route: "v2_vc" });
  assert.deepEqual(Object.keys(v2.options ?? {}).sort(), [
    "intelligibility_guidance_scale",
    "length_adjust",
    "num_inference_steps",
    "route",
    "similarity_guidance_scale",
    "voice_anonymization",
  ]);
  for (const engine of ["v1_whisper_bigvgan_vc", "v1_xlsr_hift_vc"] as const) {
    const v1 = seedVcLogic.toRequest({ ...value, engine }, speech);
    assert.deepEqual(v1.convert, { route: engine });
    assert.deepEqual(Object.keys(v1.options ?? {}).sort(), [
      "inference_guidance_scale",
      "length_adjust",
      "num_inference_steps",
      "route",
    ]);
  }
});

test("Seed-VC singing runs the singing engine, whatever engine speech had", () => {
  const value = {
    ...seedVcLogic.initial([]),
    engine: "v1_xlsr_hift_vc" as const,
  };
  const singing = seedVcLogic.toRequest(
    value,
    ctx({ audioFamily: "seed_vc", convertMode: "singing" }),
  );
  assert.deepEqual(singing.convert, { route: "v1_svc" });
  assert.equal(singing.options?.route, "v1_svc");
  assert.equal("similarity_guidance_scale" in (singing.options ?? {}), false);
  assert.equal("inference_guidance_scale" in (singing.options ?? {}), true);
  assert.deepEqual(
    SEED_VC_ENGINES.map((engine) => engine.label),
    ["V2", "V1 Whisper", "V1 XLSR"],
  );
});

test("Seed-VC clamps out-of-range values and takes defaults from the schema", () => {
  const value = seedVcLogic.initial([
    { name: "num_inference_steps", type: "int", default: 25 },
  ]);
  assert.equal(value.steps, 25);
  const patch = seedVcLogic.toRequest(
    { ...value, length: 9, steps: 0, similarity: -1 },
    ctx({ convertMode: "speech" }),
  );
  assert.equal(patch.options?.length_adjust, 2);
  assert.equal(patch.options?.num_inference_steps, 1);
  assert.equal(patch.options?.similarity_guidance_scale, 0);
});

test("RVC and Chatterbox send their settings within range", () => {
  assert.deepEqual(rvcLogic.toRequest({ blend: 2, protect: 0.9, rms: -1 }), {
    options: { retrieval_blend: 1, unvoiced_protection: 0.5, rms_mix_rate: 0 },
  });
  assert.deepEqual(rvcLogic.initial([]), {
    blend: 0,
    protect: 0.33,
    rms: 0.25,
  });
  assert.deepEqual(chatterboxConvertLogic.initial([]), {
    guidance: 0.7,
    steps: 10,
  });
  assert.deepEqual(
    chatterboxConvertLogic.toRequest({ guidance: 0.5, steps: 12.4 }),
    {
      options: { s3gen_cfg_rate: 0.5, num_inference_steps: 12 },
    },
  );
});

test("Vevo2's style reaches the convert settings; singing always keeps the source's", () => {
  const speech = ctx({ audioFamily: "vevo2", convertMode: "speech" });
  const singing = ctx({ audioFamily: "vevo2", convertMode: "singing" });
  for (const [style, mode, sent] of [
    ["target", speech, "target"],
    ["source", speech, "source"],
    ["target", singing, "source"],
  ] as const) {
    assert.deepEqual(vevo2StyleLogic.toRequest({ style }, mode), {
      convert: { style: sent },
    });
  }
});

test("collectToolRequest merges each panel's convert part with its options", () => {
  const panels = [seedVcLogic, vevo2StyleLogic].map((logic) => ({
    ...logic,
    Component: () => null,
  }));
  const { patch, error } = collectToolRequest(
    panels,
    { "vevo2-style": { style: "target" } },
    { text: "" },
    ctx({ convertMode: "speech" }),
  );
  assert.equal(error, null);
  assert.deepEqual(patch.convert, { route: "v2_vc", style: "target" });
  assert.equal(patch.options?.route, "v2_vc");
  assert.equal(
    panelValue(seedVcLogic, { "seed-vc": { engine: "v1_xlsr_hift_vc" } }, [])
      .steps,
    30,
  );
});

test("the tool context reads the Convert caps from the status", () => {
  const context = audioModelContextFor(
    {
      audio_family: "vevo2",
      audio_workflows: ["clone", "convert"],
      audio_convert: {
        modes: ["speech", "singing"],
        target: "audio",
        builtin_voices: [],
        pitch: { speech: { auto: true }, singing: { auto: true } },
        style: true,
        route_reloads: false,
        source_max_seconds: 300,
      },
    },
    {
      musicGeneration: false,
      cudaMusicGeneration: false,
      musicNeedsDescription: false,
    },
  );
  assert.deepEqual(context.convert?.modes, ["speech", "singing"]);
  assert.equal(context.convert?.style, true);
  const none = audioModelContextFor(
    { audio_family: "kokoro", audio_convert: null },
    {
      musicGeneration: false,
      cudaMusicGeneration: false,
      musicNeedsDescription: false,
    },
  );
  assert.equal(none.convert, null);
});

test("the registry lists the Convert panels after Clone's", () => {
  const registry = readSrc("features/audio/tools/registry.tsx");
  assert.match(
    registry,
    /\.\.\.CLONE_TOOL_PANELS,\s*\.\.\.CONVERT_TOOL_PANELS,\s*\];/,
  );
});
