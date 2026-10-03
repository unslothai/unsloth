// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AudioConvertCaps } from "../src/features/chat/types/api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  CONVERT_BUILTIN_MISSING,
  CONVERT_MODEL_ORDER,
  CONVERT_SINGING_UNSUPPORTED,
  CONVERT_INPUTS_MISSING,
  CONVERT_SOURCE_MISSING,
  CONVERT_SOURCE_TEXT_MISSING,
  CONVERT_TARGET_MISSING,
  convertBlocker,
  convertCaps,
  convertPitchSupport,
  convertServerTask,
  convertSwitchNotice,
} = await import("../src/features/audio/convert-policy.ts");
const { AUDIO_CPP_REPO, audioCppModelFor, audioCppWorkflowsFor } = await import(
  "../src/features/audio/audio-cpp-catalog.ts"
);

const base: AudioConvertCaps = {
  modes: ["speech"],
  target: "audio",
  builtin_voices: [],
  pitch: {},
  style: false,
  route_reloads: false,
  source_max_seconds: 300,
};

// What the backend reports per family (PLAN-ADDENDUM pitch matrix).
const CAPS: Record<string, AudioConvertCaps> = {
  rvc: {
    ...base,
    target: "builtin",
    builtin_voices: [
      { id: "default", label: "Default" },
      { id: "manthos", label: "Manthos" },
    ],
    pitch: { speech: { auto: false } },
  },
  seed_vc: {
    ...base,
    modes: ["speech", "singing"],
    pitch: { singing: { auto: true } },
    route_reloads: true,
  },
  meanvc2: base,
  chatterbox: base,
  vevo2: {
    ...base,
    modes: ["speech", "singing"],
    pitch: { speech: { auto: true }, singing: { auto: true } },
    style: true,
  },
};

const recording = {
  kind: "clip" as const,
  id: "c1",
  name: "Take 1",
  durationS: 8,
};
const voice = {
  kind: "voice" as const,
  id: "v1",
  name: "Narrator",
  durationS: 6,
};

const ready = {
  source: recording,
  sourceBusy: false,
  sourceExpired: false,
  sourceError: null,
  target: voice,
  targetBusy: false,
  targetExpired: false,
  targetError: null,
  builtinVoice: "default",
  caps: CAPS.seed_vc,
  mode: "speech" as const,
  style: "source" as const,
  sourceText: "",
  panelError: null,
};

test("the picker order names only seeded convert models", () => {
  for (const name of CONVERT_MODEL_ORDER) {
    const model = audioCppModelFor(`${AUDIO_CPP_REPO}/${name}`);
    assert.ok(model, name);
    assert.ok(audioCppWorkflowsFor(model).includes("convert"), name);
  }
  assert.equal(CONVERT_MODEL_ORDER[0], "SeedVC-MLX-GGUF");
});

test("caps come from the tool context; a model without modes cannot convert", () => {
  assert.equal(convertCaps({ convert: null }), null);
  assert.equal(convertCaps({}), null);
  assert.equal(convertCaps({ convert: { ...base, modes: [] } }), null);
  assert.equal(convertCaps({ convert: CAPS.rvc }), CAPS.rvc);
});

test("blockers come in rail order: source, target, mode, transcript, panel", () => {
  assert.equal(convertBlocker(ready), null);
  assert.deepEqual(
    convertBlocker({ ...ready, source: null, target: null, panelError: "x" }),
    { kind: "source", reason: CONVERT_INPUTS_MISSING },
  );
  assert.deepEqual(convertBlocker({ ...ready, source: null }), {
    kind: "source",
    reason: CONVERT_SOURCE_MISSING,
  });
  assert.deepEqual(
    convertBlocker({ ...ready, target: null, panelError: "x" }),
    {
      kind: "target",
      reason: CONVERT_TARGET_MISSING,
    },
  );
  assert.deepEqual(
    convertBlocker({
      ...ready,
      caps: CAPS.vevo2,
      style: "target",
      sourceText: " ",
      panelError: "x",
    }),
    { kind: "source-text", reason: CONVERT_SOURCE_TEXT_MISSING },
  );
  assert.deepEqual(convertBlocker({ ...ready, panelError: "Pick one." }), {
    kind: "panel",
    reason: "Pick one.",
  });
  // Singing on a speech-only model.
  assert.deepEqual(
    convertBlocker({ ...ready, caps: CAPS.meanvc2, mode: "singing" }),
    { kind: "mode", reason: CONVERT_SINGING_UNSUPPORTED },
  );
});

test("expired, busy and failed inputs explain themselves", () => {
  assert.match(
    convertBlocker({ ...ready, sourceExpired: true, targetExpired: true })
      ?.reason ?? "",
    /recording and target voice are no longer on the server/,
  );
  assert.equal(
    convertBlocker({ ...ready, sourceExpired: true })?.kind,
    "source-expired",
  );
  assert.match(
    convertBlocker({ ...ready, sourceBusy: true })?.reason ?? "",
    /Waiting for the recording/,
  );
  assert.deepEqual(
    convertBlocker({ ...ready, source: null, sourceError: "Too long." }),
    { kind: "source-error", reason: "Too long." },
  );
  assert.equal(
    convertBlocker({ ...ready, targetExpired: true })?.kind,
    "target-expired",
  );
  assert.match(
    convertBlocker({ ...ready, targetBusy: true })?.reason ?? "",
    /Waiting for the target voice/,
  );
  assert.deepEqual(
    convertBlocker({ ...ready, target: null, targetError: "Bad file." }),
    { kind: "target-error", reason: "Bad file." },
  );
});

test("RVC needs a built-in voice, not a target recording", () => {
  const rvc = { ...ready, caps: CAPS.rvc, target: null };
  assert.equal(convertBlocker(rvc), null);
  // An expired target upload left over from another model does not block RVC.
  assert.equal(convertBlocker({ ...rvc, targetExpired: true }), null);
  assert.deepEqual(convertBlocker({ ...rvc, builtinVoice: "" }), {
    kind: "target",
    reason: CONVERT_BUILTIN_MISSING,
  });
  assert.deepEqual(convertBlocker({ ...rvc, builtinVoice: "gone" }), {
    kind: "target",
    reason: CONVERT_BUILTIN_MISSING,
  });
});

test("Take target style needs the transcript only in Speech", () => {
  const vevo = { ...ready, caps: CAPS.vevo2, style: "target" as const };
  assert.equal(convertBlocker(vevo)?.kind, "source-text");
  assert.equal(convertBlocker({ ...vevo, sourceText: "Hello." }), null);
  assert.equal(convertBlocker({ ...vevo, mode: "singing" }), null);
  // Models without the style choice never ask for it.
  assert.equal(convertBlocker({ ...ready, style: "target" }), null);
});

test("pitch shows per family, mode and style", () => {
  const hidden = { show: false, auto: false };
  assert.deepEqual(convertPitchSupport(CAPS.rvc, "speech", "source"), {
    show: true,
    auto: false,
  });
  assert.deepEqual(
    convertPitchSupport(CAPS.seed_vc, "speech", "source"),
    hidden,
  );
  assert.deepEqual(convertPitchSupport(CAPS.seed_vc, "singing", "source"), {
    show: true,
    auto: true,
  });
  assert.deepEqual(convertPitchSupport(CAPS.vevo2, "speech", "source"), {
    show: true,
    auto: true,
  });
  assert.deepEqual(convertPitchSupport(CAPS.vevo2, "speech", "target"), hidden);
  // Singing forces Keep source style, so pitch comes back.
  assert.deepEqual(convertPitchSupport(CAPS.vevo2, "singing", "target"), {
    show: true,
    auto: true,
  });
  assert.deepEqual(
    convertPitchSupport(CAPS.meanvc2, "speech", "source"),
    hidden,
  );
  assert.deepEqual(
    convertPitchSupport(CAPS.chatterbox, "speech", "source"),
    hidden,
  );
  assert.deepEqual(convertPitchSupport(null, "speech", "source"), hidden);
  // A mode the model does not offer has no pitch either.
  assert.deepEqual(convertPitchSupport(CAPS.rvc, "singing", "source"), hidden);
});

test("the server task per mode comes from the workflow tasks", () => {
  const tasks = { clone: "tts", convert: "vc", "convert:singing": "svc" };
  assert.equal(convertServerTask(CAPS.vevo2, tasks, "speech"), "vc");
  assert.equal(convertServerTask(CAPS.vevo2, tasks, "singing"), "svc");
  assert.equal(
    convertServerTask(CAPS.vevo2, { "convert:speech": "vc" }, "speech"),
    "vc",
  );
  assert.equal(convertServerTask(CAPS.meanvc2, tasks, "singing"), null);
  assert.equal(convertServerTask(null, tasks, "speech"), null);
  assert.equal(convertServerTask(CAPS.vevo2, null, "speech"), null);
});

test("a run that reloads the model says so before and during; a repeat run does not", () => {
  assert.deepEqual(
    convertSwitchNotice({
      modelName: "Chatterbox",
      loadedTask: "clon",
      nextTask: "vc",
      routeChange: false,
      family: "chatterbox",
    }),
    {
      before: "Reloads Chatterbox for Convert, about 2 s.",
      during: "Switching Chatterbox to Convert…",
    },
  );
  assert.deepEqual(
    convertSwitchNotice({
      modelName: "Vevo2",
      loadedTask: "vc",
      nextTask: "svc",
      routeChange: false,
      family: "vevo2",
    }),
    {
      before: "Reloads Vevo2 for singing, about 6 s.",
      during: "Switching Vevo2 to singing…",
    },
  );
  assert.deepEqual(
    convertSwitchNotice({
      modelName: "Seed-VC",
      loadedTask: "vc",
      nextTask: "vc",
      routeChange: true,
      family: "seed_vc",
    }),
    {
      before: "Reloads Seed-VC with the new engine, about 4–8 s.",
      during: "Switching Seed-VC to the new engine…",
    },
  );
  // Same task, same engine: no notice.
  assert.equal(
    convertSwitchNotice({
      modelName: "Seed-VC",
      loadedTask: "vc",
      nextTask: "vc",
      routeChange: false,
      family: "seed_vc",
    }),
    null,
  );
  // Unknown on either side: nothing to promise.
  assert.equal(
    convertSwitchNotice({
      modelName: "X",
      loadedTask: null,
      nextTask: "vc",
      routeChange: false,
    }),
    null,
  );
  // A family without a measured time gets no estimate.
  assert.equal(
    convertSwitchNotice({
      modelName: "Custom",
      loadedTask: "tts",
      nextTask: "vc",
      routeChange: false,
    })?.before,
    "Reloads Custom for Convert.",
  );
});
