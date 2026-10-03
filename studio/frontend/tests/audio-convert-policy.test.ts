// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ConvertBlockerInput } from "../src/features/audio/convert-policy.ts";
import type { AudioConvertCaps } from "../src/features/chat/types/api.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

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

const ready: ConvertBlockerInput = {
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

test("blockers come in rail order and explain themselves", () => {
  assert.equal(convertBlocker(ready), null);
  const cases: [Partial<ConvertBlockerInput>, string, RegExp | string][] = [
    [
      { source: null, target: null, panelError: "x" },
      "source",
      CONVERT_INPUTS_MISSING,
    ],
    [{ source: null }, "source", CONVERT_SOURCE_MISSING],
    [{ target: null, panelError: "x" }, "target", CONVERT_TARGET_MISSING],
    [
      { caps: CAPS.vevo2, style: "target", sourceText: " ", panelError: "x" },
      "source-text",
      CONVERT_SOURCE_TEXT_MISSING,
    ],
    [{ panelError: "Pick one." }, "panel", "Pick one."],
    [
      { caps: CAPS.meanvc2, mode: "singing" },
      "mode",
      CONVERT_SINGING_UNSUPPORTED,
    ],
    [
      { sourceExpired: true, targetExpired: true },
      "source-expired",
      /recording and target voice are no longer on the server/,
    ],
    [{ sourceExpired: true }, "source-expired", /recording is no longer/],
    [{ sourceBusy: true }, "source", /Waiting for the recording/],
    [{ source: null, sourceError: "Too long." }, "source-error", "Too long."],
    [{ targetExpired: true }, "target-expired", /target voice is no longer/],
    [{ targetBusy: true }, "target", /Waiting for the target voice/],
    [{ target: null, targetError: "Bad file." }, "target-error", "Bad file."],
  ];
  for (const [patch, kind, reason] of cases) {
    const blocker = convertBlocker({ ...ready, ...patch });
    assert.equal(blocker?.kind, kind, JSON.stringify(patch));
    if (typeof reason === "string") assert.equal(blocker?.reason, reason);
    else assert.match(blocker?.reason ?? "", reason);
  }
});

test("RVC needs a built-in voice, not a target recording", () => {
  const rvc = { ...ready, caps: CAPS.rvc, target: null };
  assert.equal(convertBlocker(rvc), null);
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
  assert.equal(convertBlocker({ ...ready, style: "target" }), null);
});

test("pitch shows per family, mode and style", () => {
  const cases: [
    AudioConvertCaps | null,
    "speech" | "singing",
    "source" | "target",
    boolean,
    boolean,
  ][] = [
    [CAPS.rvc, "speech", "source", true, false],
    [CAPS.rvc, "singing", "source", false, false],
    [CAPS.seed_vc, "speech", "source", false, false],
    [CAPS.seed_vc, "singing", "source", true, true],
    [CAPS.vevo2, "speech", "source", true, true],
    [CAPS.vevo2, "speech", "target", false, false],
    [CAPS.vevo2, "singing", "target", true, true],
    [CAPS.meanvc2, "speech", "source", false, false],
    [CAPS.chatterbox, "speech", "source", false, false],
    [null, "speech", "source", false, false],
  ];
  for (const [caps, mode, style, show, auto] of cases) {
    assert.deepEqual(convertPitchSupport(caps, mode, style), { show, auto });
  }
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
  const notice = (
    modelName: string,
    loadedTask: string | null,
    nextTask: string,
    routeChange: boolean,
    family?: string,
  ) =>
    convertSwitchNotice({
      modelName,
      loadedTask,
      nextTask,
      routeChange,
      family,
    });
  assert.deepEqual(notice("Chatterbox", "clon", "vc", false, "chatterbox"), {
    before: "Reloads Chatterbox for Convert, about 2 s.",
    during: "Switching Chatterbox to Convert…",
  });
  assert.deepEqual(notice("Vevo2", "vc", "svc", false, "vevo2"), {
    before: "Reloads Vevo2 for singing, about 6 s.",
    during: "Switching Vevo2 to singing…",
  });
  assert.deepEqual(notice("Seed-VC", "vc", "vc", true, "seed_vc"), {
    before: "Reloads Seed-VC with the new engine, about 4–8 s.",
    during: "Switching Seed-VC to the new engine…",
  });
  assert.equal(notice("Seed-VC", "vc", "vc", false, "seed_vc"), null);
  assert.equal(notice("X", null, "vc", false), null);
  assert.equal(
    notice("Custom", "tts", "vc", false)?.before,
    "Reloads Custom for Convert.",
  );
});

test("changing the recording cancels a transcription of the previous one", () => {
  const hook = readSrc("features/audio/hooks/use-convert-generation.ts");
  assert.match(
    hook,
    /useEffect\(\(\) => cancelTranscribe\(\), \[sourceKey, cancelTranscribe\]\)/,
  );
});

test("Use again restores an upload or history clip target, not only a saved voice", () => {
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(
    page,
    /clip\.target_clip_id\s*\?\s*\{ kind: "clip" as const, id: clip\.target_clip_id \}/,
  );
  assert.match(
    page,
    /clip\.target_input_id\s*\?\s*\{ kind: "input" as const, id: clip\.target_input_id \}/,
  );
});

test("Convert transcribes its source as far as it converts", () => {
  const hook = readSrc("features/audio/hooks/use-convert-generation.ts");
  assert.match(hook, /purpose: "convert",/);
  const transcribe = readSrc(
    "features/audio/hooks/use-reference-transcribe.ts",
  );
  assert.match(
    transcribe,
    /\.\.\.\(language \? \{ language \} : \{\}\),\s*purpose,/,
  );
});

test("only the expired-upload 404 expires uploads, and a re-upload clears it", () => {
  const hook = readSrc("features/audio/hooks/use-convert-generation.ts");
  assert.match(
    hook,
    /error\.status === 404 &&\s*error\.message === REFERENCE_EXPIRED_MESSAGE/,
  );
  assert.match(
    hook,
    /for \(const id of \[sourceId, targetId\]\) \{\s*if \(id\) \{\s*next\.delete\(id\);/,
  );
});

// Chatterbox restarts under clon for Clone and vc for Convert; a stale task hides the reload.
const CLONE_REFRESHES_AFTER_RUN =
  /await showRunResult\(\{[\s\S]*?\}\);\s*(?:\/\/[^\n]*\n\s*)*await refreshStatus\(\);\s*\} catch \(error\)/;
const CLONE_REFRESHES_AFTER_STOP =
  /if \(!expired\) toast\.error\(message\);\s*\}\s*(?:\/\/[^\n]*\n\s*)*await refreshStatus\(\);\s*\} finally/;
const CONVERT_REFRESHES_AFTER_STOPPED_SWITCH =
  /\} else if \(switchNotice\) \{\s*(?:\/\/[^\n]*\n\s*)*await refreshStatus\(\);\s*\}\s*\} finally/;
const CONVERTS_ONLY =
  /const loadedConvertsOnly =\s*!!status\?\.audio_workflows\?\.includes\("convert"\) &&\s*!status\.audio_workflows\.includes\("clone"\);/;
const SPEAK_BLOCKER_OPENS_CONVERT =
  /ttsWorkflow === "speak" && loadedConvertsOnly\s*\?\s*\{\s*reason: "The loaded model converts recordings\.",[\s\S]*?label: "open Convert",\s*onClick: \(\) => transitionWorkflow\("convert"\)/;
const SPEAK_STATUS_LINE_CONVERTS =
  /ttsWorkflow === "speak" && loadedConvertsOnly\s*\?\s*"The loaded model converts recordings\. Pick a speech model\."/;

test("Convert's reload notice reads a status refreshed after a Clone run, a stopped Clone or a stopped switch", () => {
  const clone = readSrc("features/audio/hooks/use-clone-generation.ts");
  assert.match(clone, CLONE_REFRESHES_AFTER_RUN);
  assert.match(clone, CLONE_REFRESHES_AFTER_STOP);
  const convert = readSrc("features/audio/hooks/use-convert-generation.ts");
  assert.match(convert, CONVERT_REFRESHES_AFTER_STOPPED_SWITCH);
});

test("Speak sends a convert-only model to Convert, not to Clone", () => {
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, CONVERTS_ONLY);
  assert.match(page, SPEAK_BLOCKER_OPENS_CONVERT);
  assert.match(page, SPEAK_STATUS_LINE_CONVERTS);
});

test("the Convert source card states Convert's own length cap", () => {
  const page = readSrc("features/audio/pages/convert-page.tsx");
  assert.match(page, /maxSeconds=\{caps\?\.source_max_seconds \?\? 300\}/);
  const card = readSrc("features/audio/components/audio-source-input.tsx");
  assert.match(card, /durationS > maxSeconds/);
});
