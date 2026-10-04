// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AudioModelContext } from "../src/features/audio/tools/types.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  audioModelContextFor,
  collectToolRequest,
  panelApplies,
  panelValue,
  toolValueKey,
} = await import("../src/features/audio/tools/select.ts");
const {
  CLONE_PANEL_LOGIC,
  COSYVOICE_INSTRUCTION_MISSING,
  EMOTION_AUDIO_MISSING,
  SAVED_VOICE_DELETED,
  SAVED_VOICE_MISSING,
  chatterboxExpressivenessLogic,
  cosyVoiceModeLogic,
  emotionSourceProblem,
  f5SpeedDialectLogic,
  indexTts2EmotionLogic,
  qwen3TimbreLogic,
  speakVoiceLogic,
  vibeVoiceDialogueLogic,
} = await import("../src/features/audio/tools/panel-logic.ts");
const {
  cloneBlocker,
  emotionVectorString,
  formatVibeVoiceScript,
  referenceTextField,
} = await import("../src/features/audio/clone-policy.ts");

const ctx = (
  overrides: Partial<AudioModelContext> = {},
): AudioModelContext => ({
  audioType: "audiocpp_tts",
  audioFamily: null,
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
  audioWorkflows: ["clone"],
  requiredInputs: [],
  referenceTextMode: null,
  ...overrides,
});

const shown = (
  family: string,
  workflow: "speak" | "clone" = "clone",
  extra = {},
) =>
  CLONE_PANEL_LOGIC.filter((panel) =>
    panelApplies(panel, workflow, ctx({ audioFamily: family, ...extra })),
  ).map((panel) => panel.id);

test("each clone family gets its one panel, only on Clone", () => {
  assert.deepEqual(shown("qwen3_tts"), ["qwen3-timbre"]);
  assert.deepEqual(shown("index_tts2"), ["index-emotion"]);
  assert.deepEqual(shown("chatterbox"), ["chatterbox-expressiveness"]);
  assert.deepEqual(shown("cosyvoice3"), ["cosyvoice-mode"]);
  assert.deepEqual(shown("f5_tts"), ["f5-speed-dialect"]);
  assert.deepEqual(shown("voxcpm2"), []);
  assert.deepEqual(shown("qwen3_tts", "speak"), []);
});

test("the saved-voice choice shows on Speak only for models that also clone", () => {
  assert.equal(
    panelApplies(
      speakVoiceLogic,
      "speak",
      ctx({ audioWorkflows: ["speak", "clone"] }),
    ),
    true,
  );
  assert.equal(
    panelApplies(speakVoiceLogic, "speak", ctx({ audioWorkflows: ["speak"] })),
    false,
  );
  assert.equal(
    panelApplies(
      speakVoiceLogic,
      "speak",
      ctx({ audioWorkflows: ["speak", "clone"], referenceTextMode: "required" }),
    ),
    false,
  );
  assert.equal(
    panelApplies(
      speakVoiceLogic,
      "clone",
      ctx({ audioWorkflows: ["speak", "clone"] }),
    ),
    false,
  );
  assert.equal(
    panelApplies(
      vibeVoiceDialogueLogic,
      "speak",
      ctx({ audioFamily: "vibevoice" }),
    ),
    true,
  );
});

test("collectToolRequest merges every panel's part and returns the first error", () => {
  const a = {
    id: "a",
    initial: () => ({ on: true }),
    toRequest: () => ({ options: { x: 1 }, instructions: "first" }),
    validate: () => "a is missing",
  };
  const b = {
    id: "b",
    initial: () => ({}),
    toRequest: () => ({
      options: { y: "z" },
      speed: 1.5,
      referenceTextMode: "hidden" as const,
    }),
    validate: () => "b is missing",
  };
  const { patch, error } = collectToolRequest(
    [a, b] as never,
    {},
    { text: "hi" },
    ctx(),
  );
  assert.deepEqual(patch, {
    options: { x: 1, y: "z" },
    instructions: "first",
    speed: 1.5,
    referenceTextMode: "hidden",
  });
  assert.equal(error, "a is missing");
});

test("a kept value is laid over the panel's defaults", () => {
  const value = panelValue(
    indexTts2EmotionLogic,
    { "index-emotion": { mode: "mixer" } },
    [],
  );
  assert.equal(value.mode, "mixer");
  assert.equal(value.alpha, 0.7);
  assert.equal(value.vector.length, 8);
  assert.equal(
    toolValueKey("repo/model", "clone", "index-emotion"),
    "repo/model:clone:index-emotion",
  );
});

test("the IndexTTS2 mixer sends eight weights and the strength", () => {
  const vector = [0.8, 0, 0, 0, 0, 0, 0.2, 0];
  assert.equal(emotionVectorString(vector), "0.8,0,0,0,0,0,0.2,0");
  const patch = indexTts2EmotionLogic.toRequest({
    ...indexTts2EmotionLogic.initial([]),
    mode: "mixer",
    vector,
    alpha: 0.6,
  });
  assert.deepEqual(patch, {
    options: { emotion_vector: "0.8,0,0,0,0,0,0.2,0", emotion_alpha: 0.6 },
  });
  assert.deepEqual(
    indexTts2EmotionLogic.toRequest({
      ...indexTts2EmotionLogic.initial([]),
      mode: "mixer",
    }),
    {},
  );
});

test("IndexTTS2 emotion from text and from audio", () => {
  const base = indexTts2EmotionLogic.initial([]);
  assert.deepEqual(
    indexTts2EmotionLogic.toRequest({ ...base, text: "  joyful " }),
    {
      options: {
        use_emotion_text: true,
        emotion_text: "joyful",
        emotion_alpha: 0.7,
      },
    },
  );
  assert.deepEqual(indexTts2EmotionLogic.toRequest(base), {});
  const source = { kind: "clip" as const, id: "c1", name: "x", durationS: 2 };
  assert.deepEqual(
    indexTts2EmotionLogic.toRequest({ ...base, mode: "audio", source }),
    {
      inputs: { emotion: { clip_id: "c1" } },
      options: { emotion_alpha: 0.7 },
    },
  );
  assert.equal(
    indexTts2EmotionLogic.validate?.(
      { ...base, mode: "audio" },
      { text: "" },
      ctx(),
    ),
    EMOTION_AUDIO_MISSING,
  );
  // A replacement still uploading, a failed one, or an expired clip blocks the run.
  for (const status of [
    { phase: "uploading" as const, name: "y", progress: 0.5 },
    { phase: "error" as const, message: "Upload failed." },
    { phase: "expired" as const },
  ]) {
    const sourceProblem = emotionSourceProblem(status);
    assert.ok(sourceProblem);
    assert.equal(
      indexTts2EmotionLogic.validate?.({ ...base, mode: "audio", source, sourceProblem }, { text: "" }, ctx()),
      sourceProblem,
    );
  }
  assert.equal(emotionSourceProblem({ phase: "ready" }), null);
});

test("Qwen3 Timbre only sends x_vector_only_mode and drops the transcript requirement", () => {
  const required = ctx({
    audioFamily: "qwen3_tts",
    referenceTextMode: "required",
  });
  const on = qwen3TimbreLogic.toRequest({ timbreOnly: true });
  assert.deepEqual(on.options, { x_vector_only_mode: true });
  assert.equal(referenceTextField(required, on), "hidden");
  assert.equal(
    referenceTextField(
      required,
      qwen3TimbreLogic.toRequest({ timbreOnly: false }),
    ),
    "required",
  );
  const blocked = cloneBlocker({
    reference: { kind: "input", id: "i", name: "a.wav", durationS: 4 },
    referenceBusy: false,
    referenceExpired: false,
    referenceError: null,
    referenceText: "",
    referenceTextField: referenceTextField(required, on),
    text: "Hello",
    panelError: null,
  });
  assert.equal(blocked, null);
  // A failed replacement hides the kept clip, so it blocks until dismissed.
  const failed = cloneBlocker({
    reference: { kind: "input", id: "i", name: "a.wav", durationS: 4 },
    referenceBusy: false,
    referenceExpired: false,
    referenceError: "Upload failed.",
    referenceText: "",
    referenceTextField: referenceTextField(required, on),
    text: "Hello",
    panelError: null,
  });
  assert.equal(failed?.kind, "reference-error");
});

test("CosyVoice3 Cross-lingual needs no transcript; Instruct needs its instruction", () => {
  const required = ctx({
    audioFamily: "cosyvoice3",
    referenceTextMode: "required",
  });
  const cross = cosyVoiceModeLogic.toRequest({
    mode: "cross_lingual",
    instruction: "",
  });
  assert.deepEqual(cross.options, { template_name: "cross_lingual" });
  assert.equal(referenceTextField(required, cross), "hidden");
  const same = cosyVoiceModeLogic.toRequest({
    mode: "zero_shot",
    instruction: "",
  });
  assert.equal(referenceTextField(required, same), "required");
  assert.equal(
    cosyVoiceModeLogic.validate?.(
      { mode: "instruct", instruction: " " },
      { text: "" },
      required,
    ),
    COSYVOICE_INSTRUCTION_MISSING,
  );
  assert.deepEqual(
    cosyVoiceModeLogic.toRequest({ mode: "instruct", instruction: "Calm." }),
    {
      options: { template_name: "instruct" },
      instructions: "Calm.",
      referenceTextMode: "optional",
    },
  );
});

test("Chatterbox and F5 send their options, F5 speed top-level", () => {
  assert.deepEqual(
    chatterboxExpressivenessLogic.toRequest({ exaggeration: 3, guidance: 0.4 }),
    {
      options: { exaggeration: 2, guidance_scale: 0.4 },
    },
  );
  assert.deepEqual(
    f5SpeedDialectLogic.toRequest({ speed: 1, dialect: "UNK" }),
    {},
  );
  assert.deepEqual(
    f5SpeedDialectLogic.toRequest({ speed: 1.3, dialect: "EGY" }),
    {
      speed: 1.3,
      options: { dialect: "EGY" },
    },
  );
});

test("Maya1's required voice description reaches the tool context", () => {
  const maya = audioModelContextFor(
    {
      audio_type: "audiocpp_tts",
      audio_family: "maya1",
      audio_required_inputs: ["instruct"],
    },
    {
      musicGeneration: false,
      cudaMusicGeneration: false,
      musicNeedsDescription: false,
    },
  );
  assert.deepEqual(maya.requiredInputs, ["instruct"]);
});

test("a saved voice on Speak becomes the run's reference", () => {
  assert.deepEqual(
    speakVoiceLogic.toRequest({ source: "saved", voiceId: "v1" }),
    {
      inputs: { reference: { voice_id: "v1" } },
    },
  );
  assert.deepEqual(
    speakVoiceLogic.toRequest({ source: "builtin", voiceId: "v1" }),
    {},
  );
  assert.equal(
    speakVoiceLogic.validate?.(
      { source: "saved", voiceId: null },
      { text: "" },
      ctx(),
    ),
    SAVED_VOICE_MISSING,
  );
});

test("a saved voice deleted on Clone holds Speak's Generate", () => {
  const saved = { source: "saved" as const, voiceId: "v1" };
  const check = (savedVoiceIds: readonly string[] | null) =>
    speakVoiceLogic.validate?.(saved, { text: "hi" }, ctx({ savedVoiceIds }));
  assert.equal(check(["v2"]), SAVED_VOICE_DELETED);
  assert.equal(check(["v1", "v2"]), null);
  // Not loaded yet, or the list failed: nothing to compare against.
  assert.equal(check(null), null);
});

test("formatVibeVoiceScript prefixes plain text only", () => {
  assert.equal(
    formatVibeVoiceScript("Hello there."),
    "Speaker 1: Hello there.",
  );
  assert.equal(
    formatVibeVoiceScript("Speaker 2: Hi\nSpeaker 1: Yo"),
    "Speaker 2: Hi\nSpeaker 1: Yo",
  );
  assert.equal(formatVibeVoiceScript("  speaker 3 : hey"), "speaker 3 : hey");
  assert.equal(formatVibeVoiceScript("   "), "");
});

test("the status context reads the clone fields defensively", () => {
  const page = {
    musicGeneration: false,
    cudaMusicGeneration: false,
    musicNeedsDescription: false,
  };
  const none = audioModelContextFor(null, page);
  assert.deepEqual(none.audioWorkflows, []);
  assert.equal(none.referenceTextMode, null);
  assert.equal(
    audioModelContextFor({ audio_reference_text: "bogus" }, page)
      .referenceTextMode,
    null,
  );
  assert.equal(
    audioModelContextFor({ audio_reference_text: "unused" }, page)
      .referenceTextMode,
    "unused",
  );
  assert.equal(
    referenceTextField({ referenceTextMode: "unused" }, {}),
    "hidden",
  );
  assert.equal(referenceTextField({ referenceTextMode: null }, {}), "optional");
});

test("Clone's blockers come in rail order", () => {
  const base = {
    reference: null,
    referenceBusy: false,
    referenceExpired: false,
    referenceError: null,
    referenceText: "",
    referenceTextField: "required" as const,
    text: "",
    panelError: "panel says no",
  };
  assert.equal(cloneBlocker(base)?.kind, "reference");
  assert.equal(
    cloneBlocker({ ...base, referenceExpired: true })?.reason,
    "This reference expired.",
  );
  const reference = {
    kind: "input" as const,
    id: "i",
    name: "a",
    durationS: 3,
  };
  assert.equal(cloneBlocker({ ...base, reference })?.kind, "reference-text");
  assert.equal(
    cloneBlocker({ ...base, reference, referenceText: "hi" })?.kind,
    "text",
  );
  assert.equal(
    cloneBlocker({ ...base, reference, referenceText: "hi", text: "go" })?.kind,
    "panel",
  );
  assert.equal(
    cloneBlocker({
      ...base,
      reference,
      referenceText: "hi",
      text: "go",
      panelError: null,
    }),
    null,
  );
});

test("panel labels are plain words, never option names", () => {
  const panels = readSrc("features/audio/tools/clone-panels.tsx");
  for (const label of [
    "Timbre only",
    "Emotion",
    "Strength",
    "Expressiveness",
    "Same language",
    "Cross-lingual",
    "Instruct",
  ]) {
    assert.ok(panels.includes(label), label);
  }
  assert.doesNotMatch(
    panels,
    />\s*(x_vector_only_mode|emotion_alpha|guidance_scale|template_name)\s*</,
  );
});

test("deleting the selected saved voice clears Speak's pick", () => {
  const picker = readSrc("features/audio/components/voice-picker.tsx");
  assert.match(picker, /if \(voice\.id === selectedId\) onDeselect\?\.\(\);/);
  const speak = readSrc("features/audio/tools/speak-panels.tsx");
  assert.match(
    speak,
    /onDeselect=\{\(\) => onChange\(\{ \.\.\.value, voiceId: null \}\)\}/,
  );
});

test("reference transcription sends no language hint", () => {
  const hook = readSrc("features/audio/hooks/use-reference-transcribe.ts");
  assert.doesNotMatch(hook, /\blanguage\s*[:,}]/);
});

test("a run's inline fallback clip stays on the page that started it", () => {
  const clone = readSrc("features/audio/hooks/use-clone-generation.ts");
  const body = clone.slice(clone.indexOf("export async function showRunResult("));
  assert.doesNotMatch(body.slice(0, body.indexOf("\n}\n")), /workflow: "clone"/);
  assert.match(readSrc("features/audio/hooks/use-speech-generation.ts"), /showRunResult\(\{[^}]*workflow: "speak"/);
});
