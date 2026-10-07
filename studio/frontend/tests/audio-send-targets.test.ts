// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { STEM_SEND_TARGETS, clipSendTargets, transcriptSendTargets } =
  await import("../src/features/audio/send-targets.ts");

const clip = (workflow: string, duration_s: number | null = 4) => ({
  workflow,
  duration_s,
});

test("speech clips can be edited, and Edit takes its own results again", () => {
  assert.deepEqual(clipSendTargets(clip("speak"), "speak"), [
    "clone",
    "convert",
    "separate",
    "transcribe",
    "edit",
  ]);
  assert.ok(clipSendTargets(clip("clone"), "clone").includes("edit"));
  assert.ok(clipSendTargets(clip("convert"), "convert").includes("edit"));
  assert.deepEqual(clipSendTargets(clip("edit"), "edit"), [
    "clone",
    "convert",
    "separate",
    "transcribe",
    "edit",
  ]);
  assert.ok(!clipSendTargets(clip("clone"), "clone").includes("clone"));
  assert.ok(!clipSendTargets(clip("convert"), "convert").includes("convert"));
});

test("music is split into stems but not sent to speech Edit", () => {
  assert.deepEqual(clipSendTargets(clip("music", 30), "music"), [
    "clone",
    "convert",
    "separate",
    "transcribe",
  ]);
  // older clips carry only an audio type
  assert.ok(
    !clipSendTargets(
      { audio_type: "minimax_music3", duration_s: 10 },
      "music",
    ).includes("edit"),
  );
});

test("a clip longer than Edit takes is not offered to Edit", () => {
  assert.ok(clipSendTargets(clip("speak", 30.05), "speak").includes("edit"));
  assert.ok(!clipSendTargets(clip("speak", 31), "speak").includes("edit"));
  // an unknown length is left to Edit's own check
  assert.ok(clipSendTargets(clip("speak", null), "speak").includes("edit"));
});

test("stems go to Clone, Convert, Music edit and Transcribe, or become a voice", () => {
  assert.deepEqual(
    STEM_SEND_TARGETS.map((target) => target.workflow),
    ["clone", "convert", "music", "transcribe", "voice"],
  );
  assert.equal(
    STEM_SEND_TARGETS.find((target) => target.workflow === "music")?.label,
    "Music edit",
  );
});

test("a transcript's text goes to Speak; Edit and Clone also need its audio", () => {
  const input = { kind: "input" as const, id: "a1", name: "take.wav" };
  assert.deepEqual(
    transcriptSendTargets({ text: "hello", source: input, duration: 12 }),
    ["speak", "edit", "clone"],
  );
  assert.deepEqual(
    transcriptSendTargets({ text: "hello", source: null, duration: 12 }),
    ["speak"],
  );
  assert.deepEqual(
    transcriptSendTargets({
      text: "hello",
      source: { kind: "voice", id: "v1", name: "Narrator" },
      duration: 8,
    }),
    ["speak", "clone"],
  );
  assert.deepEqual(
    transcriptSendTargets({ text: "hello", source: input, duration: 95 }),
    ["speak", "clone"],
  );
  assert.deepEqual(
    transcriptSendTargets({ text: "  ", source: input, duration: 3 }),
    [],
  );
});

test("the Audio page wires every send, and waits out a running task first", () => {
  const host = readSrc("features/audio/audio-page.tsx");
  assert.match(host, /clipSendTargets\(clip, ttsWorkflow\)/);
  assert.match(host, /transcriptSendTargets\(\{ text, source, duration \}\)/);
  assert.match(host, /if \(!runBusy\(\)\) handlers\[id\]\?\.\(\)/);
  assert.match(
    host,
    /if \(transitionWorkflow\("edit"\)\) adoptEditSource\(reference\(\)\)/,
  );
  assert.match(host, /useAudioSeparateStore\.getState\(\)\.setSource\(\{/);
  assert.match(host, /sendClipToMusic\(clip, "edit", name\)/);
  const music = readSrc("features/audio/hooks/use-music-generation.ts");
  assert.match(
    music,
    /return loaded && withWaitingEdit\(loaded, editWaiting\);/,
  );
  assert.match(host, /source: \{ clip_id: voiceClip\.id \}/);
  assert.match(host, /<SaveVoiceDialog\s+open=\{active && savingVoice\}/);
  assert.match(
    host,
    /useEffect\(\(\) => \{\s*if \(!active\) return;\s*return \(\) => setSavingVoice\(false\);\s*\}, \[active\]\);/,
  );
  const transcriptSend = host.slice(
    host.indexOf("const sendTranscriptHandlersFor"),
    host.indexOf("const handleUseTextAgain"),
  );
  assert.doesNotMatch(transcriptSend, /runTranscription|handleTranscribe/);
  assert.match(
    transcriptSend,
    /adoptEditSource\(selection\);\s*useAudioEditStore\s*\.getState\(\)\s*\.setTranscript\(text\.trim\(\), selection\.id\);/,
  );
  assert.match(
    transcriptSend,
    /adoptReference\(selection\);[^]*?useAudioCloneStore\s*\.getState\(\)\s*\.applyTranscript\(selection, referenceTranscript\(selection\)\.trim\(\)\);/,
  );

  const menu = readSrc("features/audio/pages/tts-workspace.tsx");
  assert.match(menu, /onClick=\{\(\) => onSaveVoice\(clip\)\}/);
  const transcribe = readSrc("features/audio/pages/transcribe-page.tsx");
  assert.match(transcribe, /<SendToItems handlers=\{sendHandlers\} \/>/);
  const gallery = readSrc("features/audio/transcript-gallery.tsx");
  assert.match(
    gallery,
    /<ClipSendToMenu handlers=\{sendHandlersFor\(record\)\} \/>/,
  );
  const separate = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(separate, /sendTargets=\{STEM_SEND_TARGETS\}/);
});
