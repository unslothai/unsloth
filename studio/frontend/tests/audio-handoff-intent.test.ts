// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  audioPickSearch,
  audioRouteIntent,
  audioWorkflowForPick,
  audioWorkflowUrlAction,
  validateAudioSearch,
} = await import("../src/features/audio/route-search.ts");

test("every task Audio runs survives validation, including text-to-audio", () => {
  for (const task of [
    "text-to-speech",
    "automatic-speech-recognition",
    "text-to-audio",
  ]) {
    assert.deepEqual(validateAudioSearch({ task }), { task });
  }
});

test("unknown tasks, workflows and audio types are dropped", () => {
  assert.deepEqual(
    validateAudioSearch({
      task: "text-to-image",
      workflow: "remix",
      audioType: "not-a-type",
      loadId: "   ",
    }),
    {},
  );
  assert.deepEqual(
    validateAudioSearch({ workflow: 3, task: ["text-to-speech"] }),
    {},
  );
  assert.deepEqual(validateAudioSearch({ workflow: "separate" }), {
    workflow: "separate",
  });
});

test("the params the route accepted before pass through unchanged", () => {
  const search = {
    model: "unsloth/orpheus-3b-0.1-ft-GGUF",
    quant: "orpheus-3b-0.1-ft-Q4_K_M.gguf",
    ggufQuant: "Q4_K_M",
    task: "text-to-speech",
    audioType: "audiocpp_tts",
    loadId: "load-1",
    item: "clip-1",
  };
  assert.deepEqual(validateAudioSearch({ ...search, extra: "x" }), search);
  assert.deepEqual(validateAudioSearch({ audioType: "minimax_music3" }), {
    audioType: "minimax_music3",
  });
});

test("a workflow param is kept when it names a workflow", () => {
  for (const workflow of ["speak", "clone", "edit", "music", "transcribe"]) {
    assert.deepEqual(validateAudioSearch({ workflow }), { workflow });
  }
});

test("the task maps to its workflow", () => {
  assert.equal(audioRouteIntent({ task: "text-to-speech" }), "speak");
  assert.equal(
    audioRouteIntent({ task: "automatic-speech-recognition" }),
    "transcribe",
  );
  assert.equal(audioRouteIntent({ task: "text-to-audio" }), "music");
  assert.equal(audioRouteIntent({}), null);
  assert.equal(audioRouteIntent({ task: "text-to-image" }), null);
});

test("an explicit workflow beats the one the task implies", () => {
  assert.equal(
    audioRouteIntent({ task: "text-to-speech", workflow: "music" }),
    "music",
  );
  assert.equal(
    audioRouteIntent({ task: "text-to-audio", workflow: "bogus" }),
    "music",
  );
});

test("a chat pick opens on the workflow its model runs in", () => {
  assert.equal(
    audioWorkflowForPick({ id: "hexgrad/Kokoro-82M", task: "text-to-speech" }),
    "speak",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "openai/whisper-small",
      task: "automatic-speech-recognition",
    }),
    "transcribe",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "MiniMaxAI/MiniMax-Music3",
      task: "text-to-speech",
      audioType: "minimax_music3",
    }),
    "music",
  );
  assert.equal(
    audioWorkflowForPick({
      id: "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF",
      task: "text-to-speech",
    }),
    "music",
  );
  assert.equal(audioWorkflowForPick({ id: "x/y", task: null }), null);
  const clone = (folder: string) =>
    audioWorkflowForPick({
      id: `audio-cpp/audio.cpp-gguf/${folder}`,
      task: "text-to-speech",
    });
  assert.equal(clone("Qwen3-TTS-12Hz-0.6B-Base-GGUF"), "clone");
  assert.equal(clone("VoxCPM2-GGUF"), "speak");
});

test("the route validates through the shared helper", () => {
  const route = readSrc("app/routes/audio.tsx");
  assert.match(route, /validateSearch: validateAudioSearch,/);
});

test("the chat picker routes music and separation rows to Audio with their workflow", () => {
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(
    pickers,
    /audioPickSearch\(id, \{ \.\.\.meta, task: pickedTask \}\)/,
  );
  const tasks = pickers.match(/export const AUDIO_GEN_TASKS = \[([^\]]*)\]/);
  assert.ok(tasks);
  assert.match(tasks[1], /"text-to-audio"/);
  assert.match(tasks[1], /"audio-to-audio"/);
});

test("a pick's Audio search names its file, quant label, load and workflow", () => {
  assert.deepEqual(
    audioPickSearch("audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF", {
      ggufFilename: "ace-step-q8_0.gguf",
      ggufVariant: "Q8_0",
      task: "text-to-audio",
      loadId: "/cache/snap",
      isGguf: true,
    }),
    {
      model: "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF",
      quant: "ace-step-q8_0.gguf",
      ggufQuant: "Q8_0",
      task: "text-to-audio",
      audioType: undefined,
      loadId: "/cache/snap",
      gguf: true,
      workflow: "music",
    },
  );
  // Without a filename the quant label rides ggufQuant, never `quant`.
  const labelOnly = audioPickSearch("x/Kokoro-82M-GGUF", {
    ggufVariant: "Q4_K_M",
    task: "text-to-speech",
  });
  assert.equal(labelOnly.quant, undefined);
  assert.equal(labelOnly.ggufQuant, "Q4_K_M");
  assert.equal(labelOnly.workflow, "speak");
});

test("a routed audio-to-audio pick opens Separate", () => {
  assert.equal(
    audioWorkflowForPick({
      id: "someone/BSRoformer-GGUF",
      task: "audio-to-audio",
    }),
    "separate",
  );
});

test("a chat picker handoff opens its page before the load, not only after it succeeds", () => {
  const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
  assert.match(
    handoff,
    /const key = `\$\{wanted\}\|[^`]*\|\$\{routeSearch\.workflow \?\? ""\}`;/,
  );
  assert.match(
    handoff,
    /if \(busyRef\.current !== null\) return;[\s\S]*?if \(\s*isAudioWorkflowId\(routedWorkflow\) &&\s*!transitionWorkflow\(routedWorkflow\)\s*\) \{\s*return;\s*\}\s*handledRouteModel\.current = key;\s*handleModelSelect\(wanted,/,
  );
});

test("a model deep link without a workflow opens the page its task names before loading", () => {
  const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
  assert.match(
    handoff,
    /const routedWorkflow = isAudioWorkflowId\(routeSearch\.workflow\)\s*\?\s*routeSearch\.workflow\s*:\s*audioWorkflowForPick\(\{\s*id: wanted,\s*task: routeSearch\.task,\s*audioType: routeSearch\.audioType,\s*\}\);/,
  );
  assert.equal(
    audioWorkflowForPick({ id: "some/model", task: "text-to-audio" }),
    "music",
  );
  assert.equal(audioWorkflowForPick({ id: "some/model" }), null);
  assert.equal(
    audioWorkflowForPick({ id: "some/model", audioType: "minimax_music3" }),
    "music",
  );
});

test("a chat pick of an uncatalogued clone-only family opens Clone", () => {
  assert.equal(
    audioWorkflowForPick({ id: "x/MioTTS-GGUF", task: "text-to-speech" }),
    "clone",
  );
  assert.equal(
    audioWorkflowForPick({ id: "x/Kokoro-82M-GGUF", task: "text-to-speech" }),
    "speak",
  );
});

test("a GGUF pick keeps its format through the Audio route", async () => {
  const { audioPickSearch, validateAudioSearch } = await import(
    "../src/features/audio/route-search.ts"
  );
  const search = audioPickSearch("someone/parakeet-audiocpp", {
    task: "automatic-speech-recognition",
    isGguf: true,
    ggufVariant: "Q8_0",
  });
  assert.equal(search.gguf, true);
  assert.equal(validateAudioSearch({ ...search }).gguf, true);
  assert.equal(validateAudioSearch({ gguf: "true" }).gguf, true);
  assert.equal(validateAudioSearch({ gguf: "1" }).gguf, undefined);
  assert.equal(
    audioPickSearch("org/whisper-small", {
      task: "automatic-speech-recognition",
    }).gguf,
    undefined,
  );
  const { readFile } = await import("node:fs/promises");
  const handoff = await readFile(
    new URL(
      "../src/features/audio/hooks/use-audio-handoff.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // Without it the slot picks the STT engine from the repo name alone.
  assert.match(handoff, /isGguf: routeSearch\.gguf/);
});

test("?workflow= follows the committed page, and a moved URL is a request", () => {
  const base = { hasModel: false, hasItem: false } as const;
  // A reload (nothing synced yet) on another page's URL asks for that page.
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: "transcribe",
      lastSynced: null,
      committed: "speak",
    }),
    { kind: "intent", workflow: "transcribe" },
  );
  // Back/forward off the synced value is a request too.
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: "clone",
      lastSynced: "music",
      committed: "music",
    }),
    { kind: "intent", workflow: "clone" },
  );
  // A refused request leaves the inputs unchanged, so it is asked again once busy settles.
  const refused = {
    ...base,
    urlWorkflow: "music",
    lastSynced: "speak",
    committed: "speak",
  } as const;
  assert.deepEqual(audioWorkflowUrlAction(refused), {
    kind: "intent",
    workflow: "music",
  });
  assert.deepEqual(audioWorkflowUrlAction(refused), {
    kind: "intent",
    workflow: "music",
  });
  // The page moved on (a tab or title-menu pick): the URL catches up.
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: "speak",
      lastSynced: "speak",
      committed: "separate",
    }),
    { kind: "write", workflow: "separate" },
  );
  // A bare /audio gets the committed page written in.
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: null,
      lastSynced: null,
      committed: "speak",
    }),
    { kind: "write", workflow: "speak" },
  );
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: null,
      lastSynced: "clone",
      committed: "clone",
    }),
    { kind: "write", workflow: "clone" },
  );
  // In sync: nothing to do, and no transition (Transcribe's has side effects).
  assert.deepEqual(
    audioWorkflowUrlAction({
      ...base,
      urlWorkflow: "transcribe",
      lastSynced: "transcribe",
      committed: "transcribe",
    }),
    { kind: "none" },
  );
});

test("the model and Library handoffs own the URL until they clean it up", () => {
  for (const flags of [
    { hasModel: true, hasItem: false },
    { hasModel: false, hasItem: true },
  ]) {
    assert.deepEqual(
      audioWorkflowUrlAction({
        ...flags,
        urlWorkflow: "music",
        lastSynced: null,
        committed: "speak",
      }),
      { kind: "none" },
    );
    assert.deepEqual(
      audioWorkflowUrlAction({
        ...flags,
        urlWorkflow: null,
        lastSynced: "speak",
        committed: "clone",
      }),
      { kind: "none" },
    );
  }
});

test("the URL keeps the page after a handoff instead of clearing it", () => {
  const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
  // A model handoff leaves its page in the URL, so a reload reopens it.
  assert.match(
    handoff,
    /const settledWorkflow = routedWorkflow \?\? committedWorkflow;\s*lastSyncedWorkflow\.current = settledWorkflow;\s*void navigateSelf\(\{\s*to: "\/audio",\s*search: \{ workflow: settledWorkflow \},\s*replace: true,\s*\}\);/,
  );
  assert.doesNotMatch(handoff, /search: \{\}/);
  // Every write the handoff makes replaces the entry, so syncing never stacks history.
  const navigations = handoff.match(/navigateSelf\(\{[\s\S]*?\}\);/g) ?? [];
  assert.ok(navigations.length >= 4, String(navigations.length));
  for (const call of navigations) assert.match(call, /replace: true/);
});

test("the first URL commits its page, so the loaded model's page cannot replace it", () => {
  const base = { hasModel: false, hasItem: false, lastSynced: null } as const;
  // A reload of ?workflow=speak, or a ?task=text-to-speech link, while the store still holds
  // its default Speak: without a commit, a loaded music model moved the page to Music.
  assert.deepEqual(
    audioWorkflowUrlAction({ ...base, urlWorkflow: "speak", committed: "speak" }),
    { kind: "intent", workflow: "speak" },
  );
  // A bare /audio asks for nothing, so the loaded model's page may still be adopted.
  assert.deepEqual(
    audioWorkflowUrlAction({ ...base, urlWorkflow: null, committed: "speak" }),
    { kind: "write", workflow: "speak" },
  );
  const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
  assert.match(
    handoff,
    /const urlWorkflow = audioRouteIntent\(routeSearch\);/,
  );
});
