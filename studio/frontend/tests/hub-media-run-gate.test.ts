// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { SelectedModelView } from "../src/features/hub/types.ts";
import { studioPageForTask } from "../src/features/hub/lib/unsloth-support.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { hubAudioRunRefused, hubModelRunsOnAudioPage, isHubModelRunEligible } =
  await import("../src/features/hub/lib/model-run-selection.ts");
const { taskForMediaPick } = await import(
  "../src/features/model-picker/components/model-selector/audio-picker-policy.ts"
);

function mediaModel(
  task: SelectedModelView["pipelineTag"],
  kind: "discover" | "cache" | "local",
  localSource?: SelectedModelView["localSource"],
): SelectedModelView {
  const loadId = "/cache/models--Org--Model/snapshots/revision";
  return {
    id: kind === "local" ? loadId : "Org/Model",
    loadId,
    kind,
    displayId: "Org/Model",
    hubRepoId: "Org/Model",
    owner: "Org",
    title: "Model",
    summary: "Model",
    sourceLabel: "Hub cache",
    path: loadId,
    localSource,
    isLocal: kind === "local",
    isGguf: false,
    modelFormat: "safetensors",
    isDownloaded: true,
    runtimeCanChat: false,
    capabilities: [],
    license: null,
    pipelineTag: task,
  };
}

function mediaRunEligible(model: SelectedModelView): boolean {
  const task = taskForMediaPick(model.pipelineTag, model.task);
  return isHubModelRunEligible({
    model,
    isDataset: false,
    mediaRuntime: studioPageForTask(task ?? undefined) !== undefined,
    nonGgufRuntimeAvailable: false,
  });
}

// The backend tags every local non-GGUF diffusers checkpoint text-to-image, whatever its source.
test("a filesystem diffusion row never counts as running on a media page", () => {
  assert.equal(
    mediaRunEligible(mediaModel("text-to-image", "local", "models_dir")),
    false,
  );
  assert.equal(
    mediaRunEligible(mediaModel("text-to-image", "local", "lmstudio")),
    false,
  );
  assert.equal(
    mediaRunEligible(mediaModel("text-to-image", "local", "ollama")),
    false,
  );
  assert.equal(
    mediaRunEligible(mediaModel("text-to-video", "local", "custom")),
    false,
  );
  assert.equal(mediaRunEligible(mediaModel("text-to-image", "local")), false);
});

test("complete Hub-backed diffusion rows stay runnable on their page", () => {
  assert.equal(
    mediaRunEligible(mediaModel("text-to-image", "local", "hf_cache")),
    true,
  );
  assert.equal(mediaRunEligible(mediaModel("text-to-image", "cache")), true);
  assert.equal(
    mediaRunEligible(mediaModel("image-text-to-video", "discover")),
    true,
  );
});

test("chat tasks are not eligible through the media route", () => {
  assert.equal(mediaRunEligible(mediaModel("text-generation", "cache")), false);
  assert.equal(mediaRunEligible(mediaModel(undefined, "discover")), false);
  assert.equal(
    mediaRunEligible(mediaModel("text-generation", "local", "hf_cache")),
    false,
  );
});

test("an audio model the Audio page runs opens there, the rest keep their chat Run", () => {
  const audio = (overrides: Partial<SelectedModelView>): SelectedModelView => ({
    ...mediaModel(undefined, "cache"),
    runtimeCanChat: true,
    ...overrides,
  });
  const opensAudio = (model: SelectedModelView) =>
    hubModelRunsOnAudioPage(
      model,
      taskForMediaPick(model.pipelineTag, model.task),
    );
  // A cached audio runtime GGUF carries its header task and no Hub tag.
  assert.equal(
    opensAudio(
      audio({
        id: "x/ACE-GGUF",
        hubRepoId: "x/ACE-GGUF",
        isGguf: true,
        task: "text-to-audio",
      }),
    ),
    true,
  );
  assert.equal(
    opensAudio(
      audio({
        hubRepoId: "unsloth/orpheus-3b-0.1-ft",
        pipelineTag: "text-to-speech",
      }),
    ),
    true,
  );
  assert.equal(
    opensAudio(
      audio({
        hubRepoId: "openai/whisper-small",
        pipelineTag: "automatic-speech-recognition",
      }),
    ),
    true,
  );
  // Tagged audio, but nothing here decodes it: Run stays where it was.
  assert.equal(
    opensAudio(
      audio({
        hubRepoId: "facebook/musicgen-small",
        pipelineTag: "text-to-audio",
      }),
    ),
    false,
  );
  assert.equal(
    opensAudio(
      audio({ hubRepoId: "suno/bark", pipelineTag: "text-to-speech" }),
    ),
    false,
  );
  assert.equal(
    opensAudio(
      audio({ hubRepoId: "unsloth/Qwen3-8B", pipelineTag: "text-generation" }),
    ),
    false,
  );
  // The shared GGUF audio repo is many models, not one Run target.
  assert.equal(
    opensAudio(
      audio({
        id: "audio-cpp/audio.cpp-gguf",
        hubRepoId: "audio-cpp/audio.cpp-gguf",
        isGguf: true,
        pipelineTag: "text-to-speech",
        task: "audio-to-audio",
        tags: ["gguf", "audio.cpp"],
      }),
    ),
    false,
  );
  // A filesystem row has no Hub id to hand Audio, so it keeps its chat Run.
  assert.equal(
    opensAudio({
      ...mediaModel("text-to-speech", "local", "models_dir"),
      hubRepoId: "unsloth/orpheus-3b-0.1-ft",
    }),
    false,
  );
});

// #13775: NVIDIA's own Magpie GGUF (arch magpietts) and the VoxCPM2 safetensors are tagged
// text-to-speech but only their audio.cpp builds run; Run opened them in chat, which failed.
test("Run refuses an audio model neither Audio nor chat can run", () => {
  const audio = (overrides: Partial<SelectedModelView>): SelectedModelView => ({
    ...mediaModel(undefined, "cache"),
    runtimeCanChat: true,
    ...overrides,
  });
  const refused = (model: SelectedModelView) =>
    hubAudioRunRefused(model, taskForMediaPick(model.pipelineTag, model.task));
  const magpie = audio({
    id: "nvidia/magpie_tts_multilingual_357m",
    hubRepoId: "nvidia/magpie_tts_multilingual_357m",
    isGguf: true,
    modelFormat: "gguf",
    pipelineTag: "text-to-speech",
    task: "text-generation",
    libraryName: "nemo",
    tags: ["nemo", "gguf", "text-to-speech"],
  });
  assert.equal(refused(magpie), true);
  assert.equal(
    refused(
      audio({
        hubRepoId: "openbmb/VoxCPM2",
        pipelineTag: "text-to-speech",
        libraryName: "voxcpm",
      }),
    ),
    true,
  );
  assert.equal(
    refused(
      audio({
        hubRepoId: "facebook/musicgen-small",
        pipelineTag: "text-to-audio",
      }),
    ),
    true,
  );
  // What Audio runs opens there, and the refusal never fires for it.
  assert.equal(
    refused(
      audio({
        id: "audio-cpp/MiniMax-Music3-GGUF",
        hubRepoId: "audio-cpp/MiniMax-Music3-GGUF",
        isGguf: true,
        modelFormat: "gguf",
        pipelineTag: "text-to-audio",
        task: "text-to-audio",
        libraryName: "audio.cpp",
      }),
    ),
    false,
  );
  assert.equal(
    refused(
      audio({
        hubRepoId: "unsloth/orpheus-3b-0.1-ft",
        pipelineTag: "text-to-speech",
      }),
    ),
    false,
  );
  // An Orpheus fine-tune GGUF named for no family loads in llama.cpp, and chat speaks it.
  assert.equal(
    refused(
      audio({
        id: "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
        hubRepoId: "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
        isGguf: true,
        modelFormat: "gguf",
        pipelineTag: "text-to-speech",
        task: "text-generation",
        libraryName: "transformers",
        baseModel: "canopylabs/3b-fr-ft-research_release",
        tags: ["transformers", "gguf", "text-to-speech"],
      }),
    ),
    false,
  );
  assert.equal(refused({ ...magpie, libraryName: undefined }), false);
  // The codec vocabulary is a speech exemption: a music GGUF Audio cannot run is refused.
  assert.equal(
    refused(
      audio({
        id: "calcuis/ace-gguf",
        hubRepoId: "calcuis/ace-gguf",
        isGguf: true,
        modelFormat: "gguf",
        pipelineTag: "text-to-audio",
        task: "text-generation",
        tags: ["gguf", "text-to-audio"],
      }),
    ),
    true,
  );
  // llama.cpp has no CSM decoder, whatever library the card names.
  assert.equal(
    refused(
      audio({
        id: "ggml-org/sesame-csm-1b-GGUF",
        hubRepoId: "ggml-org/sesame-csm-1b-GGUF",
        isGguf: true,
        modelFormat: "gguf",
        pipelineTag: "text-to-speech",
        task: "text-generation",
        tags: ["gguf", "text-to-speech"],
      }),
    ),
    true,
  );
  // Chat models, transcription and filesystem rows keep their chat Run.
  assert.equal(
    refused(
      audio({ hubRepoId: "unsloth/Qwen3-8B", pipelineTag: "text-generation" }),
    ),
    false,
  );
  assert.equal(
    refused(
      audio({
        hubRepoId: "nvidia/parakeet-tdt-0.6b-v3",
        pipelineTag: "automatic-speech-recognition",
        libraryName: "nemo",
      }),
    ),
    false,
  );
  assert.equal(refused({ ...magpie, pipelineTag: undefined }), false);
  assert.equal(
    refused({
      ...mediaModel("text-to-speech", "local", "models_dir"),
      hubRepoId: "openbmb/VoxCPM2",
    }),
    false,
  );
});

test("a cached separation GGUF opens Separate: its audio type reaches the gate", () => {
  const model: SelectedModelView = {
    ...mediaModel(undefined, "cache"),
    id: "x/HTDemucs-GGUF",
    hubRepoId: "x/HTDemucs-GGUF",
    isGguf: true,
    runtimeCanChat: true,
    task: "audio-to-audio",
    audioType: "audiocpp_sep",
  };
  const task = taskForMediaPick(model.pipelineTag, model.task);
  assert.equal(hubModelRunsOnAudioPage(model, task), true);
  // Any other audio-to-audio row (enhancers, codecs) still keeps its chat Run.
  assert.equal(
    hubModelRunsOnAudioPage({ ...model, audioType: null }, task),
    false,
  );
});

test("the Hub forwards the row's audio type into the Audio handoff", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/features/hub/hub-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /audioPickSearch\([\s\S]{0,400}audioType: selectedModel\.audioType/,
  );
});

test("an offline cached curated audio GGUF still opens Audio from its catalog task", async () => {
  const { hubAudioTask } = await import(
    "../src/features/hub/lib/model-run-selection.ts"
  );
  const model: SelectedModelView = {
    ...mediaModel(undefined, "cache"),
    id: "unsloth/orpheus-3b-0.1-ft-GGUF",
    hubRepoId: "unsloth/orpheus-3b-0.1-ft-GGUF",
    isGguf: true,
    runtimeCanChat: true,
    // No Hub tag offline: the backend reports the generic task for a cached GGUF.
    task: "text-generation",
  };
  const task = taskForMediaPick(model.pipelineTag, model.task);
  assert.equal(hubAudioTask(model, task), "text-to-speech");
  assert.equal(hubModelRunsOnAudioPage(model, task), true);
  // A chat GGUF with the same generic task keeps its chat Run.
  const chat = {
    ...model,
    id: "unsloth/Qwen3-8B-GGUF",
    hubRepoId: "unsloth/Qwen3-8B-GGUF",
  };
  assert.equal(hubAudioTask(chat, task), "text-generation");
  assert.equal(hubModelRunsOnAudioPage(chat, task), false);
});
