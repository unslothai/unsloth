// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { SelectedModelView } from "../src/features/hub/types.ts";
import { studioPageForTask } from "../src/features/hub/lib/unsloth-support.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { hubModelRunsOnAudioPage, isHubModelRunEligible } = await import(
  "../src/features/hub/lib/model-run-selection.ts"
);
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

// The backend tags a local non-GGUF diffusers checkpoint text-to-image (_local_model_task),
// and it does that for every local row whatever its source, so these rows are real.
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
  // An hf_cache row is a complete Hub snapshot, so it routes like a cached repo.
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
