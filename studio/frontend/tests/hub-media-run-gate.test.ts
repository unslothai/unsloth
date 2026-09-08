// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { SelectedModelView } from "../src/features/hub/types.ts";
import { studioPageForTask } from "../src/features/hub/lib/unsloth-support.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isHubModelRunEligible } = await import(
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
