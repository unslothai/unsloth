// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  matchesRememberedModel,
  readImageModel,
  redirectRetiredImagePick,
  rememberImageModel,
} from "../src/features/images/image-model-recall.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();

test("a pipeline pick of a retired FLUX.3 full copy loads the base it copies", () => {
  assert.deepEqual(redirectRetiredImagePick({ repoId: "unsloth/flux-3-image-klein", kind: "pipeline" }), {
    repoId: "black-forest-labs/flux-3-image-klein",
    kind: "pipeline",
  });
  assert.equal(
    redirectRetiredImagePick({ repoId: "Unsloth/FLUX-3-Image-Klein-Base", kind: "pipeline" as const }).repoId,
    "black-forest-labs/flux-3-image-klein-base",
  );
});

test("its root bf16 transformer is the pipeline load; other picks stay as named", () => {
  assert.deepEqual(
    redirectRetiredImagePick({
      repoId: "unsloth/flux-3-image-klein",
      kind: "single_file" as const,
      filename: "model.safetensors",
      vaeFile: "x/vae.safetensors",
    }),
    { repoId: "black-forest-labs/flux-3-image-klein", kind: "pipeline" },
  );
  const other = { repoId: "unsloth/flux-3-image-klein-FP8", kind: "pipeline" as const };
  assert.equal(redirectRetiredImagePick(other), other);
});

test("a remembered retired copy is recalled as its base, so it matches the status the load reports", () => {
  rememberImageModel({ repoId: "unsloth/flux-3-image-klein", kind: "pipeline" });
  const recalled = readImageModel();
  assert.deepEqual(recalled, { repoId: "black-forest-labs/flux-3-image-klein", kind: "pipeline" });
  assert.equal(
    matchesRememberedModel(recalled!, {
      loaded: true,
      repo_id: "black-forest-labs/flux-3-image-klein",
      model_kind: "pipeline",
    }),
    true,
  );
});
