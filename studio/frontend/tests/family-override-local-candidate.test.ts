// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isFamilyOverrideLocalCandidate,
  localArtifactPassesOverrideGate,
} from "../src/features/model-picker/components/model-selector/family-override-local-candidate.ts";
import { familyOverrideOptions } from "../src/features/model-picker/components/model-selector/family-override-options.ts";

const row = (
  artifactKind:
    | "diffusers_pipeline"
    | "diffusers_modular_pipeline"
    | "transformers_model"
    | "single_file_checkpoint"
    | "gguf"
    | "adapter"
    | "unknown",
  task: string | null = null,
) => ({ artifact_kind: artifactKind, task });

test("only an opaque pipeline directory qualifies under an explicit family", () => {
  assert.equal(
    isFamilyOverrideLocalCandidate(row("diffusers_pipeline"), "z-image", "image"),
    true,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(row("diffusers_pipeline"), "auto", "image"),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(row("diffusers_pipeline"), undefined, "image"),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(
      row("diffusers_pipeline", "text-to-image"),
      "z-image",
      "image",
    ),
    false,
  );
});

test("the same structural contract qualifies an opaque cached snapshot", () => {
  const cachedSnapshot = {
    repo_id: "unsloth/custom-finetune",
    artifact_kind: "diffusers_pipeline" as const,
    task: null,
  };
  assert.equal(isFamilyOverrideLocalCandidate(cachedSnapshot, "z-image", "image"), true);
});

test("components, shards, adapters and ordinary model directories never qualify", () => {
  for (const kind of [
    "transformers_model",
    "single_file_checkpoint",
    "gguf",
    "adapter",
    "unknown",
  ] as const) {
    assert.equal(
      isFamilyOverrideLocalCandidate(row(kind), "z-image", "image"),
      false,
      kind,
    );
  }
});

test("an opaque pipeline stays hidden until the explicit contract applies", () => {
  assert.equal(
    localArtifactPassesOverrideGate(row("diffusers_pipeline"), "auto", "image"),
    false,
  );
  assert.equal(
    localArtifactPassesOverrideGate(row("diffusers_pipeline"), "z-image", "image"),
    true,
  );
  assert.equal(
    localArtifactPassesOverrideGate(row("transformers_model"), "auto", "image"),
    true,
  );
});

test("a modular-only root is eligible for video but never for images", () => {
  const modular = row("diffusers_modular_pipeline");
  assert.equal(
    isFamilyOverrideLocalCandidate(modular, "z-image", "image"),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(modular, "ltx-video", "video"),
    true,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(modular, "auto", "video"),
    false,
  );
  assert.equal(
    localArtifactPassesOverrideGate(modular, "z-image", "image"),
    false,
  );
  assert.equal(
    localArtifactPassesOverrideGate(modular, "ltx-video", "video"),
    true,
  );
});

test("family options follow the backend registry and deduplicate names", () => {
  assert.deepEqual(familyOverrideOptions(["z-image", "z-image", "flux.1"]), [
    ["auto", "Auto (detect)"],
    ["z-image", "Z-Image"],
    ["flux.1", "FLUX.1"],
  ]);
});
