// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { shouldShowTrainingArtifactsDeleted } from "../src/features/training/lib/run-display.ts";

type TrainingStatus = "completed" | "error" | "running" | "stopped";

function artifactState(
  status: TrainingStatus,
  outputDir: string | null,
  artifactsAvailable: boolean,
) {
  return {
    status,
    // biome-ignore lint/style/useNamingConvention: API schema
    output_dir: outputDir,
    // biome-ignore lint/style/useNamingConvention: API schema
    artifacts_available: artifactsAvailable,
  };
}

test("labels artifacts as deleted only when a recorded output directory is gone", () => {
  assert.equal(
    shouldShowTrainingArtifactsDeleted(
      artifactState("completed", "/outputs/run-1", false),
    ),
    true,
  );
  for (const run of [
    artifactState("error", null, false),
    artifactState("stopped", "", false),
    artifactState("running", "/outputs/run-1", false),
    artifactState("completed", "/outputs/run-1", true),
  ]) {
    assert.equal(shouldShowTrainingArtifactsDeleted(run), false);
  }
});
