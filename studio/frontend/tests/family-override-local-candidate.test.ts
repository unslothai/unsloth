// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  artifactKindSupportsFamilyOverride,
  familyOverrideArtifactKind,
  resolvedFamilyOverrideSelection,
  taskOpaqueArtifactSupportsFamilyOverride,
} from "../src/features/model-picker/components/model-selector/family-override-local-candidate.ts";
import { familyOverrideOptions } from "../src/features/model-picker/components/model-selector/family-override-options.ts";

test("an explicit family selects exactly one structurally loadable artifact kind", () => {
  assert.equal(familyOverrideArtifactKind("auto", "image"), undefined);
  assert.equal(familyOverrideArtifactKind(undefined, "video"), undefined);
  assert.equal(
    familyOverrideArtifactKind("z-image", "image"),
    "diffusers_pipeline",
  );
  assert.equal(
    familyOverrideArtifactKind("ltx-2", "video", ["minimax-h3"]),
    "diffusers_pipeline",
  );
  assert.equal(
    familyOverrideArtifactKind(" MINIMAX-H3 ", "video", ["minimax-h3"]),
    "diffusers_modular_pipeline",
  );
});

test("a dual-manifest root satisfies either family loader contract", () => {
  assert.equal(
    artifactKindSupportsFamilyOverride(
      "diffusers_dual_pipeline",
      "diffusers_pipeline",
    ),
    true,
  );
  assert.equal(
    artifactKindSupportsFamilyOverride(
      "diffusers_dual_pipeline",
      "diffusers_modular_pipeline",
    ),
    true,
  );
  assert.equal(
    artifactKindSupportsFamilyOverride(
      "diffusers_pipeline",
      "diffusers_modular_pipeline",
    ),
    false,
  );
});

test("a structural family override never crosses a known task boundary", () => {
  assert.equal(
    taskOpaqueArtifactSupportsFamilyOverride(
      null,
      "diffusers_pipeline",
      "diffusers_pipeline",
    ),
    true,
  );
  assert.equal(
    taskOpaqueArtifactSupportsFamilyOverride(
      "text-to-image",
      "diffusers_pipeline",
      "diffusers_pipeline",
    ),
    false,
  );
  assert.equal(
    taskOpaqueArtifactSupportsFamilyOverride(
      "text-to-video",
      "diffusers_dual_pipeline",
      "diffusers_modular_pipeline",
    ),
    false,
  );
});

test("selector restoration prefers the canonical engaged family over an alias", () => {
  assert.equal(
    resolvedFamilyOverrideSelection({
      source: "explicit",
      requested: "h3",
      value: "minimax-h3",
    }),
    "minimax-h3",
  );
  assert.equal(
    resolvedFamilyOverrideSelection({
      source: "auto",
      requested: null,
      value: "minimax-h3",
    }),
    "auto",
  );
});

test("family options follow the backend registry and deduplicate names", () => {
  assert.deepEqual(familyOverrideOptions(["z-image", "z-image", "flux.1"]), [
    ["auto", "Auto (detect)"],
    ["z-image", "Z-Image"],
    ["flux.1", "FLUX.1"],
  ]);
});
