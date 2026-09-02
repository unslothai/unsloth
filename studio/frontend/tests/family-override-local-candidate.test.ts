// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  familyOverrideArtifactKind,
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

test("family options follow the backend registry and deduplicate names", () => {
  assert.deepEqual(familyOverrideOptions(["z-image", "z-image", "flux.1"]), [
    ["auto", "Auto (detect)"],
    ["z-image", "Z-Image"],
    ["flux.1", "FLUX.1"],
  ]);
});
