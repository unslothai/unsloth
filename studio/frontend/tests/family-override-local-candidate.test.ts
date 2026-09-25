// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import {
  artifactKindSupportsFamilyOverride,
  familyOverrideArtifactKind,
  familyOverrideForPick,
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

test("normal picks drop stale overrides while opaque picks keep their classifier", () => {
  assert.equal(familyOverrideForPick("z-image", false), undefined);
  assert.equal(familyOverrideForPick(" z-image ", true), "z-image");
  assert.equal(familyOverrideForPick("auto", true), undefined);
});

test("family options follow the backend registry and deduplicate names", () => {
  assert.deepEqual(familyOverrideOptions(["z-image", "z-image", "flux.1"]), [
    ["auto", "Auto (detect)"],
    ["z-image", "Z-Image"],
    ["flux.1", "FLUX.1"],
  ]);
});

test("every backend media family has a display label", () => {
  const root = new URL("../../backend/core/inference/", import.meta.url);
  const names = ["diffusion_families.py", "video_families.py"].flatMap((f) =>
    [
      ...readFileSync(new URL(f, root), "utf8").matchAll(
        /^\s+name = "([^"]+)"/gm,
      ),
    ].map((m) => m[1]),
  );
  assert.ok(names.includes("flux.1") && names.includes("ltx-2"));
  const unlabeled = familyOverrideOptions(names)
    .slice(1)
    .filter(([value, label]) => value === label)
    .map(([value]) => value);
  assert.deepEqual(unlabeled, []);
});
