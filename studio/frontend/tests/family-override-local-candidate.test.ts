// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isFamilyOverrideLocalCandidate } from "../src/features/model-picker/components/model-selector/family-override-local-candidate.ts";

const CANONICAL_FAMILY_RESEED =
  /resolvedCanonicalSelectValue\(record\.family_override/;
const FAMILY_OVERRIDE_INVENTORY_ESCAPE =
  /row\.capabilities\.canChat \|\|\s*studioPageForTask\(row\.task\) !== undefined \|\|\s*isFamilyOverrideInventoryCandidate\(row\)/;
const BARE_CHECKPOINT_CAPABILITY_SIGNATURE =
  /row\.modelFormat === "safetensors"[\s\S]*row\.task == null[\s\S]*!row\.capabilities\.canChat[\s\S]*!row\.capabilities\.canTrain[\s\S]*!row\.capabilities\.supportsLora/;

test("an explicit family surfaces only unclassified local safetensors", () => {
  const opaqueSafetensors = { model_format: "safetensors", task: null };
  assert.equal(isFamilyOverrideLocalCandidate(opaqueSafetensors, true), true);
  assert.equal(isFamilyOverrideLocalCandidate(opaqueSafetensors, false), false);
  assert.equal(
    isFamilyOverrideLocalCandidate(
      { model_format: "safetensors", task: "text-generation" },
      true,
    ),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate({ model_format: "gguf", task: null }, true),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(
      { model_format: "unknown", task: null },
      true,
    ),
    false,
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(opaqueSafetensors, true, "ideogram-4"),
    false,
    "pipeline-only image families cannot load a lone checkpoint",
  );
  assert.equal(
    isFamilyOverrideLocalCandidate(opaqueSafetensors, true, "minimax-h3"),
    false,
    "the modular H3 workflow cannot load a lone checkpoint",
  );
});

test("image and video family selectors opt all local inventories into the narrow fallback", () => {
  for (const page of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const source = readFileSync(new URL(page, import.meta.url), "utf8");
    assert.match(
      source,
      /allowUnknownLocalModels=\{familyOverride !== "auto"\}/,
    );
    assert.match(source, /unknownLocalModelFamily=\{familyOverride\}/);
    assert.match(
      source,
      CANONICAL_FAMILY_RESEED,
      "family aliases must reseed from the canonical family that actually loaded",
    );
  }

  const picker = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    picker.match(
      /isFamilyOverrideLocalCandidate\(\s*m,\s*allowUnknownLocalModels,\s*unknownLocalModelFamily,\s*\)/g,
    )?.length ?? 0,
    3,
    "LM Studio, ./models, and custom-folder rows must use the same fallback",
  );
  assert.equal(
    picker.split("Boolean(task) || m.capabilities?.canChat !== false").length - 1,
    3,
    "architecture-less rows stay out of chat while task-scoped media overrides may admit them",
  );

  const inventory = readFileSync(
    new URL(
      "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(inventory, /capabilities: row\.capabilities/);
  assert.match(
    inventory,
    FAMILY_OVERRIDE_INVENTORY_ESCAPE,
    "the shared inventory must preserve the narrow architecture-less row for a later explicit family override",
  );
  assert.match(
    inventory,
    BARE_CHECKPOINT_CAPABILITY_SIGNATURE,
    "only the backend's deliberately non-chat bare-checkpoint capability signature may bypass the inventory guard",
  );
});
