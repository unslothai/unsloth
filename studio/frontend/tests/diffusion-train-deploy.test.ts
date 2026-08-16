// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  resolveDiffusionDeployBase,
  resolveDiffusionTrainingBase,
} from "../src/features/images/train/diffusion-train-deploy.ts";

const klein = {
  name: "flux.2-klein",
  label: "FLUX.2 Klein",
  default_base: "black-forest-labs/FLUX.2-klein-base-4B",
  base_repos: [
    "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-base-9B",
  ],
  deploy_bases: {
    "black-forest-labs/FLUX.2-klein-base-4B":
      "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-base-9B":
      "black-forest-labs/FLUX.2-klein-9B",
    "unsloth/FLUX.2-klein-base-9B": "unsloth/FLUX.2-klein-9B",
  },
};

test("deploys each Klein base adapter on its matching distilled checkpoint", () => {
  assert.equal(
    resolveDiffusionDeployBase(klein, "black-forest-labs/FLUX.2-klein-base-4B"),
    "black-forest-labs/FLUX.2-klein-4B",
  );
  assert.equal(
    resolveDiffusionDeployBase(klein, "BLACK-FOREST-LABS/FLUX.2-KLEIN-BASE-9B"),
    "black-forest-labs/FLUX.2-klein-9B",
  );
  assert.equal(
    resolveDiffusionDeployBase(klein, "unsloth/FLUX.2-klein-base-9B"),
    "unsloth/FLUX.2-klein-9B",
  );
});

test("keeps custom bases and the legacy family-wide mapping working", () => {
  assert.equal(
    resolveDiffusionDeployBase(klein, "/models/custom-klein"),
    "/models/custom-klein",
  );
  assert.equal(
    resolveDiffusionDeployBase(
      {
        ...klein,
        base_repos: ["krea/Krea-2-Raw"],
        deploy_bases: {},
        deploy_base: "krea/Krea-2-Turbo",
      },
      "krea/Krea-2-Raw",
    ),
    "krea/Krea-2-Turbo",
  );
});

test("preselects the training base paired with a loaded distilled checkpoint", () => {
  assert.equal(
    resolveDiffusionTrainingBase(klein, "black-forest-labs/FLUX.2-klein-9B"),
    "black-forest-labs/FLUX.2-klein-base-9B",
  );
  assert.equal(
    resolveDiffusionTrainingBase(klein, "BLACK-FOREST-LABS/FLUX.2-KLEIN-4B"),
    "black-forest-labs/FLUX.2-klein-base-4B",
  );
});

test("returns null rather than inventing a base the backend would refuse", () => {
  assert.equal(resolveDiffusionTrainingBase(undefined, "black-forest-labs/FLUX.2-klein-9B"), null);
  assert.equal(resolveDiffusionTrainingBase(klein, ""), null);
  // Loaded checkpoint the family declares no pairing for.
  assert.equal(resolveDiffusionTrainingBase(klein, "krea/Krea-2-Turbo"), null);
  // A repo whose name matches nothing the family offers stays null.
  assert.equal(resolveDiffusionTrainingBase(klein, "unsloth/FLUX.2-klein-42B"), null);
});

test("a mirror-loaded checkpoint preselects the vendor base it copies", () => {
  // Deploy hands a LoRA trained on unsloth/FLUX.2-klein-base-9B the mirror checkpoint
  // unsloth/FLUX.2-klein-9B, so that is what /images/status reports afterwards. Its pairing names
  // the mirror TRAINING id, which base_repos does not offer, and the panel then fell back to the
  // first base: the 4B, seeding a 9B workflow from 4B weights. A mirror keeps the upstream name.
  assert.equal(
    resolveDiffusionTrainingBase(klein, "unsloth/FLUX.2-klein-9B"),
    "black-forest-labs/FLUX.2-klein-base-9B",
  );
  assert.equal(
    resolveDiffusionTrainingBase(klein, "UNSLOTH/FLUX.2-KLEIN-9B"),
    "black-forest-labs/FLUX.2-klein-base-9B",
  );
});

test("the Train panel preselect actually consults the pairing", async () => {
  // The helper on its own changes nothing: the bug was in the preselect chain, which fell from an
  // exact base_repos match straight to base_repos[0]. Assert the pairing sits BETWEEN the two.
  const source = await readFile(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  const paired = source.indexOf("pairedTrainingBase ??");
  const first = source.indexOf("family.base_repos[0]");
  assert.ok(paired > 0, "the preselect no longer falls back to the paired training base");
  assert.ok(paired < first && first - paired < 40, "base_repos[0] is no longer the last resort");
  assert.match(source, /resolveDiffusionTrainingBase\(reportedFamily, loadedBaseRepo\)/);
});
