// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  resolveDiffusionDeployBase,
  resolveDiffusionTrainingBase,
} from "../src/features/images/train/diffusion-train-deploy.ts";

import { readSrcAsync } from "./helpers/kit.ts";

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
  assert.equal(resolveDiffusionTrainingBase(klein, "krea/Krea-2-Turbo"), null);
  assert.equal(resolveDiffusionTrainingBase(klein, "unsloth/FLUX.2-klein-42B"), null);
});

test("a mirror-loaded checkpoint preselects the vendor base it copies", () => {
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
  const source = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");
  const paired = source.indexOf("pairedTrainingBase ??");
  const first = source.indexOf("family.base_repos[0]");
  assert.ok(paired > 0, "the preselect no longer falls back to the paired training base");
  assert.ok(paired < first && first - paired < 40, "base_repos[0] is no longer the last resort");
  assert.match(source, /resolveDiffusionTrainingBase\(reportedFamily, loadedBaseRepo\)/);
});

// What the backend's family_train_infos returns for Qwen-Image-2.1: both training ids deploy on upstream Turbo.
const qwen21 = {
  name: "qwen-image-2.1",
  label: "Qwen-Image-2.1",
  default_base: "Qwen/Qwen-Image-2.1",
  base_repos: ["Qwen/Qwen-Image-2.1"],
  deploy_bases: {
    "Qwen/Qwen-Image-2.1": "Qwen/Qwen-Image-2.1-Turbo",
    "unsloth/Qwen-Image-2.1": "Qwen/Qwen-Image-2.1-Turbo",
  },
};

test("a Qwen-Image-2.1 adapter deploys on Turbo, and a loaded Turbo preselects 2.1", () => {
  assert.equal(
    resolveDiffusionDeployBase(qwen21, "Qwen/Qwen-Image-2.1"),
    "Qwen/Qwen-Image-2.1-Turbo",
  );
  assert.equal(
    resolveDiffusionDeployBase(qwen21, "unsloth/Qwen-Image-2.1"),
    "Qwen/Qwen-Image-2.1-Turbo",
  );
  assert.equal(
    resolveDiffusionTrainingBase(qwen21, "Qwen/Qwen-Image-2.1-Turbo"),
    "Qwen/Qwen-Image-2.1",
  );
  // 2.1 itself is the training base, not a distilled pair, so the reverse lookup does not claim it.
  assert.equal(resolveDiffusionTrainingBase(qwen21, "Qwen/Qwen-Image-2.1"), null);
});

// Qwen-Image-2.1 and Turbo load int8 / fp8 by default, and those builds refuse an adapter picked after the load.
test("Deploy to Create bakes the trained adapter into the load", async () => {
  const source = await readSrcAsync("features/images/images-page.tsx");
  const start = source.indexOf("const handleDeployAdapter = useCallback(");
  const deploy = source.slice(start, source.indexOf("const deployFromVideo", start));
  assert.match(deploy, /const deployLoras: LoraSpecInput\[\] = \[\{ id: stem, weight: 1 \}\];/);
  assert.match(
    deploy,
    /handleLoad\(args\.baseRepo, \{ kind: "pipeline" \}, \{\s*\.\.\.currentLoadAdvanced\(args\.baseRepo\),\s*loras: deployLoras,\s*\}\)/,
  );
  // A load that never starts leaves the previous model resident, so its own selection comes back.
  assert.match(deploy, /if \(!started\) \{\s*pendingDeploy\.current = null;\s*setLoras\(prevLoras\);/);
});
