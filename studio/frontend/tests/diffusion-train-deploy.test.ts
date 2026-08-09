// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveDiffusionDeployBase } from "../src/features/images/train/diffusion-train-deploy.ts";

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
