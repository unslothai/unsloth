// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runningTrainingSelection } from "../src/features/images/train/diffusion-train-restore.ts";

const families = [
  { name: "flux.1", base_repos: ["black-forest-labs/FLUX.1-dev"] },
  { name: "moss-image", base_repos: ["OpenMOSS/MOSS-Image"] },
  { name: "ltx-2.5", base_repos: ["Lightricks/LTX-2.5-Diffusers"] },
];
// Shape of /training/status a few seconds into an LTX-2.5 run (Colab L4 capture).
const loading = {
  active: true,
  status: "running",
  job_id: "bee76c44",
  family: null,
  base_model: "Lightricks/LTX-2.5-Diffusers",
};

test("a running LTX-2.5 job restores its family while family is still null", () => {
  assert.deepEqual(runningTrainingSelection(loading, families), {
    familyName: "ltx-2.5",
    baseModel: "Lightricks/LTX-2.5-Diffusers",
  });
});

test("the reported family wins once the backend sets it", () => {
  assert.equal(
    runningTrainingSelection({ ...loading, family: "moss-image", base_model: "me/moss-finetune" }, families)
      ?.familyName,
    "moss-image",
  );
  assert.equal(
    runningTrainingSelection({ ...loading, base_model: "black-forest-labs/FLUX.1-dev" }, families)?.familyName,
    "flux.1",
  );
});

test("idle, terminal and unknown-family statuses leave the form alone", () => {
  assert.equal(runningTrainingSelection(null, families), null);
  assert.equal(runningTrainingSelection({ ...loading, active: false, status: "completed" }, families), null);
  assert.equal(runningTrainingSelection({ ...loading, base_model: "x/unknown" }, families), null);
});

test("the train panel seeds the form from the running job", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /runningTrainingSelection\(status, families\)/);
});
