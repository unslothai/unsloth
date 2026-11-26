// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  runningTrainingSelection,
  shownTrainingSelection,
  trainingRunSelection,
} from "../src/features/images/train/diffusion-train-restore.ts";

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
  assert.match(src, /shownTrainingSelection\(status, families, dismissedJobId\)/);
});

// NVIDIA feedback p.40: after a restart the form read FLUX.1-dev beside a completed or resumed Z-Image run.
const zimage = [...families, { name: "z-image", base_repos: ["Tongyi-MAI/Z-Image"] }];
const completed = { active: false, status: "completed", job_id: "z1", family: "z-image", base_model: "Tongyi-MAI/Z-Image" };

test("a finished run the panel still shows seeds the form; a dismissed or idle one does not", () => {
  assert.deepEqual(shownTrainingSelection(completed, zimage, null), {
    familyName: "z-image",
    baseModel: "Tongyi-MAI/Z-Image",
  });
  assert.equal(shownTrainingSelection({ ...completed, status: "stopped" }, zimage, null)?.familyName, "z-image");
  assert.equal(shownTrainingSelection(completed, zimage, "z1"), null);
  assert.equal(shownTrainingSelection({ ...completed, status: "idle" }, zimage, null), null);
  // A running job is still followed.
  assert.equal(shownTrainingSelection({ ...loading }, families, null)?.familyName, "ltx-2.5");
});

test("a run record (resume or history view) maps to its family and base", () => {
  assert.deepEqual(trainingRunSelection({ family: null, base_model: "Tongyi-MAI/Z-Image" }, zimage), {
    familyName: "z-image",
    baseModel: "Tongyi-MAI/Z-Image",
  });
  assert.equal(trainingRunSelection({ family: "z-image", base_model: "me/z-finetune" }, zimage)?.baseModel, "me/z-finetune");
  assert.equal(trainingRunSelection(null, zimage), null);
});

test("resuming and opening a run from history point the form at that run", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  const open = src.slice(src.indexOf("const openPrevRun = useCallback("));
  assert.match(open.slice(0, 400), /const run = trainingRunSelection\(detail, families\);\s*if \(run\) followRun\(run\);/);
  const resume = src.slice(src.indexOf("const onResume = useCallback("));
  assert.match(
    resume.slice(0, resume.indexOf("const onStop")),
    /await startDiffusionTraining\(payload\);[\s\S]*const run = trainingRunSelection\(detail, families\);\s*if \(run\) followRun\(run\);/,
  );
});
