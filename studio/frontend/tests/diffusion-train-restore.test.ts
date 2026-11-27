// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runningTrainingSelection, trainingRunSelection } from "../src/features/images/train/diffusion-train-restore.ts";

const families = [
  { name: "flux.1", base_repos: ["black-forest-labs/FLUX.1-dev"] },
  { name: "moss-image", base_repos: ["OpenMOSS/MOSS-Image"] },
  { name: "ltx-2.5", base_repos: ["Lightricks/LTX-2.5-Diffusers"] },
  { name: "z-image", base_repos: ["Tongyi-MAI/Z-Image-Turbo"] },
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
  assert.match(src, /runningTrainingSelection\(status, families/);
});

// A Z-Image-Turbo run resumed from a checkpoint that has finished: the backend keeps this status until the next start.
const finishedZImage = {
  active: false,
  status: "completed",
  job_id: "c0ffee11",
  family: "z-image",
  base_model: "Tongyi-MAI/Z-Image-Turbo",
};

test("a finished run seeds an untouched form, not the FLUX.1-dev default", () => {
  for (const st of ["completed", "stopped", "error"]) {
    assert.deepEqual(
      runningTrainingSelection({ ...finishedZImage, status: st }, families, { formUntouched: true }),
      { familyName: "z-image", baseModel: "Tongyi-MAI/Z-Image-Turbo" },
      st,
    );
  }
});

test("a finished run leaves a form the user changed alone", () => {
  assert.equal(runningTrainingSelection(finishedZImage, families, { formUntouched: false }), null);
  assert.equal(
    runningTrainingSelection({ ...finishedZImage, status: "idle" }, families, { formUntouched: true }),
    null,
  );
});

test("a resumed run's record names its family and base", () => {
  assert.deepEqual(trainingRunSelection({ family: null, base_model: "Tongyi-MAI/Z-Image-Turbo" }, families), {
    familyName: "z-image",
    baseModel: "Tongyi-MAI/Z-Image-Turbo",
  });
  assert.equal(trainingRunSelection({ family: "nope", base_model: "x/y" }, families), null);
});

test("the train panel seeds from finished runs and from the resumed run's detail", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /runningTrainingSelection\(status, families, \{\s*formUntouched: presetsSettled && !modelPickedByUser\.current/);
  const resume = src.slice(src.indexOf("const onResume"), src.indexOf("const onStop"));
  assert.ok(
    resume.indexOf("trainingRunSelection(detail, families)") > -1 &&
      resume.indexOf("trainingRunSelection(detail, families)") < resume.indexOf("startDiffusionTraining("),
    "onResume sets family and base from detail before starting",
  );
});

test("a restored named preset keeps its model when a finished run's status arrives", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  const apply = src.slice(src.indexOf("const applyPresetParams"), src.indexOf("setFamilyName(params.familyName)"));
  assert.match(apply, /if \(owned\) \{[^}]*modelPickedByUser\.current = true/);
});

test("a finished run waits for the stored preset before seeding the form", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /useState\(!savedPresets\)/);
  assert.match(src, /if \(presets\.presetsReady\) onReady\(\)/);
  assert.match(src, /onReady=\{markPresetsSettled\}/);
});

test("opening a run from history points an untouched form at it, never over a user's pick", () => {
  const src = readFileSync(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  const open = src.slice(src.indexOf("const openPrevRun = useCallback("), src.indexOf("const viewLossHistory"));
  assert.match(
    open,
    /if \(!modelPickedByUser\.current\) \{\s*const opened = trainingRunSelection\(detail, families\);\s*if \(opened\) applyRunSelection\(opened\);/,
  );
});
