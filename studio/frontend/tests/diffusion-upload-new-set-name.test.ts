// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  datasetNamesForCreation,
  existingDatasetName,
  freeDatasetName,
} from "../src/features/images/train/dataset-files.ts";

import { readSrcAsync } from "./helpers/kit.ts";

const source = await readSrcAsync(
  "features/images/train/diffusion-train-panel.tsx",
);
const apiSource = await readSrcAsync("features/images/api.ts");

const datasets = [{ name: "my-images" }, { name: "Dogs" }];

test("a new set name that an existing set already uses is caught, whatever its case", () => {
  assert.equal(existingDatasetName("my-images", datasets), "my-images");
  assert.equal(existingDatasetName("  dogs ", datasets), "Dogs");
  assert.equal(existingDatasetName("cats", datasets), null);
  assert.equal(existingDatasetName("my-images", []), null);
});

test("the prefilled name for a new set is one no set uses yet", () => {
  assert.equal(freeDatasetName([]), "my-images");
  assert.equal(freeDatasetName(datasets), "my-images-2");
  assert.equal(
    freeDatasetName([...datasets, { name: "MY-IMAGES-2" }]),
    "my-images-3",
  );
});

test("names of occupied but untrainable folders still reserve a new-set name", () => {
  assert.deepEqual(
    datasetNamesForCreation({
      datasets: [],
      dataset_names: ["captions-only"],
    }),
    [{ name: "captions-only" }],
  );
  assert.deepEqual(
    datasetNamesForCreation({ datasets }),
    datasets,
  );
});

test("the new-set form waits for the set list before it uploads", () => {
  assert.match(source, /const namesLoading = uploadMode && info === null;/);
});

test("the new-set form does not upload into a set that already exists", () => {
  const form = source.slice(source.indexOf("{uploadMode ? ("));
  const newSet = form.slice(0, form.indexOf(") : ("));
  assert.equal(
    newSet.match(/disabled=\{uploading \|\| namesLoading \|\| takenName !== null\}/g)?.length,
    2,
  );
  assert.match(newSet, /\{takenName && \(/);

  const drop = source.slice(
    source.indexOf("const onDrop"),
    source.indexOf("const onStart"),
  );
  assert.ok(drop.indexOf("if (takenName)") >= 0);
  assert.ok(drop.indexOf("if (takenName)") < drop.indexOf("await uploadTo("));
  assert.ok(drop.indexOf("if (namesLoading)") >= 0);
  assert.ok(drop.indexOf("if (namesLoading)") < drop.indexOf("await uploadTo("));
  assert.match(drop, /await uploadTo\(dropTarget, dropped, uploadMode\)/);

  assert.match(newSet, /void uploadTo\(uploadName\.trim\(\), files, true\)/);
  assert.match(newSet, /pickFolder\(uploadName\.trim\(\), true\)/);
  assert.match(source, /uploadDiffusionDataset\(name, chunks\[0\], createOnly\)/);
  assert.match(apiSource, /form\.append\("create_only", createOnly \? "true" : "false"\)/);
});

test("adding to the selected set still uploads into it", () => {
  const add = source.slice(
    source.indexOf("{!uploadMode && selectedDataset && ("),
  );
  const buttons = add.slice(0, add.indexOf("</>"));
  assert.match(buttons, /void uploadTo\(dataset, files\)/);
  assert.doesNotMatch(buttons, /takenName|namesLoading/);
});
