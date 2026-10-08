// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  datasetNamesForCreation,
  existingDatasetName,
  freeDatasetName,
  isDatasetContinuation,
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

test("a captions-only set created in this form remains available for follow-up files", () => {
  assert.equal(isDatasetContinuation("captions-only", "captions-only"), true);
  assert.equal(isDatasetContinuation(" captions-only ", "captions-only"), true);
  assert.equal(isDatasetContinuation("CAPTIONS-ONLY", "captions-only"), false);
  assert.equal(isDatasetContinuation("another-set", "captions-only"), false);
  assert.equal(isDatasetContinuation("captions-only", null), false);
  assert.match(source, /setContinuationDatasetName\(res\.name\)/);
  assert.match(source, /const createsDataset = uploadMode && !continuingUploadName;/);
  assert.match(
    source,
    /const resultCaptionsOnly = res\.image_count === 0 && \(res\.clip_count \?\? 0\) === 0;/,
  );
  assert.match(
    source,
    /\(info\?\.continuation_dataset_names \?\? \[\]\)\.includes\(takenName\)/,
  );
  assert.match(source, /\{takenNameUnlisted && \(/);
});

test("the new-set form waits for the set list before it uploads", () => {
  assert.match(
    source,
    /uploadMode && \(infoLoadState === "idle" \|\| infoLoadState === "loading"\)/,
  );
  assert.match(source, /const namesUnavailable = uploadMode && infoLoadState === "failed";/);
});

test("a failed initial or stale set-list request can be retried without enabling uploads", () => {
  const refresh = source.slice(
    source.indexOf("const infoRequestId"),
    source.indexOf("// On first activation"),
  );
  assert.match(refresh, /setInfoLoadState\("loading"\)/);
  assert.match(refresh, /setInfoLoadState\("loaded"\)/);
  assert.match(
    refresh,
    /catch \{\s+if \(requestId === infoRequestId\.current\) setInfoLoadState\("failed"\)/,
  );
  assert.match(refresh, /if \(requestId === infoRequestId\.current\) \{\s+setInfo\(i\);/);

  const form = source.slice(source.indexOf("{uploadMode ? ("));
  const newSet = form.slice(0, form.indexOf(") : ("));
  assert.match(newSet, /\{namesUnavailable && \(/);
  assert.match(newSet, /onClick=\{\(\) => void refreshInfo\(\)\}/);
  assert.match(newSet, />\s*Retry\s*<\/Button>/);
});

test("the initial upload form replaces an occupied default name after inventory loads", () => {
  assert.match(
    source,
    /setUploadName\(\(current\) =>\s+existingDatasetName\(current, occupiedDatasets\)\s+\? freeDatasetName\(occupiedDatasets\)\s+: current/,
  );
  assert.match(source, /\[uploadMode, continuingUploadName, occupiedDatasets\]/);
});

test("the new-set form does not upload into a set that already exists", () => {
  const form = source.slice(source.indexOf("{uploadMode ? ("));
  const newSet = form.slice(0, form.indexOf(") : ("));
  assert.equal(
    newSet.match(
      /disabled=\{uploading \|\| namesLoading \|\| namesUnavailable \|\| takenName !== null\}/g,
    )?.length,
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
  assert.ok(drop.indexOf("if (namesUnavailable)") >= 0);
  assert.ok(drop.indexOf("if (namesUnavailable)") < drop.indexOf("await uploadTo("));
  assert.match(drop, /await uploadTo\(dropTarget, dropped, createsDataset\)/);

  assert.match(newSet, /void uploadTo\(uploadName\.trim\(\), files, createsDataset\)/);
  assert.match(newSet, /pickFolder\(uploadName\.trim\(\), createsDataset\)/);
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

test("a safe captions-only folder can be continued from the new-set form", () => {
  assert.match(
    source,
    /const takenNameUnlisted =\s+takenName !== null && \(info\?\.continuation_dataset_names \?\? \[\]\)\.includes\(takenName\);/,
  );
  const form = source.slice(source.indexOf("{uploadMode ? ("));
  const newSet = form.slice(0, form.indexOf(") : ("));
  assert.match(
    newSet,
    /\{takenNameUnlisted && \([\s\S]*?setUploadName\(takenName\);\s+setContinuationDatasetName\(takenName\);[\s\S]*?Add to it/,
  );
});

test("picking any entry in the dataset list ends a continuation", () => {
  const change = source.slice(source.indexOf("onValueChange={(v) => {"));
  const body = change.slice(0, change.indexOf("setDataset(v);"));
  assert.match(body, /setContinuationDatasetName\(null\);/);
  assert.ok(body.indexOf("return;") < body.indexOf("setContinuationDatasetName(null)"));
});

test("a name the user typed is never replaced by a generated one", () => {
  assert.match(
    source,
    /if \(!uploadMode \|\| continuingUploadName \|\| uploadNameEdited\.current\) return;/,
  );
  const form = source.slice(source.indexOf("{uploadMode ? ("));
  const newSet = form.slice(0, form.indexOf(") : ("));
  assert.match(newSet, /uploadNameEdited\.current = true;\s+setUploadName\(e\.target\.value\);/);
});
