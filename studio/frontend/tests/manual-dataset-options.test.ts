// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ManualDatasetOptionDrafts,
  createManualDatasetOptionDrafts,
  manualDatasetOptionsFormKey,
  manualDatasetSplitDefault,
  normalizeManualDatasetOption,
  synchronizeManualDatasetOptionDrafts,
  validateManualDatasetSplit,
  validateManualDatasetSubset,
} from "../src/features/training/lib/manual-dataset-options.ts";

test("manual split defaults preserve remote behavior but require local certainty", () => {
  assert.equal(manualDatasetSplitDefault(false), "train");
  assert.equal(manualDatasetSplitDefault(true), "");
});

test("manual dataset options accept backend-compatible config and split names", () => {
  assert.equal(validateManualDatasetSubset("en_US-v2.1"), null);
  assert.equal(validateManualDatasetSubset("config with spaces"), null);
  assert.equal(validateManualDatasetSubset("cönfig"), null);
  assert.equal(validateManualDatasetSubset("v1..v2"), null);
  assert.equal(validateManualDatasetSubset(""), null);
  assert.equal(validateManualDatasetSplit("validation", true), null);
  assert.equal(validateManualDatasetSplit("tréin", true), null);
  assert.equal(validateManualDatasetSplit("train.clean", true), null);
  assert.equal(validateManualDatasetSplit("train[:10%]", true), null);
  assert.equal(validateManualDatasetSplit("train[1_000:2_000]", true), null);
  assert.equal(
    validateManualDatasetSplit(
      "test[:-5%](pct1_dropremainder) + train[40%:60%](pct1_dropremainder)",
      true,
    ),
    null,
  );
  assert.equal(validateManualDatasetSplit("", false), null);
  assert.equal(normalizeManualDatasetOption("  validation  "), "validation");
});

test("manual dataset options reject missing, traversing, and unsupported values", () => {
  assert.equal(validateManualDatasetSplit("", true), "required");
  assert.equal(validateManualDatasetSubset("../private"), "invalid");
  assert.equal(validateManualDatasetSubset(".."), "invalid");
  assert.equal(validateManualDatasetSubset("config:name"), "invalid");
  assert.equal(validateManualDatasetSubset("config\nname"), "invalid");
  assert.equal(validateManualDatasetSplit("train/value", true), "invalid");
  assert.equal(validateManualDatasetSplit("train-clean", true), "invalid");
  assert.equal(validateManualDatasetSplit("train evil", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[10%", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[101%:]", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[101:20%]", true), "invalid");
  assert.equal(validateManualDatasetSplit("train[-101:20%]", true), "invalid");
  assert.equal(validateManualDatasetSubset("config\u200bname"), "invalid");
  assert.equal(validateManualDatasetSubset("config\ud800name"), "invalid");
  assert.equal(
    validateManualDatasetSplit("train[10:20](closest)", true),
    "invalid",
  );
  assert.equal(
    validateManualDatasetSplit(
      "test[:-5%] + train[40%:60%](pct1_dropremainder)",
      true,
    ),
    "invalid",
  );
  assert.equal(validateManualDatasetSplit("x".repeat(129), true), "too_long");
});

test("manual option length counts Unicode code points like the backend", () => {
  const astralLetter = "\u{10400}";

  assert.equal(validateManualDatasetSubset(astralLetter.repeat(128)), null);
  assert.equal(
    validateManualDatasetSubset(astralLetter.repeat(129)),
    "too_long",
  );
  assert.equal(validateManualDatasetSplit(astralLetter.repeat(128), true), null);
  assert.equal(
    validateManualDatasetSplit(astralLetter.repeat(129), true),
    "too_long",
  );
});

test("manual form identity and field synchronization preserve unrelated drafts", () => {
  const formKey = manualDatasetOptionsFormKey("org/dataset", null, false);
  const initial = createManualDatasetOptionDrafts({
    datasetSubset: null,
    datasetSplit: "train",
    datasetEvalSplit: null,
    defaultSplit: "train",
  });
  const withInvalidSubset: ManualDatasetOptionDrafts = {
    ...initial,
    subset: {
      ...initial.subset,
      value: "../private",
      error: "invalid",
    },
  };
  const synchronized = synchronizeManualDatasetOptionDrafts(
    withInvalidSubset,
    {
      datasetSubset: null,
      datasetSplit: "validation",
      datasetEvalSplit: null,
      defaultSplit: "train",
    },
  );

  assert.equal(
    manualDatasetOptionsFormKey("org/dataset", null, false),
    formKey,
  );
  assert.strictEqual(synchronized.subset, withInvalidSubset.subset);
  assert.deepEqual(synchronized.subset, {
    committedValue: null,
    value: "../private",
    error: "invalid",
  });
  assert.deepEqual(synchronized.split, {
    committedValue: "validation",
    value: "validation",
    error: null,
  });
  assert.strictEqual(synchronized.evalSplit, withInvalidSubset.evalSplit);
  assert.notEqual(
    manualDatasetOptionsFormKey("org/other-dataset", null, false),
    formKey,
  );
});

test("manual field synchronization does not revive a draft after A to B to A", () => {
  const atTrain: ManualDatasetOptionDrafts = {
    ...createManualDatasetOptionDrafts({
      datasetSubset: null,
      datasetSplit: "train",
      datasetEvalSplit: null,
      defaultSplit: "train",
    }),
    split: {
      committedValue: "train",
      value: "train[",
      error: "invalid",
    },
  };
  const atValidation = synchronizeManualDatasetOptionDrafts(atTrain, {
    datasetSubset: null,
    datasetSplit: "validation",
    datasetEvalSplit: null,
    defaultSplit: "train",
  });
  const backAtTrain = synchronizeManualDatasetOptionDrafts(atValidation, {
    datasetSubset: null,
    datasetSplit: "train",
    datasetEvalSplit: null,
    defaultSplit: "train",
  });

  assert.deepEqual(atValidation.split, {
    committedValue: "validation",
    value: "validation",
    error: null,
  });
  assert.deepEqual(backAtTrain.split, {
    committedValue: "train",
    value: "train",
    error: null,
  });
});

test("streaming manual options require bare split names", () => {
  assert.equal(validateManualDatasetSplit("train", true, false), null);
  assert.equal(
    validateManualDatasetSplit("train[:10%]", true, false),
    "invalid",
  );
  assert.equal(
    validateManualDatasetSplit("train + validation", true, false),
    "invalid",
  );
});
