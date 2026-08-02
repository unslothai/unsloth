// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createHfBrowseDatasetSelection, datasetSelectionStreamingPatch } =
  await import("../src/features/training/stores/training-config-policy.ts");

test("selecting an on-device Hub dataset disables streaming", () => {
  const deviceOptions = {
    knownCached: true,
    localPath: "/cache/datasets--org--dataset",
    preferLocalCache: true,
  };
  const cachedSelection = createHfBrowseDatasetSelection(
    "org/dataset",
    deviceOptions,
  );

  assert.deepEqual(
    datasetSelectionStreamingPatch(cachedSelection, deviceOptions),
    { datasetStreaming: false },
  );
  assert.deepEqual(datasetSelectionStreamingPatch(cachedSelection), {});
  assert.deepEqual(
    datasetSelectionStreamingPatch(
      createHfBrowseDatasetSelection("org/remote-dataset"),
    ),
    {},
  );
});
