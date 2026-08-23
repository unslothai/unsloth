// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { stagedDownloadStartFeedback } from "../src/features/hub/download-manager/staged-download-feedback.ts";

test("a refused staged download reports the server reason", () => {
  assert.deepEqual(
    stagedDownloadStartFeedback("error", "The model is loaded"),
    {
      tone: "error",
      title: "Could not start the download",
      description: "The model is loaded",
    },
  );
});

test("a staged download network failure has actionable fallback copy", () => {
  assert.deepEqual(stagedDownloadStartFeedback("error"), {
    tone: "error",
    title: "Could not start the download",
    description: "Check the connection, then select the model again.",
  });
  assert.equal(stagedDownloadStartFeedback("started"), null);
});

test("occupied staged downloads direct the user to the existing job", () => {
  assert.deepEqual(stagedDownloadStartFeedback("conflict"), {
    tone: "info",
    title: "Resume this download from Models",
    description:
      "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
  });
  assert.deepEqual(stagedDownloadStartFeedback("busy"), {
    tone: "info",
    title: "Download already in progress",
    description:
      "Reselect this model once the running download finishes to load it.",
  });
});
