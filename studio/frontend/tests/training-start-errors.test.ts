// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeTrainingStartError } = await import(
  "../src/features/training/lib/training-start-errors.ts"
);

test("training start errors localize stable backend codes", () => {
  assert.equal(
    normalizeTrainingStartError({
      message: "server fallback",
      errorCode: "hf_model_access_denied",
    }),
    "Hugging Face denied access to this model. Add a valid Hugging Face token with repository access and accept any required access terms, then try again.",
  );
  assert.equal(
    normalizeTrainingStartError(
      "server fallback",
      "hf_model_metadata_unavailable",
    ),
    "Hugging Face model metadata is temporarily unavailable. Retry before starting training.",
  );
});

test("training start errors preserve unknown backend messages", () => {
  assert.equal(
    normalizeTrainingStartError("Specific backend failure", "future_code"),
    "Specific backend failure",
  );
});
