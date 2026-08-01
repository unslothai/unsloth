// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { trainingModelMatchesTypeConstraint } from "../src/features/training/lib/model-type-constraint.ts";

test("enforces optional onboarding model-type constraints", () => {
  assert.equal(trainingModelMatchesTypeConstraint("vision", undefined), true);
  assert.equal(trainingModelMatchesTypeConstraint("text", "text"), true);
  assert.equal(trainingModelMatchesTypeConstraint("vision", "vision"), true);
  assert.equal(trainingModelMatchesTypeConstraint("audio", "audio"), true);
  assert.equal(
    trainingModelMatchesTypeConstraint("embeddings", "embeddings"),
    true,
  );
  assert.equal(trainingModelMatchesTypeConstraint("text", "vision"), false);
  assert.equal(trainingModelMatchesTypeConstraint("vision", "audio"), false);
  assert.equal(trainingModelMatchesTypeConstraint("audio", "embeddings"), false);
  assert.equal(trainingModelMatchesTypeConstraint("embeddings", "text"), false);
});
