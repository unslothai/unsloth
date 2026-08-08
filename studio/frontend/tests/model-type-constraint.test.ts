// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { trainingModelMatchesTypeConstraint } from "../src/features/training/lib/model-type-constraint.ts";

test("enforces optional onboarding model-type constraints", () => {
  const text = { hasModelTypeSignal: true };
  const vision = { isVision: true, hasModelTypeSignal: true };
  const audio = { isAudio: true, hasModelTypeSignal: true };
  const embeddings = { isEmbedding: true, hasModelTypeSignal: true };

  assert.equal(trainingModelMatchesTypeConstraint(vision, undefined), true);
  assert.equal(trainingModelMatchesTypeConstraint(text, "text"), true);
  assert.equal(trainingModelMatchesTypeConstraint(vision, "vision"), true);
  assert.equal(trainingModelMatchesTypeConstraint(audio, "audio"), true);
  assert.equal(
    trainingModelMatchesTypeConstraint(embeddings, "embeddings"),
    true,
  );
  assert.equal(trainingModelMatchesTypeConstraint(text, "vision"), false);
  assert.equal(trainingModelMatchesTypeConstraint(vision, "audio"), false);
  assert.equal(trainingModelMatchesTypeConstraint(audio, "embeddings"), false);
  assert.equal(trainingModelMatchesTypeConstraint(embeddings, "text"), false);
});

test("allows an unclassified local model through an onboarding constraint", () => {
  assert.equal(trainingModelMatchesTypeConstraint({}, "vision"), true);
  assert.equal(trainingModelMatchesTypeConstraint({}, "audio"), true);
  assert.equal(trainingModelMatchesTypeConstraint({}, "embeddings"), true);
});

test("preserves every capability when enforcing onboarding constraints", () => {
  const audioVision = {
    isAudio: true,
    isVision: true,
    hasModelTypeSignal: true,
  };

  assert.equal(trainingModelMatchesTypeConstraint(audioVision, "vision"), true);
  assert.equal(trainingModelMatchesTypeConstraint(audioVision, "audio"), true);
  assert.equal(trainingModelMatchesTypeConstraint(audioVision, "text"), false);
});
