// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ownsModelLoadRun,
  releaseOwnedModelLoadRun,
} from "../../studio/frontend/src/features/chat/utils/model-load-run.ts";

test("the active run owns shared loading state", () => {
  const run = { attemptId: 1 };
  assert.equal(ownsModelLoadRun(run, run), true);
  assert.equal(releaseOwnedModelLoadRun(run, run), null);
});

test("late cleanup cannot release a replacement run", () => {
  const staleRun = { attemptId: 1 };
  const replacementRun = { attemptId: 2 };
  assert.equal(ownsModelLoadRun(replacementRun, staleRun), false);
  assert.equal(
    releaseOwnedModelLoadRun(replacementRun, staleRun),
    replacementRun,
  );
});
