// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveStartTrainingButtonLabelKey } from "../src/features/studio/wizard/start-training-cta-state.ts";

const readyState = {
  stopRequested: false,
  scheduled: false,
  scheduling: false,
  startBlocked: false,
  isLoadingModel: false,
  isCheckingDataset: false,
  hasModel: true,
  hasDataset: true,
  scheduleMode: false,
};

test("the stop label takes priority over every other start state", () => {
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      stopRequested: true,
      scheduled: true,
      scheduling: true,
      startBlocked: true,
      isLoadingModel: true,
      isCheckingDataset: true,
    }),
    "studio.training.stopping",
  );
});

test("a waiting schedule and a schedule being saved win over a normal start", () => {
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      scheduled: true,
      scheduling: true,
      startBlocked: true,
    }),
    "studio.schedule.scheduled",
  );
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      scheduling: true,
      startBlocked: true,
    }),
    "studio.schedule.scheduling",
  );
});

test("start button labels cover pending work and incomplete selections", () => {
  const cases = [
    [{ ...readyState, startBlocked: true }, "studio.training.starting"],
    [{ ...readyState, isLoadingModel: true }, "studio.training.loadingModel"],
    [
      { ...readyState, isCheckingDataset: true },
      "studio.training.checkingDataset",
    ],
    [
      { ...readyState, hasModel: false, hasDataset: false },
      "studio.training.chooseModelAndDataset",
    ],
    [{ ...readyState, hasModel: false }, "studio.training.chooseModel"],
    [{ ...readyState, hasDataset: false }, "studio.training.chooseDataset"],
    [readyState, "studio.training.startTraining"],
    [
      { ...readyState, scheduleMode: true },
      "studio.schedule.scheduleTraining",
    ],
  ] as const;

  for (const [state, expected] of cases) {
    assert.equal(resolveStartTrainingButtonLabelKey(state), expected);
  }
});
