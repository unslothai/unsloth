// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);

for (const objective of ["dpo", "orpo"] as const) {
  test(`GRPO's zero KL coefficient falls back to ${objective}'s default beta`, () => {
    const store = useTrainingConfigStore;
    store.getState().reset();
    store.setState({ trainingObjective: "grpo", rlBeta: 0 });
    store.getState().setTrainingObjective(objective);
    assert.equal(store.getState().trainingObjective, objective);
    assert.equal(store.getState().rlBeta, null);
  });
}

test("a positive beta carries over between objectives", () => {
  const store = useTrainingConfigStore;
  store.getState().reset();
  store.setState({ trainingObjective: "grpo", rlBeta: 0.2 });
  store.getState().setTrainingObjective("dpo");
  assert.equal(store.getState().rlBeta, 0.2);
});
