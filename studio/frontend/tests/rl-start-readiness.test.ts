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
const { deriveTrainingReadiness } = await import(
  "../src/features/training/hooks/use-training-readiness.ts"
);
const { rlStartMissing } = await import(
  "../src/features/training/lib/rl-roles.ts"
);

function readyConfig(trainingObjective: "sft" | "grpo" | "dpo") {
  return {
    ...useTrainingConfigStore.getState(),
    selectedModel: "unsloth/Qwen2.5-1.5B-Instruct",
    modelType: "text" as const,
    datasetSource: "huggingface" as const,
    dataset: "yahma/alpaca-cleaned",
    trainingObjective,
  };
}

test("GRPO without the answer a reward compares to, and DPO without rejected, keep Start off", () => {
  const grpoMissing = rlStartMissing("grpo", ["question", "text"], {}, ["answer"]);
  assert.deepEqual(grpoMissing, ["answer"]);
  const dpoMissing = rlStartMissing("dpo", ["question", "answer"], {}, []);
  assert.deepEqual(dpoMissing, ["chosen", "rejected"]);

  for (const [objective, missing] of [
    ["grpo", grpoMissing],
    ["dpo", dpoMissing],
  ] as const) {
    const readiness = deriveTrainingReadiness(
      readyConfig(objective),
      "cuda",
      true,
      missing,
    );
    assert.equal(readiness.isReady, false, objective);
    assert.deepEqual(readiness.rlMissing, missing);
  }
  // Same config with nothing missing is Ready, and SFT ignores a stale RL list.
  assert.equal(
    deriveTrainingReadiness(readyConfig("grpo"), "cuda", true, []).isReady,
    true,
  );
  assert.equal(
    deriveTrainingReadiness(readyConfig("sft"), "cuda", true, ["answer"]).isReady,
    true,
  );
});
