// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ten shipped model_defaults express warmup as a ratio and set no warmup_steps,
// so the form has to derive it or those recommendations never arrive.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import type { BackendModelConfig } from "../src/features/training/api/models-api.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const yaml = await import("js-yaml");
const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);

const MODEL_DEFAULTS_DIR = new URL(
  "../../backend/assets/configs/model_defaults/",
  import.meta.url,
);

function shippedConfigs(): { name: string; config: BackendModelConfig }[] {
  const found: { name: string; config: BackendModelConfig }[] = [];
  const walk = (dir: URL, prefix: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        walk(new URL(`${entry.name}/`, dir), `${prefix}${entry.name}/`);
      } else if (entry.name.endsWith(".yaml")) {
        const text = readFileSync(new URL(entry.name, dir), "utf8");
        found.push({
          name: `${prefix}${entry.name}`,
          config: (yaml.load(text) ?? {}) as BackendModelConfig,
        });
      }
    }
  };
  walk(MODEL_DEFAULTS_DIR, "");
  return found;
}

test("a warmup_ratio default reaches the form as steps", () => {
  const patch = mapBackendModelConfigToTrainingPatch({
    training: { warmup_ratio: 0.1, max_steps: 30 },
  });
  assert.equal(patch.warmupSteps, 3);
});

test("an explicit warmup_steps still wins over a ratio", () => {
  const patch = mapBackendModelConfigToTrainingPatch({
    training: { warmup_steps: 7, warmup_ratio: 0.1, max_steps: 30 },
  });
  assert.equal(patch.warmupSteps, 7);
});

test("a ratio with no usable max_steps leaves warmup alone", () => {
  for (const training of [
    { warmup_ratio: 0.1 },
    { warmup_ratio: 0.1, max_steps: 0 },
  ]) {
    const patch = mapBackendModelConfigToTrainingPatch({ training });
    assert.equal(patch.warmupSteps, undefined, JSON.stringify(training));
  }
});

test("every shipped model default carries its warmup into the patch", () => {
  const configs = shippedConfigs();
  // Guard the fixture: a move or rename should fail loudly rather than leave
  // this test silently checking nothing.
  assert.ok(
    configs.length > 50,
    `only found ${configs.length} shipped configs`,
  );

  const ratioOnly = configs.filter(
    ({ config }) =>
      config.training?.warmup_ratio !== undefined &&
      config.training?.warmup_steps === undefined,
  );
  assert.ok(
    ratioOnly.length > 0,
    "expected at least one shipped config to express warmup as a ratio",
  );

  for (const { name, config } of configs) {
    const training = config.training ?? {};
    if (
      training.warmup_steps === undefined &&
      training.warmup_ratio === undefined
    ) {
      continue;
    }
    const patch = mapBackendModelConfigToTrainingPatch(config);
    assert.equal(
      typeof patch.warmupSteps,
      "number",
      `${name} declares a warmup but none reached the patch`,
    );
  }
});
