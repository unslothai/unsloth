// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { DEFAULT_PER_MODEL_CONFIG, normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

test("the toggle defaults to vision enabled", () => {
  assert.equal(DEFAULT_PER_MODEL_CONFIG.disableVision, false);
});

test("a config blob saved before the toggle existed still normalizes", () => {
  // The compatibility case: persisted JSON with no disableVision key at all.
  // Omitted by rest-destructuring rather than `delete`, so the key is genuinely
  // absent (not present-and-undefined) the way an old blob would have it.
  const { disableVision: _omitted, ...legacy } = DEFAULT_PER_MODEL_CONFIG;

  assert.equal(
    normalizePerModelConfig(legacy as Record<string, unknown>).disableVision,
    false,
  );
});

test("a non-boolean value falls back rather than dropping vision by accident", () => {
  for (const bad of ["true", 1, null, {}]) {
    const normalized = normalizePerModelConfig({
      ...DEFAULT_PER_MODEL_CONFIG,
      disableVision: bad,
    });
    assert.equal(normalized.disableVision, false, `for ${JSON.stringify(bad)}`);
  }
});

test("the toggle round-trips through normalization", () => {
  const normalized = normalizePerModelConfig({
    ...DEFAULT_PER_MODEL_CONFIG,
    disableVision: true,
  });

  assert.equal(normalized.disableVision, true);
});

test("the signature changes with the toggle, so a reload is not deduped away", () => {
  // Without this the model would stay up with its projector still resident
  // while the UI showed Vision as off.
  const on: PerModelConfig = {
    ...DEFAULT_PER_MODEL_CONFIG,
    disableVision: false,
  };
  const off: PerModelConfig = {
    ...DEFAULT_PER_MODEL_CONFIG,
    disableVision: true,
  };

  assert.notEqual(loadedConfigSignature(on), loadedConfigSignature(off));
});
