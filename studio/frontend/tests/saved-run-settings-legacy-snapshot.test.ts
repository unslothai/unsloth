// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

// Its own file: the legacy import is latched per process, so it must be the first read here.
registerStoreStubResolver();
const { store } = installLocalStorageFake();

const REPO = "unsloth/gemma-3-270m-it-GGUF";
const QUANT = "UD-Q4_K_XL";
store.set(
  "unsloth_load_settings",
  JSON.stringify({
    [`${REPO}::${QUANT}`]: { contextLength: 4096, kvCacheDtype: "q8_0" },
  }),
);

const { savedRunSettings } = await import(
  "../src/features/model-picker/model-config/saved-run-settings.ts"
);
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);

test("the read that migrates a legacy record returns the same snapshot twice", () => {
  const target = modelConfigTarget(REPO, {
    source: "hub",
    isLora: false,
    ggufVariant: QUANT,
    isDownloaded: true,
    isGguf: true,
  });
  const first = savedRunSettings(target);
  assert.equal(first?.kvCacheDtype, "q8_0");
  assert.ok(store.get("unsloth_model_configs"), "the first read migrated the record");
  assert.equal(savedRunSettings(target), first);
});
