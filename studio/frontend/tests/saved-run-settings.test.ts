// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store, fireWindowEvent } = installLocalStorageFake();
Object.assign(globalThis.window, {
  dispatchEvent: (event: Event) => {
    fireWindowEvent(event.type, event);
    return true;
  },
});

const { forgetRunSettings, savedRunSettings, subscribeSavedRunSettings } =
  await import(
    "../src/features/model-picker/model-config/saved-run-settings.ts"
  );
const { DEFAULT_PER_MODEL_CONFIG, resolveInitialConfig, savePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const REPO = "unsloth/gemma-4-12B-it-qat-GGUF";
const QUANT = "UD-Q4_K_XL";

const ggufTarget = () =>
  modelConfigTarget(REPO, {
    source: "hub",
    isLora: false,
    ggufVariant: QUANT,
    isDownloaded: true,
    isGguf: true,
  });

const tuned = () => ({
  ...DEFAULT_PER_MODEL_CONFIG,
  kvCacheDtype: "q8_0",
  nParallel: 4,
});

function captureOverridePuts(): { modelKey: string; body: unknown }[] {
  const puts: { modelKey: string; body: unknown }[] = [];
  setAuthFetchHandler((_input, init) => {
    const body = JSON.parse(String(init?.body ?? "{}"));
    puts.push({ modelKey: body.model_key ?? body.key ?? "", body });
    return new Response(JSON.stringify({ overrides: {}, removed_keys: [] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  return puts;
}

test("a model with nothing saved reports none", () => {
  store.clear();
  assert.equal(savedRunSettings(ggufTarget()), null);
});

test("saving without loading is what the row reads back", () => {
  store.clear();
  assert.ok(savePerModelConfig(REPO, QUANT, tuned()));
  assert.equal(savedRunSettings(ggufTarget())?.kvCacheDtype, "q8_0");
  // Another quant of the same repo keeps its own answer.
  const other = modelConfigTarget(REPO, {
    source: "hub",
    isLora: false,
    ggufVariant: "Q8_0",
    isGguf: true,
  });
  assert.equal(savedRunSettings(other), null);
});

test("a subscribed reader sees a save without remounting", () => {
  store.clear();
  let changes = 0;
  const stop = subscribeSavedRunSettings(() => {
    changes += 1;
  });
  try {
    assert.equal(savedRunSettings(ggufTarget()), null);
    savePerModelConfig(REPO, QUANT, tuned());
    assert.ok(changes > 0, "a save must notify subscribers");
    assert.equal(savedRunSettings(ggufTarget())?.nParallel, 4);
  } finally {
    stop();
  }
});

test("forget drops the saved settings and undo restores the same values", async () => {
  store.clear();
  const puts = captureOverridePuts();
  try {
    savePerModelConfig(REPO, QUANT, tuned());
    const undo = forgetRunSettings(ggufTarget());
    assert.ok(
      undo,
      "forget returns an undo when there was something to forget",
    );
    assert.equal(resolveInitialConfig(REPO, QUANT).remembered, false);

    assert.equal(undo(), true);
    const restored = resolveInitialConfig(REPO, QUANT);
    assert.equal(restored.remembered, true);
    assert.equal(restored.config.kvCacheDtype, "q8_0");
    assert.equal(restored.config.nParallel, 4);

    // GGUF is API-loadable, so both the forget and the undo reach the server mirror.
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(puts.length, 2);
  } finally {
    setAuthFetchHandler(null);
  }
});

test("forget with nothing saved does nothing", () => {
  store.clear();
  assert.equal(forgetRunSettings(ggufTarget()), null);
});

test("forget marks a mounted editor's draft unsaved, and undo marks it saved again", async () => {
  store.clear();
  const {
    modelConfigDraftKey,
    primeModelConfigDraft,
    readModelConfigDraft,
    retainModelConfigDraft,
  } = await import(
    "../src/features/model-picker/model-config/model-config-draft.ts"
  );
  setAuthFetchHandler(() => new Response("{}", { status: 200 }));
  const key = modelConfigDraftKey(REPO, QUANT);
  // The sidebar keeps the loaded model's editor mounted, so its draft outlives the picker.
  const release = retainModelConfigDraft(key);
  try {
    savePerModelConfig(REPO, QUANT, tuned());
    primeModelConfigDraft(key, { config: tuned(), remembered: true }, "none");
    const undo = forgetRunSettings(ggufTarget());
    assert.ok(undo);
    assert.equal(readModelConfigDraft(key)?.savedRemember, false);
    assert.equal(readModelConfigDraft(key)?.remember, false);
    // The editor keeps showing what is loaded; only the saved flags change.
    assert.equal(readModelConfigDraft(key)?.config.kvCacheDtype, "q8_0");
    assert.equal(undo(), true);
    assert.equal(readModelConfigDraft(key)?.savedRemember, true);
    assert.equal(readModelConfigDraft(key)?.remember, true);
  } finally {
    release();
    setAuthFetchHandler(null);
  }
});
