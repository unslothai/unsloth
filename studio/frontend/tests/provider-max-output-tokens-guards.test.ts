// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  LEGACY_CUSTOM_PROVIDER_TYPE,
  loadExternalProviders,
  normalizeProviderMaxOutputTokens,
  PROVIDER_MAX_OUTPUT_TOKENS_MIN,
  saveExternalProviders,
  supportsProviderMaxOutputTokens,
} = await import("../src/features/chat/external-providers.ts");

const {
  EXTERNAL_MAX_OUTPUT_TOKENS,
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  resolveExternalMaxTokensClamp,
} = await import("../src/features/chat/provider-capabilities.ts");

// The UI type is what the dialog draws (`resolveUiProviderTypeFromConfig`); the backend
// type is what the server row holds. They disagree for a row saved as `openai` with a
// custom display name or base URL.
test("the override is offered on every connection except a ChatGPT subscription", () => {
  assert.equal(supportsProviderMaxOutputTokens("custom", "custom"), true);
  assert.equal(supportsProviderMaxOutputTokens("custom", "openai"), true);
  assert.equal(supportsProviderMaxOutputTokens("openrouter", "openrouter"), true);
  assert.equal(supportsProviderMaxOutputTokens("anthropic", "anthropic"), true);

  // a codex override would be stored and never read, and either type saying so is enough
  assert.equal(supportsProviderMaxOutputTokens("openai_codex", "openai_codex"), false);
  assert.equal(supportsProviderMaxOutputTokens("openai_codex", null), false);
  assert.equal(supportsProviderMaxOutputTokens("openai", "openai_codex"), false);

  // Unknown backend type: a connection being created has no server row yet, and an entry
  // synced before the field existed carries no value. The UI type decides both.
  assert.equal(supportsProviderMaxOutputTokens("custom", null), true);
  assert.equal(supportsProviderMaxOutputTokens("custom", undefined), true);
  assert.equal(supportsProviderMaxOutputTokens("custom", ""), true);
  assert.equal(supportsProviderMaxOutputTokens("openai", null), true);
  assert.equal(supportsProviderMaxOutputTokens(null, null), false);
});

test("a stored override is dropped unless it is a usable value", () => {
  assert.equal(normalizeProviderMaxOutputTokens(384000), 384000);
  assert.equal(normalizeProviderMaxOutputTokens(PROVIDER_MAX_OUTPUT_TOKENS_MIN), 64);

  // anything a hand-edited or older localStorage entry could hold
  for (const junk of [
    "384000", 384000.5, -1, 0, 63, Number.NaN, Number.POSITIVE_INFINITY,
    null, undefined, {}, [], 2 ** 53,
  ]) {
    assert.equal(normalizeProviderMaxOutputTokens(junk), undefined);
  }
});

test("a documented per-model cap bounds the connection override", () => {
  const cases: Array<[string, string, number]> = [
    ["openai", "gpt-5.6", 128000],
    ["openai", "gpt-5.3", 16384],
    ["gemini", "gemini-3-pro", 65536],
    ["deepseek", "deepseek-reasoner", 384000],
    ["openrouter", "deepseek/deepseek-reasoner", 384000],
    ["openai_codex", "gpt-5.3-codex", 128000],
  ];
  for (const [providerType, modelId, documented] of cases) {
    assert.equal(getExternalMaxOutputTokens(providerType, modelId), documented);
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, null), documented);
    // a router connection fronts models of every size, so one limit must not raise them all
    assert.equal(
      getExternalMaxOutputTokens(providerType, modelId, 999999),
      documented,
    );
    // lowering is the whole point of a limit, so that half is honoured
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, 8192), 8192);
  }
});

test("every OpenAI family the picker admits carries its documented cap", () => {
  // A missing or too-generous row turns a raised Max Tokens into a failed
  // request; the 4,096 pair is under the 8,192 default, so those two fail on
  // an untouched config. The bare `gpt-5` and `gpt-4` rows are last, so this
  // also pins that a more specific family keeps its own cap.
  const caps: Array<[string, number]> = [
    ["gpt-5.6-sol", 128000],
    ["gpt-5.5", 128000],
    ["gpt-5.4", 65536],
    ["gpt-5.3", 16384],
    ["gpt-5.2", 128000],
    ["gpt-5.1", 128000],
    // The chat aliases cap at 16,384 whatever their family does.
    ["gpt-5-chat-latest", 16384],
    ["gpt-5.1-chat-latest", 16384],
    ["gpt-5.2-chat-latest", 16384],
    ["gpt-5.3-chat-latest", 16384],
    ["gpt-5", 128000],
    ["gpt-5-mini", 128000],
    ["gpt-4.1", 32768],
    ["gpt-4.1-mini", 32768],
    ["gpt-4.5-preview", 16384],
    ["gpt-4o", 16384],
    ["gpt-4o-mini", 16384],
    ["chatgpt-4o-latest", 16384],
    ["gpt-3.5-turbo", 4096],
    ["gpt-3.5-turbo-16k", 4096],
    ["gpt-4-turbo", 4096],
    ["gpt-4-turbo-preview", 4096],
    ["gpt-4", 8192],
  ];
  for (const [modelId, cap] of caps) {
    assert.equal(getExternalMaxOutputTokens("openai", modelId), cap);
    assert.equal(getExternalMaxOutputTokens("openai", modelId, 1000000), cap);
  }
});

test("the dated Anthropic ids carry their documented cap", () => {
  // Opus 4.1 and Opus 4 sit at 32,000, under the 32,768 fallback, so without
  // a row a raised Max Tokens overshoots them.
  const caps: Array<[string, number]> = [
    ["claude-opus-4-5-20251101", 64000],
    ["claude-sonnet-4-5-20250929", 64000],
    ["claude-haiku-4-5-20251001", 64000],
    ["claude-sonnet-4-20250514", 64000],
    ["claude-opus-4-1-20250805", 32000],
    ["claude-opus-4-20250514", 32000],
  ];
  for (const [modelId, cap] of caps) {
    assert.equal(getExternalMaxOutputTokens("anthropic", modelId), cap);
    assert.equal(getExternalMaxOutputTokens("anthropic", modelId, 1000000), cap);
  }
});

test("a model with no documented cap takes the connection override", () => {
  // the reported case: a router id no capability row matches pinned at 32,768
  const undocumented: Array<[string, string | null]> = [
    ["openrouter", "minimax/minimax-m3"],
    ["openai", "o3"],
    ["vllm", "some/local-model"],
    ["ollama", null],
    [LEGACY_CUSTOM_PROVIDER_TYPE, "any-model"],
    [LEGACY_CUSTOM_PROVIDER_TYPE, "gpt-4o"],
  ];
  for (const [providerType, modelId] of undocumented) {
    assert.equal(
      getExternalMaxOutputTokens(providerType, modelId),
      EXTERNAL_MAX_OUTPUT_TOKENS,
    );
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, 262144), 262144);
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, 1024), 1024);
    // an unusable stored value fails closed to the fallback rather than to 0
    assert.equal(
      getExternalMaxOutputTokens(providerType, modelId, 0),
      EXTERNAL_MAX_OUTPUT_TOKENS,
    );
  }
});

/** Kimi is the one provider with an output floor of its own (16,000) and no documented
 * per-model cap, so without this its override alone could set the slider's max below
 * the slider's min. */
test("a provider's own output floor outranks a lower connection override", () => {
  const kimiFloor = getExternalMinOutputTokens("kimi");
  assert.equal(kimiFloor, 16000);

  assert.equal(getExternalMaxOutputTokens("kimi", "kimi-k2.6", 8000), kimiFloor);
  assert.equal(getExternalMaxOutputTokens("kimi", "kimi-k2.6", 262144), 262144);
  assert.equal(
    getExternalMaxOutputTokens("kimi", "kimi-k2.6"),
    EXTERNAL_MAX_OUTPUT_TOKENS,
  );
});

// The settings panel PERSISTS what this returns and it only ever lowers, so the
// availability guards are what stop a blink in the provider list from destroying a
// configured override.
test("a live Max Tokens is only lowered when the cap is actually known", () => {
  const base = {
    settingsHydrated: true,
    hasActiveExternalProvider: true,
    isExternalModel: true,
    maxTokens: 384000,
    maxTokensMax: 32768,
  };
  assert.equal(resolveExternalMaxTokensClamp(base), 32768);

  // No resolved provider: connections toggled off, settings hydrated before the sync
  // lands, or a connection deleted while selected. The cap reads as the 32,768 fallback
  // in all of those, and clamping would be permanent.
  assert.equal(
    resolveExternalMaxTokensClamp({ ...base, hasActiveExternalProvider: false }),
    null,
  );
  assert.equal(resolveExternalMaxTokensClamp({ ...base, settingsHydrated: false }), null);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, isExternalModel: false }), null);
  // already within the cap, and exactly at it
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: 8192 }), null);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: 32768 }), null);

  // converges in one pass: feeding the result back asks for no further change
  const once = resolveExternalMaxTokensClamp(base);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: once as number }), null);
});

// deliberately not a Custom row: that one type round-trips the same with or without the gate
test("an entry saved by an older install loads without gaining a cap", () => {
  const legacy = [
    {
      id: "legacy-1",
      providerType: "openrouter",
      name: "Old Router",
      baseUrl: "https://example.com/v1",
      models: ["vendor/model"],
      createdAt: 1,
      updatedAt: 1,
    },
  ];
  store.set("unsloth_chat_external_providers", JSON.stringify(legacy));
  const [loaded] = loadExternalProviders();
  assert.equal(loaded.maxOutputTokens, undefined);
  assert.equal(loaded.backendProviderType, undefined);
  assert.equal(loaded.models.length, 1);

  // a round trip through save keeps the stored type once it is known
  saveExternalProviders([{ ...loaded, backendProviderType: "openai", maxOutputTokens: 384000 }]);
  const [reloaded] = loadExternalProviders();
  assert.equal(reloaded.backendProviderType, "openai");
  assert.equal(reloaded.maxOutputTokens, 384000);
});

// The guard has to hold at every clamp site, not just the effect: a preset or checkpoint
// applied while the provider is unresolved would lower the value permanently. Source-level
// assertions, since neither call site is reachable without a DOM.
test("every clamp site waits for a resolved provider", () => {
  const settings = readFileSync(
    new URL("../src/features/chat/chat-settings-sheet.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    settings,
    /function applyPresetParamsWithinCurrentLimits\([\s\S]*?if \(!isExternalModel \|\| activeExternalProvider == null\) return nextParams;/,
  );

  const store = readFileSync(
    new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    store,
    /if \(provider\) \{\s*const cap = getExternalMaxOutputTokens\(\s*provider\.providerType/,
  );
});
