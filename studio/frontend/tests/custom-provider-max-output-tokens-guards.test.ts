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
  CUSTOM_MAX_OUTPUT_TOKENS_MIN,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  loadExternalProviders,
  normalizeCustomMaxOutputTokens,
  saveExternalProviders,
  supportsCustomMaxOutputTokens,
} = await import("../src/features/chat/external-providers.ts");

const {
  EXTERNAL_MAX_OUTPUT_TOKENS,
  getExternalMaxOutputTokens,
  resolveExternalMaxTokensClamp,
} = await import("../src/features/chat/provider-capabilities.ts");

// Two provider types are in play and they are NOT interchangeable. The UI type is
// what the connections dialog draws, from `resolveUiProviderTypeFromConfig`; the
// backend type is what the server row actually holds. They disagree for a row saved
// as `openai` with a custom display name or base URL, which the dialog shows as
// Custom while the server still rejects an override on it.
test("the override is offered only when both provider types say custom", () => {
  assert.equal(supportsCustomMaxOutputTokens("custom", "custom"), true);
  assert.equal(supportsCustomMaxOutputTokens("custom", "openai"), false);
  assert.equal(supportsCustomMaxOutputTokens("openai", "custom"), false);
  assert.equal(supportsCustomMaxOutputTokens("openai", "openai"), false);

  // Unknown backend type: a connection being created has no server row yet, and an
  // entry synced before the field existed carries no value. Both fall back to the
  // outgoing mapping, so a brand new Custom connection keeps working as before.
  assert.equal(supportsCustomMaxOutputTokens("custom", null), true);
  assert.equal(supportsCustomMaxOutputTokens("custom", undefined), true);
  assert.equal(supportsCustomMaxOutputTokens("custom", ""), true);
  assert.equal(supportsCustomMaxOutputTokens("openai", null), false);
  assert.equal(supportsCustomMaxOutputTokens(null, null), false);
});

test("a stored override is dropped unless it is a usable custom-connection value", () => {
  const custom = LEGACY_CUSTOM_PROVIDER_TYPE;
  assert.equal(normalizeCustomMaxOutputTokens(custom, 384000), 384000);
  assert.equal(normalizeCustomMaxOutputTokens(custom, CUSTOM_MAX_OUTPUT_TOKENS_MIN), 64);

  // Anything a hand-edited or older localStorage entry could hold.
  for (const junk of [
    "384000", 384000.5, -1, 0, 63, Number.NaN, Number.POSITIVE_INFINITY,
    null, undefined, {}, [], 2 ** 53,
  ]) {
    assert.equal(normalizeCustomMaxOutputTokens(custom, junk), undefined);
  }

  // And never for a provider that has documented caps of its own.
  for (const type of ["openai", "anthropic", "gemini", "vllm", "ollama", "llama_cpp"]) {
    assert.equal(normalizeCustomMaxOutputTokens(type, 384000), undefined);
  }
});

test("named providers keep the caps they had before the override existed", () => {
  // Real entries from the capability table, each one a provider with a documented
  // cap of its own. The override argument is passed on every call, so this fails if
  // it ever leaks past the custom-only early return.
  const cases: Array<[string, string | null, number]> = [
    ["openai", "gpt-5.6", 128000],
    ["openai", "gpt-5.3", 16384],
    ["openai", "gpt-4o", EXTERNAL_MAX_OUTPUT_TOKENS],
    ["gemini", "gemini-3-pro", 65536],
    ["deepseek", "deepseek-reasoner", 384000],
    ["openrouter", "deepseek/deepseek-reasoner", 384000],
    ["vllm", "some/local-model", EXTERNAL_MAX_OUTPUT_TOKENS],
    ["ollama", null, EXTERNAL_MAX_OUTPUT_TOKENS],
  ];
  for (const [providerType, modelId, expected] of cases) {
    assert.equal(getExternalMaxOutputTokens(providerType, modelId), expected);
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, 384000), expected);
    assert.equal(getExternalMaxOutputTokens(providerType, modelId, null), expected);
  }
});

test("a custom connection takes its own cap, above or below the fallback", () => {
  const custom = LEGACY_CUSTOM_PROVIDER_TYPE;
  assert.equal(getExternalMaxOutputTokens(custom, "any-model"), EXTERNAL_MAX_OUTPUT_TOKENS);
  assert.equal(getExternalMaxOutputTokens(custom, "any-model", 384000), 384000);
  assert.equal(getExternalMaxOutputTokens(custom, "any-model", 1024), 1024);
  // A model id that matches a known family is still ignored: on a custom endpoint the
  // id space is user-controlled and means nothing.
  assert.equal(getExternalMaxOutputTokens(custom, "gpt-4o"), EXTERNAL_MAX_OUTPUT_TOKENS);
  // An unusable stored value fails closed to the fallback rather than to 0.
  assert.equal(getExternalMaxOutputTokens(custom, "any-model", 0), EXTERNAL_MAX_OUTPUT_TOKENS);
});

// The settings panel PERSISTS what this returns, and it only ever lowers, so the
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

  // No resolved provider: connections toggled off, a cold browser where settings
  // hydrate before the sync lands, or a connection deleted while selected. The cap
  // reads as the 32,768 fallback in all of those, and clamping would be permanent.
  assert.equal(
    resolveExternalMaxTokensClamp({ ...base, hasActiveExternalProvider: false }),
    null,
  );
  assert.equal(resolveExternalMaxTokensClamp({ ...base, settingsHydrated: false }), null);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, isExternalModel: false }), null);
  // Already within the cap, and exactly at it.
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: 8192 }), null);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: 32768 }), null);

  // Converges in one pass: feeding the result back in asks for no further change.
  const once = resolveExternalMaxTokensClamp(base);
  assert.equal(resolveExternalMaxTokensClamp({ ...base, maxTokens: once as number }), null);
});

test("an entry saved by an older install loads without gaining a cap", () => {
  const legacy = [
    {
      id: "legacy-1",
      providerType: "custom",
      name: "Old Custom",
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

  // And a round trip through save keeps the stored type once it is known.
  saveExternalProviders([{ ...loaded, backendProviderType: "openai", maxOutputTokens: 384000 }]);
  const [reloaded] = loadExternalProviders();
  assert.equal(reloaded.backendProviderType, "openai");
  assert.equal(reloaded.maxOutputTokens, 384000);
});

// The same guard has to hold everywhere a cap is applied, not just in the effect:
// a preset applied, or a checkpoint selected, while the provider is unresolved would
// otherwise lower the value permanently. Source-level assertions, since neither call
// site is reachable without a DOM.
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
