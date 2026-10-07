// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-level: these rules live inside component effects with no renderer here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  deletePerModelConfig,
  perModelConfigStorageChanged,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PANEL = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

test("a list typed while hydration was in flight is not sanitized", () => {
  // Only legacy stored data is sanitized; text typed during the fetch is live input.
  assert.match(PANEL, /const localAtStart = configAtStart\.llamaExtraArgs;/);
  assert.match(
    PANEL,
    /local !== null && local\.length > 0 && local === localAtStart|local != null && local\.length > 0 && local === localAtStart/,
  );
});

test("a collapsed section is re-judged once the catalogue lands", () => {
  // A verdict reached before the flag probe answered must not freeze when Advanced collapses.
  assert.match(
    PANEL,
    /if \(showAdvanced \|\| !target\.isGguf \|\| resolvedIsDiffusion\) \{/,
  );
  assert.match(PANEL, /loadLlamaFlagCatalog\(\)\.then\(\(catalog\) => \{/);
});

const ADAPTER = readFileSync(
  path.join(HERE, "..", "src/features/chat/api/chat-adapter.ts"),
  "utf8",
);

test("a background auto-load hydrates a server-only override", () => {
  // Local storage misses overrides written via the API or another browser.
  assert.match(ADAPTER, /let resolvedExtraArgs = config\.llamaExtraArgs;/);
  assert.match(ADAPTER, /const stored = await fetchLoadExtraArgs\(/);
  assert.match(ADAPTER, /sanitizeStoredExtraArgs\(tokens, managed\?\.managed/);
  // Older builds may have stored a flag that is managed now, which /load would 400.
  assert.match(ADAPTER, /const cleaned = clean\(resolvedExtraArgs\);/);
  // Cached inventory can return a different loadId than the one the row was written under.
  assert.match(ADAPTER, /candidate\.id,\n\s*candidate\.ggufVariant \?\? null,/);
  assert.equal(
    ADAPTER.match(/llama_extra_args: resolvedExtraArgs \?\? \[\]/g)?.length,
    2,
  );
  assert.match(ADAPTER, /candidate\.kind === "gguf" &&\s*\n?\s*!isDiffusion/);
});

test("a collapsed section stops objecting once nothing is left to object to", () => {
  // Reset with Advanced collapsed unmounts the row, so the panel must restore loadability.
  assert.match(PANEL, /setExtraArgsLoadable\(true\);\n\s*return;/);
});

const COMPOSER = readFileSync(
  path.join(HERE, "..", "src/features/chat/shared-composer.tsx"),
  "utf8",
);

test("a compare pane sanitizes the local list as well as the fetched one", () => {
  // An older build's config may name a now-managed flag, which /load rejects with 400.
  assert.match(COMPOSER, /const local = ownConfig\.llamaExtraArgs;/);
  assert.match(COMPOSER, /const cleaned = clean\(local\);/);
});

test("the hidden revalidation can only tighten the verdict", () => {
  // formatExtraArgs rebalances quotes, so only lowering the verdict here is safe.
  assert.match(PANEL, /if \(!loadable\) \{\n\s*setExtraArgsLoadable\(false\);/);
});

test("a mounted row re-reads the catalogue when the binary changes", () => {
  // An in-app llama.cpp update replaces the binary while the panel stays open.
  assert.match(PANEL, /subscribeLlamaFlagCatalog\(\(\) => setCatalogEpoch/);
  assert.match(PANEL, /\}, \[catalogEpoch\]\);/);
});

test("the hidden validation re-runs when the binary changes", () => {
  // The row is unmounted while this check runs, so it cannot subscribe itself.
  assert.match(
    PANEL,
    /subscribeLlamaFlagCatalog\(\(\) =>\s*\n?\s*setHiddenCatalogEpoch/,
  );
  assert.match(PANEL, /hiddenCatalogEpoch,\n\s*\]\);/);
});

const OVERRIDES = readFileSync(
  path.join(HERE, "..", "src/features/model-picker/api/model-overrides.ts"),
  "utf8",
);

test("the legacy overrides fallback searches the caller's own identities", () => {
  // Older backends return the whole map, so an empty default key list found nothing.
  assert.match(OVERRIDES, /fallbackKeys\.length > 0\s*\n?\s*\? fallbackKeys/);
  assert.match(OVERRIDES, /modelOverrideKey\(loadId, ggufVariant\)/);
  assert.match(OVERRIDES, /modelOverrideKey\(aliasId, ggufVariant\)/);
});

test("the hidden hydration check knows the slot floor", () => {
  // The backend deterministically refuses a batch below the floor.
  assert.match(
    PANEL,
    /serverConfig\?\.nParallel \?\? configRef\.current\.nParallel/,
  );
});

test("the panel adopts a shared server config without overwriting a live edit", () => {
  assert.match(PANEL, /fetchLoadModelOverride\(/);
  assert.match(
    PANEL,
    /const serverConfig = resolvedRow\s*\n?\s*\? fromApiOverride\(resolvedRow, \{/,
  );
  // Fields the row lacks keep the local value; the list travels sanitized.
  assert.match(
    PANEL,
    /\.\.\.configAtStart,\s*\n?\s*llamaExtraArgs: sanitizedLocal,/,
  );
  assert.match(PANEL, /let sanitizedLocal = localAtStart;/);
  assert.match(
    PANEL,
    /sanitizedLocal = cleaned\.length > 0 \? cleaned : null;/,
  );
  assert.match(
    PANEL,
    /savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*rememberedConfig,/,
  );
  assert.match(PANEL, /configRef\.current === configAtStart/);
  assert.match(PANEL, /rememberRef\.current === rememberAtStart/);
  assert.match(
    PANEL,
    /replaceModelConfigDraft\(draftKey, serverConfig, \{\s*\n?\s*remember: true,\s*\n?\s*savedRemember: true,\s*\n?\s*\}\);/,
  );
});

test("the panel strips llama-server arguments from a non-GGUF row", () => {
  // API auto-switch applies the row to MLX and safetensors loads too.
  assert.match(PANEL, /target\.isGguf \? loadManagedLlamaFlags\(\) : null,/);
  assert.match(PANEL, /\(row\) => panelOverrideRow\(row, target\.isGguf\)/);
  assert.match(
    PANEL,
    /const local = target\.isGguf\s*\?\s*configRef\.current\.llamaExtraArgs\s*:\s*undefined;/,
  );
});

test("hydration detects a newer save or forget", () => {
  const modelId = "unsloth/Hydration-Race-GGUF";
  const variant = "Q4_K_M";
  assert.ok(
    savePerModelConfig(modelId, variant, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 2048,
    }),
  );
  const atStart = resolveInitialConfig(modelId, variant);

  assert.ok(
    savePerModelConfig(modelId, variant, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 4096,
    }),
  );
  assert.equal(
    perModelConfigStorageChanged(
      atStart,
      resolveInitialConfig(modelId, variant),
    ),
    true,
  );

  assert.ok(deletePerModelConfig(modelId, variant));
  assert.equal(
    perModelConfigStorageChanged(
      atStart,
      resolveInitialConfig(modelId, variant),
    ),
    true,
  );
  assert.equal(
    perModelConfigStorageChanged(atStart, {
      config: { ...atStart.config },
      remembered: atStart.remembered,
    }),
    false,
  );
});

test("the hydration write-back rejects a stale server response", () => {
  const requestStart = PANEL.indexOf("Promise.all([");
  const storageSnapshot = PANEL.indexOf(
    "const storedAtStart = resolveInitialConfig(configId, target.ggufVariant);",
  );
  const responseStart = PANEL.indexOf(
    ".then(([resolvedOverride, managed]) => {",
    requestStart,
  );
  const adoptionStart = PANEL.indexOf(
    "if (\n          resolvedRow &&",
    responseStart,
  );
  const writeBackEnd = PANEL.indexOf(
    "setSavedRemember(hydrationSaved);",
    adoptionStart,
  );
  assert.ok(
    requestStart >= 0 &&
      storageSnapshot >= 0 &&
      storageSnapshot < requestStart &&
      responseStart > requestStart &&
      adoptionStart > responseStart &&
      writeBackEnd > adoptionStart,
    "the storage snapshot must precede the request and guard its write-back",
  );

  const writeBack = PANEL.slice(adoptionStart, writeBackEnd);
  assert.match(
    writeBack,
    /const storedConfig = resolveInitialConfig\(\s*configId,\s*target\.ggufVariant,\s*\);\s*if \(perModelConfigStorageChanged\(storedAtStart, storedConfig\)\) \{\s*return;\s*\}[\s\S]*const rememberedConfig = fromApiOverride\(\s*resolvedRow,\s*storedConfig\.config,\s*\);[\s\S]*savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*rememberedConfig,/,
  );
});

test("a build that serves one slot does not raise the floor", () => {
  // Without --kv-unified the backend clamps to one slot, so an explicit Slots value is not effective.
  assert.match(PANEL, /if \(limits\?\.parallelSlotsClamped\) \{\n\s*return 2;/);
  assert.equal(
    PANEL.match(/effectiveBatchFloor\(/g)?.length,
    4,
    "one definition and three call sites",
  );
});

const ADAPTER_AUTOLOAD = ADAPTER;

test("an auto-load records what it launched with", () => {
  // The loading lease blocks the status applier, so the load itself must record the baseline.
  assert.match(
    ADAPTER_AUTOLOAD,
    /loadedLlamaExtraArgs:\s*\n?\s*loadResp\.requested_llama_extra_args !== undefined/,
  );
  assert.match(ADAPTER_AUTOLOAD, /loadedLlamaExtraArgs: null,/);
});

test("a compare pane records what it launched with", () => {
  assert.match(
    COMPOSER,
    /loadedLlamaExtraArgs:\s*\n?\s*resp\.requested_llama_extra_args !== undefined/,
  );
});

test("a stored empty list hydrates as a clear, not as nothing stored", () => {
  // Explicit [] is a tombstone; treating it as absent lets /load carry cleared flags over.
  assert.match(OVERRIDES, /explicit: Array\.isArray\(tokens\)/);
  assert.match(OVERRIDES, /\{ tokens: \[\], explicit: false \}/);
  assert.match(PANEL, /resolvedArgs\.explicit && local === undefined/);
  assert.match(ADAPTER, /\} else if \(stored\.explicit\) \{/);
  assert.match(COMPOSER, /\} else if \(resolvedArgs\.explicit\) \{/);
});

const CHAT_PAGE = readFileSync(
  path.join(HERE, "..", "src/features/chat/chat-page.tsx"),
  "utf8",
);

test("a Chat launch that applies remembered config carries its arguments", () => {
  // /load inherits flags only from the same resident model, so cold launches must send them.
  assert.match(
    CHAT_PAGE,
    /const remembered = rememberedConfigFor\(selection\);/,
  );
  assert.match(
    CHAT_PAGE,
    /\.\.\.\(remembered \? \{ config: remembered \} : \{\}\),/,
  );
});
