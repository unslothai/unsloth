// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

import { readSrc } from "./helpers/kit.ts";

let vite: ViteDevServer;
let withSavedModelIds: (catalog: string[], saved: string[]) => string[];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/chat-providers-dialog.tsx",
  );
  withSavedModelIds = loaded.withSavedModelIds;
});

after(async () => {
  await vite.close();
});

test("saved ids missing from the live catalog stay pickable", () => {
  assert.deepEqual(
    withSavedModelIds(
      ["openai/gpt-5.5", "deepseek/deepseek-v4-pro"],
      ["deepseek/deepseek-v4-pro:nitro", "openai/gpt-5.5"],
    ),
    ["openai/gpt-5.5", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro:nitro"],
  );
});

test("automatic catalog loads ignore the typed API key and Base URL", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /apiKey: silent \? "" : apiKey\.trim\(\),/);
  assert.match(
    dialog,
    /const baseUrl = silent\s*\? editingProvider\?\.baseUrl \|\| null\s*: parseBaseUrlForProvider\(\s*baseUrlDraft,/,
  );
});

test("explicit OpenRouter reloads keep saved routing variants too", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(
    dialog,
    /if \(silent \|\| hasPublicModelCatalog\(providerType\)\) \{\s*setAvailableModels\(withSavedModelIds\(modelIds, selectedModelIds\)\);/,
  );
});

test("automatic catalog loads do not disable saving", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /if \(silent\) setCatalogAutoLoad\(autoLoad\);\s*else setModelsLoading\(true\);/);
  const save = dialog.match(/disabled=\{\s*registryLoading \|\|[^}]*\}\s*onClick=\{\(\) =>\s*editingProviderId\s*\? void saveProviderEdits\(\)/);
  assert.ok(save, "Save button not found");
  assert.doesNotMatch(save[0], /catalogAutoLoading/);
});

test("a response from a retired automatic load is dropped by identity, not key", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /const autoLoad = \{ key: autoLoadCatalogKey \};\s*autoLoadedCatalogRef\.current = autoLoad;/);
  assert.match(dialog, /if \(silent && autoLoadedCatalogRef\.current !== autoLoad\) return;/);
  assert.match(
    dialog,
    /return \(\) => \{\s*if \(autoLoadedCatalogRef\.current === autoLoad\) autoLoadedCatalogRef\.current = null;\s*\};\s*\}, \[autoLoadCatalogKey\]\);/,
  );
});
