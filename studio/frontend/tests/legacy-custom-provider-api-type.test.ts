// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test, { after } from "node:test";
import { createServer } from "vite";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
Object.assign(window.location, { href: "http://localhost/" });
const vite = await createServer({ server: { middlewareMode: true } });
after(() => vite.close());

test("legacy OpenAI rows require built-in labels and preserve managed endpoints", async () => {
  const sync = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const registry = [
    {
      provider_type: "openai",
      display_name: "OpenAI",
      base_url: "https://api.openai.com/v1",
    },
  ];

  for (const [displayName, baseUrl, existingType, expectedType] of [
    ["My OpenAI Key", "https://api.openai.com/v1", undefined, "openai"],
    ["vLLM", "https://api.openai.com/v1/", "vllm", "openai"],
    [
      "Team Azure",
      "https://team.openai.azure.com/openai/v1",
      "custom",
      "openai",
    ],
    [
      "Team Foundry",
      "https://team.services.ai.azure.com/openai/v1",
      "custom",
      "openai",
    ],
    ["vLLM", "https://team.services.ai.azure.com.attacker.example/v1", undefined, "vllm"],
    ["vLLM", "https://gateway.example/v1", undefined, "vllm"],
    ["Custom", "https://gateway.example/v1", undefined, "custom"],
    ["OpenAI", "https://gateway.example/v1", undefined, "openai"],
    ["My OpenAI Proxy", "https://gateway.example/v1", "custom", "openai"],
    ["My Gateway", "http://localhost:8080/v1", undefined, "openai"],
  ] as const) {
    const uiType = sync.resolveUiProviderTypeFromConfig(
      "openai",
      displayName,
      baseUrl,
      registry,
      existingType,
    );
    assert.equal(uiType, expectedType);
  }

  const dialog = await readFile(
    new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    dialog,
    /function shouldShowProviderApiType\(providerType: string\): boolean \{\s*return providerType === LEGACY_CUSTOM_PROVIDER_TYPE;\s*\}/,
  );
  assert.match(dialog, /\{shouldShowProviderApiType\(providerType\) \? \(/);
  assert.doesNotMatch(
    dialog,
    /editingBackendProviderType === LEGACY_CUSTOM_PROVIDER_TYPE/,
  );
});
