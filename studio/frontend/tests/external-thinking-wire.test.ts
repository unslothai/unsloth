// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { clearProviderModelCatalog, setProviderModelCatalog } = await import("../src/features/chat/model-catalog.ts");
const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

type Caps = ReturnType<typeof getExternalReasoningCapabilities>;

// Runs the adapter's own external reasoning expressions, so the test follows the shipped code.
function externalReasoningFields(caps: Caps, reasoningEnabled: boolean): unknown {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const enabledAt = adapter.indexOf("const externalReasoningEnabled =");
  const fieldsAt = adapter.indexOf("...(externalReasoningCaps.supportsReasoning", enabledAt);
  assert.ok(enabledAt > 0 && fieldsAt > enabledAt, "the adapter's external reasoning fields moved");
  const enabled = adapter.slice(adapter.indexOf("=", enabledAt) + 1, adapter.indexOf(";", enabledAt));
  const fields = adapter.slice(fieldsAt + 3, adapter.indexOf(": {}),", fieldsAt) + ": {})".length);
  const build = new Function(
    "externalReasoningCaps",
    "reasoningEnabled",
    "selectedExternalEffort",
    "fallbackExternalEffort",
    `const externalReasoningEnabled = ${enabled};\nreturn ${fields};`,
  );
  return build(caps, reasoningEnabled, "high", "high");
}

test("an always-on catalog model sends thinking on even when the chat stored it off", () => {
  setProviderModelCatalog("openrouter", [{ id: "acme/always-thinking", reasoning: { mandatory: true } }], 1);
  try {
    const caps = getExternalReasoningCapabilities("openrouter", "acme/always-thinking");
    assert.equal(caps.reasoningStyle, "enable_thinking");
    assert.equal(caps.supportsReasoningOff, false);
    assert.deepEqual(externalReasoningFields(caps, false), { thinking: { type: "enabled" } });
  } finally {
    clearProviderModelCatalog("openrouter");
  }
});

test("every Thinking control resolves reasoning for the id the adapter sends, not the router's last pick", () => {
  const modelArguments = (relative: string): string[] => {
    const calls = [...readSrc(relative).matchAll(/getExternalReasoningCapabilities\(\s*[^,]+,\s*(?:\/\/[^\n]*\n\s*)?([^,]+),/g)];
    assert.ok(calls.length > 0, relative);
    return calls.map((match) => match[1].trim().replace("?.", "."));
  };
  assert.deepEqual(modelArguments("features/chat/api/chat-adapter.ts"), ["externalSelection.modelId"]);
  for (const file of ["features/chat/shared-composer.tsx", "components/assistant-ui/thread.tsx"]) {
    assert.deepEqual(modelArguments(file), ["externalSelection.modelId"], file);
  }
  assert.equal(getExternalReasoningCapabilities("openrouter", "openrouter/free").reasoningStyle, "enable_thinking");
});

test("a toggleable catalog model still sends the stored choice", () => {
  setProviderModelCatalog("openrouter", [{ id: "acme/toggle-thinking", reasoning: { mandatory: false } }], 1);
  try {
    const caps = getExternalReasoningCapabilities("openrouter", "acme/toggle-thinking");
    assert.equal(caps.reasoningStyle, "enable_thinking");
    assert.equal(caps.supportsReasoningOff, true);
    assert.deepEqual(externalReasoningFields(caps, false), { thinking: { type: "disabled" } });
    assert.deepEqual(externalReasoningFields(caps, true), { thinking: { type: "enabled" } });
  } finally {
    clearProviderModelCatalog("openrouter");
  }
});
