// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);
const { requestParsesThinkTags } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

type Caps = ReturnType<typeof getExternalReasoningCapabilities>;

// Runs the adapter's own external reasoning expressions, so the test follows the shipped code.
function externalReasoningFields(caps: Caps, reasoningEnabled: boolean, effort = "high"): unknown {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const enabledAt = adapter.indexOf("const externalReasoningEnabled =");
  const fieldsAt = adapter.indexOf("const externalReasoningFields", enabledAt);
  assert.ok(enabledAt > 0 && fieldsAt > enabledAt, "the adapter's external reasoning fields moved");
  const enabled = adapter.slice(adapter.indexOf("=", enabledAt) + 1, adapter.indexOf(";", enabledAt));
  const fields = adapter.slice(adapter.indexOf("=", fieldsAt) + 1, adapter.indexOf(";", fieldsAt)).trim();
  const build = new Function(
    "externalReasoningCaps",
    "reasoningEnabled",
    "selectedExternalEffort",
    "fallbackExternalEffort",
    `const externalReasoningEnabled = ${enabled};\nreturn ${fields};`,
  );
  return build(caps, reasoningEnabled, effort, "high");
}

test("an always-on catalog model sends thinking on even when the chat stored it off", () => {
  const magistral = getExternalReasoningCapabilities("mistral", "magistral-small");
  assert.equal(magistral.reasoningStyle, "enable_thinking");
  assert.equal(magistral.supportsReasoningOff, false);
  assert.deepEqual(externalReasoningFields(magistral, false), { thinking: { type: "enabled" } });
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
  const qwen = getExternalReasoningCapabilities("qwen", "qwen3.5-plus");
  assert.equal(qwen.reasoningStyle, "enable_thinking");
  assert.equal(qwen.supportsReasoningOff, true);
  assert.deepEqual(externalReasoningFields(qwen, false), { thinking: { type: "disabled" } });
  assert.deepEqual(externalReasoningFields(qwen, true), { thinking: { type: "enabled" } });
});

test("an external effort of none reads as thinking off", () => {
  const gpt = getExternalReasoningCapabilities("openai", "gpt-5.1");
  assert.equal(gpt.reasoningStyle, "reasoning_effort");
  assert.ok(gpt.reasoningEffortLevels.includes("none"));
  assert.equal(requestParsesThinkTags(externalReasoningFields(gpt, true, "none") as object), false);
  assert.equal(requestParsesThinkTags(externalReasoningFields(gpt, true, "high") as object), true);
});
