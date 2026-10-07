// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

for (const style of [
  "reasoning_effort",
  "reasoning",
  "thinking",
  "chat_template_kwargs.enable_thinking",
]) {
  test(`Custom explicitly opts into ${style}, independently of model or URL`, () => {
    // Structural options keep the red control failing on behavior, not imports.
    const options = { reasoningConfig: { enabled: true, style } };
    const caps = getExternalReasoningCapabilities(
      "custom",
      "unlisted-model",
      options,
    );
    assert.equal(caps.supportsReasoning, true);
    assert.equal(caps.supportsReasoningOff, true);
    assert.equal(
      caps.reasoningStyle,
      ["reasoning_effort", "reasoning"].includes(style)
        ? "reasoning_effort"
        : "enable_thinking",
    );
    if (caps.reasoningStyle === "reasoning_effort") {
      assert.deepEqual(
        [...caps.reasoningEffortLevels],
        ["none", "low", "medium", "high"],
      );
    }
  });
}

test("unconfigured Custom keeps mandatory-reasoning routes always on", () => {
  const caps = getExternalReasoningCapabilities("custom", "deepseek/deepseek-r1", {});
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningAlwaysOn, true);
  assert.equal(caps.supportsReasoningOff, false);
  assert.equal(
    getExternalReasoningCapabilities("custom", "unlisted-model", {}).supportsReasoning,
    false,
  );
});
