// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  thinkingPresentation,
  stepThinkingEffort,
  effortLabel,
  type ThinkingCapabilities,
} from "../src/features/chat/lib/thinking-presentation.ts";

const caps: ThinkingCapabilities = {
  supportsReasoning: true,
  reasoningStyle: "reasoning_effort",
  reasoningAlwaysOn: false,
  supportsReasoningOff: true,
  reasoningEffortLevels: ["none", "low", "high", "xhigh"],
};
test("only named, supported levels become slider stops; Off stays separate", () => {
  assert.deepEqual(thinkingPresentation(caps), {
    kind: "adjustable",
    levels: ["low", "high", "xhigh"],
    canDisable: true,
    description: "Choose how much effort the model spends thinking.",
  });
  assert.equal(effortLabel("xhigh"), "Extra High");
});
test("fixed, toggle, mandatory, unknown and unsupported remain distinct", () => {
  assert.equal(
    thinkingPresentation({ ...caps, reasoningEffortLevels: ["high"] }).kind,
    "fixed",
  );
  const toggle = { ...caps, reasoningStyle: "enable_thinking" };
  assert.deepEqual(thinkingPresentation(toggle).levels, []);
  assert.equal(thinkingPresentation(toggle).kind, "toggle");
  assert.equal(
    thinkingPresentation({ ...toggle, reasoningAlwaysOn: true }).kind,
    "always-on",
  );
  assert.equal(
    thinkingPresentation({ ...toggle, reasoningKnown: false }).kind,
    "unknown",
  );
  assert.equal(
    thinkingPresentation({ ...caps, supportsReasoning: false }).kind,
    "unsupported",
  );
  assert.equal(
    thinkingPresentation({ ...caps, reasoningAlwaysOn: true }).canDisable,
    false,
  );
});
test("local effort-only templates with none can disable thinking", () => {
  assert.equal(
    thinkingPresentation({ ...caps, supportsReasoningOff: false }).canDisable,
    true,
  );
});
test("shortcuts clamp or wrap real steps and cannot move a fixed level", () => {
  const levels = thinkingPresentation(caps).levels;
  assert.equal(stepThinkingEffort(levels, "high", 1, false), "xhigh");
  assert.equal(stepThinkingEffort(levels, "xhigh", 1, false), "xhigh");
  assert.equal(stepThinkingEffort(levels, "xhigh", 1, true), "low");
  assert.equal(stepThinkingEffort(levels, "max", 1, false), "low");
  assert.equal(stepThinkingEffort(["high"], "high", 1, true), null);
});
