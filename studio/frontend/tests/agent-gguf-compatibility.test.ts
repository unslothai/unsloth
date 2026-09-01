// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  agentRunsOnActiveModel,
  fallbackAgent,
  pickCompatibleAgent,
} from "../src/features/settings/components/agent-command.ts";

test("only the llama-server-only agents are gated on GGUF", () => {
  for (const agent of ["codex", "claude"]) {
    assert.equal(agentRunsOnActiveModel(agent, false), false);
    assert.equal(agentRunsOnActiveModel(agent, true), true);
  }
  for (const agent of ["opencode", "openclaw", "hermes"]) {
    assert.equal(agentRunsOnActiveModel(agent, false), true);
    assert.equal(agentRunsOnActiveModel(agent, true), true);
  }
});

test("a safetensors model never falls back to a GGUF-only agent", () => {
  // The old fallback was the literal default, "claude". Now that `unsloth start
  // claude` refuses a non-GGUF model, resetting to it would hand the user a
  // second command the CLI rejects.
  assert.equal(fallbackAgent(false), "opencode");
  assert.equal(fallbackAgent(true), "claude");
});

test("the first detected agent that can run the active model wins", () => {
  assert.equal(
    pickCompatibleAgent(["claude", "codex", "opencode"], "opencode", true),
    "claude",
  );
  assert.equal(
    pickCompatibleAgent(["claude", "codex", "opencode"], "claude", false),
    "opencode",
  );
});

test("only GGUF-only agents detected on a safetensors model resets the pick", () => {
  assert.equal(pickCompatibleAgent(["claude"], "claude", false), "opencode");
  assert.equal(pickCompatibleAgent(["codex"], "codex", false), "opencode");
  assert.equal(pickCompatibleAgent(["claude", "codex"], "claude", false), "opencode");
});

test("a compatible current pick is left alone", () => {
  assert.equal(pickCompatibleAgent(["codex"], "opencode", false), null);
  assert.equal(pickCompatibleAgent([], "opencode", false), null);
  assert.equal(pickCompatibleAgent([], "claude", true), null);
});

test("nothing detected still corrects a pick that cannot run", () => {
  // The non-loopback reset clears the detected list and drops back to the
  // default agent, so this is the only place that correction can happen.
  assert.equal(pickCompatibleAgent([], "claude", false), "opencode");
});

test("loading a GGUF re-steers back to the GGUF-only agents", () => {
  // Both directions, so the picker is not one-way sticky.
  assert.equal(pickCompatibleAgent(["claude", "codex"], "opencode", true), "claude");
  assert.equal(pickCompatibleAgent(["codex", "opencode"], "codex", false), "opencode");
});
