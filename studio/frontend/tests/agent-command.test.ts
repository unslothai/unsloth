// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildAgentCommand,
  buildAgentShellCommands,
  quoteShellArg,
} from "../src/features/settings/components/agent-command.ts";

const REMOTE_STUDIO = "https://shed-topic-remark-arguments.trycloudflare.com/";

test("the Windows variant reproduces the reported PowerShell command", () => {
  assert.equal(
    buildAgentCommand(REMOTE_STUDIO, null, "windows", "claude"),
    '$env:UNSLOTH_STUDIO_URL="https://shed-topic-remark-arguments.trycloudflare.com"; unsloth start claude',
  );
});

test("the Unix override emits a Bash-compatible remote command", () => {
  assert.equal(
    buildAgentCommand(REMOTE_STUDIO, null, "unix", "claude"),
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start claude",
  );
});

test("safe model IDs do not get unnecessary quotes", () => {
  assert.equal(
    quoteShellArg("unsloth/Qwen3.8:UD-Q4_K_XL", "unix"),
    "unsloth/Qwen3.8:UD-Q4_K_XL",
  );
  assert.equal(
    quoteShellArg("unsloth/Qwen3.8:UD-Q4_K_XL", "windows"),
    "unsloth/Qwen3.8:UD-Q4_K_XL",
  );
});

test("model paths use the selected shell's single-quote escaping", () => {
  assert.equal(
    quoteShellArg("/tmp/Lee's model.gguf", "unix"),
    "'/tmp/Lee'\\''s model.gguf'",
  );
  assert.equal(
    quoteShellArg("C:\\Models\\Lee's model.gguf", "windows"),
    "'C:\\Models\\Lee''s model.gguf'",
  );
});

test("the Unix selection reaches every generated command", () => {
  const commands = buildAgentShellCommands(
    REMOTE_STUDIO,
    "unix",
    "codex",
    "--model '/tmp/My model.gguf'",
  );
  assert.equal(
    commands.primary,
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start codex --model '/tmp/My model.gguf'",
  );
  assert.equal(
    commands.subagent,
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start codex --as-subagent --model '/tmp/My model.gguf'",
  );
  assert.deepEqual(commands.passThrough, [
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start claude --continue",
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start codex --persist resume --last",
  ]);
  assert.equal(
    commands.dryRun,
    "UNSLOTH_STUDIO_URL=https://shed-topic-remark-arguments.trycloudflare.com unsloth start claude --no-launch",
  );
  assert.match(commands.remoteSetup, /^export UNSLOTH_STUDIO_URL=/);
});

test("the Windows selection reaches every generated command", () => {
  const commands = buildAgentShellCommands(
    REMOTE_STUDIO,
    "windows",
    "claude",
    "--model 'C:\\Models\\My model.gguf'",
  );
  for (const command of [
    commands.primary,
    commands.subagent,
    ...commands.passThrough,
    commands.dryRun,
  ]) {
    assert.match(command, /^\$env:UNSLOTH_STUDIO_URL=/);
  }
  assert.equal(
    commands.subagent,
    '$env:UNSLOTH_STUDIO_URL="https://shed-topic-remark-arguments.trycloudflare.com"; unsloth start claude --as-subagent --model \'C:\\Models\\My model.gguf\'',
  );
  assert.match(commands.remoteSetup, /^\$env:UNSLOTH_STUDIO_URL =/);
});
