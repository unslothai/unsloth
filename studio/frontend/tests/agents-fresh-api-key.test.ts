// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { buildAgentShellCommands } from "../src/features/settings/components/agent-command.ts";
import { useFreshApiKeyStore } from "../src/features/settings/stores/fresh-api-key-store.ts";
import { readSrc } from "./helpers/kit.ts";

// `revealed` is local to the API tab, so it is lost on a tab switch or Done; examples read the store.
test("Settings > API examples and MCP take the stored key, not the tab-local one", () => {
  const tab = readSrc("features/settings/tabs/api-keys-tab.tsx");
  assert.match(tab, /<ApiExamplesPanel\s+apiKey=\{freshKey\}/);
  assert.match(tab, /<McpAccessSection apiKey=\{freshKey\} \/>/);
});

test("the API monitor page's examples carry the new key", () => {
  const page = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(page, /apiKey: freshKey \?\? undefined/);
  assert.match(page, /<MediaApiExamples\s+apiKey=\{freshKey\}/);
  assert.match(page, /<AudioApiExamples\s+apiKey=\{freshKey\}/);
  const dialog = readSrc(
    "features/api-monitor/components/api-example-dialog.tsx",
  );
  assert.match(dialog, /apiKey: freshKey \?\? undefined/);
});

const LAN_STUDIO = "http://10.4.0.110:8888";
const KEY = "sk-unsloth-abcd1234efgh5678";

test("a key just created reaches the Agents command for a LAN Studio", () => {
  const commands = buildAgentShellCommands(
    LAN_STUDIO,
    "unix",
    "opencode",
    "--model XHToken/Spark-X2.6-4B-GGUF:Q8_0",
    KEY,
  );
  assert.equal(
    commands.primary,
    `UNSLOTH_STUDIO_URL=http://10.4.0.110:8888 unsloth start opencode --api-key ${KEY} --model XHToken/Spark-X2.6-4B-GGUF:Q8_0`,
  );
  assert.match(
    commands.subagent,
    /--api-key sk-unsloth-abcd1234efgh5678 --as-subagent/,
  );
});

test("without a fresh key the command stays keyless", () => {
  const commands = buildAgentShellCommands(LAN_STUDIO, "unix", "opencode", "");
  assert.doesNotMatch(commands.primary, /--api-key/);
});

test("loopback never carries the key: the CLI mints its own there", () => {
  assert.equal(
    buildAgentShellCommands("http://127.0.0.1:8888", "unix", "claude", "", KEY)
      .primary,
    "unsloth start claude",
  );
});

test("revoking that key drops it, revoking another keeps it", () => {
  const store = useFreshApiKeyStore.getState();
  store.setKey(KEY);
  store.forgetIfRevoked("zzzz9999");
  assert.equal(useFreshApiKeyStore.getState().key, KEY);
  store.forgetIfRevoked("abcd1234");
  assert.equal(useFreshApiKeyStore.getState().key, null);
});

test("the Agents commands that can carry the key are kept out of reload snapshots", () => {
  const tab = readSrc("features/settings/tabs/agents-tab.tsx");
  assert.match(tab, /data-reload-snapshot-sensitive=\{sensitive \? "" : undefined\}/);
  assert.equal(tab.match(/sensitive=\{freshApiKey !== null\}/g)?.length, 2);
});
