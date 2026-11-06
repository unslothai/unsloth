// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
Object.assign((globalThis.window as { location: object }).location, {
  href: "http://localhost/",
});

const providers = await import("../src/features/chat/external-providers.ts");
const { buildTitleRefreshRequest, buildTitleRequest } = await import(
  "../src/features/chat/utils/chat-title.ts"
);
const { claudeAgentModelLabels, isClaudeAgentOffMessage } = await import(
  "../src/features/chat/claude-agent.ts"
);
const vite = await createServer({ server: { middlewareMode: true } });
after(() => vite.close());

function connection(id: string, providerType: string) {
  return {
    id,
    providerType,
    name: id,
    baseUrl: "",
    models: ["opus"],
    hasApiKey: true,
    createdAt: 1,
    updatedAt: 1,
  };
}

test("Claude Agent takes images in chat and from MCP tools, as the backend does", () => {
  providers.setProviderModelCapabilities("claude_agent", undefined);
  assert.equal(
    providers.providerModelSupportsVision("claude_agent", "opus"),
    true,
  );
  assert.equal(
    providers.providerModelTakesMcpImages("claude_agent", "opus"),
    true,
  );
  providers.setProviderModelCapabilities("claude_agent", {
    haiku: { vision: false },
  });
  assert.equal(
    providers.providerModelSupportsVision("claude_agent", "haiku"),
    false,
  );
  assert.equal(
    providers.providerModelTakesMcpImages("claude_agent", "haiku"),
    false,
  );
  providers.setProviderModelCapabilities("claude_agent", undefined);
});

test("speech, transcription and image pickers leave Claude Agent out", () => {
  assert.equal(
    providers.servesMediaRequests({ providerType: "claude_agent" }),
    false,
  );
  assert.equal(providers.servesMediaRequests({ providerType: "openai" }), true);
  assert.equal(
    providers.servesMediaRequests({ providerType: "typesafe" }),
    false,
  );
  const voice = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    voice,
    /connections\.filter\(\(connection\) => servesMediaRequests\(connection\)\)/,
  );
  for (const file of [
    "features/audio/audio-page.tsx",
    "features/audio/hooks/use-speak-connection.ts",
  ]) {
    assert.match(
      readSrc(file),
      /\.filter\(\(c\) => servesMediaRequests\(c\)\)/,
      file,
    );
  }
});

test("a Claude Agent chat is titled locally, never by another Claude Code run", async () => {
  providers.saveExternalProviders([
    connection("claude", "claude_agent"),
    connection("anthropic", "anthropic"),
  ] as never);
  const claude = providers.buildExternalModelId("claude", "opus");
  const anthropic = providers.buildExternalModelId("anthropic", "opus");
  assert.equal(await buildTitleRequest(claude, "User: hi"), null);
  assert.equal(await buildTitleRefreshRequest(claude, "User: hi"), null);
  assert.notEqual(await buildTitleRequest(anthropic, "User: hi"), null);
});

test("the status card names models by display name or short id", () => {
  assert.deepEqual(
    claudeAgentModelLabels([
      { id: "opus", display_name: "Opus 5.5" },
      { id: "claude-opus-5-5-20261001" },
      "sonnet",
      "claude-sonnet-5-20260901",
      { id: "haiku" },
      "sonnet",
    ]),
    ["Opus 5.5", "sonnet", "haiku"],
  );
  assert.equal(
    isClaudeAgentOffMessage(
      "Claude Agent is off. Turn it on in Settings > System > Developer mode.",
    ),
    true,
  );
  assert.equal(
    isClaudeAgentOffMessage("Run `claude` in a terminal and sign in."),
    false,
  );
});

test("Developer mode reads the Claude Agent switch and its env override", async () => {
  const { developerModeFromApi } = await vite.ssrLoadModule(
    "/src/features/settings/api/developer-mode.ts",
  );
  const base = {
    video_cpu: false,
    video_cpu_setting: false,
    default_video_cpu: false,
    video_cpu_env_override: null,
    video_cpu_applicable: true,
    host_ram_bytes: null,
    min_ram_bytes: 1,
  };
  const on = developerModeFromApi({
    ...base,
    claude_agent: true,
    claude_agent_env_override: true,
  });
  assert.equal(on.claudeAgent, true);
  assert.equal(on.claudeAgentEnvOverride, true);
  const unset = developerModeFromApi(base);
  assert.equal(unset.claudeAgent, false);
  assert.equal(unset.claudeAgentEnvOverride, null);
});

test("saved Claude Agent connections leave the pickers while the registry drops it", async () => {
  const { connectionTypeListed } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  assert.equal(
    connectionTypeListed("claude_agent", new Set(["openai"])),
    false,
  );
  assert.equal(
    connectionTypeListed("claude_agent", new Set(["claude_agent"])),
    true,
  );
  assert.equal(connectionTypeListed("openai", new Set()), true);
  const sync = readSrc("features/chat/sync-external-providers.ts");
  assert.match(
    sync,
    /config\.is_enabled && connectionTypeListed\(config\.provider_type, registryTypes\)/,
  );
});

test("the Claude Agent switch lives in Developer mode and resyncs connections", () => {
  const section = readSrc(
    "features/settings/components/developer-mode-section.tsx",
  );
  assert.match(section, /data-testid="developer-mode-claude-agent"/);
  assert.match(
    section,
    /disabled=\{!settings \|\| saving \|\| claudeAgentForced !== null\}/,
  );
  assert.match(
    section,
    /if \(patch\.claudeAgent !== undefined\) void bootstrapPersistedCredentials\(\);/,
  );
});
