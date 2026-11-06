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
const claudeAgent = await import("../src/features/chat/claude-agent.ts");
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

const STATUS_MODELS = [
  { id: "opus", display_name: "Opus 5.5", alias: true },
  { id: "fable", display_name: "Fable 5.1", alias: true },
  { id: "sonnet", display_name: "Sonnet 5.5", alias: true },
  { id: "haiku", display_name: "Haiku 5.5", alias: true },
  { id: "claude-haiku-4-5-20251001", display_name: "Haiku 4.5", alias: false },
  { id: "claude-opus-4-8", display_name: "Opus 4.8", alias: false },
];

test("models are named by display name, aliases first, with aliases picked by default", () => {
  const options = claudeAgent.claudeAgentModelOptions([
    ...STATUS_MODELS,
    "sonnet",
    { id: "claude-sonnet-5" },
    "custom",
  ]);
  const groups = claudeAgent.claudeAgentModelGroups(options);
  assert.deepEqual(
    groups.main.map((option: { name: string }) => option.name),
    ["Opus 5.5", "Fable 5.1", "Sonnet 5.5", "Haiku 5.5", "Custom"],
  );
  assert.deepEqual(
    groups.more.map((option: { name: string }) => option.name),
    ["Haiku 4.5", "Opus 4.8", "Sonnet 5"],
  );
  const listed = claudeAgent.claudeAgentModelOptions(STATUS_MODELS);
  assert.deepEqual(claudeAgent.claudeAgentDefaultModels(listed), [
    "opus",
    "fable",
    "sonnet",
    "haiku",
  ]);
  assert.deepEqual(
    claudeAgent.claudeAgentInitialSelection(listed, ["haiku", "claude-x-1"]),
    { selected: ["haiku"], extra: ["claude-x-1"] },
  );
  assert.deepEqual(
    claudeAgent.claudeAgentModelsToSave(
      listed,
      ["claude-opus-4-8", "opus", "gone"],
      "claude-x-1, opus\n",
    ),
    ["opus", "claude-opus-4-8", "claude-x-1"],
  );
});

test("unknown Claude Agent ids fall back to a title-cased name", () => {
  assert.equal(claudeAgent.claudeAgentFallbackName("opus"), "Opus");
  assert.equal(claudeAgent.claudeAgentFallbackName("claude-opus-4-8"), "Opus 4.8");
  assert.equal(
    claudeAgent.claudeAgentFallbackName("claude-haiku-4-5-20251001"),
    "Haiku 4.5",
  );
});

test("connection labels change for Claude Agent only, and remember the CLI's names", () => {
  claudeAgent.rememberClaudeAgentModels(STATUS_MODELS);
  assert.equal(claudeAgent.connectionModelLabel("claude_agent", "opus"), "Opus 5.5");
  assert.equal(claudeAgent.connectionModelLabel("claude_agent", "claude-sonnet-9"), "Sonnet 9");
  assert.equal(claudeAgent.connectionModelLabel("anthropic", "opus"), "opus");
  assert.equal(claudeAgent.connectionModelLabel(undefined, "haiku"), "haiku");
  const stored = JSON.parse(
    globalThis.localStorage.getItem("unsloth_claude_agent_model_names") ?? "{}",
  );
  assert.equal(stored.fable, "Fable 5.1");
  claudeAgent.rememberClaudeAgentModels([]);
  assert.equal(claudeAgent.connectionModelLabel("claude_agent", "fable"), "Fable 5.1");
});

test("the setup view walks off, checklist, repair and ready", () => {
  const step = (overrides: Record<string, unknown>) =>
    claudeAgent.claudeAgentSetupStep({
      off: false,
      forcedOff: false,
      checking: false,
      status: null,
      error: "",
      ...overrides,
    });
  const status = (overrides: Record<string, unknown>) => ({
    ready: false,
    version: null,
    account: null,
    models: [],
    message: "x",
    ...overrides,
  });
  assert.equal(step({ off: true }), "off");
  assert.equal(step({ off: true, forcedOff: true }), "forced_off");
  assert.equal(
    step({ error: "Claude Agent is off. Turn it on in Settings > System > Developer mode." }),
    "off",
  );
  assert.equal(step({ checking: true }), "checking");
  assert.equal(step({ error: "Network down" }), "failed");
  assert.equal(step({ status: status({ problem: "cli_missing" }) }), "install_cli");
  assert.equal(step({ status: status({ problem: "signed_out" }) }), "sign_in");
  assert.equal(step({ status: status({ sdk_missing: true }) }), "sdk_missing");
  assert.equal(step({ status: status({ problem: "failed" }) }), "failed");
  assert.equal(step({ status: status({ ready: true }), checking: true }), "ready");

  assert.deepEqual(claudeAgent.claudeAgentChecklist("install_cli"), [
    { id: "install", state: "current" },
    { id: "sign_in", state: "next" },
    { id: "check", state: "next" },
  ]);
  assert.deepEqual(
    claudeAgent.claudeAgentChecklist("sign_in")?.map((item: { state: string }) => item.state),
    ["done", "current", "next"],
  );
  assert.equal(claudeAgent.claudeAgentChecklist("sdk_missing"), null);
  assert.equal(claudeAgent.claudeAgentChecklist("ready"), null);
  for (const value of ["install_cli", "sign_in", "sdk_missing", "failed"] as const) {
    assert.equal(claudeAgent.claudeAgentRechecksOnReturn(value), true);
  }
  for (const value of ["ready", "off", "forced_off", "checking"] as const) {
    assert.equal(claudeAgent.claudeAgentRechecksOnReturn(value), false);
  }
});

test("the checklist shows the official install command for the backend's OS", () => {
  assert.equal(
    claudeAgent.claudeAgentInstallCommand("unix"),
    "curl -fsSL https://claude.ai/install.sh | bash",
  );
  assert.equal(claudeAgent.claudeAgentInstallCommand(undefined), claudeAgent.claudeAgentInstallCommand("unix"));
  assert.equal(
    claudeAgent.claudeAgentInstallCommand("windows"),
    // lint-allow: AV005 asserts the displayed install command
    "irm https://claude.ai/install.ps1 | iex",
  );
  assert.equal(claudeAgent.CLAUDE_CODE_NPM_INSTALL, "npm install -g @anthropic-ai/claude-code");
  assert.equal(
    claudeAgent.claudeAgentSdkCommand(
      'Install the Claude Agent SDK in the Unsloth environment: uv pip install --python "C:/py" "claude-agent-sdk>=0.2.178"',
    ),
    'uv pip install --python "C:/py" "claude-agent-sdk>=0.2.178"',
  );
  assert.equal(
    claudeAgent.claudeAgentStatusLine({
      ready: true,
      version: "2.1.318",
      account: "Claude Max",
      models: [],
      message: null,
    }),
    "Claude Max · Claude Code 2.1.318",
  );
  assert.equal(
    claudeAgent.isClaudeAgentOffMessage("Run `claude` in a terminal and sign in."),
    false,
  );
});

test("Add connection offers Claude Agent with its own setup view and Connect button", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /entry\.provider_type === CLAUDE_AGENT_PROVIDER_TYPE \? \(\s*<ClaudeExperimentalBadge \/>/);
  assert.match(dialog, /\{isClaudeAgent \? \(\s*<ClaudeAgentSetup/);
  assert.match(dialog, /off=\{claudeAgentOff\}/);
  assert.match(dialog, /available_but_off === true/);
  assert.match(dialog, /\{isClaudeAgent \? null : \(<>/);
  assert.match(dialog, /connectionModelLabel\(provider\.providerType, model, claudeAgentNames\)/);
  const setup = readSrc("features/chat/claude-agent-status.tsx");
  assert.match(setup, /updateDeveloperModeSettings\(\{ claudeAgent: true \}\)/);
  assert.match(setup, /disabled=\{step === "forced_off" \|\| turningOn\}/);
  assert.match(setup, /window\.addEventListener\("focus", onReturn\)/);
  assert.match(setup, /document\.addEventListener\("visibilitychange", onReturn\)/);
  assert.doesNotMatch(setup, /upstream provider/);
});

test("pickers and the header show Claude Agent display names", () => {
  assert.match(
    readSrc("features/chat/chat-page.tsx"),
    /let displayName = connectionModelLabel\(\s*provider\.providerType,\s*model,\s*claudeAgentNames,\s*\)/,
  );
  assert.match(readSrc("features/scheduled-tasks/task-dialog.tsx"), /\{option\.label\}/);
  assert.match(
    readSrc("features/chat/lib/external-model-label.ts"),
    /connectionModelLabel\(provider\?\.providerType, parsed\.modelId\)/,
  );
});

test("a Claude Agent reply says Starting Claude until its first chunk", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /runtime\.setLiveStarting\(liveThreadKey\(serverCancel\), true\)/);
  assert.match(adapter, /runtime\.setLiveStarting\(liveThreadKey\(serverCancel\), false\)/);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /data-slot="starting-claude"/);
  assert.match(thread, /t\("generationStats\.startingClaude"\)/);
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
  const { connectionTypeListed, usableRegistryTypes } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const offered = usableRegistryTypes([
    { provider_type: "openai" },
    { provider_type: "claude_agent", available_but_off: true },
  ]);
  assert.equal(connectionTypeListed("claude_agent", offered), false);
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
