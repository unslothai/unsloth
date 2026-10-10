// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// source assertions avoid the auth barrel, which cannot load in this test harness.

import assert from "node:assert/strict";
import test from "node:test";

import { ar } from "../src/i18n/locales/ar.ts";
import { de } from "../src/i18n/locales/de.ts";
import { en } from "../src/i18n/locales/en.ts";
import { es } from "../src/i18n/locales/es.ts";
import { fr } from "../src/i18n/locales/fr.ts";
import { he } from "../src/i18n/locales/he.ts";
import { hi } from "../src/i18n/locales/hi.ts";
import { it } from "../src/i18n/locales/it.ts";
import { ja } from "../src/i18n/locales/ja.ts";
import { ko } from "../src/i18n/locales/ko.ts";
import { ptBR } from "../src/i18n/locales/pt-br.ts";
import { ru } from "../src/i18n/locales/ru.ts";
import { sv } from "../src/i18n/locales/sv.ts";
import { zhCN } from "../src/i18n/locales/zh-CN.ts";
import { readSrc } from "./helpers/kit.ts";

const SECTION = readSrc("features/settings/components/mcp-access-section.tsx");
const API_TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const USAGE = readSrc("features/settings/components/usage-examples.tsx");

const LOCALES = { ar, de, en, es, fr, he, hi, it, ja, ko, ptBR, ru, sv, zhCN };

test("only the owner sees it, after the Decision API section, and settings search finds it", () => {
  assert.match(
    API_TAB,
    /\{isOwner \? <DecisionApiSection \/> : null\}\s*\{isOwner \? <McpAccessSection apiKey=\{revealed\} \/> : null\}/,
  );
  assert.doesNotMatch(SECTION, /useIsAccountOwner/);
  assert.match(
    SECTION,
    /data-settings-label=\{t\("settings.apiKeys.mcp.title"\)\}/,
  );
});

test("the switch is the shared primitive and the environment locks it in text", () => {
  assert.match(
    SECTION,
    /import \{ Switch \} from "@\/components\/ui\/switch";/,
  );
  assert.match(SECTION, /<SettingsRow/);
  assert.match(SECTION, /disabled=\{busy \|\| settings\.forcedByEnv\}/);
  assert.match(SECTION, /const ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP";/);
  assert.match(
    SECTION,
    /forcedByEnv\s*\?\s*t\("settings.apiKeys.mcp.lockedByEnv", \{ name: ENV_FORCE \}\)/,
  );
  assert.equal(en.settings.apiKeys.mcp.lockedByEnv, "Set by {name}.");
});

test("a rejected change resyncs and shows the error inline, and a failed first load still renders the section", () => {
  const apply = SECTION.slice(
    SECTION.indexOf("const apply = async"),
    SECTION.indexOf("if (!(settings || error))"),
  );
  assert.ok(
    apply.indexOf("updateMcpAccess(enabled)") <
      apply.lastIndexOf("loadMcpAccess()"),
  );
  assert.match(apply, /t\("settings.apiKeys.mcp.saveError"\)/);
  assert.match(SECTION, /text-xs leading-snug text-destructive/);
  assert.doesNotMatch(SECTION, /toast/);
  assert.match(SECTION, /if \(!\(settings \|\| error\)\) \{\s*return null;/);
  assert.ok(SECTION.indexOf("{error ? (") < SECTION.indexOf("{settings ? ("));
  assert.match(SECTION, /translate\("settings.apiKeys.mcp.loadError"\)/);
});

test("no emerald and no unscaled lengths", () => {
  assert.doesNotMatch(SECTION, /emerald/);
  assert.doesNotMatch(SECTION, /-\[\d+(?:\.\d+)?(?:px|rem)\]/);
});

test("the copy stays plain in every locale", () => {
  const keys = Object.keys(en.settings.apiKeys.mcp);
  for (const [name, locale] of Object.entries(LOCALES)) {
    const copy = (
      locale.settings.apiKeys as unknown as {
        mcp: Record<string, string>;
      }
    ).mcp;
    assert.deepEqual(Object.keys(copy), keys, name);
    for (const value of Object.values(copy)) {
      assert.doesNotMatch(value, /[;—]/, `${name}: ${value}`);
    }
  }
});

test("the setup snippet shows only while on, targets the usage examples' address and names the variable and file", () => {
  assert.match(SECTION, /const snippet = settings\?\.enabled\s*\?/);
  assert.match(SECTION, /\{snippet \? \(/);
  assert.match(SECTION, /<CommandBlock command=\{snippet\.text\} \/>/);
  assert.match(SECTION, /buildMcpSnippet\(agent, settings\.url, os, apiKey\)/);
  assert.match(
    SECTION,
    /useTunnel && cloudflareUrl\s*\?\s*cloudflareUrl\s*:\s*\(serverUrl \?\? origin\)/,
  );
  // tunnel preference changes must notify this section immediately.
  assert.match(
    SECTION,
    /useSyncExternalStore\(\s*subscribeUseTunnelPref,\s*readUseTunnelPref,/,
  );
  assert.match(
    USAGE,
    /function writeUseTunnelPref[\s\S]*?for \(const listener of useTunnelListeners\) listener\(\);/,
  );
  assert.match(
    SECTION,
    /snippet\.readsKeyEnv \? \([\s\S]*?t\("settings.apiKeys.mcp.exportKeyHint", \{\s*name: MCP_API_KEY_ENV,?\s*\}\)/,
  );
  assert.match(
    SECTION,
    /t\("settings.apiKeys.mcp.configFileHint", \{\s*path: snippet\.configPath,?\s*\}\)/,
  );
  assert.equal(
    en.settings.apiKeys.mcp.exportKeyHint,
    "Set {name} to an access token from this page before you start the agent.",
  );
});

test("a key created on this page fills the snippet and is kept out of reload snapshots", () => {
  assert.match(SECTION, /buildMcpSnippet\(agent, base, os, apiKey\)/);
  assert.match(
    SECTION,
    /data-reload-snapshot-sensitive=\{apiKey \? "" : undefined\}\s*>\s*<CommandBlock command=\{snippet\.text\} \/>/,
  );
});

test("the agent picker keeps the pill trigger and shared list, CLI agents get the shell choice, and both picks persist", () => {
  const trigger = SECTION.slice(
    SECTION.indexOf("<SelectTrigger"),
    SECTION.indexOf("</SelectTrigger>"),
  );
  assert.ok(trigger.length > 0);
  assert.doesNotMatch(trigger, /rounded-/);
  assert.match(SECTION, /SUPPORTED_AGENTS\.map\(/);
  assert.match(SECTION, /from "\.\/coding-agent-list";/);
  assert.match(SECTION, /from "\.\/agent-command-block";/);
  assert.match(SECTION, /MCP_SHELL_AGENT_IDS\.has\(agent\) \? \(\s*<fieldset/);
  assert.match(SECTION, /className="hub-tab-toggle inline-flex/);
  assert.match(SECTION, /aria-pressed=\{os === option\.os\}/);
  assert.match(SECTION, /useSettingsPanelPrefsStore\(\(s\) => s\.mcpAgent\)/);
  assert.match(SECTION, /useSettingsPanelPrefsStore\(\(s\) => s\.mcpOs\)/);
  assert.match(SECTION, /onValueChange=\{setStoredAgent\}/);
  assert.match(SECTION, /onClick=\{\(\) => setStoredOs\(option\.os\)\}/);
});
