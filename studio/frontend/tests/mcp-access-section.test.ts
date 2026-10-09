// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > API gets an owner-only switch for agent access (MCP). The section reaches
// the auth barrel, which cannot be imported here, so this asserts on source.

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

const LOCALES = { ar, de, en, es, fr, he, hi, it, ja, ko, ptBR, ru, sv, zhCN };

test("only the owner sees it, after the Decision API section", () => {
  assert.match(
    API_TAB,
    /\{isOwner \? <DecisionApiSection \/> : null\}\s*\{isOwner \? <McpAccessSection[^>]*\/> : null\}/,
  );
  assert.doesNotMatch(SECTION, /useIsAccountOwner/);
});

test("the section labels itself for settings search", () => {
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
  assert.match(SECTION, /disabled=\{busy \|\| forcedByEnv\}/);
  assert.match(SECTION, /const ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP";/);
  assert.match(
    SECTION,
    /forcedByEnv\s*\?\s*t\("settings.apiKeys.mcp.lockedByEnv", \{ name: ENV_FORCE \}\)/,
  );
  assert.equal(en.settings.apiKeys.mcp.lockedByEnv, "Set by {name}.");
});

test("a rejected change resyncs from the server and shows the error inline", () => {
  const apply = SECTION.slice(
    SECTION.indexOf("const apply = async"),
    SECTION.indexOf("const header ="),
  );
  assert.ok(
    apply.indexOf("updateMcpAccess(enabled)") <
      apply.lastIndexOf("loadMcpAccess()"),
  );
  assert.match(apply, /t\("settings.apiKeys.mcp.saveError"\)/);
  assert.match(SECTION, /text-xs leading-snug text-destructive/);
  assert.doesNotMatch(SECTION, /toast/);
});

test("a failed first load still renders the section with its error", () => {
  assert.match(SECTION, /if \(!settings\) \{\s*return error \?/);
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
