// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// source assertions avoid the auth barrel, which cannot load in this test harness.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MCP_TOOL_LABEL_KEYS,
  mcpToolLabelKey,
} from "../src/features/settings/components/mcp-tool-labels.ts";
import { OWNER_ONLY_SETTINGS_TABS } from "../src/features/settings/settings-tab-visibility.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";
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

const DIALOG = readSrc("features/settings/settings-dialog.tsx");
const PALETTE = readSrc("components/command-palette.tsx");
const TAB = readSrc("features/settings/tabs/mcp-tab.tsx");
const TOOLS = readSrc("features/settings/components/mcp-tools-section.tsx");
const ROW = readSrc("features/settings/components/unsloth-studio-mcp-row.tsx");
const AGENTS_TAB = readSrc("features/settings/tabs/agents-tab.tsx");

const LOCALES = { ar, de, en, es, fr, he, hi, it, ja, ko, ptBR, ru, sv, zhCN };

// Same list and order as the backend's tool_catalog().
const CATALOG = [
  "studio_status",
  "list_models",
  "load_model",
  "unload_model",
  "plan_load",
  "download_model",
  "chat",
  "embed",
  "system_one",
  "generate_image",
  "generate_audio",
  "transcribe",
  "analyze_audio",
  "generate_video",
  "run_recipe",
  "get_job",
  "datasets",
  "start_training",
  "list_training_runs",
  "export_model",
  "list_media",
  "voices",
  "cancel",
];

function lookup(root: unknown, key: string): unknown {
  return key
    .split(".")
    .reduce<unknown>(
      (node, part) => (node as Record<string, unknown> | undefined)?.[part],
      root,
    );
}

function leaves(node: unknown, prefix = ""): [string, string][] {
  return Object.entries(node as Record<string, unknown>).flatMap(
    ([key, value]) =>
      typeof value === "string"
        ? [[`${prefix}${key}`, value] as [string, string]]
        : leaves(value, `${prefix}${key}.`),
  );
}

test("the MCP tab is registered after Agents, lazy, owner-only and in the palette", () => {
  assert.equal(
    SETTINGS_TABS.indexOf("mcp"),
    SETTINGS_TABS.indexOf("agents") + 1,
  );
  assert.ok(OWNER_ONLY_SETTINGS_TABS.has("mcp"));
  assert.match(
    DIALOG,
    /mcp: \(\) => import\("\.\/tabs\/mcp-tab"\)\.then\(\(m\) => \(\{ default: m\.McpTab \}\)\)/,
  );
  assert.match(
    DIALOG,
    /id: "agents",[\s\S]*?\},\s*\{\s*id: "mcp",\s*labelKey: "settings\.tabs\.mcp",\s*icon: McpServerIcon,\s*badgeKey: "common\.new",/,
  );
  assert.match(PALETTE, /mcp: "settings\.tabs\.mcp",/);
  assert.equal(en.settings.tabs.mcp, "MCP");
});

test("the page names Unsloth, keeps the access card and links to API keys", () => {
  assert.equal(en.settings.mcp.title, "Unsloth MCP");
  assert.match(TAB, /data-settings-label=\{t\("settings\.mcp\.title"\)\}/);
  assert.match(TAB, /className="text-xl font-semibold font-heading"/);
  const access = TAB.indexOf("<McpAccessSection");
  const tools = TAB.indexOf("<McpToolsSection");
  assert.ok(access >= 0 && tools >= 0 && access < tools);
  assert.match(TAB, /setActiveTab\("api-keys"\)/);
  assert.match(TAB, /t\("settings\.mcp\.createKey"\)/);
});

test("every catalog tool has a short label, and an unknown one falls back to its name", () => {
  assert.deepEqual(Object.keys(MCP_TOOL_LABEL_KEYS), CATALOG);
  for (const name of CATALOG) {
    const key = mcpToolLabelKey(name);
    assert.ok(key, name);
    assert.equal(typeof lookup(en, key), "string", key);
  }
  assert.deepEqual(
    Object.keys(en.settings.mcp.toolLabels).length,
    CATALOG.length,
  );
  assert.equal(mcpToolLabelKey("brand_new_tool"), null);
  assert.equal(mcpToolLabelKey("constructor"), null);
  assert.match(TOOLS, /const label = labelKey \? t\(labelKey\) : tool\.name;/);
  assert.match(
    TOOLS,
    /\{labelKey \? \(\s*<span className="text-ui-11 text-muted-foreground font-mono/,
  );
});

test("tool rows reuse the chat menu row, are named by their label and save one at a time", () => {
  assert.match(TOOLS, /import \{ Switch \} from "@\/components\/ui\/switch";/);
  assert.match(
    TOOLS,
    /className="flex min-h-9 items-center gap-2\.5 rounded-lg px-2 py-1"/,
  );
  assert.match(
    TOOLS,
    /className="mt-3 flex flex-col rounded-xl border border-border\/70 p-1\.5"/,
  );
  // One flat list like the chat menu, each tool with its own icon.
  assert.doesNotMatch(TOOLS, /border-t/);
  for (const name of CATALOG) {
    assert.match(TOOLS, new RegExp(`\\n  ${name}: \\w+Icon,`), name);
  }
  assert.match(TOOLS, /<Switch\s+className="ml-auto"\s+aria-label=\{label\}/);
  assert.match(TOOLS, /disabled=\{saving\}/);
  assert.doesNotMatch(TOOLS, /forcedByEnv/);
  assert.doesNotMatch(TOOLS, /truncate|(?<![\w-])h-9\b/);
  assert.doesNotMatch(TOOLS, /DragDrop|Reorder/);
});

test("a rejected tool change resyncs and shows the error inline, and an older backend hides the list", () => {
  const toggle = TOOLS.slice(
    TOOLS.indexOf("const toggle = async"),
    TOOLS.indexOf("if (!(settings || error))"),
  );
  assert.ok(
    toggle.indexOf("updateMcpTool(name, on)") <
      toggle.lastIndexOf("loadMcpAccess()"),
  );
  // One tool's state, merged on the server: never a full list built from this tab's copy.
  assert.doesNotMatch(toggle, /updateMcpTools|\.filter\(/);
  assert.match(toggle, /t\("settings\.mcp\.tools\.saveError"\)/);
  assert.match(TOOLS, /translate\("settings\.mcp\.tools\.loadError"\)/);
  assert.match(TOOLS, /text-xs leading-snug text-destructive/);
  assert.match(TOOLS, /settings\.tools === null\) \{\s*return null;/);
  assert.doesNotMatch(TOOLS, /toast/);
});

test("the Agents row shows the status as text and opens the MCP tab", () => {
  assert.match(ROW, /t\("settings\.mcp\.status\.on"\)/);
  assert.match(ROW, /t\("settings\.mcp\.status\.off"\)/);
  assert.match(ROW, /<Button\s+variant="outline"\s+size="sm"/);
  assert.match(ROW, /setActiveTab\("mcp"\)/);
  assert.match(ROW, /t\("settings\.mcp\.agentsRow\.open"\)/);
  assert.match(
    AGENTS_TAB,
    // Last on the page, after the dry-run section.
    /<CommandBlock command=\{shellCommands\.dryRun\} \/>\s*<\/div>\s*<\/SettingsSection>\s*<UnslothStudioMcpRow \/>\s*<\/div>\s*\);\s*\}\s*$/,
  );
  assert.doesNotMatch(AGENTS_TAB, /tabs\/mcp-tab/);
});

test("no emerald and no unscaled lengths", () => {
  for (const source of [TAB, TOOLS, ROW]) {
    assert.doesNotMatch(source, /emerald/);
    assert.doesNotMatch(source, /-\[\d+(?:\.\d+)?(?:px|rem)\]/);
  }
});

test("the English copy says Unsloth, never Studio", () => {
  for (const [key, value] of leaves(en.settings.mcp)) {
    assert.doesNotMatch(value, /Studio/, key);
  }
});

test("every locale has the same settings.mcp keys, with plain copy", () => {
  const keys = leaves(en.settings.mcp)
    .map(([key]) => key)
    .sort();
  for (const [name, locale] of Object.entries(LOCALES)) {
    const copy = lookup(locale, "settings.mcp");
    assert.ok(copy, name);
    const entries = leaves(copy);
    assert.deepEqual(entries.map(([key]) => key).sort(), keys, name);
    for (const [key, value] of entries) {
      assert.doesNotMatch(value, /[;—]/, `${name}: ${key}`);
    }
  }
});
