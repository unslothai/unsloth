// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// source assertions avoid the auth barrel, which cannot load in this test harness.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MCP_TOOL_DESCRIPTION_KEYS,
  MCP_TOOL_LABEL_KEYS,
  groupMcpTools,
  mcpToolDescriptionKey,
  mcpToolLabelKey,
} from "../src/features/settings/components/mcp-tool-labels.ts";
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

const SECTION = readSrc("features/settings/components/mcp-access-section.tsx");
const TOOLS = readSrc("features/settings/components/mcp-tools-dialog.tsx");
const ROW = readSrc("features/settings/components/unsloth-studio-mcp-row.tsx");
const DIALOG = readSrc("features/settings/settings-dialog.tsx");
const STORE = readSrc("features/settings/stores/settings-dialog-store.ts");

const AREAS: Record<string, string> = {
  studio_status: "status",
  list_models: "models",
  load_model: "models",
  unload_model: "models",
  plan_load: "models",
  download_model: "models",
  chat: "text",
  embed: "text",
  system_one: "text",
  generate_image: "images",
  generate_audio: "audio",
  transcribe: "audio",
  analyze_audio: "audio",
  generate_video: "video",
  run_recipe: "jobs",
  get_job: "jobs",
  datasets: "data",
  start_training: "training",
  list_training_runs: "training",
  export_model: "export",
  list_media: "library",
  voices: "library",
  cancel: "cancel",
};

const LOCALES = { ar, de, en, es, fr, he, hi, it, ja, ko, ptBR, ru, sv, zhCN };

// Same list and order as the backend's tool_catalog().
const CATALOG = Object.keys(AREAS);

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

test("there is no MCP tab; the settings live on Agents", () => {
  assert.ok(!(SETTINGS_TABS as readonly string[]).includes("mcp"));
  assert.doesNotMatch(DIALOG, /tabs\/mcp-tab|McpServerIcon/);
  assert.equal(lookup(en, "settings.tabs.mcp"), undefined);
});

test("the card shows a tools count and opens the dialog, and an older backend hides both", () => {
  assert.match(SECTION, /\{settings\.tools \? \(\s*<SettingsRow\s+label=\{t\("settings\.mcp\.tools\.title"\)\}/);
  assert.match(SECTION, /t\("settings\.mcp\.tools\.count", \{/);
  assert.match(SECTION, /t\("settings\.mcp\.tools\.choose"\)/);
  assert.match(SECTION, /\{settings\?\.tools \? \(\s*<McpToolsDialog/);
  assert.equal(en.settings.mcp.tools.count, "{on} of {total} on");
});

test("tools fall into the five groups in order, with unknown areas last", () => {
  const tools = Object.entries(AREAS).map(([name, area]) => ({ name, area }));
  const groups = groupMcpTools([...tools, { name: "brand_new", area: "future" }]);
  assert.deepEqual(
    groups.map((group) => [group.id, group.tools.map((tool) => tool.name)]),
    [
      ["models", ["studio_status", "list_models", "load_model", "unload_model", "plan_load", "download_model"]],
      ["generate", ["chat", "embed", "system_one", "generate_image", "generate_audio", "transcribe", "analyze_audio", "generate_video"]],
      ["training", ["start_training", "list_training_runs", "export_model"]],
      ["library", ["datasets", "list_media", "voices"]],
      ["jobs", ["run_recipe", "get_job", "cancel"]],
      ["other", ["brand_new"]],
    ],
  );
  assert.deepEqual(
    groupMcpTools([{ name: "embed", area: "text" }]).map((group) => group.id),
    ["generate"],
  );
  for (const group of groups) {
    assert.equal(typeof lookup(en, group.labelKey), "string", group.labelKey);
  }
});

test("each group and every tool get one row that says what a click does, and every change sends only the changed tools", () => {
  assert.doesNotMatch(TOOLS, /<Switch|groupTurnO|"mixed"/);
  assert.match(TOOLS, /const allOn = on === tools\.length;/);
  assert.match(
    TOOLS,
    /allOn\s+\? "settings\.mcp\.tools\.selectNone"\s+: "settings\.mcp\.tools\.selectAll"/,
  );
  assert.match(
    TOOLS,
    /tools\s+\.filter\(\(tool\) => tool\.enabled === allOn\)\s+\.map\(\(tool\) => tool\.name\),\s+!allOn,/,
  );
  assert.match(TOOLS, /label=\{t\("settings\.mcp\.tools\.allTools"\)\}\s+prominent=\{true\}/);
  assert.match(TOOLS, /t\("settings\.mcp\.tools\.count", \{ on, total: tools\.length \}\)/);
  assert.equal(en.settings.mcp.tools.selectAll, "Select all");
  assert.equal(en.settings.mcp.tools.selectNone, "Select none");
  assert.match(
    TOOLS,
    /onSave\(Object\.fromEntries\(names\.map\(\(name\) => \[name, on\]\)\)\)/,
  );
  assert.match(TOOLS, /moreBelow \? "border-border" : "border-transparent"/);
});

test("each tool is a checkbox pill with its own icon", () => {
  assert.match(
    TOOLS,
    /<button\s+type="button"\s+role="checkbox"\s+aria-checked=\{on\}\s+data-mcp-tool=\{tool\.name\}/,
  );
  assert.match(TOOLS, /onClick=\{\(\) => onToggle\(!on\)\}/);
  for (const name of CATALOG) {
    assert.match(TOOLS, new RegExp(`\\n  ${name}: \\w+Icon,`), name);
  }
});

test("saves flip at once, a late reply never undoes a newer change, and a rejected one resyncs", () => {
  const save = SECTION.slice(
    SECTION.indexOf("const saveTools = async"),
    SECTION.indexOf("// subscribe to same-tab tunnel changes"),
  );
  const apply = SECTION.slice(SECTION.indexOf("const apply = async"), SECTION.indexOf("// Settings > API links here"));
  assert.ok(save.indexOf("setSettings((current)") < save.indexOf("updateMcpTools(states)"));
  assert.ok(save.indexOf("updateMcpTools(states)") < save.lastIndexOf("loadMcpAccess()"));
  // The switch and the tools share one request counter, so neither full snapshot undoes the other.
  for (const body of [save, apply]) {
    assert.match(body, /const request = \+\+saveRequest\.current;/);
    assert.match(body, /if \(request === saveRequest\.current\) \{\s+setSettings\(next\);/);
  }
  assert.doesNotMatch(apply, /setSettings\(await/);
  // The API client queues every request (mcp-access-api.test.ts), so no second queue here.
  assert.doesNotMatch(SECTION, /enqueueSave|saveQueue|disabled_tools|disabledTools/);
  assert.match(save, /t\("settings\.mcp\.tools\.saveError"\)/);
});

test("the API tab row matches the keyless link beside it, shows the status as text and jumps to the card on Agents", () => {
  assert.match(ROW, /t\("settings\.mcp\.status\.on"\)/);
  assert.match(ROW, /t\("settings\.mcp\.status\.off"\)/);
  assert.match(ROW, /label=\{t\("settings\.apiKeys\.mcp\.title"\)\}/);
  assert.match(ROW, /openDialog\("agents", \{ scrollTarget: "agents-mcp" \}\)/);
  assert.match(STORE, /"agents-mcp": "agents",/);
  assert.match(SECTION, /scrollTarget !== "agents-mcp" \|\| !loaded/);
  assert.match(SECTION, /consumeScrollTarget\("agents-mcp"\)/);
  assert.equal(en.settings.mcp.apiKeysRow.open, "Open in Agents");
});

test("tools a chat never gets are named once, in a note under the groups", () => {
  assert.match(
    SECTION,
    /\.filter\(\(tool\) => tool\.chat === false\)[\s\S]*?t\("settings\.mcp\.tools\.chatNote", \{/,
  );
  assert.match(SECTION, /footnote=\{chatNote\}/);
  assert.match(TOOLS, /\{footnote \? \(\s*<p /);
  assert.doesNotMatch(TOOLS, /notInChat/);
});

test("every catalog tool has a label and a description, and an unknown one falls back to its name", () => {
  for (const [keys, keyOf] of [
    [MCP_TOOL_LABEL_KEYS, mcpToolLabelKey],
    [MCP_TOOL_DESCRIPTION_KEYS, mcpToolDescriptionKey],
  ] as const) {
    assert.deepEqual(Object.keys(keys), CATALOG);
    for (const name of CATALOG) {
      assert.equal(typeof lookup(en, keyOf(name) ?? ""), "string", name);
    }
    assert.equal(keyOf("brand_new_tool"), null);
    assert.equal(keyOf("constructor"), null);
  }
  assert.match(TOOLS, /const label = labelKey \? t\(labelKey\) : tool\.name;/);
});

test("every locale has the same settings.mcp keys with plain copy, and English says Unsloth, not Studio", () => {
  for (const [key, value] of leaves(en.settings.mcp)) {
    assert.doesNotMatch(value, /Studio/, key);
  }
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
