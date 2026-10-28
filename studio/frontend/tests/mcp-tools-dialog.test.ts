// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// source assertions avoid the auth barrel, which cannot load in this test harness.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MCP_TOOL_LABEL_KEYS,
  groupMcpTools,
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

// Each catalog tool's backend area (its module's short name).
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
  // Empty groups are left out.
  assert.deepEqual(
    groupMcpTools([{ name: "embed", area: "text" }]).map((group) => group.id),
    ["generate"],
  );
  for (const group of groups) {
    assert.equal(typeof lookup(en, group.labelKey), "string", group.labelKey);
  }
});

test("a group switch is on only when all its tools are, and every change sends the full off list", () => {
  assert.match(TOOLS, /checked=\{on === group\.tools\.length\}/);
  // Only the changed tools go to the server, which merges them into its stored list.
  assert.match(
    TOOLS,
    /onSave\(Object\.fromEntries\(names\.map\(\(name\) => \[name, on\]\)\)\)/,
  );
  assert.match(TOOLS, /t\("settings\.mcp\.tools\.groupToggle", \{\s*group: label,/);
  // "Turn all on" is disabled until something is off, and turns on only the tools that are off.
  assert.match(
    TOOLS,
    /<Button\s+variant="outline"\s+disabled=\{saving \|\| !anyOff\}\s+onClick=\{\(\) =>\s+setEnabled\(\s+tools\.filter\(\(tool\) => !tool\.enabled\)\.map\(\(tool\) => tool\.name\),\s+true,/,
  );
  assert.match(TOOLS, /disabled=\{saving\}/);
});

test("tool rows are one line, reuse the chat menu row, toggle from the name and each has its own icon", () => {
  assert.match(
    TOOLS,
    /<label\s+data-mcp-tool=\{tool\.name\}\s+className="flex min-h-9 cursor-pointer items-center gap-2\.5 rounded-lg px-2 py-1 has-disabled:cursor-default"/,
  );
  assert.match(TOOLS, /<Switch\s+size="sm"\s+aria-label=\{label\}/);
  for (const name of CATALOG) {
    assert.match(TOOLS, new RegExp(`\\n  ${name}: \\w+Icon,`), name);
  }
  assert.doesNotMatch(TOOLS, /forcedByEnv|toast/);
});

test("a rejected tool change resyncs and shows the error in the dialog", () => {
  const save = SECTION.slice(
    SECTION.indexOf("const saveTools = async"),
    SECTION.indexOf("// subscribe to same-tab tunnel changes"),
  );
  assert.ok(
    save.indexOf("updateMcpTools(states)") <
      save.lastIndexOf("loadMcpAccess()"),
  );
  // Changed tools only, merged on the server: never a full list built from this copy.
  assert.doesNotMatch(SECTION, /disabled_tools|disabledTools/);
  assert.doesNotMatch(TOOLS, /\.filter\(\(tool\) => \(names/);
  assert.match(save, /t\("settings\.mcp\.tools\.saveError"\)/);
  assert.match(TOOLS, /text-xs leading-snug text-destructive/);
});

test("the API tab row matches the keyless link beside it, shows the status as text and jumps to the card on Agents", () => {
  assert.match(ROW, /t\("settings\.mcp\.status\.on"\)/);
  assert.match(ROW, /t\("settings\.mcp\.status\.off"\)/);
  assert.match(ROW, /label=\{t\("settings\.apiKeys\.mcp\.title"\)\}/);
  assert.doesNotMatch(ROW, /SettingsSection/);
  assert.match(
    ROW,
    /className="inline-flex items-center gap-1 rounded-full px-1\.5 py-0\.5 text-ui-11 font-medium text-muted-foreground/,
  );
  assert.match(ROW, /openDialog\("agents", \{ scrollTarget: "agents-mcp" \}\)/);
  assert.match(STORE, /"agents-mcp": "agents",/);
  assert.match(SECTION, /scrollTarget !== "agents-mcp" \|\| !loaded/);
  assert.match(SECTION, /consumeScrollTarget\("agents-mcp"\)/);
  assert.equal(en.settings.mcp.apiKeysRow.open, "Open in Agents");
});

test("tools a chat never gets are named once, in a muted note under the groups", () => {
  assert.equal(en.settings.mcp.tools.chatNote, "The chat never gets {tools}.");
  assert.match(
    SECTION,
    /\.filter\(\(tool\) => tool\.chat === false\)[\s\S]*?t\("settings\.mcp\.tools\.chatNote", \{/,
  );
  assert.match(SECTION, /new Intl\.ListFormat\(locale, \{/);
  assert.match(SECTION, /footnote=\{chatNote\}/);
  assert.match(
    TOOLS,
    /\{footnote \? \(\s*<p className="px-3 text-xs leading-snug text-muted-foreground">/,
  );
  assert.doesNotMatch(TOOLS, /notInChat/);
});

test("no emerald and no unscaled lengths", () => {
  for (const source of [TOOLS, ROW]) {
    assert.doesNotMatch(source, /emerald/);
    assert.doesNotMatch(source, /-\[\d+(?:\.\d+)?(?:px|rem)\]/);
  }
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
  // The raw name moves behind a hover; an unknown tool shows it as the label.
  assert.match(
    TOOLS,
    /<TooltipContent className="font-mono text-ui-11">\s*\{tool\.name\}/,
  );
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
