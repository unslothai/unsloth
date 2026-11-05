// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THEME_FILE_KEYS = [
  "settings.appearance.themeFile.title",
  "settings.appearance.themeFile.importLabel",
  "settings.appearance.themeFile.exportLabel",
];

test("the Appearance tab shows the theme file section between Code and Rendering", () => {
  const tab = readSrc("features/settings/tabs/appearance-tab.tsx");
  assert.match(tab, /import \{ ThemeFileSection \} from "\.\.\/appearance-theme\/ui\/theme-file-section";/);
  const section = tab.indexOf("<ThemeFileSection />");
  assert.ok(section !== -1, "ThemeFileSection is not rendered");
  assert.ok(section > tab.indexOf('t("settings.appearance.code.title")'), "ThemeFileSection should follow Code");
  assert.ok(section < tab.indexOf("<GpuRenderingRow />"), "ThemeFileSection should precede Rendering");
});

test("settings search finds the theme file rows", () => {
  const search = readSrc("features/settings/settings-search.ts");
  const appearance = /appearance: \[([\s\S]*?)\]/.exec(search);
  assert.ok(appearance, "no appearance entry in SETTINGS_SEARCH_INDEX");
  for (const key of THEME_FILE_KEYS) {
    assert.ok(appearance[1].includes(`"${key}"`), `${key} is not searchable`);
  }
});

test("theme file code never writes raw HTML", () => {
  const root = new URL("../src/features/settings/appearance-theme/", import.meta.url);
  const files = readdirSync(root, { recursive: true, encoding: "utf8" }).filter((name) =>
    /\.tsx?$/.test(name),
  );
  assert.ok(files.length > 0);
  for (const name of files) {
    const source = readFileSync(new URL(name, root), "utf8");
    assert.doesNotMatch(source, /dangerouslySetInnerHTML|innerHTML/, name);
  }
});
