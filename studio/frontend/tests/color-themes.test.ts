// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  COLOR_THEMES,
  COLOR_THEME_IDS,
  FLAVOR_THEME_IDS,
  isColorThemeId,
} from "../src/features/settings/lib/color-themes.ts";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const indexCss = read("../src/index.css");
const themeBoot = read("../public/theme-boot.js");
const backendSettings = read("../../backend/routes/settings.py");

test("every non-standard theme is applied by the boot script", () => {
  for (const id of COLOR_THEME_IDS) {
    if (id === "standard") continue;
    assert.match(themeBoot, new RegExp(`"${id}"`), id);
  }
});

test("the backend accepts every theme id", () => {
  const literal = backendSettings.match(/palette: Literal\[([\s\S]*?)\]/);
  assert.ok(literal);
  const accepted = [...literal[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...accepted].sort(), [...COLOR_THEME_IDS].sort());
});

test("every flavor theme has light and dark seed blocks", () => {
  for (const id of FLAVOR_THEME_IDS) {
    assert.ok(
      indexCss.includes(`:root[data-palette="${id}"]:not(.dark) {`),
      `${id} light`,
    );
    assert.ok(
      indexCss.includes(`:root[data-palette="${id}"].dark {`),
      `${id} dark`,
    );
    assert.ok(
      indexCss.includes(`[data-palette="${id}"],`) ||
        indexCss.includes(`[data-palette="${id}"])`),
      `${id} mapping`,
    );
  }
});

test("every theme id has registry colors", () => {
  assert.deepEqual(
    Object.keys(COLOR_THEMES).sort(),
    [...COLOR_THEME_IDS].sort(),
  );
});

test("isColorThemeId rejects unknown values", () => {
  assert.equal(isColorThemeId("matcha"), true);
  assert.equal(isColorThemeId("dracula"), false);
  assert.equal(isColorThemeId(null), false);
});
