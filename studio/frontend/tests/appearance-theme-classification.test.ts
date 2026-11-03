// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  CUSTOMIZATION_FIELD_CATEGORY,
  FILE_PATH_FOR_FIELD,
  LAYOUT_FIELDS,
  LOOK_FIELDS,
  mapIssuePath,
  themeLayoutFromCustomization,
  themeLookFromCustomization,
} from "../src/features/settings/appearance-theme/domain/classification.ts";
import { DEFAULT_CUSTOMIZATION } from "../src/features/settings/lib/appearance-customization.ts";

function fieldsWith(category: string): string[] {
  return Object.entries(CUSTOMIZATION_FIELD_CATEGORY)
    .filter(([, value]) => value === category)
    .map(([key]) => key);
}

test("every customization field has a category and no category names an unknown field", () => {
  assert.deepEqual(
    Object.keys(CUSTOMIZATION_FIELD_CATEGORY).sort(),
    Object.keys(DEFAULT_CUSTOMIZATION).sort(),
  );
});

test("personal fields are exactly pointer cursors and reduce motion", () => {
  assert.deepEqual(fieldsWith("personal"), ["pointerCursors", "reduceMotion"]);
});

test("look and layout lists are the classified fields, in map order", () => {
  assert.deepEqual([...LOOK_FIELDS].sort(), [
    "chatFont",
    "chatWidth",
    "codeFont",
    "codeFontSize",
    "codeLigatures",
    "codeLineNumbers",
    "codeWordWrap",
    "colors",
    "composerOutline",
    "contrast",
    "fontSmoothing",
    "headingFont",
    "uiFont",
    "uiFontSize",
  ]);
  assert.deepEqual([...LAYOUT_FIELDS].sort(), [
    "sentAttachments",
    "sidebarMenu",
    "sidebarNav",
    "sidebarNavAuto",
  ]);
});

test("the categories partition every customization field", () => {
  const all = [
    ...LOOK_FIELDS,
    ...LAYOUT_FIELDS,
    ...fieldsWith("asset"),
    ...fieldsWith("personal"),
  ].sort();
  assert.deepEqual(all, Object.keys(DEFAULT_CUSTOMIZATION).sort());
  assert.deepEqual(fieldsWith("asset"), ["importedFonts"]);
});

test("every look, layout and asset field has a file path under a known section", () => {
  const expected = [...LOOK_FIELDS, ...LAYOUT_FIELDS, "importedFonts"].sort();
  assert.deepEqual(Object.keys(FILE_PATH_FOR_FIELD).sort(), expected);

  const sections = ["colors", "typography", "code", "chat", "layout", "assets"];
  for (const path of Object.values(FILE_PATH_FOR_FIELD)) {
    const section = path.split(".")[0];
    assert.ok(
      sections.includes(section),
      `${path} is not under a known section`,
    );
  }
  assert.equal(FILE_PATH_FOR_FIELD.codeLineNumbers, "code.lineNumbers");
  assert.equal(FILE_PATH_FOR_FIELD.chatWidth, "chat.width");
  assert.equal(FILE_PATH_FOR_FIELD.importedFonts, "assets.fonts");
});

test("sanitizer issue paths map to their file paths", () => {
  assert.equal(mapIssuePath("colors.light.accent"), "colors.light.accent");
  assert.equal(mapIssuePath("contrast"), "colors.contrast");
  assert.equal(mapIssuePath("uiFontSize"), "typography.uiFontSize");
  assert.equal(mapIssuePath("chatWidth"), "chat.width");
  assert.equal(mapIssuePath("importedFonts[1]"), "assets.fonts[1]");
  assert.equal(mapIssuePath("importedFonts[0].name"), "assets.fonts[0].name");
  assert.equal(
    mapIssuePath("sidebarNav[2].pinned"),
    "layout.sidebarNav[2].pinned",
  );
  assert.equal(mapIssuePath("sidebarNavAuto[0]"), "layout.sidebarNavAuto[0]");
  assert.equal(mapIssuePath("pointerCursors"), "pointerCursors");
});

test("the look of the defaults holds exactly the look fields", () => {
  const look = themeLookFromCustomization(DEFAULT_CUSTOMIZATION);
  assert.deepEqual(Object.keys(look).sort(), [...LOOK_FIELDS].sort());
  assert.equal(look.contrast, DEFAULT_CUSTOMIZATION.contrast);
  assert.deepEqual(look.colors, DEFAULT_CUSTOMIZATION.colors);
});

test("the layout of the defaults holds exactly the layout fields", () => {
  const layout = themeLayoutFromCustomization(DEFAULT_CUSTOMIZATION);
  assert.deepEqual(Object.keys(layout).sort(), [...LAYOUT_FIELDS].sort());
  assert.deepEqual(layout.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});

test("personal and asset values never leak into the look or layout", () => {
  const changed = {
    ...DEFAULT_CUSTOMIZATION,
    pointerCursors: true,
    reduceMotion: "on" as const,
    importedFonts: [{ name: "Brand", dataUrl: "data:font/woff2;base64,AAAA" }],
    contrast: 80,
  };
  const look = themeLookFromCustomization(changed);
  const layout = themeLayoutFromCustomization(changed);

  assert.equal(look.contrast, 80);
  assert.equal(Object.hasOwn(look, "pointerCursors"), false);
  assert.equal(Object.hasOwn(look, "reduceMotion"), false);
  assert.equal(Object.hasOwn(look, "importedFonts"), false);
  assert.equal(Object.hasOwn(layout, "pointerCursors"), false);
  assert.equal(Object.hasOwn(layout, "contrast"), false);
});
