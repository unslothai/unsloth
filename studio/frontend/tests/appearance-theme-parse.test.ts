// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  LOOK_FIELDS,
  themeLookFromCustomization,
} from "../src/features/settings/appearance-theme/domain/classification.ts";
import { parseThemeFile } from "../src/features/settings/appearance-theme/domain/parse.ts";
import {
  DEFAULT_THEME_GLASS,
  MAX_BACKGROUND_DATA_URL_CHARS,
  MAX_THEME_FILE_CHARS,
  MAX_THEME_FILE_CHARS_NO_ASSETS,
  THEME_FILE_KIND,
  type ThemeBackground,
  type ThemeFileV1,
} from "../src/features/settings/appearance-theme/domain/schema.ts";
import {
  DEFAULT_CUSTOMIZATION,
  type ImportedFont,
  SIDEBAR_MENU_DEFAULT_VISIBLE,
  SIDEBAR_MENU_ITEM_IDS,
  SIDEBAR_NAV_DEFAULT_PINNED,
  SIDEBAR_NAV_ITEM_IDS,
} from "../src/features/settings/lib/appearance-customization.ts";

const FULL_LAYOUT: NonNullable<ThemeFileV1["layout"]> = {
  sentAttachments: "chips",
  sidebarMenu: [...SIDEBAR_MENU_ITEM_IDS].reverse().map((id) => ({
    id,
    visible: !SIDEBAR_MENU_DEFAULT_VISIBLE[id],
  })),
  sidebarNav: [...SIDEBAR_NAV_ITEM_IDS].reverse().map((id) => ({
    id,
    pinned: !SIDEBAR_NAV_DEFAULT_PINNED[id],
  })),
  sidebarNavAuto: [],
};

const FULL_FONTS: ImportedFont[] = [
  { name: "Brand Sans", dataUrl: "data:font/woff2;base64,d09GMgABAAAA" },
];

const FULL_BACKGROUND: ThemeBackground = {
  dataUrl: "data:image/png;base64,iVBORw0KGgo=",
  adjustments: {
    scope: "all",
    effect: "Dither",
    visibility: 40,
    conversation: 20,
    blur: 5,
    fade: false,
    outline: 10,
    sizing: "contain",
    zoom: 150,
    position: "0% 100%",
    glassLightVisibility: 50,
    glassDarkVisibility: 30,
    glassFade: true,
  },
};

/** A valid v1 file in which every field differs from its default. */
function fullTheme(): ThemeFileV1 {
  return {
    $schema: "unsloth-theme.schema.json",
    kind: THEME_FILE_KIND,
    schemaVersion: 1,
    meta: {
      name: "Harbor Dusk",
      author: "Ada Lovelace",
      description: "Warm accents for late sessions",
      studioVersion: "2026.11.1",
    },
    base: { palette: "matcha" },
    colors: {
      light: {
        accent: "#2f6fd6",
        background: "#fbfaf7",
        foreground: "#1a1c20",
      },
      dark: { accent: "#ff7a59", background: "#101318", foreground: "#e8eaef" },
      contrast: 72,
    },
    typography: {
      uiFont: "Brand Sans",
      headingFont: "Brand Serif",
      chatFont: "Chat Mono",
      codeFont: "Code Mono",
      uiFontSize: 16,
      codeFontSize: 13,
      fontSmoothing: false,
    },
    code: { lineNumbers: true, wordWrap: true, ligatures: false },
    chat: { width: "wide", composerOutline: 35 },
    glass: {
      material: "frosted",
      strength: { light: 60, dark: 25 },
      lens: "magnify",
      menus: false,
    },
    layout: FULL_LAYOUT,
    assets: { fonts: FULL_FONTS, background: FULL_BACKGROUND },
  };
}

function parseOk(text: string) {
  const result = parseThemeFile(text);
  if (!result.ok) throw new Error(`expected ok, got ${result.error}`);
  return result;
}

test("a full v1 file parses back to every field it wrote, with no warnings", () => {
  const source = fullTheme();
  const { theme, warnings } = parseOk(JSON.stringify(source));

  assert.deepEqual(warnings, []);
  assert.deepEqual(theme.meta, source.meta);
  assert.equal(theme.palette, "matcha");

  assert.deepEqual(theme.look.colors, {
    light: source.colors.light,
    dark: source.colors.dark,
  });
  assert.equal(theme.look.contrast, 72);
  assert.equal(theme.look.uiFont, "Brand Sans");
  assert.equal(theme.look.headingFont, "Brand Serif");
  assert.equal(theme.look.chatFont, "Chat Mono");
  assert.equal(theme.look.codeFont, "Code Mono");
  assert.equal(theme.look.uiFontSize, 16);
  assert.equal(theme.look.codeFontSize, 13);
  assert.equal(theme.look.fontSmoothing, false);
  assert.equal(theme.look.codeLineNumbers, true);
  assert.equal(theme.look.codeWordWrap, true);
  assert.equal(theme.look.codeLigatures, false);
  assert.equal(theme.look.chatWidth, "wide");
  assert.equal(theme.look.composerOutline, 35);
  assert.deepEqual(Object.keys(theme.look).sort(), [...LOOK_FIELDS].sort());

  assert.deepEqual(theme.glass, {
    material: "frosted",
    strength: { light: 60, dark: 25 },
    lens: "magnify",
    menus: false,
  });

  assert.deepEqual(theme.layout, {
    sentAttachments: FULL_LAYOUT.sentAttachments,
    sidebarMenu: FULL_LAYOUT.sidebarMenu,
    sidebarNav: FULL_LAYOUT.sidebarNav,
    sidebarNavAuto: [],
  });
  assert.deepEqual(theme.fonts, FULL_FONTS);
  assert.deepEqual(theme.background, FULL_BACKGROUND);
});

test("missing sections fall back to the defaults without warnings", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1 }),
  );

  assert.deepEqual(warnings, []);
  assert.deepEqual(theme.meta, { name: "Untitled theme" });
  assert.equal(theme.palette, "standard");
  assert.deepEqual(
    theme.look,
    themeLookFromCustomization(DEFAULT_CUSTOMIZATION),
  );
  assert.deepEqual(theme.glass, DEFAULT_THEME_GLASS);
  assert.equal(theme.layout, null);
  assert.equal(theme.fonts, null);
  assert.equal(theme.background, null);
});

test("a null section counts as absent and is not reported", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      typography: null,
      layout: null,
      assets: null,
    }),
  );

  assert.deepEqual(warnings, []);
  assert.equal(theme.layout, null);
  assert.equal(theme.fonts, null);
  assert.equal(theme.background, null);
});

test("an empty layout section is present and resolves to the shipped layout", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1, layout: {} }),
  );

  assert.deepEqual(warnings, []);
  assert.deepEqual(theme.layout, {
    sentAttachments: DEFAULT_CUSTOMIZATION.sentAttachments,
    sidebarMenu: DEFAULT_CUSTOMIZATION.sidebarMenu,
    sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav,
    sidebarNavAuto: ["projects"],
  });
});

test("a customized sidebar nav without sidebarNavAuto gets no automatic rows", () => {
  const { theme } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      layout: { sidebarNav: [{ id: "hub", pinned: false }] },
    }),
  );

  assert.deepEqual(theme.layout?.sidebarNavAuto, []);
});

test("unknown keys are reported at their file path", () => {
  const text = JSON.stringify({
    kind: THEME_FILE_KIND,
    schemaVersion: 1,
    extra: 1,
    typography: { uiFont: "Brand", bogus: true },
    colors: { light: { accent: "#112233", surface: "#ffffff" } },
    code: { wordWrap: true, wrap: 1 },
  });
  const { theme, warnings } = parseOk(text);

  assert.deepEqual(warnings, [
    { kind: "unknown", path: "extra" },
    { kind: "unknown", path: "colors.light.surface" },
    { kind: "unknown", path: "typography.bogus" },
    { kind: "unknown", path: "code.wrap" },
  ]);
  assert.equal(theme.look.uiFont, "Brand");
  assert.equal(theme.look.colors.light.accent, "#112233");
  assert.equal(theme.look.codeWordWrap, true);
});

test("__proto__ keys stay plain data and never touch Object.prototype", () => {
  const text = `{"kind":"${THEME_FILE_KIND}","schemaVersion":1,"__proto__":{"polluted":1},"typography":{"__proto__":{"polluted":"yes"}},"colors":{"light":{"__proto__":{"accent":"#ff0000"}}}}`;
  const { theme, warnings } = parseOk(text);

  assert.equal(Object.hasOwn(Object.prototype, "polluted"), false);
  assert.equal(({} as Record<string, unknown>).polluted, undefined);
  assert.equal(theme.look.colors.light.accent, null);
  assert.deepEqual(warnings, [
    { kind: "unknown", path: "__proto__" },
    { kind: "unknown", path: "colors.light.__proto__" },
    { kind: "unknown", path: "typography.__proto__" },
  ]);
});

test("an invalid accent becomes null with an invalid warning", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      colors: { light: { accent: "url(x)" } },
    }),
  );

  assert.equal(theme.look.colors.light.accent, null);
  assert.deepEqual(warnings, [
    { kind: "invalid", path: "colors.light.accent" },
  ]);
});

test("a font name with CSS-breaking characters is sanitized and reported", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      typography: { uiFont: 'Evil"; x' },
    }),
  );

  assert.equal(theme.look.uiFont, "Evil x");
  assert.deepEqual(warnings, [{ kind: "invalid", path: "typography.uiFont" }]);
});

test("an out-of-range font size is clamped and reported at its file path", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      typography: { uiFontSize: 99 },
    }),
  );

  assert.equal(theme.look.uiFontSize, 20);
  assert.deepEqual(warnings, [
    { kind: "clamped", path: "typography.uiFontSize" },
  ]);
});

test("an unknown palette falls back to standard with a warning", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      base: { palette: "neon" },
    }),
  );

  assert.equal(theme.palette, "standard");
  assert.deepEqual(warnings, [{ kind: "unknownPalette", palette: "neon" }]);
});

test("a non-string palette is reported as invalid", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      base: { palette: 5 },
    }),
  );

  assert.equal(theme.palette, "standard");
  assert.deepEqual(warnings, [{ kind: "invalid", path: "base.palette" }]);
});

test("a newer schema version parses as current and says so first", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({ ...fullTheme(), schemaVersion: 2 }),
  );

  assert.deepEqual(warnings, [{ kind: "newerVersion" }]);
  assert.equal(theme.look.uiFont, "Brand Sans");
});

test("warnings list the version first, then reading order", () => {
  const { warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 2,
      extra: 1,
      base: { palette: "neon" },
      typography: { uiFontSize: 99 },
      colors: { light: { accent: "url(x)" } },
    }),
  );

  assert.deepEqual(warnings, [
    { kind: "newerVersion" },
    { kind: "unknown", path: "extra" },
    { kind: "unknownPalette", palette: "neon" },
    { kind: "invalid", path: "colors.light.accent" },
    { kind: "clamped", path: "typography.uiFontSize" },
  ]);
});

test("a file that is not a theme is rejected", () => {
  const notThemes = [
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 0 }),
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: "1" }),
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1.5 }),
    JSON.stringify({ schemaVersion: 1 }),
    JSON.stringify({ kind: "something-else", schemaVersion: 1 }),
    "[]",
    "null",
    "42",
  ];
  for (const text of notThemes) {
    assert.deepEqual(parseThemeFile(text), { ok: false, error: "notTheme" });
  }
});

test("text that is not JSON is rejected as invalid JSON", () => {
  assert.deepEqual(parseThemeFile("{"), { ok: false, error: "invalidJson" });
});

test("a file over 8,000,000 characters is too large before it is parsed", () => {
  assert.deepEqual(parseThemeFile(" ".repeat(MAX_THEME_FILE_CHARS + 1)), {
    ok: false,
    error: "tooLarge",
  });
});

test("a file without assets over the settings-only cap is too large", () => {
  // Trailing whitespace is valid JSON, so the file is only large, not malformed.
  const padded =
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1 }) +
    " ".repeat(300_000);
  assert.ok(padded.length > MAX_THEME_FILE_CHARS_NO_ASSETS);
  assert.deepEqual(parseThemeFile(padded), { ok: false, error: "tooLarge" });

  const withAssets =
    JSON.stringify({ kind: THEME_FILE_KIND, schemaVersion: 1, assets: {} }) +
    " ".repeat(300_000);
  assert.equal(parseThemeFile(withAssets).ok, true);
});

test("an svg background is skipped with a warning", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      assets: {
        background: { dataUrl: "data:image/svg+xml;base64,PHN2Zy8+" },
      },
    }),
  );

  assert.equal(theme.background, null);
  assert.deepEqual(warnings, [{ kind: "backgroundSkipped" }]);
});

test("a background data URL over its length cap is skipped", () => {
  const dataUrl = `data:image/png;base64,${"A".repeat(MAX_BACKGROUND_DATA_URL_CHARS)}`;
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      assets: { background: { dataUrl } },
    }),
  );

  assert.equal(theme.background, null);
  assert.deepEqual(warnings, [{ kind: "backgroundSkipped" }]);
});

test("background adjustments ignore unknown keys without a warning", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      assets: {
        background: {
          dataUrl: FULL_BACKGROUND.dataUrl,
          adjustments: { scope: "all", bogus: 1 },
        },
      },
    }),
  );

  assert.deepEqual(warnings, []);
  assert.equal(theme.background?.adjustments.scope, "all");
  assert.equal(
    Object.hasOwn(theme.background?.adjustments ?? {}, "bogus"),
    false,
  );
});

test("a font that is not a valid font data URL is dropped with its index", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      assets: {
        fonts: [
          { name: "Brand", dataUrl: "data:font/woff2;base64,AAAA" },
          { name: "Broken", dataUrl: "https://example.com/font.woff2" },
        ],
      },
    }),
  );

  assert.deepEqual(theme.fonts, [
    { name: "Brand", dataUrl: "data:font/woff2;base64,AAAA" },
  ]);
  assert.deepEqual(warnings, [{ kind: "dropped", path: "assets.fonts[1]" }]);
});

test("a section of the wrong type is reported and read as absent", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      typography: "bold",
    }),
  );

  assert.deepEqual(warnings, [{ kind: "invalid", path: "typography" }]);
  assert.equal(theme.look.uiFont, null);
});

test("glass values that are wrong are reported and replaced", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      glass: {
        material: "hologram",
        strength: { light: 150, dark: "x" },
        lens: "spin",
        menus: "yes",
      },
    }),
  );

  assert.deepEqual(theme.glass, {
    material: "solid",
    strength: { light: 100, dark: DEFAULT_THEME_GLASS.strength.dark },
    lens: "off",
    menus: true,
  });
  assert.deepEqual(warnings, [
    { kind: "invalid", path: "glass.material" },
    { kind: "clamped", path: "glass.strength.light" },
    { kind: "invalid", path: "glass.strength.dark" },
    { kind: "invalid", path: "glass.lens" },
    { kind: "invalid", path: "glass.menus" },
  ]);
});

test("a long description is clamped to 280 characters", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      meta: { description: "d".repeat(300) },
    }),
  );

  assert.equal(theme.meta.description?.length, 280);
  assert.deepEqual(warnings, [{ kind: "clamped", path: "meta.description" }]);
});

test("the name is trimmed, control characters become spaces, and it is capped at 80", () => {
  const trimmed = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      meta: { name: "  Tab\tTitle\u0007  " },
    }),
  );
  assert.equal(trimmed.theme.meta.name, "Tab Title");
  assert.deepEqual(trimmed.warnings, []);

  const long = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      meta: { name: "n".repeat(81) },
    }),
  );
  assert.equal(long.theme.meta.name.length, 80);
  assert.deepEqual(long.warnings, [{ kind: "clamped", path: "meta.name" }]);
});

test("a non-string name is reported and the default name is used", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      meta: { name: 5 },
    }),
  );

  assert.equal(theme.meta.name, "Untitled theme");
  assert.deepEqual(warnings, [{ kind: "invalid", path: "meta.name" }]);
});

test("personal settings never reach the result", () => {
  const { theme, warnings } = parseOk(
    JSON.stringify({
      kind: THEME_FILE_KIND,
      schemaVersion: 1,
      pointerCursors: true,
      typography: { reduceMotion: "on" },
      layout: { pointerCursors: true },
    }),
  );

  assert.doesNotMatch(JSON.stringify(theme), /pointerCursors|reduceMotion/);
  assert.deepEqual(
    warnings
      .map((warning) => ("path" in warning ? warning.path : warning.kind))
      .sort(),
    ["layout.pointerCursors", "pointerCursors", "typography.reduceMotion"],
  );
});
