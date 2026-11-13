// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  CUSTOMIZATION_FIELDS,
  LAYOUT_FIELDS,
  LOOK_FIELDS,
  mapIssuePath,
  pickFields,
} from "../src/features/settings/appearance-theme/domain/classification.ts";
import { parseThemeFile } from "../src/features/settings/appearance-theme/domain/parse.ts";
import {
  MAX_THEME_FILE_CHARS,
  MAX_THEME_FILE_CHARS_NO_ASSETS,
  MAX_THEME_WARNINGS,
  type ThemeParseResult,
  THEME_FILE_KIND,
  type ThemeWarning,
} from "../src/features/settings/appearance-theme/domain/schema.ts";
import {
  DEFAULT_CUSTOMIZATION,
  SIDEBAR_MENU_DEFAULT_VISIBLE,
  SIDEBAR_MENU_ITEM_IDS,
  SIDEBAR_NAV_DEFAULT_PINNED,
  SIDEBAR_NAV_ITEM_IDS,
} from "../src/features/settings/lib/appearance-customization.ts";

const file = (extra: Record<string, unknown>) => ({
  kind: THEME_FILE_KIND,
  schemaVersion: 1,
  ...extra,
});
const parse = (extra: Record<string, unknown>) =>
  parseThemeFile(JSON.stringify(file(extra)));
const pad = (json: string, length: number) => json.padEnd(length, " ");

function okOf(result: ThemeParseResult): Extract<ThemeParseResult, { ok: true }> {
  if (!result.ok) throw new Error(`expected a theme, got ${result.error}`);
  return result;
}

test("every customization field is classified, and only two are personal", () => {
  assert.deepEqual(
    Object.keys(DEFAULT_CUSTOMIZATION).filter((key) => !Object.hasOwn(CUSTOMIZATION_FIELDS, key)),
    [],
  );
  const personal = Object.entries(CUSTOMIZATION_FIELDS)
    .filter(([, spec]) => spec.category === "personal")
    .map(([key]) => key);
  assert.deepEqual(personal, ["pointerCursors", "reduceMotion"]);
});

test("sanitizer issue paths map to their file paths", () => {
  const rows = [
    ["uiFontSize", "typography.uiFontSize"],
    ["importedFonts[1].name", "assets.fonts[1].name"],
    ["colors.light.accent", "colors.light.accent"],
    ["pointerCursors", "pointerCursors"],
  ] as const;
  for (const [path, expected] of rows) assert.equal(mapIssuePath(path), expected);
});

const FULL_LAYOUT = {
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

const FULL = {
  meta: { name: "Dusk", author: "Ada", description: "Warm evening" },
  base: { palette: "cherry" },
  colors: {
    contrast: 72,
    elementOutline: 30,
    light: { accent: "#3366ff", background: "#fafafa", foreground: "#111111" },
    dark: { accent: "#ff8800", background: null, foreground: "#eeeeee" },
  },
  typography: {
    uiFont: "Lato", headingFont: "Georgia", chatFont: "Merriweather", codeFont: "Cascadia Code",
    uiFontSize: 17, codeFontSize: 13, fontSmoothing: false,
  },
  code: { lineNumbers: true, wordWrap: true, ligatures: false },
  chat: { width: "wide", composerOutline: 40 },
  layout: FULL_LAYOUT,
  glass: { material: "frosted", strength: { light: 62, dark: 33 }, lens: "magnify", menus: false },
  assets: { fonts: [{ name: "Brand", dataUrl: "data:font/woff2;base64,QUJDRA==" }] },
};

test("a full non-default file parses back to every value it holds, with no warnings", () => {
  const { theme, warnings } = okOf(parse(FULL));
  assert.deepEqual(warnings, []);
  assert.deepEqual(theme.meta, FULL.meta);
  assert.equal(theme.palette, "cherry");
  assert.deepEqual(theme.look, {
    colors: { light: FULL.colors.light, dark: FULL.colors.dark },
    contrast: 72,
    elementOutline: 30,
    uiFont: "Lato",
    headingFont: "Georgia",
    chatFont: "Merriweather",
    codeFont: "Cascadia Code",
    uiFontSize: 17,
    codeFontSize: 13,
    fontSmoothing: false,
    codeLineNumbers: true,
    codeWordWrap: true,
    codeLigatures: false,
    chatWidth: "wide",
    composerOutline: 40,
  });
  assert.deepEqual(theme.layout, FULL_LAYOUT);
  assert.deepEqual(theme.glass, FULL.glass);
  assert.deepEqual(theme.fonts, FULL.assets.fonts);
  assert.equal(theme.background, null);
});

test("missing sections fall back to the defaults, and an empty layout is present", () => {
  const bare = okOf(parse({}));
  assert.deepEqual(bare.warnings, []);
  assert.deepEqual(bare.theme.meta, { name: "Untitled theme" });
  assert.equal(bare.theme.palette, "standard");
  assert.deepEqual(bare.theme.look, pickFields(DEFAULT_CUSTOMIZATION, LOOK_FIELDS));
  assert.deepEqual(bare.theme.glass, {
    material: "solid",
    strength: { light: 40, dark: 15 },
    lens: "off",
    menus: true,
  });
  assert.equal(bare.theme.layout, null);
  assert.equal(bare.theme.fonts, null);
  assert.equal(bare.theme.background, null);
  assert.deepEqual(
    okOf(parse({ layout: {} })).theme.layout,
    pickFields(DEFAULT_CUSTOMIZATION, LAYOUT_FIELDS),
  );
});

const WARNING_CASES: [string, Record<string, unknown>, ThemeWarning[]][] = [
  ["unknown root key", { bogus: 1 }, [{ kind: "unknown", path: "bogus" }]],
  ["unknown nested key", { typography: { uiFontFace: "x" } }, [{ kind: "unknown", path: "typography.uiFontFace" }]],
  ["a null section is absent", { typography: null, meta: null }, []],
  ["a non-object section is invalid", { typography: "lato" }, [{ kind: "invalid", path: "typography" }]],
  ["invalid accent", { colors: { light: { accent: "url(x)" } } }, [{ kind: "invalid", path: "colors.light.accent" }]],
  ["font with CSS-breaking characters", { typography: { uiFont: 'Evil"; x' } }, [{ kind: "invalid", path: "typography.uiFont" }]],
  ["out-of-range font size", { typography: { uiFontSize: 99 } }, [{ kind: "clamped", path: "typography.uiFontSize" }]],
  ["unknown palette", { base: { palette: "neon" } }, [{ kind: "unknownPalette", palette: "neon" }]],
  ["non-string palette", { base: { palette: 5 } }, [{ kind: "invalid", path: "base.palette" }]],
  ["glass strength out of range", { glass: { strength: { light: 500 } } }, [{ kind: "clamped", path: "glass.strength.light" }]],
  ["unknown glass material", { glass: { material: "x" } }, [{ kind: "invalid", path: "glass.material" }]],
  [
    "newer schema version is read and reported first",
    { schemaVersion: 2, bogus: 1 },
    [{ kind: "newerVersion" }, { kind: "unknown", path: "bogus" }],
  ],
  [
    "svg background is skipped",
    { assets: { background: { dataUrl: "data:image/svg+xml;base64,PHN2Zz4=" } } },
    [{ kind: "backgroundSkipped" }],
  ],
  ["a 200-character name is clamped", { meta: { name: "a".repeat(200) } }, [{ kind: "clamped", path: "meta.name" }]],
  [
    "background adjustments that had to change are reported",
    { assets: { background: { dataUrl: "data:image/png;base64,iVBORw0KGgo=", adjustments: { visibility: 500, glow: 1 } } } },
    [
      { kind: "unknown", path: "assets.background.adjustments.glow" },
      { kind: "clamped", path: "assets.background.adjustments.visibility" },
    ],
  ],
];

test("hidden bidi and zero-width characters are stripped from the name", () => {
  assert.equal(okOf(parse({ meta: { name: "Safe\u202Eeman\u200B" } })).theme.meta.name, "Safe eman");
});

for (const [name, extra, expected] of WARNING_CASES) {
  test(`warnings: ${name}`, () => {
    assert.deepEqual(okOf(parse(extra)).warnings, expected);
  });
}

const ERROR_CASES: [string, string, string][] = [
  ["invalid JSON", "{not json", "invalidJson"],
  ["an array root", "[1]", "notTheme"],
  ["another kind", JSON.stringify({ kind: "unsloth-studio.other", schemaVersion: 1 }), "notTheme"],
  ["schemaVersion 0", JSON.stringify(file({ schemaVersion: 0 })), "notTheme"],
  ["schemaVersion as a string", JSON.stringify(file({ schemaVersion: "1" })), "notTheme"],
  ["no schemaVersion", JSON.stringify(file({ schemaVersion: undefined })), "notTheme"],
  ["over 8,000,000 characters", pad(JSON.stringify(file({})), MAX_THEME_FILE_CHARS + 1), "tooLarge"],
  ["over the settings-only cap without assets", pad(JSON.stringify(file({})), MAX_THEME_FILE_CHARS_NO_ASSETS + 1), "tooLarge"],
];

for (const [name, text, error] of ERROR_CASES) {
  test(`rejected: ${name}`, () => {
    assert.deepEqual(parseThemeFile(text), { ok: false, error });
  });
}

test("the same size is accepted only when assets carry fonts or a background", () => {
  const withFont = file({ assets: { fonts: [{ name: "Brand", dataUrl: "data:font/woff2;base64,QUJD" }] } });
  assert.equal(okOf(parseThemeFile(pad(JSON.stringify(withFont), MAX_THEME_FILE_CHARS_NO_ASSETS + 1))).warnings.length, 0);
  for (const assets of [{}, { fonts: [] }]) {
    const text = pad(JSON.stringify(file({ assets })), MAX_THEME_FILE_CHARS_NO_ASSETS + 1);
    assert.deepEqual(parseThemeFile(text), { ok: false, error: "tooLarge" });
  }
});

test("__proto__ inside a section is reported unknown and never touches Object.prototype", () => {
  const text = `{"kind":"${THEME_FILE_KIND}","schemaVersion":1,"typography":{"__proto__":{"polluted":true}}}`;
  assert.deepEqual(okOf(parseThemeFile(text)).warnings, [{ kind: "unknown", path: "typography.__proto__" }]);
  assert.equal(Object.hasOwn(Object.prototype, "polluted"), false);
  assert.equal(({} as { polluted?: unknown }).polluted, undefined);
});

test("personal keys inside sections change nothing but are reported unknown", () => {
  const bare = okOf(parse({ typography: {}, layout: {} }));
  const personal = okOf(
    parse({ typography: { pointerCursors: true, reduceMotion: "on" }, layout: { pointerCursors: true } }),
  );
  assert.deepEqual(personal.theme, bare.theme);
  assert.deepEqual(personal.warnings, [
    { kind: "unknown", path: "typography.pointerCursors" },
    { kind: "unknown", path: "typography.reduceMotion" },
    { kind: "unknown", path: "layout.pointerCursors" },
  ]);
});

test("warnings stop at the cap and the last one counts the rest", () => {
  const extra = Object.fromEntries(Array.from({ length: 60 }, (_, i) => [`key${i}`, i]));
  const { warnings } = okOf(parse(extra));
  assert.equal(warnings.length, MAX_THEME_WARNINGS + 1);
  assert.deepEqual(warnings.at(-1), { kind: "more", count: 10 });
});

test("a huge invalid layout list ends in one counted warning", () => {
  const rows = Array.from({ length: 5000 }, () => ({ id: "nope", pinned: true }));
  const { warnings } = okOf(parse({ layout: { sidebarNav: rows } }));
  assert.equal(warnings.length, 51);
  assert.deepEqual(warnings.at(-1), { kind: "more", count: 4950 });
});
