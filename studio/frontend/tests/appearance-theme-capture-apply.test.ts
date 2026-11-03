// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { captureTheme } from "../src/features/settings/appearance-theme/application/capture.ts";
import { planThemeApply } from "../src/features/settings/appearance-theme/application/plan-apply.ts";
import type { AppearanceSnapshot } from "../src/features/settings/appearance-theme/application/types.ts";
import {
  LAYOUT_FIELDS,
  LOOK_FIELDS,
} from "../src/features/settings/appearance-theme/domain/classification.ts";
import { parseThemeFile } from "../src/features/settings/appearance-theme/domain/parse.ts";
import {
  DEFAULT_THEME_GLASS,
  type ThemeGlass,
} from "../src/features/settings/appearance-theme/domain/schema.ts";
import {
  type AppearanceCustomization,
  DEFAULT_CUSTOMIZATION,
  type ImportedFont,
  isDefaultCustomization,
  sanitizeCustomization,
} from "../src/features/settings/lib/appearance-customization.ts";

const TYPOGRAPHY_FIELDS: (keyof AppearanceCustomization)[] = [
  "uiFont",
  "headingFont",
  "chatFont",
  "codeFont",
  "uiFontSize",
  "codeFontSize",
  "fontSmoothing",
];

const BACKGROUND_DATA_URL = "data:image/png;base64,iVBORw0KGgo=";

const FONTS: ImportedFont[] = [
  { name: "Brand Sans", dataUrl: "data:font/woff2;base64,d09GMgABAAAA" },
  { name: "Mono Pro", dataUrl: "data:font/otf;base64,T1RUTwAL" },
];

const FULL_CUSTOMIZATION = sanitizeCustomization({
  ...DEFAULT_CUSTOMIZATION,
  colors: {
    light: { accent: "#2f6fd6", background: "#fbf7f0", foreground: null },
    dark: { accent: "#e0a458", background: "#15110d", foreground: "#f3ece2" },
  },
  uiFont: "Brand Sans",
  headingFont: "Mono Pro",
  chatFont: null,
  codeFont: "Mono Pro",
  uiFontSize: 17,
  codeFontSize: 13,
  contrast: 72,
  composerOutline: 40,
  chatWidth: "wide",
  sentAttachments: "chips",
  importedFonts: FONTS,
  fontSmoothing: false,
  codeLineNumbers: true,
  codeWordWrap: true,
  codeLigatures: false,
  sidebarMenu: [...DEFAULT_CUSTOMIZATION.sidebarMenu].reverse(),
  sidebarNav: [...DEFAULT_CUSTOMIZATION.sidebarNav]
    .reverse()
    .map((item) => ({ ...item, pinned: !item.pinned })),
  sidebarNavAuto: [],
});

const FULL_GLASS: ThemeGlass = {
  material: "frosted",
  strength: { light: 62, dark: 33 },
  lens: "magnify",
  menus: false,
};

const FULL: AppearanceSnapshot = {
  palette: "cherry",
  customization: FULL_CUSTOMIZATION,
  glass: FULL_GLASS,
  background: null,
};

const DEFAULTS: AppearanceSnapshot = {
  palette: "standard",
  customization: DEFAULT_CUSTOMIZATION,
  glass: {
    ...DEFAULT_THEME_GLASS,
    strength: { ...DEFAULT_THEME_GLASS.strength },
  },
  background: null,
};

function exportAndParse(
  snapshot: AppearanceSnapshot,
  opts: Parameters<typeof captureTheme>[1],
) {
  const text = JSON.stringify(captureTheme(snapshot, opts));
  const parsed = parseThemeFile(text);
  if (!parsed.ok) assert.fail(`parse failed: ${parsed.error}`);
  return { text, theme: parsed.theme, warnings: parsed.warnings };
}

test("a full non-default look survives export, parse and apply onto defaults", () => {
  const { text, theme, warnings } = exportAndParse(FULL, {
    name: "  Night Garden  ",
    author: "Ada",
    includeLayout: true,
    includeAssets: true,
  });
  assert.deepEqual(warnings, []);
  assert.deepEqual(Object.keys(JSON.parse(text)), [
    "kind",
    "schemaVersion",
    "meta",
    "base",
    "colors",
    "typography",
    "code",
    "chat",
    "glass",
    "layout",
    "assets",
  ]);

  const plan = planThemeApply(DEFAULTS, theme, {
    applyLayout: true,
    applyAssets: true,
  });
  assert.equal(plan.palette, "cherry");
  assert.deepEqual(plan.glass, FULL_GLASS);
  for (const field of LOOK_FIELDS) {
    assert.deepEqual(
      plan.customization[field],
      FULL_CUSTOMIZATION[field],
      field,
    );
  }
  for (const field of LAYOUT_FIELDS) {
    assert.deepEqual(
      plan.customization[field],
      FULL_CUSTOMIZATION[field],
      field,
    );
  }
  assert.deepEqual(plan.customization.importedFonts, FONTS);
  assert.deepEqual(plan.customization, FULL_CUSTOMIZATION);
  assert.equal(plan.background, "keep");
  assert.deepEqual(plan.changes, [
    "palette",
    "colors",
    "typography",
    "code",
    "chat",
    "glass",
    "layout",
    "fonts",
  ]);
});

test("applyLayout false keeps the current layout", () => {
  const { theme } = exportAndParse(FULL, {
    name: "layout",
    includeLayout: true,
    includeAssets: false,
  });
  assert.ok(theme.layout);
  const plan = planThemeApply(DEFAULTS, theme, {
    applyLayout: false,
    applyAssets: false,
  });
  for (const field of LAYOUT_FIELDS) {
    assert.deepEqual(
      plan.customization[field],
      DEFAULT_CUSTOMIZATION[field],
      field,
    );
  }
  assert.ok(!plan.changes.includes("layout"));
  assert.ok(plan.changes.includes("colors"));
});

test("applyAssets false keeps the current fonts and background", () => {
  const { theme } = exportAndParse(FULL, {
    name: "assets",
    includeLayout: false,
    includeAssets: true,
    backgroundDataUrl: BACKGROUND_DATA_URL,
  });
  assert.ok(theme.fonts && theme.background);

  const kept = planThemeApply(DEFAULTS, theme, {
    applyLayout: false,
    applyAssets: false,
  });
  assert.deepEqual(kept.customization.importedFonts, []);
  assert.equal(kept.background, "keep");
  assert.ok(!kept.changes.includes("fonts"));
  assert.ok(!kept.changes.includes("background"));

  const applied = planThemeApply(DEFAULTS, theme, {
    applyLayout: false,
    applyAssets: true,
  });
  assert.deepEqual(applied.customization.importedFonts, FONTS);
  assert.deepEqual(applied.background, theme.background);
  assert.ok(applied.changes.includes("fonts"));
  assert.ok(applied.changes.includes("background"));
});

test("imported fonts from the theme come first, win by name, and cap at three", () => {
  const themeFonts: ImportedFont[] = [
    { name: "Theme A", dataUrl: "data:font/woff2;base64,QUFBQQ==" },
    { name: "Brand Sans", dataUrl: "data:font/woff2;base64,QkJCQg==" },
  ];
  const themeSnapshot: AppearanceSnapshot = {
    ...DEFAULTS,
    customization: sanitizeCustomization({
      ...DEFAULT_CUSTOMIZATION,
      importedFonts: themeFonts,
    }),
  };
  const { theme } = exportAndParse(themeSnapshot, {
    name: "fonts",
    includeLayout: false,
    includeAssets: true,
  });

  const currentFonts: ImportedFont[] = [
    { name: "Old One", dataUrl: "data:font/ttf;base64,Q0NDQw==" },
    { name: "Brand Sans", dataUrl: "data:font/otf;base64,REREUA==" },
    { name: "Old Two", dataUrl: "data:font/otf;base64,RUVFRQ==" },
  ];
  const current: AppearanceSnapshot = {
    ...DEFAULTS,
    customization: sanitizeCustomization({
      ...DEFAULT_CUSTOMIZATION,
      importedFonts: currentFonts,
    }),
  };
  const plan = planThemeApply(current, theme, {
    applyLayout: false,
    applyAssets: true,
  });
  assert.deepEqual(plan.customization.importedFonts, [
    themeFonts[0],
    themeFonts[1],
    currentFonts[0],
  ]);
  assert.ok(plan.changes.includes("fonts"));
});

test("a theme without a typography section resets typography to defaults", () => {
  const { text } = exportAndParse(FULL, {
    name: "no typography",
    includeLayout: false,
    includeAssets: false,
  });
  const file = JSON.parse(text) as Record<string, unknown>;
  delete file.typography;
  const parsed = parseThemeFile(JSON.stringify(file));
  if (!parsed.ok) assert.fail(parsed.error);

  const plan = planThemeApply(FULL, parsed.theme, {
    applyLayout: false,
    applyAssets: false,
  });
  for (const field of TYPOGRAPHY_FIELDS) {
    assert.deepEqual(
      plan.customization[field],
      DEFAULT_CUSTOMIZATION[field],
      field,
    );
  }
  assert.ok(plan.changes.includes("typography"));
});

test("personal settings stay on this device", () => {
  const current: AppearanceSnapshot = {
    ...DEFAULTS,
    customization: sanitizeCustomization({
      ...DEFAULT_CUSTOMIZATION,
      pointerCursors: true,
      reduceMotion: "on",
    }),
  };
  const { text, theme } = exportAndParse(current, {
    name: "personal",
    includeLayout: true,
    includeAssets: true,
  });
  assert.doesNotMatch(text, /pointerCursors|reduceMotion/);

  const plan = planThemeApply(current, theme, {
    applyLayout: true,
    applyAssets: true,
  });
  assert.equal(plan.customization.pointerCursors, true);
  assert.equal(plan.customization.reduceMotion, "on");

  // A file that tries to smuggle personal keys into a look section is ignored.
  const file = JSON.parse(text) as Record<string, Record<string, unknown>>;
  file.typography.pointerCursors = true;
  file.typography.reduceMotion = "off";
  const hostile = parseThemeFile(JSON.stringify(file));
  if (!hostile.ok) assert.fail(hostile.error);
  const fromDefaults = planThemeApply(DEFAULTS, hostile.theme, {
    applyLayout: true,
    applyAssets: true,
  });
  assert.equal(fromDefaults.customization.pointerCursors, false);
  assert.equal(fromDefaults.customization.reduceMotion, "system");
});

test("the default look exports and applies as no change", () => {
  const { theme, warnings } = exportAndParse(DEFAULTS, {
    name: "",
    includeLayout: true,
    includeAssets: true,
  });
  assert.deepEqual(warnings, []);

  const plan = planThemeApply(DEFAULTS, theme, {
    applyLayout: true,
    applyAssets: true,
  });
  assert.deepEqual(plan.changes, []);
  assert.equal(isDefaultCustomization(plan.customization), true);
  assert.equal(
    JSON.stringify(plan.customization),
    JSON.stringify(DEFAULT_CUSTOMIZATION),
  );
});

test("export falls back to a default name and drops an empty author", () => {
  const file = captureTheme(DEFAULTS, {
    name: "   ",
    author: "  ",
    includeLayout: false,
    includeAssets: false,
  });
  assert.deepEqual(file.meta, { name: "Untitled theme" });
  assert.equal("assets" in file, false);
  assert.equal("layout" in file, false);
});
