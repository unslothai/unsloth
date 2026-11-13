// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { sanitizeAdjustments } from "../src/features/chat-background/model.ts";
import { captureTheme } from "../src/features/settings/appearance-theme/application/capture.ts";
import { type AppearanceSnapshot, planThemeApply, planUndo } from "../src/features/settings/appearance-theme/application/plan-apply.ts";
import { LAYOUT_FIELDS } from "../src/features/settings/appearance-theme/domain/classification.ts";
import { parseThemeFile } from "../src/features/settings/appearance-theme/domain/parse.ts";
import { type AppearanceTheme, THEME_FILE_KIND } from "../src/features/settings/appearance-theme/domain/schema.ts";
import { DEFAULT_CUSTOMIZATION, sanitizeCustomization } from "../src/features/settings/lib/appearance-customization.ts";

const BACKGROUND = "data:image/png;base64,iVBORw0KGgo=";
const TYPOGRAPHY = ["uiFont", "headingFont", "chatFont", "codeFont", "uiFontSize", "codeFontSize", "fontSmoothing"] as const;
const GLASS = ["material", "strength", "lens", "menus"] as const;

const font = (name: string, data = "QUJD") => ({ name, dataUrl: `data:font/woff2;base64,${data}` });
const pick = <T extends object, K extends keyof T>(source: T, keys: readonly K[]) =>
  Object.fromEntries(keys.map((key) => [key, source[key]]));
const file = (extra: Record<string, unknown>) => ({ kind: THEME_FILE_KIND, schemaVersion: 1, ...extra });
const apply = (current: AppearanceSnapshot, theme: AppearanceTheme, applyLayout = true, applyAssets = true) =>
  planThemeApply(current, theme, { applyLayout, applyAssets });

function parseOk(text: string): AppearanceTheme {
  const result = parseThemeFile(text);
  if (!result.ok) throw new Error(result.error);
  return result.theme;
}
/** Exports a snapshot the way the app does, then parses the text back. */
function roundTrip(snapshot: AppearanceSnapshot, includeAssets = false) {
  const exported = captureTheme(snapshot, {
    name: "Dusk",
    includeLayout: true,
    includeAssets,
    backgroundDataUrl: includeAssets && snapshot.background ? BACKGROUND : null,
  });
  return parseOk(JSON.stringify(exported));
}

const BASE: AppearanceSnapshot = {
  palette: "standard",
  customization: DEFAULT_CUSTOMIZATION,
  glass: { material: "solid", style: "frosted", strength: { light: 40, dark: 15 }, lens: "off", menus: true },
  background: null,
};
const withFonts = (fonts: ReturnType<typeof font>[]): AppearanceSnapshot => ({
  ...BASE,
  customization: sanitizeCustomization({ importedFonts: fonts }),
});

const FULL: AppearanceSnapshot = {
  palette: "cherry",
  customization: sanitizeCustomization({
    colors: { light: { accent: "#3366ff", background: "#fafafa", foreground: "#111111" }, dark: { accent: "#ff8800" } },
    uiFont: "Lato", headingFont: "Georgia", chatFont: "Merriweather", codeFont: "Cascadia Code",
    chatWidth: "wide", sentAttachments: "chips", uiFontSize: 17, codeFontSize: 13, contrast: 72, composerOutline: 40, elementOutline: 25,
    fontSmoothing: false, codeLineNumbers: true, codeWordWrap: true, codeLigatures: false,
    pointerCursors: true, reduceMotion: "off",
    importedFonts: [font("Brand")],
    sidebarNav: [{ id: "tasks", pinned: true }, { id: "hub", pinned: false }],
    sidebarNavAuto: [],
  }),
  glass: { material: "frosted", style: "frosted", strength: { light: 62, dark: 33 }, lens: "magnify", menus: false },
  background: null,
};

test("a non-default look survives export, parse and apply onto defaults", () => {
  const plan = apply(BASE, roundTrip(FULL, true));
  assert.equal(plan.palette, "cherry");
  // Personal fields never leave the device, so they come from the target.
  assert.deepEqual(plan.customization, { ...FULL.customization, pointerCursors: false, reduceMotion: "on" });
  assert.deepEqual(pick(plan.glass, GLASS), pick(FULL.glass, GLASS));
});

test("applyLayout false keeps the target layout and still applies the look", () => {
  const plan = apply(BASE, roundTrip(FULL, true), false);
  assert.deepEqual(pick(plan.customization, LAYOUT_FIELDS), pick(DEFAULT_CUSTOMIZATION, LAYOUT_FIELDS));
  assert.equal(plan.customization.chatWidth, "wide");
});

test("applyAssets false keeps the current fonts and a background of 'keep'", () => {
  const withBackground = { ...FULL, background: { adjustments: sanitizeAdjustments({ effect: "Dither" }), blob: new Blob(["x"]) } };
  const exported = captureTheme(withBackground, { name: "Dusk", includeLayout: true, includeAssets: true, backgroundDataUrl: BACKGROUND });
  const plan = apply(withFonts([font("Old")]), parseOk(JSON.stringify(exported)), true, false);
  assert.deepEqual(plan.customization.importedFonts, [font("Old")]);
  assert.equal(plan.background, "keep");
});

test("applyAssets merges fonts theme-first, the theme wins by name, and the cap is three", () => {
  const theme = roundTrip(withFonts([font("Brand", "QUJDRA=="), font("New")]), true);
  const plan = apply(withFonts([font("Brand"), font("Old"), font("Keep")]), theme);
  assert.deepEqual(plan.customization.importedFonts, [font("Brand", "QUJDRA=="), font("New"), font("Old")]);
});

test("a theme without a typography section resets typography to the defaults", () => {
  const current = { ...BASE, customization: sanitizeCustomization({ uiFont: "Lato", uiFontSize: 17, codeFontSize: 13, fontSmoothing: false }) };
  const theme = parseOk(JSON.stringify(file({ colors: { contrast: 60 } })));
  const plan = apply(current, theme);
  assert.deepEqual(pick(plan.customization, TYPOGRAPHY), pick(DEFAULT_CUSTOMIZATION, TYPOGRAPHY));
  assert.equal(plan.customization.contrast, 60);
});

test("personal settings stay on this device", () => {
  const current = { ...BASE, customization: sanitizeCustomization({ pointerCursors: true, reduceMotion: "system" }) };
  const exported = JSON.stringify(captureTheme(FULL, { name: "Dusk", includeLayout: true, includeAssets: true }));
  assert.equal(exported.includes("pointerCursors"), false);
  assert.equal(exported.includes("reduceMotion"), false);
  const plan = apply(current, parseOk(exported));
  assert.equal(plan.customization.pointerCursors, true);
  assert.equal(plan.customization.reduceMotion, "system");
});

test("the default look exports and applies as no change", () => {
  const plan = apply(BASE, roundTrip(BASE, true));
  assert.deepEqual(plan.changes, []);
  assert.equal(plan.background, "keep");
  assert.equal(JSON.stringify(plan.customization), JSON.stringify(DEFAULT_CUSTOMIZATION));
});

test("a solid theme keeps the remembered glass style; any other material sets it", () => {
  const current = { ...BASE, glass: { ...BASE.glass, style: "liquid" as const } };
  const solid = parseOk(JSON.stringify(file({ glass: { material: "solid" } })));
  const frosted = parseOk(JSON.stringify(file({ glass: { material: "frosted" } })));
  assert.equal(apply(current, solid).glass.style, "liquid");
  assert.equal(apply(current, frosted).glass.style, "frosted");
});

test("changes follow THEME_CHANGES order whatever order the file lists them in", () => {
  const theme = parseOk(JSON.stringify(file({
    assets: { background: { dataUrl: BACKGROUND }, fonts: [font("Brand")] },
    layout: { sentAttachments: "chips" },
    glass: { material: "frosted" },
    chat: { width: "wide" },
    code: { wordWrap: true },
    typography: { uiFont: "Lato" },
    base: { palette: "cherry" },
  })));
  assert.deepEqual(apply(BASE, theme).changes, [
    "palette", "typography", "code", "chat", "glass", "layout", "fonts", "background",
  ]);
});

test("undo reverts only what the theme set and nobody changed since", () => {
  const before = { ...BASE, customization: { ...DEFAULT_CUSTOMIZATION, contrast: 40, uiFontSize: 14 } };
  const theme = parseOk(JSON.stringify(file({ base: { palette: "matcha" }, colors: { contrast: 70 }, typography: { uiFontSize: 18 } })));
  const applied = apply(before, theme);
  // After applying, the user edits the font size and turns on pointer cursors.
  const now = { ...BASE, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, uiFontSize: 16, pointerCursors: true } };
  const undone = planUndo(before, applied, now);
  assert.equal(undone.palette, "standard");
  assert.equal(undone.customization.contrast, 40);
  assert.equal(undone.customization.uiFontSize, 16);
  assert.equal(undone.customization.pointerCursors, true);
});

test("fonts pushed out by the cap are named, and export drops names of fonts it leaves out", () => {
  const current = { ...BASE, customization: sanitizeCustomization({ ...DEFAULT_CUSTOMIZATION, importedFonts: [font("Mine A"), font("Mine B")] }) };
  const theme = parseOk(JSON.stringify(file({ assets: { fonts: [font("Theirs A"), font("Theirs B")] } })));
  assert.deepEqual(apply(current, theme).droppedFonts, ["Mine B"]);
  const withBrand = { ...current, customization: { ...current.customization, uiFont: "Mine A", codeFont: "Fira Code" } };
  const exported = captureTheme(withBrand, { name: "x", includeLayout: false, includeAssets: false }) as { typography: Record<string, unknown> };
  assert.equal(exported.typography.uiFont, null);
  assert.equal(exported.typography.codeFont, "Fira Code");
});

test("undo keeps an imported font the user selected after applying", () => {
  const theme = parseOk(JSON.stringify(file({ assets: { fonts: [font("Theirs")] } })));
  const applied = apply(BASE, theme);
  const now = { ...BASE, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, uiFont: "Theirs" } };
  const undone = planUndo(BASE, applied, now);
  assert.equal(undone.customization.uiFont, "Theirs");
  assert.deepEqual(undone.customization.importedFonts.map((f) => f.name), ["Theirs"]);
  // Without the later pick, Undo removes the theme's font as before.
  const untouched = { ...now, customization: applied.customization };
  assert.deepEqual(planUndo(BASE, applied, untouched).customization.importedFonts, []);
});

test("export without assets blanks only font selections, not a setting that shares a font's name", () => {
  const current = { ...BASE, customization: sanitizeCustomization({ importedFonts: [font("wide")], chatWidth: "wide", uiFont: "wide" }) };
  const exported = captureTheme(current, { name: "x", includeLayout: false, includeAssets: false }) as Record<string, Record<string, unknown>>;
  assert.equal(exported.chat.width, "wide");
  assert.equal(exported.typography.uiFont, null);
});

test("a selected font survives the cap when a theme adds fonts", () => {
  const current = { ...BASE, customization: sanitizeCustomization({ importedFonts: [font("A"), font("B"), font("Mine")], codeFont: "Mine" }) };
  const theme = parseOk(JSON.stringify(file({ typography: { codeFont: "Mine" }, assets: { fonts: [font("T1"), font("T2")] } })));
  const plan = apply(current, theme);
  assert.equal(plan.customization.codeFont, "Mine");
  assert.ok(plan.customization.importedFonts.some((f) => f.name === "Mine"));
});

test("undo keeps every selected font under the cap", () => {
  const before = { ...BASE, customization: sanitizeCustomization({ importedFonts: [font("A"), font("B"), font("C")], codeFont: "C" }) };
  const theme = parseOk(JSON.stringify(file({ typography: { codeFont: "C" }, assets: { fonts: [font("New")] } })));
  const applied = apply(before, theme);
  const now = { ...before, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, uiFont: "New" } };
  const undone = planUndo(before, applied, now);
  const names = undone.customization.importedFonts.map((f) => f.name);
  for (const field of ["uiFont", "codeFont"] as const) {
    const name = undone.customization[field];
    assert.ok(name === null || names.includes(name), `${field}=${name} not in ${names}`);
  }
  assert.equal(undone.customization.uiFont, "New");
  assert.equal(undone.customization.codeFont, "C");
});

test("undo does not restore a selection whose font was deleted after applying", () => {
  const before = { ...BASE, customization: sanitizeCustomization({ importedFonts: [font("Brand")], uiFont: "Brand" }) };
  const theme = parseOk(JSON.stringify(file({ typography: { uiFont: "Lato" } })));
  const applied = apply(before, theme);
  const now = { ...before, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, importedFonts: [] } };
  const undone = planUndo(before, applied, now);
  assert.deepEqual(undone.customization.importedFonts, []);
  assert.equal(undone.customization.uiFont, null);
});

test("undo reverts each color override on its own", () => {
  const theme = parseOk(JSON.stringify(file({ colors: { light: { accent: "#112233", background: "#fafafa" } } })));
  const applied = apply(BASE, theme);
  const edited = { ...applied.customization.colors.light, accent: "#445566" };
  const now = { ...BASE, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, colors: { ...applied.customization.colors, light: edited } } };
  const undone = planUndo(BASE, applied, now);
  assert.equal(undone.customization.colors.light.accent, "#445566");
  assert.equal(undone.customization.colors.light.background, null);
});

test("no selection points at a font the cap dropped while fonts are put back", () => {
  const before = { ...BASE, customization: sanitizeCustomization({
    importedFonts: [font("A"), font("B"), font("C")], uiFont: "A", headingFont: "B", chatFont: "C" }) };
  const theme = parseOk(JSON.stringify(file({ typography: { uiFont: "Lato", headingFont: "B", chatFont: "C" }, assets: { fonts: [font("T")] } })));
  const applied = apply(before, theme);
  const now = { ...before, palette: applied.palette, glass: applied.glass,
    customization: { ...applied.customization, codeFont: "T" } };
  const undone = planUndo(before, applied, now);
  const names = undone.customization.importedFonts.map((f) => f.name);
  for (const field of ["uiFont", "headingFont", "chatFont", "codeFont"] as const) {
    const name = undone.customization[field];
    assert.ok(name === null || names.includes(name), `${field}=${name} not in ${names}`);
  }
});

test("undo reverts light and dark glass strength on their own", () => {
  const theme = parseOk(JSON.stringify(file({ glass: { material: "frosted", strength: { light: 70, dark: 60 } } })));
  const applied = apply(BASE, theme);
  const now = { ...BASE, palette: applied.palette, customization: applied.customization,
    glass: { ...applied.glass, strength: { ...applied.glass.strength, dark: 25 } } };
  const undone = planUndo(BASE, applied, now);
  assert.deepEqual(undone.glass.strength, { light: 40, dark: 25 });
});

test("undo brings back a font the theme's cap evicted, even after the theme's font was removed", () => {
  const before = { ...BASE, customization: sanitizeCustomization({ importedFonts: [font("A"), font("B"), font("C")], codeFont: "C" }) };
  const theme = parseOk(JSON.stringify(file({ typography: { codeFont: "T" }, assets: { fonts: [font("T")] } })));
  const applied = apply(before, theme);
  assert.ok(!applied.customization.importedFonts.some((f) => f.name === "C"));
  const now = { ...before, palette: applied.palette, glass: applied.glass, customization: { ...applied.customization,
    importedFonts: applied.customization.importedFonts.filter((f) => f.name !== "T"), codeFont: null } };
  const undone = planUndo(before, applied, now);
  assert.ok(undone.customization.importedFonts.some((f) => f.name === "C"));
});
