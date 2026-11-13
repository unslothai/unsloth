// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  BackgroundAdjustments,
  GlassSettings,
  WindowMaterial,
} from "../../../chat-background/model.ts";
import {
  type AppearanceCustomization,
  type CustomModeColors,
  fontFamilyKey,
  type ImportedFont,
  sanitizeCustomization,
} from "../../lib/appearance-customization.ts";
import type { ColorThemeId } from "../../lib/color-themes.ts";
import {
  changeOf,
  PLACED_FIELDS,
  THEME_CHANGES,
  type ThemeChange,
} from "../domain/classification.ts";
import type { AppearanceTheme, ThemeBackground } from "../domain/schema.ts";

/** The live appearance a theme replaces, and what undo puts back. */
export type AppearanceSnapshot = {
  palette: ColorThemeId;
  customization: AppearanceCustomization;
  /** `style` is the glass look the on/off switch remembers while glass is off. */
  glass: GlassSettings & { style: Exclude<WindowMaterial, "solid"> };
  background: { adjustments: BackgroundAdjustments; blob: Blob } | null;
};

export type ThemeApplyPlan = Omit<AppearanceSnapshot, "background"> & {
  background: "keep" | ThemeBackground;
  changes: ThemeChange[];
  /** Fonts that no longer fit under the imported-font cap once the theme's are added. */
  droppedFonts: string[];
};

export const FONT_FIELDS = ["uiFont", "headingFont", "chatFont", "codeFont"] as const;

const fontKey = fontFamilyKey;

/** Families `c` selects that it does not import: installed on this device. */
function localFamilies(c: AppearanceCustomization): Set<string> {
  const imported = new Set(c.importedFonts.map((font) => fontKey(font.name)));
  return new Set(FONT_FIELDS.map((field) => fontKey(c[field])).filter((name) => name && !imported.has(name)));
}

/** Selected fonts first, so the imported-font cap drops unused ones before any in use. */
function selectedFirst(c: Pick<AppearanceCustomization, (typeof FONT_FIELDS)[number]>, fonts: ImportedFont[]) {
  const selected = new Set(FONT_FIELDS.map((field) => fontKey(c[field])));
  const inUse = (f: ImportedFont) => selected.has(fontKey(f.name));
  return [...fonts.filter(inUse), ...fonts.filter((f) => !inUse(f))];
}

/**
 * Keeps font selections and imported fonts consistent: a selection naming one of `importedNames`
 * that `keep` can supply is put back, and one whose font is still missing after the cap is reset.
 */
function reconcileFonts(
  customization: AppearanceCustomization,
  importedNames: Set<string>,
  keep: ImportedFont[] = [],
): AppearanceCustomization {
  const key = fontKey;
  const imported = new Set([...importedNames].map(key));
  const selected = new Set(FONT_FIELDS.map((field) => key(customization[field])));
  const present = new Set(customization.importedFonts.map((font) => key(font.name)));
  const restored = keep.filter(
    (font) => !present.has(key(font.name)) && imported.has(key(font.name)) && selected.has(key(font.name)),
  );
  const next = sanitizeCustomization({
    ...customization,
    importedFonts: selectedFirst(customization, [...restored, ...customization.importedFonts]),
  });
  // Selections take the imported font's own spelling, so exact-name removal still finds them.
  const kept = new Map(next.importedFonts.map((font) => [key(font.name), font.name]));
  for (const field of FONT_FIELDS) {
    const name = next[field];
    if (!name) continue;
    if (kept.has(key(name))) next[field] = kept.get(key(name))!;
    else if (imported.has(key(name))) next[field] = null;
  }
  return next;
}

/** Equality for customization values; font lists compare entries without serializing their data. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b) && (a[0] as ImportedFont | undefined)?.dataUrl !== undefined) {
    const fonts = b as ImportedFont[];
    return a.length === b.length &&
      (a as ImportedFont[]).every((font, i) => font.name === fonts[i].name && font.dataUrl === fonts[i].dataUrl);
  }
  return JSON.stringify(a) === JSON.stringify(b);
}

function mergeFonts(themeFonts: ImportedFont[], currentFonts: ImportedFont[]): ImportedFont[] {
  // Case-insensitive like CSS family matching and the font uploader.
  const names = new Set(themeFonts.map((font) => font.name.toLowerCase()));
  return [...themeFonts, ...currentFonts.filter((font) => !names.has(font.name.toLowerCase()))];
}

const glassKey = ({ material, strength, lens, menus }: GlassSettings) =>
  JSON.stringify([material, strength.light, strength.dark, lens, menus]);

/**
 * What applying a parsed theme would do, without touching the app. The parser fills every look
 * field, so a section the theme leaves out resets to its defaults. Personal fields stay as they are.
 */
export function planThemeApply(
  current: AppearanceSnapshot,
  theme: AppearanceTheme,
  opts: { applyLayout: boolean; applyAssets: boolean },
): ThemeApplyPlan {
  const picked = {
    ...current.customization,
    ...theme.look,
    ...(opts.applyLayout && theme.layout ? theme.layout : {}),
  };
  const importedFonts = selectedFirst(
    picked,
    opts.applyAssets && theme.fonts
      ? mergeFonts(theme.fonts, current.customization.importedFonts)
      : current.customization.importedFonts,
  );
  const customization = sanitizeCustomization({ ...picked, importedFonts });
  // A selection whose imported font was left out or pushed past the cap would point at nothing.
  const reconciled = reconcileFonts(
    customization,
    new Set(
      [...(theme.fonts ?? []), ...current.customization.importedFonts]
        .map((font) => font.name)
        .filter((name) => !localFamilies(current.customization).has(fontKey(name))),
    ),
  );
  const kept = new Set(reconciled.importedFonts.map((font) => font.name));
  const { material } = theme.glass;
  const glass = { ...theme.glass, style: material === "solid" ? current.glass.style : material };
  const background = opts.applyAssets && theme.background ? theme.background : "keep";

  const changed = new Set<ThemeChange>();
  if (current.palette !== theme.palette) changed.add("palette");
  for (const field of PLACED_FIELDS) {
    if (!sameValue(current.customization[field], reconciled[field])) {
      changed.add(changeOf(field));
    }
  }
  if (glassKey(current.glass) !== glassKey(glass)) changed.add("glass");
  if (background !== "keep") changed.add("background");

  return {
    palette: theme.palette,
    customization: reconciled,
    glass,
    background,
    changes: THEME_CHANGES.filter((change) => changed.has(change)),
    droppedFonts: importedFonts.filter((font) => !kept.has(font.name)).map((font) => font.name),
  };
}

/**
 * What Undo writes: each value the theme set goes back to `before`, unless it changed again since
 * the theme was applied, so later edits (and personal settings) survive an Undo.
 */
export function planUndo(
  before: AppearanceSnapshot,
  applied: ThemeApplyPlan,
  now: AppearanceSnapshot,
): Omit<AppearanceSnapshot, "background"> {
  const revert = <T>(was: T, set: T, is: T): T => (sameValue(is, set) ? was : is);
  const customization = { ...now.customization } as Record<string, unknown>;
  for (const key of Object.keys(now.customization) as (keyof AppearanceCustomization)[]) {
    customization[key] = revert(before.customization[key], applied.customization[key], now.customization[key]);
  }
  // Per color, so editing one override after applying does not pin the theme's others.
  const colors = {} as Record<string, Record<string, unknown>>;
  for (const mode of ["light", "dark"] as const) {
    colors[mode] = {};
    for (const key of Object.keys(now.customization.colors[mode]) as (keyof CustomModeColors)[]) {
      colors[mode][key] = revert(
        before.customization.colors[mode][key],
        applied.customization.colors[mode][key],
        now.customization.colors[mode][key],
      );
    }
  }
  customization.colors = colors;
  // Per font: entries untouched since Apply go back to their pre-Apply bytes (or away if Apply
  // added them and nothing selects them); fonts Apply evicted under the cap come back.
  const byName = (list: ImportedFont[]) => new Map(list.map((font) => [fontKey(font.name), font]));
  const was = byName(before.customization.importedFonts);
  const set = byName(applied.customization.importedFonts);
  const is = byName(now.customization.importedFonts);
  // Only selections made after Apply keep a theme-added face; a restored local family does not.
  const selected = new Set(
    FONT_FIELDS.filter((field) => now.customization[field] !== applied.customization[field])
      .map((field) => fontKey(now.customization[field])),
  );
  const fonts: ImportedFont[] = [];
  for (const font of now.customization.importedFonts) {
    const k = fontKey(font.name);
    if (set.get(k)?.dataUrl !== font.dataUrl) fonts.push(font);
    else if (was.has(k)) fonts.push(was.get(k)!);
    else if (selected.has(k)) fonts.push(font);
  }
  for (const font of before.customization.importedFonts) {
    if (!set.has(fontKey(font.name)) && !is.has(fontKey(font.name))) fonts.push(font);
  }
  customization.importedFonts = selectedFirst(customization as AppearanceCustomization, fonts);
  const glass = { ...now.glass } as Record<string, unknown>;
  for (const key of Object.keys(now.glass) as (keyof AppearanceSnapshot["glass"])[]) {
    glass[key] = revert(before.glass[key], applied.glass[key], now.glass[key]);
  }
  glass.strength = {
    light: revert(before.glass.strength.light, applied.glass.strength.light, now.glass.strength.light),
    dark: revert(before.glass.strength.dark, applied.glass.strength.dark, now.glass.strength.dark),
  };
  const reverted = sanitizeCustomization(customization);
  // Every imported name seen, so a restored selection whose font was since deleted is cleared.
  // A family that was local (not imported) before Apply stays selectable after Undo.
  const localBefore = localFamilies(before.customization);
  const knownNames = new Set(
    [before, applied, now].flatMap((s) => s.customization.importedFonts.map((font) => font.name))
      .filter((name) => !localBefore.has(fontKey(name))),
  );
  return {
    palette: revert(before.palette, applied.palette, now.palette),
    customization: reconcileFonts(reverted, knownNames, fonts),
    glass: glass as AppearanceSnapshot["glass"],
  };
}
