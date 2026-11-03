// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  BackgroundAdjustments,
  GlassSettings,
  WindowMaterial,
} from "../../../chat-background/model.ts";
import {
  type AppearanceCustomization,
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

/** Equality for customization values; font lists compare entries without serializing their data. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b) && (a[0] as ImportedFont | undefined)?.dataUrl !== undefined) {
    const fonts = b as ImportedFont[];
    return a.length === b.length &&
      (a as ImportedFont[]).every((font, i) => font.name === fonts[i].name && font.dataUrl === fonts[i].dataUrl);
  }
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Theme fonts first, then the current fonts the theme does not replace by name. */
function mergeFonts(themeFonts: ImportedFont[], currentFonts: ImportedFont[]): ImportedFont[] {
  const names = new Set(themeFonts.map((font) => font.name));
  return [...themeFonts, ...currentFonts.filter((font) => !names.has(font.name))];
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
  const importedFonts =
    opts.applyAssets && theme.fonts
      ? mergeFonts(theme.fonts, current.customization.importedFonts)
      : current.customization.importedFonts;
  const customization = sanitizeCustomization({
    ...current.customization,
    ...theme.look,
    ...(opts.applyLayout && theme.layout ? theme.layout : {}),
    importedFonts,
  });
  const kept = new Set(customization.importedFonts.map((font) => font.name));
  const { material } = theme.glass;
  const glass = { ...theme.glass, style: material === "solid" ? current.glass.style : material };
  const background = opts.applyAssets && theme.background ? theme.background : "keep";

  const changed = new Set<ThemeChange>();
  if (current.palette !== theme.palette) changed.add("palette");
  for (const field of PLACED_FIELDS) {
    if (!sameValue(current.customization[field], customization[field])) {
      changed.add(changeOf(field));
    }
  }
  if (glassKey(current.glass) !== glassKey(glass)) changed.add("glass");
  if (background !== "keep") changed.add("background");

  return {
    palette: theme.palette,
    customization,
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
  const glass = { ...now.glass } as Record<string, unknown>;
  for (const key of Object.keys(now.glass) as (keyof AppearanceSnapshot["glass"])[]) {
    glass[key] = revert(before.glass[key], applied.glass[key], now.glass[key]);
  }
  return {
    palette: revert(before.palette, applied.palette, now.palette),
    customization: sanitizeCustomization(customization),
    glass: glass as AppearanceSnapshot["glass"],
  };
}
