// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AppearanceCustomization } from "../../lib/appearance-customization.ts";
import type { ThemeLayout, ThemeLook } from "./schema.ts";

/**
 * look and layout travel in a theme file. asset (fonts, background) travels only when asked.
 * personal never leaves the device.
 */
export type FieldCategory = "look" | "layout" | "asset" | "personal";

/**
 * Every AppearanceCustomization key must be listed here. `satisfies` fails the typecheck when a
 * field is added without a category.
 */
export const CUSTOMIZATION_FIELD_CATEGORY = {
  colors: "look",
  contrast: "look",
  uiFont: "look",
  headingFont: "look",
  chatFont: "look",
  codeFont: "look",
  uiFontSize: "look",
  codeFontSize: "look",
  fontSmoothing: "look",
  codeLineNumbers: "look",
  codeWordWrap: "look",
  codeLigatures: "look",
  chatWidth: "look",
  composerOutline: "look",
  sentAttachments: "layout",
  sidebarMenu: "layout",
  sidebarNav: "layout",
  sidebarNavAuto: "layout",
  importedFonts: "asset",
  pointerCursors: "personal",
  reduceMotion: "personal",
} as const satisfies Record<keyof AppearanceCustomization, FieldCategory>;

type CategoryMap = typeof CUSTOMIZATION_FIELD_CATEGORY;

type FieldsIn<C extends FieldCategory> = {
  [K in keyof CategoryMap]: CategoryMap[K] extends C ? K : never;
}[keyof CategoryMap];

export type LookField = FieldsIn<"look">;
export type LayoutField = FieldsIn<"layout">;

function fieldsIn(category: FieldCategory): string[] {
  return Object.keys(CUSTOMIZATION_FIELD_CATEGORY).filter(
    (key) =>
      CUSTOMIZATION_FIELD_CATEGORY[key as keyof AppearanceCustomization] ===
      category,
  );
}

export const LOOK_FIELDS = fieldsIn("look") as LookField[];
export const LAYOUT_FIELDS = fieldsIn("layout") as LayoutField[];

/** Where each look, layout, or asset field is written in the file. */
export const FILE_PATH_FOR_FIELD: Record<
  LookField | LayoutField | "importedFonts",
  string
> = {
  colors: "colors",
  contrast: "colors.contrast",
  uiFont: "typography.uiFont",
  headingFont: "typography.headingFont",
  chatFont: "typography.chatFont",
  codeFont: "typography.codeFont",
  uiFontSize: "typography.uiFontSize",
  codeFontSize: "typography.codeFontSize",
  fontSmoothing: "typography.fontSmoothing",
  codeLineNumbers: "code.lineNumbers",
  codeWordWrap: "code.wordWrap",
  codeLigatures: "code.ligatures",
  chatWidth: "chat.width",
  composerOutline: "chat.composerOutline",
  sentAttachments: "layout.sentAttachments",
  sidebarMenu: "layout.sidebarMenu",
  sidebarNav: "layout.sidebarNav",
  sidebarNavAuto: "layout.sidebarNavAuto",
  importedFonts: "assets.fonts",
};

/** Turns a sanitizer path such as "uiFontSize" or "importedFonts[1].name" into its file path. */
export function mapIssuePath(path: string): string {
  const field = /^[A-Za-z]+/.exec(path)?.[0];
  if (field === undefined || !Object.hasOwn(FILE_PATH_FOR_FIELD, field)) {
    return path;
  }
  return (
    FILE_PATH_FOR_FIELD[field as keyof typeof FILE_PATH_FOR_FIELD] +
    path.slice(field.length)
  );
}

function pickFields<K extends keyof AppearanceCustomization>(
  customization: AppearanceCustomization,
  keys: readonly K[],
): Pick<AppearanceCustomization, K> {
  const picked = {} as Pick<AppearanceCustomization, K>;
  for (const key of keys) picked[key] = customization[key];
  return picked;
}

export function themeLookFromCustomization(
  customization: AppearanceCustomization,
): ThemeLook {
  return pickFields(customization, LOOK_FIELDS);
}

export function themeLayoutFromCustomization(
  customization: AppearanceCustomization,
): ThemeLayout {
  return pickFields(customization, LAYOUT_FIELDS);
}
