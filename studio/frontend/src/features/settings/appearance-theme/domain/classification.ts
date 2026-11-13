// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AppearanceCustomization } from "../../lib/appearance-customization.ts";

/** The groups the import preview lists, in display order. */
export const THEME_CHANGES = [
  "palette",
  "colors",
  "typography",
  "code",
  "chat",
  "glass",
  "layout",
  "fonts",
  "background",
] as const;
export type ThemeChange = (typeof THEME_CHANGES)[number];

type FieldSpec =
  /** Travels in the file at `section.key` (`colors` is the light/dark pair inside `colors`). */
  | {
      category: "look" | "layout" | "asset";
      section: string;
      key: string;
      change: ThemeChange;
    }
  /** Never leaves the device. */
  | { category: "personal" };

/**
 * The one description of where every customization field goes. Capture, parse, warning paths and
 * the change list all read it, and `satisfies` fails the typecheck for a field without an entry.
 */
export const CUSTOMIZATION_FIELDS = {
  colors: { category: "look", section: "colors", key: "", change: "colors" },
  contrast: { category: "look", section: "colors", key: "contrast", change: "colors" },
  elementOutline: { category: "look", section: "colors", key: "elementOutline", change: "colors" },
  uiFont: { category: "look", section: "typography", key: "uiFont", change: "typography" },
  headingFont: { category: "look", section: "typography", key: "headingFont", change: "typography" },
  chatFont: { category: "look", section: "typography", key: "chatFont", change: "typography" },
  codeFont: { category: "look", section: "typography", key: "codeFont", change: "typography" },
  uiFontSize: { category: "look", section: "typography", key: "uiFontSize", change: "typography" },
  codeFontSize: { category: "look", section: "typography", key: "codeFontSize", change: "typography" },
  fontSmoothing: { category: "look", section: "typography", key: "fontSmoothing", change: "typography" },
  codeLineNumbers: { category: "look", section: "code", key: "lineNumbers", change: "code" },
  codeWordWrap: { category: "look", section: "code", key: "wordWrap", change: "code" },
  codeLigatures: { category: "look", section: "code", key: "ligatures", change: "code" },
  chatWidth: { category: "look", section: "chat", key: "width", change: "chat" },
  composerOutline: { category: "look", section: "chat", key: "composerOutline", change: "chat" },
  sentAttachments: { category: "layout", section: "layout", key: "sentAttachments", change: "layout" },
  sidebarMenu: { category: "layout", section: "layout", key: "sidebarMenu", change: "layout" },
  sidebarNav: { category: "layout", section: "layout", key: "sidebarNav", change: "layout" },
  sidebarNavAuto: { category: "layout", section: "layout", key: "sidebarNavAuto", change: "layout" },
  importedFonts: { category: "asset", section: "assets", key: "fonts", change: "fonts" },
  pointerCursors: { category: "personal" },
  reduceMotion: { category: "personal" },
} as const satisfies Record<keyof AppearanceCustomization, FieldSpec>;

type Fields = typeof CUSTOMIZATION_FIELDS;
type Field = keyof Fields;
type FieldsIn<C extends FieldSpec["category"]> = {
  [K in Field]: Fields[K]["category"] extends C ? K : never;
}[Field];
export type LookField = FieldsIn<"look">;
export type LayoutField = FieldsIn<"layout">;
export type PlacedField = FieldsIn<"look" | "layout" | "asset">;

const entries = Object.entries(CUSTOMIZATION_FIELDS) as [Field, FieldSpec][];

function fieldsIn<C extends FieldSpec["category"]>(category: C): FieldsIn<C>[] {
  return entries.filter(([, spec]) => spec.category === category).map(([field]) => field as FieldsIn<C>);
}

export const LOOK_FIELDS = fieldsIn("look");
export const LAYOUT_FIELDS = fieldsIn("layout");
export const PLACED_FIELDS: PlacedField[] = [...LOOK_FIELDS, ...LAYOUT_FIELDS, ...fieldsIn("asset")];

/** [field, key] pairs stored in one file section, in table order. */
export function sectionFields(section: string): [PlacedField, string][] {
  return entries.flatMap(([field, spec]) =>
    spec.category !== "personal" && spec.section === section
      ? [[field as PlacedField, spec.key] as [PlacedField, string]]
      : [],
  );
}

/** Turns a sanitizer path such as "uiFontSize" or "importedFonts[1].name" into its file path. */
export function mapIssuePath(path: string): string {
  const field = /^[A-Za-z]+/.exec(path)?.[0];
  const spec = field && Object.hasOwn(CUSTOMIZATION_FIELDS, field)
    ? (CUSTOMIZATION_FIELDS[field as Field] as FieldSpec)
    : null;
  if (!spec || spec.category === "personal") return path;
  const prefix = spec.key ? `${spec.section}.${spec.key}` : spec.section;
  return prefix + path.slice(field!.length);
}

/** The change group each customization field belongs to. */
export function changeOf(field: PlacedField): ThemeChange {
  return CUSTOMIZATION_FIELDS[field].change;
}

export function pickFields<K extends keyof AppearanceCustomization>(
  customization: AppearanceCustomization,
  keys: readonly K[],
): Pick<AppearanceCustomization, K> {
  const picked = {} as Pick<AppearanceCustomization, K>;
  for (const key of keys) picked[key] = customization[key];
  return picked;
}
