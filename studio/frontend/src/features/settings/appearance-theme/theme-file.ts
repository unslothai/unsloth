// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type BackgroundAdjustments,
  DEFAULT_GLASS,
  type GlassSettings,
  sanitizeAdjustments,
  sanitizeLens,
  sanitizeMaterial,
  sanitizeScope,
  sanitizeStrength,
  type WindowMaterial,
} from "../../chat-background/model.ts";
import { type ColorThemeId, isColorThemeId } from "../lib/color-themes.ts";
import {
  type AppearanceCustomization,
  DEFAULT_CUSTOMIZATION,
  type ImportedFont,
  sanitizeCustomization,
} from "../stores/appearance-custom-store.ts";

export const THEME_FILE_KIND = "unsloth-studio.appearance";
/** Bump only for a change older builds would misread; newer files are read best-effort. */
export const THEME_SCHEMA_VERSION = 1;
export const THEME_FILE_EXTENSION = ".unsloth-theme.json";
export const MAX_THEME_FILE_CHARS = 8_000_000;
const MAX_SETTINGS_ONLY_CHARS = 262_144;
export const MAX_BACKGROUND_DATA_URL_CHARS = 7_000_000;
export const MAX_THEME_NAME = 80;
const MAX_WARNINGS = 50;
const BACKGROUND_DATA_URL =
  /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/;

/** Where each customization field goes; `satisfies` fails the typecheck for a new field without one. */
const FIELD_GROUP = {
  colors: "colors",
  contrast: "colors",
  elementOutline: "colors",
  uiFont: "typography",
  headingFont: "typography",
  chatFont: "typography",
  codeFont: "typography",
  uiFontSize: "typography",
  codeFontSize: "typography",
  fontSmoothing: "typography",
  codeLineNumbers: "code",
  codeWordWrap: "code",
  codeLigatures: "code",
  chatWidth: "chat",
  composerOutline: "chat",
  sentAttachments: "layout",
  sidebarMenu: "layout",
  sidebarNav: "layout",
  sidebarNavAuto: "layout",
  importedFonts: "fonts",
  pointerCursors: "personal",
  reduceMotion: "personal",
} as const satisfies Record<keyof AppearanceCustomization, string>;

type Field = keyof typeof FIELD_GROUP;
const fieldsIn = (...groups: string[]) =>
  (Object.keys(FIELD_GROUP) as Field[]).filter((field) =>
    groups.includes(FIELD_GROUP[field]),
  );
const LOOK_FIELDS = fieldsIn("colors", "typography", "code", "chat");
const LAYOUT_FIELDS = fieldsIn("layout");
export const FONT_FIELDS = [
  "uiFont",
  "headingFont",
  "chatFont",
  "codeFont",
] as const;

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

export type ThemeBackground = {
  dataUrl: string;
  adjustments: BackgroundAdjustments;
};
export type AppearanceTheme = {
  name: string;
  palette: ColorThemeId;
  /** Defaults plus the file's look (and layout, when `hasLayout`) fields. */
  customization: AppearanceCustomization;
  hasLayout: boolean;
  glass: GlassSettings;
  fonts: ImportedFont[] | null;
  background: ThemeBackground | null;
};
export type ThemeWarning =
  | { kind: "newerVersion" | "backgroundSkipped" }
  | { kind: "unknownPalette"; palette: string }
  | { kind: "unknown" | "invalid"; path: string }
  | { kind: "more"; count: number };
export type ThemeParseResult =
  | { ok: true; theme: AppearanceTheme; warnings: ThemeWarning[] }
  | { ok: false; error: "tooLarge" | "invalid" };

/** The live look a theme replaces. `style` is the glass look the on/off switch remembers. */
export type AppearanceSnapshot = {
  palette: ColorThemeId;
  customization: AppearanceCustomization;
  glass: GlassSettings & { style: Exclude<WindowMaterial, "solid"> };
  background: { adjustments: BackgroundAdjustments; blob: Blob } | null;
};

/** Font families match case-insensitively, as in CSS and the font uploader. */
export const fontKey = (name: string | null | undefined) =>
  name?.toLowerCase() ?? "";

type Dict = Record<string, unknown>;
const isDict = (value: unknown): value is Dict =>
  typeof value === "object" && value !== null && !Array.isArray(value);
/** Own properties only, so a JSON "__proto__" key is never followed. */
const own = (source: unknown, key: string) =>
  isDict(source) && Object.hasOwn(source, key) ? source[key] : undefined;
const same = (a: unknown, b: unknown) =>
  JSON.stringify(a) === JSON.stringify(b);
/** Whether the sanitizer kept every value the file gave (objects may be partial; hex case is ignored). */
const kept = (sanitized: unknown, given: unknown): boolean =>
  isDict(given) && isDict(sanitized)
    ? Object.keys(given).every((key) => kept(own(sanitized, key), given[key]))
    : JSON.stringify(sanitized)?.toLowerCase() ===
      JSON.stringify(given)?.toLowerCase();
const pick = (c: AppearanceCustomization, fields: readonly Field[]) =>
  Object.fromEntries(
    fields.map((field) => [field, c[field]]),
  ) as Partial<AppearanceCustomization>;

/** Control, bidi-override and zero-width characters that could make a name read as another. */
const HIDDEN_CHARS =
  /[\p{Cc}\u061C\u200B\u200C\u200E\u200F\u202A-\u202E\u2060-\u2069\uFEFF]+/gu;
export const cleanThemeName = (value: string) =>
  Array.from(
    value
      .slice(0, MAX_THEME_NAME * 2)
      .replace(HIDDEN_CHARS, " ")
      .trim(),
  )
    .slice(0, MAX_THEME_NAME)
    .join("")
    .trim();

/** Personal fields, the theme mode and the interface scale never leave the device. */
export function captureTheme(
  snapshot: AppearanceSnapshot,
  opts: {
    name: string;
    includeLayout: boolean;
    includeAssets: boolean;
    backgroundDataUrl?: string | null;
  },
): Dict {
  const c = snapshot.customization;
  const look = pick(c, LOOK_FIELDS);
  if (!opts.includeAssets) {
    // A name that points at a font left out of the file would mean nothing elsewhere.
    const omitted = new Set(c.importedFonts.map((font) => fontKey(font.name)));
    for (const field of FONT_FIELDS)
      if (omitted.has(fontKey(look[field]))) look[field] = null;
  }
  const { material, strength, lens, menus, scope } = snapshot.glass;
  const file: Dict = {
    kind: THEME_FILE_KIND,
    schemaVersion: THEME_SCHEMA_VERSION,
    name: cleanThemeName(opts.name) || "Theme",
    palette: snapshot.palette,
    look,
    glass: { material, strength, lens, menus, scope },
  };
  if (opts.includeLayout) file.layout = pick(c, LAYOUT_FIELDS);
  if (opts.includeAssets && c.importedFonts.length > 0)
    file.fonts = c.importedFonts;
  if (opts.includeAssets && opts.backgroundDataUrl && snapshot.background) {
    file.background = {
      dataUrl: opts.backgroundDataUrl,
      adjustments: snapshot.background.adjustments,
    };
  }
  return file;
}

/** Never throws on hostile input: anything wrong falls back to a default with a warning. */
export function parseThemeFile(text: string): ThemeParseResult {
  if (text.length > MAX_THEME_FILE_CHARS)
    return { ok: false, error: "tooLarge" };
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch {
    return { ok: false, error: "invalid" };
  }
  const version = own(root, "schemaVersion");
  if (
    own(root, "kind") !== THEME_FILE_KIND ||
    !Number.isInteger(version) ||
    (version as number) < 1
  ) {
    return { ok: false, error: "invalid" };
  }
  // Only fonts or a background justify a big file.
  const fontsValue = own(root, "fonts");
  const carriesAssets =
    (Array.isArray(fontsValue) && fontsValue.length > 0) ||
    isDict(own(root, "background"));
  if (text.length > MAX_SETTINGS_ONLY_CHARS && !carriesAssets)
    return { ok: false, error: "tooLarge" };

  const warnings: ThemeWarning[] = [];
  let hidden = 0;
  const warn = (warning: ThemeWarning) => {
    if (warnings.length < MAX_WARNINGS) warnings.push(warning);
    else hidden += 1;
  };
  const object = (
    value: unknown,
    path: string,
    known: readonly string[],
  ): Dict | null => {
    if (value === undefined || value === null) return null;
    if (!isDict(value)) {
      warn({ kind: "invalid", path });
      return null;
    }
    for (const key of Object.keys(value)) {
      if (!known.includes(key))
        warn({
          kind: "unknown",
          path: `${path ? `${path}.` : ""}${key.slice(0, 64)}`,
        });
    }
    return value;
  };
  const read = <T>(
    value: unknown,
    fallback: T,
    sanitize: (value: unknown) => T,
    path: string,
  ): T => {
    if (value === undefined || value === null) return fallback;
    const next = sanitize(value);
    if (!kept(next, value)) warn({ kind: "invalid", path });
    return next;
  };

  if ((version as number) > THEME_SCHEMA_VERSION)
    warn({ kind: "newerVersion" });
  object(root, "", [
    "$schema",
    "kind",
    "schemaVersion",
    "name",
    "palette",
    "look",
    "glass",
    "layout",
    "fonts",
    "background",
  ]);

  const rawName = own(root, "name");
  const name = typeof rawName === "string" ? cleanThemeName(rawName) : "";
  if (rawName != null && typeof rawName !== "string")
    warn({ kind: "invalid", path: "name" });

  const rawPalette = own(root, "palette");
  let palette: ColorThemeId = "standard";
  if (typeof rawPalette === "string" && isColorThemeId(rawPalette))
    palette = rawPalette;
  else if (typeof rawPalette === "string")
    warn({ kind: "unknownPalette", palette: rawPalette.slice(0, 64) });
  else if (rawPalette != null) warn({ kind: "invalid", path: "palette" });

  const raw: Dict = { ...DEFAULT_CUSTOMIZATION };
  const given: [Field, string][] = [];
  const look = object(own(root, "look"), "look", LOOK_FIELDS);
  for (const field of LOOK_FIELDS) {
    const value = own(look, field);
    if (value === undefined || value === null) continue;
    raw[field] = value;
    given.push([field, `look.${field}`]);
  }
  const layout = object(own(root, "layout"), "layout", LAYOUT_FIELDS);
  // Every layout field is copied, absent ones too, so sidebarNavAuto follows sidebarNav.
  for (const field of layout ? LAYOUT_FIELDS : []) {
    raw[field] = own(layout, field);
    given.push([field, `layout.${field}`]);
  }
  const fonts = Array.isArray(fontsValue) ? fontsValue : null;
  if (fontsValue != null && !fonts) warn({ kind: "invalid", path: "fonts" });
  if (fonts) raw.importedFonts = fonts;

  const customization = sanitizeCustomization(raw);
  for (const [field, path] of given) {
    if (raw[field] != null && !kept(customization[field], raw[field]))
      warn({ kind: "invalid", path });
  }
  // Two names differing only in case are one CSS family; the first wins.
  const seen = new Set<string>();
  const themeFonts = customization.importedFonts.filter(
    (font) => !seen.has(fontKey(font.name)) && seen.add(fontKey(font.name)),
  );
  if (fonts && themeFonts.length !== fonts.length)
    warn({ kind: "invalid", path: "fonts" });

  const glassIn = object(own(root, "glass"), "glass", [
    "material",
    "strength",
    "lens",
    "menus",
    "scope",
  ]);
  const strengthIn = object(own(glassIn, "strength"), "glass.strength", [
    "light",
    "dark",
  ]);
  const strength = (mode: "light" | "dark") =>
    read(
      own(strengthIn, mode),
      DEFAULT_GLASS.strength[mode],
      (v) => sanitizeStrength(v, DEFAULT_GLASS.strength[mode]),
      `glass.strength.${mode}`,
    );
  const glass: GlassSettings = {
    material: read(
      own(glassIn, "material"),
      DEFAULT_GLASS.material,
      sanitizeMaterial,
      "glass.material",
    ),
    strength: { light: strength("light"), dark: strength("dark") },
    lens: read(
      own(glassIn, "lens"),
      DEFAULT_GLASS.lens,
      sanitizeLens,
      "glass.lens",
    ),
    menus: read(
      own(glassIn, "menus"),
      DEFAULT_GLASS.menus,
      (v) => (typeof v === "boolean" ? v : DEFAULT_GLASS.menus),
      "glass.menus",
    ),
    scope: read(
      own(glassIn, "scope"),
      DEFAULT_GLASS.scope,
      (v) => sanitizeScope(v, DEFAULT_GLASS.scope),
      "glass.scope",
    ),
  };

  let background: ThemeBackground | null = null;
  const backgroundIn = object(own(root, "background"), "background", [
    "dataUrl",
    "adjustments",
  ]);
  const dataUrl = own(backgroundIn, "dataUrl");
  if (
    backgroundIn &&
    typeof dataUrl === "string" &&
    dataUrl.length <= MAX_BACKGROUND_DATA_URL_CHARS &&
    BACKGROUND_DATA_URL.test(dataUrl)
  ) {
    const given = own(backgroundIn, "adjustments");
    const adjustments = sanitizeAdjustments(isDict(given) ? given : {});
    if (given != null && !kept(adjustments, given)) {
      warn({ kind: "invalid", path: "background.adjustments" });
    }
    background = { dataUrl, adjustments };
  } else if (own(root, "background") != null) {
    warn({ kind: "backgroundSkipped" });
  }

  return {
    ok: true,
    theme: {
      name,
      palette,
      customization: { ...customization, importedFonts: themeFonts },
      hasLayout: layout !== null,
      glass,
      fonts: fonts ? themeFonts : null,
      background,
    },
    warnings:
      hidden > 0 ? [...warnings, { kind: "more", count: hidden }] : warnings,
  };
}

/** Families `c` selects without importing them: installed on this device. */
function localFamilies(c: AppearanceCustomization): Set<string> {
  const imported = new Set(c.importedFonts.map((font) => fontKey(font.name)));
  return new Set(
    FONT_FIELDS.map((field) => fontKey(c[field])).filter(
      (name) => name && !imported.has(name),
    ),
  );
}

export type ThemeApplyPlan = Omit<AppearanceSnapshot, "background"> & {
  background: "keep" | ThemeBackground;
  changes: ThemeChange[];
  /** Current fonts that no longer fit under the imported-font cap. */
  droppedFonts: string[];
};

/** Pure. Look fields the file leaves out reset to defaults; personal fields stay as they are. */
export function planThemeApply(
  current: AppearanceSnapshot,
  theme: AppearanceTheme,
  opts: { applyLayout: boolean; applyAssets: boolean },
): ThemeApplyPlan {
  const picked = {
    ...current.customization,
    ...pick(theme.customization, LOOK_FIELDS),
    ...(opts.applyLayout && theme.hasLayout
      ? pick(theme.customization, LAYOUT_FIELDS)
      : {}),
  };
  const themeFonts = opts.applyAssets ? (theme.fonts ?? []) : [];
  const replaced = new Set(themeFonts.map((font) => fontKey(font.name)));
  const selected = new Set(FONT_FIELDS.map((field) => fontKey(picked[field])));
  const merged = [
    ...themeFonts,
    ...current.customization.importedFonts.filter(
      (font) => !replaced.has(fontKey(font.name)),
    ),
  ];
  // Selected fonts first, so the cap drops unused ones before any in use.
  const inUse = (font: ImportedFont) => selected.has(fontKey(font.name));
  const customization = sanitizeCustomization({
    ...picked,
    importedFonts: [
      ...merged.filter(inUse),
      ...merged.filter((f) => !inUse(f)),
    ],
  });

  // A selection takes its imported font's own spelling; one whose imported font is gone resets.
  const kept = new Map(
    customization.importedFonts.map((font) => [fontKey(font.name), font.name]),
  );
  const local = localFamilies(current.customization);
  const imported = new Set(
    [...(theme.fonts ?? []), ...current.customization.importedFonts].map(
      (font) => fontKey(font.name),
    ),
  );
  for (const field of FONT_FIELDS) {
    const key = fontKey(customization[field]);
    if (kept.has(key)) customization[field] = kept.get(key)!;
    else if (imported.has(key) && !local.has(key)) customization[field] = null;
  }

  const { material } = theme.glass;
  const glass = {
    ...theme.glass,
    style: material === "solid" ? current.glass.style : material,
  };
  const background =
    opts.applyAssets && theme.background ? theme.background : "keep";
  const changed = new Set<ThemeChange>();
  if (current.palette !== theme.palette) changed.add("palette");
  for (const field of Object.keys(FIELD_GROUP) as Field[]) {
    const group = FIELD_GROUP[field];
    if (
      group !== "personal" &&
      !same(current.customization[field], customization[field])
    )
      changed.add(group);
  }
  const glassKey = (g: GlassSettings) =>
    JSON.stringify([
      g.material,
      g.strength.light,
      g.strength.dark,
      g.lens,
      g.menus,
      g.scope,
    ]);
  if (glassKey(current.glass) !== glassKey(glass)) changed.add("glass");
  if (background !== "keep") changed.add("background");
  return {
    palette: theme.palette,
    customization,
    glass,
    background,
    changes: THEME_CHANGES.filter((change) => changed.has(change)),
    droppedFonts: merged
      .filter((font) => !kept.has(fontKey(font.name)))
      .map((font) => font.name),
  };
}

/** Undo: the look from before the import, keeping personal settings as they are now. */
export function undoTarget(
  before: AppearanceSnapshot,
  now: AppearanceSnapshot,
): AppearanceSnapshot {
  return {
    ...before,
    customization: {
      ...before.customization,
      ...pick(now.customization, fieldsIn("personal")),
    },
  };
}
