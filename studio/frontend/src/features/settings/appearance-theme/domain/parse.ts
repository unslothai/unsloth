// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type BackgroundAdjustments,
  DEFAULT_ADJUSTMENTS,
  sanitizeAdjustments,
  sanitizeLens,
  sanitizeMaterial,
  sanitizeStrength,
} from "../../../chat-background/model.ts";
import {
  DEFAULT_CUSTOMIZATION,
  sanitizeCustomizationWithIssues,
} from "../../lib/appearance-customization.ts";
import { type ColorThemeId, isColorThemeId } from "../../lib/color-themes.ts";
import {
  mapIssuePath,
  themeLayoutFromCustomization,
  themeLookFromCustomization,
} from "./classification.ts";
import { migrateThemeRaw } from "./migrations.ts";
import {
  type AppearanceTheme,
  CURRENT_THEME_SCHEMA_VERSION,
  DEFAULT_THEME_GLASS,
  MAX_BACKGROUND_DATA_URL_CHARS,
  MAX_THEME_AUTHOR_LENGTH,
  MAX_THEME_DESCRIPTION_LENGTH,
  MAX_THEME_FILE_CHARS,
  MAX_THEME_FILE_CHARS_NO_ASSETS,
  MAX_THEME_NAME_LENGTH,
  MAX_THEME_STUDIO_VERSION_LENGTH,
  THEME_FILE_KIND,
  type ThemeBackground,
  type ThemeGlass,
  type ThemeMeta,
  type ThemeParseError,
  type ThemeParseResult,
  type ThemeWarning,
} from "./schema.ts";

type Dict = Record<string, unknown>;

const DEFAULT_PALETTE: ColorThemeId = "standard";
const DEFAULT_THEME_NAME = "Untitled theme";
const MAX_PALETTE_WARNING_CHARS = 64;
const MAX_UNKNOWN_KEY_CHARS = 64;
const BACKGROUND_DATA_URL_PATTERN =
  /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/;

// Warnings list sections in the order a file has them.
const SECTION_ORDER = [
  "meta",
  "base",
  "colors",
  "typography",
  "code",
  "chat",
  "glass",
  "layout",
  "assets",
];

// Each map is a section's allow-list too. Keys are file names; values are customization names.
const TYPOGRAPHY_SECTION = {
  uiFont: "uiFont",
  headingFont: "headingFont",
  chatFont: "chatFont",
  codeFont: "codeFont",
  uiFontSize: "uiFontSize",
  codeFontSize: "codeFontSize",
  fontSmoothing: "fontSmoothing",
};
const CODE_SECTION = {
  lineNumbers: "codeLineNumbers",
  wordWrap: "codeWordWrap",
  ligatures: "codeLigatures",
};
const CHAT_SECTION = { width: "chatWidth", composerOutline: "composerOutline" };
const LAYOUT_SECTION = {
  sentAttachments: "sentAttachments",
  sidebarMenu: "sidebarMenu",
  sidebarNav: "sidebarNav",
  sidebarNavAuto: "sidebarNavAuto",
};

const ROOT_KEYS = [
  "$schema",
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
];
const META_KEYS = ["name", "author", "description", "studioVersion"];
const BASE_KEYS = ["palette"];
const COLORS_KEYS = ["light", "dark", "contrast"];
const MODE_KEYS = ["accent", "background", "foreground"];
const GLASS_KEYS = ["material", "strength", "lens", "menus"];
const STRENGTH_KEYS = ["light", "dark"];
const ASSETS_KEYS = ["fonts", "background"];
const BACKGROUND_KEYS = ["dataUrl", "adjustments"];

/** Reads only own properties, so a JSON "__proto__" key is never followed. */
function own(source: Dict, key: string): unknown {
  return Object.hasOwn(source, key) ? source[key] : undefined;
}

function isPlainObject(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A null field counts as absent, as it does for the customization sanitizer. */
function isPresent(value: unknown): boolean {
  return value !== undefined && value !== null;
}

function failure(error: ThemeParseError): ThemeParseResult {
  return { ok: false, error };
}

function reportUnknownKeys(
  section: Dict,
  allowed: readonly string[],
  prefix: string,
  warnings: ThemeWarning[],
): void {
  for (const key of Object.keys(section)) {
    if (!allowed.includes(key)) {
      warnings.push({
        kind: "unknown",
        path: `${prefix}${key.slice(0, MAX_UNKNOWN_KEY_CHARS)}`,
      });
    }
  }
}

/** Null when the section is absent or null; a non-object is reported and treated as absent. */
function readSection(
  parent: Dict,
  key: string,
  allowed: readonly string[],
  path: string,
  warnings: ThemeWarning[],
): Dict | null {
  const value = own(parent, key);
  if (!isPresent(value)) return null;
  if (!isPlainObject(value)) {
    warnings.push({ kind: "invalid", path });
    return null;
  }
  reportUnknownKeys(value, allowed, `${path}.`, warnings);
  return value;
}

/** Copies present fields (or every field when `all` is set) from the file to the raw customization. */
function copyFields(
  target: Dict,
  section: Dict,
  fields: Record<string, string>,
  all = false,
): void {
  for (const [key, field] of Object.entries(fields)) {
    if (all || Object.hasOwn(section, key)) target[field] = own(section, key);
  }
}

function readText(
  section: Dict | null,
  key: string,
  path: string,
  max: number,
  warnings: ThemeWarning[],
): string | undefined {
  const value = section ? own(section, key) : undefined;
  if (!isPresent(value)) return undefined;
  if (typeof value !== "string") {
    warnings.push({ kind: "invalid", path });
    return undefined;
  }
  const cleaned = value.replace(/\p{Cc}+/gu, " ").trim();
  // Two UTF-16 units per allowed code point is enough to tell whether the text is over the limit.
  const chars = Array.from(cleaned.slice(0, max * 2 + 2));
  if (chars.length > max) warnings.push({ kind: "clamped", path });
  const text = chars.slice(0, max).join("").trim();
  return text.length > 0 ? text : undefined;
}

function readMeta(section: Dict | null, warnings: ThemeWarning[]): ThemeMeta {
  const name = readText(
    section,
    "name",
    "meta.name",
    MAX_THEME_NAME_LENGTH,
    warnings,
  );
  const meta: ThemeMeta = { name: name ?? DEFAULT_THEME_NAME };
  const author = readText(
    section,
    "author",
    "meta.author",
    MAX_THEME_AUTHOR_LENGTH,
    warnings,
  );
  if (author !== undefined) meta.author = author;
  const description = readText(
    section,
    "description",
    "meta.description",
    MAX_THEME_DESCRIPTION_LENGTH,
    warnings,
  );
  if (description !== undefined) meta.description = description;
  const studioVersion = readText(
    section,
    "studioVersion",
    "meta.studioVersion",
    MAX_THEME_STUDIO_VERSION_LENGTH,
    warnings,
  );
  if (studioVersion !== undefined) meta.studioVersion = studioVersion;
  return meta;
}

function readPalette(
  base: Dict | null,
  warnings: ThemeWarning[],
): ColorThemeId {
  const value = base ? own(base, "palette") : undefined;
  if (!isPresent(value)) return DEFAULT_PALETTE;
  if (typeof value !== "string") {
    warnings.push({ kind: "invalid", path: "base.palette" });
    return DEFAULT_PALETTE;
  }
  if (isColorThemeId(value)) return value;
  warnings.push({
    kind: "unknownPalette",
    palette: value.slice(0, MAX_PALETTE_WARNING_CHARS),
  });
  return DEFAULT_PALETTE;
}

function readModeColors(
  colors: Dict,
  mode: "light" | "dark",
  warnings: ThemeWarning[],
): Dict | undefined {
  const section = readSection(
    colors,
    mode,
    MODE_KEYS,
    `colors.${mode}`,
    warnings,
  );
  if (!section) return undefined;
  return {
    accent: own(section, "accent"),
    background: own(section, "background"),
    foreground: own(section, "foreground"),
  };
}

function readStrength(
  value: unknown,
  path: string,
  fallback: number,
  warnings: ThemeWarning[],
): number {
  if (!isPresent(value)) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    warnings.push({ kind: "invalid", path });
    return fallback;
  }
  const next = sanitizeStrength(value, fallback);
  if (next !== value) warnings.push({ kind: "clamped", path });
  return next;
}

function readGlass(section: Dict | null, warnings: ThemeWarning[]): ThemeGlass {
  const glass: ThemeGlass = {
    material: DEFAULT_THEME_GLASS.material,
    strength: { ...DEFAULT_THEME_GLASS.strength },
    lens: DEFAULT_THEME_GLASS.lens,
    menus: DEFAULT_THEME_GLASS.menus,
  };
  if (!section) return glass;

  const material = own(section, "material");
  if (isPresent(material)) {
    glass.material = sanitizeMaterial(material);
    if (glass.material !== material) {
      warnings.push({ kind: "invalid", path: "glass.material" });
    }
  }

  const strength = own(section, "strength");
  if (isPresent(strength)) {
    if (isPlainObject(strength)) {
      reportUnknownKeys(strength, STRENGTH_KEYS, "glass.strength.", warnings);
      glass.strength = {
        light: readStrength(
          own(strength, "light"),
          "glass.strength.light",
          DEFAULT_THEME_GLASS.strength.light,
          warnings,
        ),
        dark: readStrength(
          own(strength, "dark"),
          "glass.strength.dark",
          DEFAULT_THEME_GLASS.strength.dark,
          warnings,
        ),
      };
    } else {
      warnings.push({ kind: "invalid", path: "glass.strength" });
    }
  }

  const lens = own(section, "lens");
  if (isPresent(lens)) {
    glass.lens = sanitizeLens(lens);
    if (glass.lens !== lens)
      warnings.push({ kind: "invalid", path: "glass.lens" });
  }

  const menus = own(section, "menus");
  if (isPresent(menus)) {
    if (typeof menus === "boolean") glass.menus = menus;
    else warnings.push({ kind: "invalid", path: "glass.menus" });
  }
  return glass;
}

/** Only the keys this build knows are copied. sanitizeAdjustments ignores the rest. */
function readAdjustments(
  value: unknown,
  warnings: ThemeWarning[],
): BackgroundAdjustments {
  if (!isPresent(value)) return sanitizeAdjustments({});
  if (!isPlainObject(value)) {
    warnings.push({ kind: "invalid", path: "assets.background.adjustments" });
    return sanitizeAdjustments({});
  }
  const picked: Dict = {};
  for (const key of Object.keys(DEFAULT_ADJUSTMENTS)) {
    if (Object.hasOwn(value, key)) picked[key] = own(value, key);
  }
  return sanitizeAdjustments(picked as Partial<BackgroundAdjustments>);
}

function readBackground(
  value: unknown,
  warnings: ThemeWarning[],
): ThemeBackground | null {
  if (!isPresent(value)) return null;
  if (!isPlainObject(value)) {
    warnings.push({ kind: "backgroundSkipped" });
    return null;
  }
  reportUnknownKeys(value, BACKGROUND_KEYS, "assets.background.", warnings);
  const dataUrl = own(value, "dataUrl");
  if (
    typeof dataUrl !== "string" ||
    dataUrl.length > MAX_BACKGROUND_DATA_URL_CHARS ||
    !BACKGROUND_DATA_URL_PATTERN.test(dataUrl)
  ) {
    warnings.push({ kind: "backgroundSkipped" });
    return null;
  }
  return {
    dataUrl,
    adjustments: readAdjustments(own(value, "adjustments"), warnings),
  };
}

function sectionRank(warning: ThemeWarning): number {
  if (warning.kind === "newerVersion") return 0;
  if (warning.kind === "unknownPalette") return SECTION_ORDER.indexOf("base");
  if (warning.kind === "backgroundSkipped") {
    return SECTION_ORDER.indexOf("assets");
  }
  const rank = SECTION_ORDER.indexOf(warning.path.split(/[.[]/, 1)[0]);
  return rank === -1 ? SECTION_ORDER.length : rank;
}

/**
 * Parses and sanitizes a theme file. Never throws on hostile input: every field is read through an
 * own-property check, and anything wrong falls back to a default with a warning.
 */
export function parseThemeFile(text: string): ThemeParseResult {
  if (text.length > MAX_THEME_FILE_CHARS) return failure("tooLarge");

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return failure("invalidJson");
  }
  if (!isPlainObject(parsed) || own(parsed, "kind") !== THEME_FILE_KIND) {
    return failure("notTheme");
  }
  const version = own(parsed, "schemaVersion");
  if (
    typeof version !== "number" ||
    !Number.isInteger(version) ||
    version < 1
  ) {
    return failure("notTheme");
  }
  if (
    text.length > MAX_THEME_FILE_CHARS_NO_ASSETS &&
    !Object.hasOwn(parsed, "assets")
  ) {
    return failure("tooLarge");
  }

  const rootWarnings: ThemeWarning[] = [];
  let root = parsed;
  if (version > CURRENT_THEME_SCHEMA_VERSION) {
    rootWarnings.push({ kind: "newerVersion" });
  } else if (version < CURRENT_THEME_SCHEMA_VERSION) {
    root = migrateThemeRaw(parsed, version);
  }
  reportUnknownKeys(root, ROOT_KEYS, "", rootWarnings);

  const warnings: ThemeWarning[] = [];
  const meta = readMeta(
    readSection(root, "meta", META_KEYS, "meta", warnings),
    warnings,
  );
  const palette = readPalette(
    readSection(root, "base", BASE_KEYS, "base", warnings),
    warnings,
  );
  const colors = readSection(root, "colors", COLORS_KEYS, "colors", warnings);
  const typography = readSection(
    root,
    "typography",
    Object.keys(TYPOGRAPHY_SECTION),
    "typography",
    warnings,
  );
  const code = readSection(
    root,
    "code",
    Object.keys(CODE_SECTION),
    "code",
    warnings,
  );
  const chat = readSection(
    root,
    "chat",
    Object.keys(CHAT_SECTION),
    "chat",
    warnings,
  );
  const glassSection = readSection(
    root,
    "glass",
    GLASS_KEYS,
    "glass",
    warnings,
  );
  const layout = readSection(
    root,
    "layout",
    Object.keys(LAYOUT_SECTION),
    "layout",
    warnings,
  );
  const assets = readSection(root, "assets", ASSETS_KEYS, "assets", warnings);

  // Start from the defaults, then copy only the fields the file actually has.
  const raw: Dict = { ...DEFAULT_CUSTOMIZATION };
  if (colors) {
    raw.colors = {
      light: readModeColors(colors, "light", warnings),
      dark: readModeColors(colors, "dark", warnings),
    };
    copyFields(raw, colors, { contrast: "contrast" });
  }
  if (typography) copyFields(raw, typography, TYPOGRAPHY_SECTION);
  if (code) copyFields(raw, code, CODE_SECTION);
  if (chat) copyFields(raw, chat, CHAT_SECTION);
  // All layout fields are copied, absent ones as undefined, so the sanitizer derives sidebarNavAuto.
  if (layout) copyFields(raw, layout, LAYOUT_SECTION, true);
  const fontsValue = assets ? own(assets, "fonts") : undefined;
  const hasFonts = isPresent(fontsValue);
  if (hasFonts) raw.importedFonts = fontsValue;

  const customization = sanitizeCustomizationWithIssues(raw);
  for (const issue of customization.issues) {
    warnings.push({ kind: issue.kind, path: mapIssuePath(issue.path) });
  }

  const glass = readGlass(glassSection, warnings);
  const background = assets
    ? readBackground(own(assets, "background"), warnings)
    : null;

  const ordered = [
    ...rootWarnings,
    ...[...warnings].sort((a, b) => sectionRank(a) - sectionRank(b)),
  ];
  const unique = [
    ...new Map(
      ordered.map((warning): [string, ThemeWarning] => [
        JSON.stringify(warning),
        warning,
      ]),
    ).values(),
  ];

  const theme: AppearanceTheme = {
    meta,
    palette,
    look: themeLookFromCustomization(customization.value),
    glass,
    layout: layout ? themeLayoutFromCustomization(customization.value) : null,
    fonts: hasFonts ? customization.value.importedFonts : null,
    background,
  };
  return { ok: true, theme, warnings: unique };
}
