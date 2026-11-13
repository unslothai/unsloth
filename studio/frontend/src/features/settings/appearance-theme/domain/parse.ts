// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type BackgroundAdjustments,
  DEFAULT_ADJUSTMENTS,
  DEFAULT_GLASS,
  type GlassSettings,
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
  LAYOUT_FIELDS,
  LOOK_FIELDS,
  mapIssuePath,
  pickFields,
  sectionFields,
} from "./classification.ts";
import {
  type AppearanceTheme,
  CURRENT_THEME_SCHEMA_VERSION,
  cleanThemeText,
  DEFAULT_THEME_NAME,
  MAX_BACKGROUND_DATA_URL_CHARS,
  MAX_THEME_FILE_CHARS,
  MAX_THEME_FILE_CHARS_NO_ASSETS,
  MAX_THEME_WARNINGS,
  META_LIMITS,
  THEME_FILE_KIND,
  type ThemeBackground,
  type ThemeMeta,
  type ThemeParseResult,
  type ThemeWarning,
} from "./schema.ts";

type Dict = Record<string, unknown>;
type Warn = (warning: ThemeWarning) => void;

const BACKGROUND_DATA_URL = /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/;
const CUSTOMIZATION_SECTIONS = ["colors", "typography", "code", "chat", "layout"];
const MODE_KEYS = ["accent", "background", "foreground"];
const keysOf = (section: string) => sectionFields(section).map(([, key]) => key);

/** Keys each object may hold, by path; anything else is reported as unknown. */
const KNOWN_KEYS: Record<string, readonly string[]> = {
  "": ["$schema", "kind", "schemaVersion", "meta", "base", "glass", "assets", ...CUSTOMIZATION_SECTIONS],
  meta: Object.keys(META_LIMITS),
  base: ["palette"],
  colors: ["light", "dark", ...keysOf("colors").filter(Boolean)],
  "colors.light": MODE_KEYS,
  "colors.dark": MODE_KEYS,
  typography: keysOf("typography"),
  code: keysOf("code"),
  chat: keysOf("chat"),
  layout: keysOf("layout"),
  glass: ["material", "strength", "lens", "menus"],
  "glass.strength": ["light", "dark"],
  assets: ["fonts", "background"],
  "assets.background": ["dataUrl", "adjustments"],
  "assets.background.adjustments": Object.keys(DEFAULT_ADJUSTMENTS),
};

/** Reads only own properties, so a JSON "__proto__" key is never followed. */
function own(source: unknown, key: string): unknown {
  return isDict(source) && Object.hasOwn(source, key) ? source[key] : undefined;
}

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readObject(value: unknown, path: string, warn: Warn): Dict | null {
  if (value === undefined || value === null) return null;
  if (!isDict(value)) {
    warn({ kind: "invalid", path });
    return null;
  }
  const known = KNOWN_KEYS[path] ?? [];
  for (const key of Object.keys(value)) {
    if (!known.includes(key)) {
      warn({ kind: "unknown", path: `${path ? `${path}.` : ""}${key.slice(0, 64)}` });
    }
  }
  return value;
}

/** A present value the sanitizer had to change is reported; absent means the fallback. */
function readValue<T>(
  value: unknown,
  fallback: T,
  sanitize: (value: unknown) => T,
  path: string,
  warn: Warn,
): T {
  if (value === undefined || value === null) return fallback;
  const next = sanitize(value);
  if (next !== value) {
    warn({ kind: typeof value === "number" && Number.isFinite(value) ? "clamped" : "invalid", path });
  }
  return next;
}

function readMeta(section: Dict | null, warn: Warn): ThemeMeta {
  const meta: ThemeMeta = { name: DEFAULT_THEME_NAME };
  for (const key of ["name", "author", "description"] as const) {
    const value = own(section, key);
    if (value === undefined || value === null) continue;
    if (typeof value !== "string") {
      warn({ kind: "invalid", path: `meta.${key}` });
      continue;
    }
    const { text, clamped } = cleanThemeText(value, META_LIMITS[key]);
    if (clamped) warn({ kind: "clamped", path: `meta.${key}` });
    if (text) meta[key] = text;
  }
  return meta;
}

function readPalette(base: Dict | null, warn: Warn): ColorThemeId {
  const value = own(base, "palette");
  if (value === undefined || value === null) return "standard";
  if (typeof value === "string" && isColorThemeId(value)) return value;
  warn(
    typeof value === "string"
      ? { kind: "unknownPalette", palette: value.slice(0, 64) }
      : { kind: "invalid", path: "base.palette" },
  );
  return "standard";
}

function readGlass(section: Dict | null, warn: Warn): GlassSettings {
  const strength = readObject(own(section, "strength"), "glass.strength", warn);
  const strengthOf = (mode: "light" | "dark") =>
    readValue(
      own(strength, mode),
      DEFAULT_GLASS.strength[mode],
      (value) => sanitizeStrength(value, DEFAULT_GLASS.strength[mode]),
      `glass.strength.${mode}`,
      warn,
    );
  return {
    material: readValue(own(section, "material"), DEFAULT_GLASS.material, sanitizeMaterial, "glass.material", warn),
    strength: { light: strengthOf("light"), dark: strengthOf("dark") },
    lens: readValue(own(section, "lens"), DEFAULT_GLASS.lens, sanitizeLens, "glass.lens", warn),
    menus: readValue(
      own(section, "menus"),
      DEFAULT_GLASS.menus,
      (value) => (typeof value === "boolean" ? value : DEFAULT_GLASS.menus),
      "glass.menus",
      warn,
    ),
  };
}

function readBackground(assets: Dict | null, warn: Warn): ThemeBackground | null {
  const value = own(assets, "background");
  if (value === undefined || value === null) return null;
  const background = readObject(value, "assets.background", warn);
  const dataUrl = own(background, "dataUrl");
  if (
    typeof dataUrl !== "string" ||
    dataUrl.length > MAX_BACKGROUND_DATA_URL_CHARS ||
    !BACKGROUND_DATA_URL.test(dataUrl)
  ) {
    warn({ kind: "backgroundSkipped" });
    return null;
  }
  // sanitizeAdjustments reads only the keys it knows; report any value it had to change.
  const given = readObject(own(background, "adjustments"), "assets.background.adjustments", warn) ?? {};
  const adjustments = sanitizeAdjustments(given);
  for (const key of Object.keys(DEFAULT_ADJUSTMENTS) as (keyof BackgroundAdjustments)[]) {
    readValue(own(given, key), adjustments[key], () => adjustments[key], `assets.background.adjustments.${key}`, warn);
  }
  return { dataUrl, adjustments };
}

/** Never throws on hostile input: anything wrong falls back to a default with a warning. */
export function parseThemeFile(text: string): ThemeParseResult {
  if (text.length > MAX_THEME_FILE_CHARS) return { ok: false, error: "tooLarge" };
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch {
    return { ok: false, error: "invalidJson" };
  }
  const version = isDict(root) ? own(root, "schemaVersion") : undefined;
  if (
    !isDict(root) ||
    own(root, "kind") !== THEME_FILE_KIND ||
    typeof version !== "number" ||
    !Number.isInteger(version) ||
    version < 1
  ) {
    return { ok: false, error: "notTheme" };
  }
  // Only fonts or a background justify a big file; an empty "assets" key does not.
  const rootAssets = own(root, "assets");
  const rootFonts = own(rootAssets, "fonts");
  const carriesAssets =
    isDict(rootAssets) &&
    ((Array.isArray(rootFonts) && rootFonts.length > 0) || isDict(own(rootAssets, "background")));
  if (text.length > MAX_THEME_FILE_CHARS_NO_ASSETS && !carriesAssets) {
    return { ok: false, error: "tooLarge" };
  }

  const warnings: ThemeWarning[] = [];
  let hidden = 0;
  const warn: Warn = (warning) => {
    if (warnings.length < MAX_THEME_WARNINGS) warnings.push(warning);
    else hidden += 1;
  };
  if (version > CURRENT_THEME_SCHEMA_VERSION) warn({ kind: "newerVersion" });
  readObject(root, "", warn);
  const section = (key: string, parent: Dict | null = root, path = key) =>
    readObject(own(parent, key), path, warn);

  const meta = readMeta(section("meta"), warn);
  const palette = readPalette(section("base"), warn);

  const raw: Dict = { ...DEFAULT_CUSTOMIZATION };
  let layoutPresent = false;
  for (const name of CUSTOMIZATION_SECTIONS) {
    const values = section(name);
    if (!values) continue;
    if (name === "colors") {
      raw.colors = {
        light: section("light", values, "colors.light"),
        dark: section("dark", values, "colors.dark"),
      };
    }
    // Every layout field is copied, absent ones as undefined, so sidebarNavAuto follows sidebarNav.
    if (name === "layout") layoutPresent = true;
    for (const [field, key] of sectionFields(name)) {
      if (key && (name === "layout" || Object.hasOwn(values, key))) raw[field] = values[key];
    }
  }
  const assets = section("assets");
  const fonts = own(assets, "fonts");
  const fontsPresent = fonts !== undefined && fonts !== null;
  if (fontsPresent) raw.importedFonts = fonts;

  const customization = sanitizeCustomizationWithIssues(raw);
  for (const issue of customization.issues) {
    warn({ kind: issue.kind, path: mapIssuePath(issue.path) });
  }
  hidden += customization.omittedIssues;
  const glass = readGlass(section("glass"), warn);
  const background = readBackground(assets, warn);

  const theme: AppearanceTheme = {
    meta,
    palette,
    look: pickFields(customization.value, LOOK_FIELDS),
    glass,
    layout: layoutPresent ? pickFields(customization.value, LAYOUT_FIELDS) : null,
    fonts: fontsPresent ? customization.value.importedFonts : null,
    background,
  };
  return {
    ok: true,
    theme,
    warnings: hidden > 0 ? [...warnings, { kind: "more", count: hidden }] : warnings,
  };
}
