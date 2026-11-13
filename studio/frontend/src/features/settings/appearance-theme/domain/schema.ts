// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  BackgroundAdjustments,
  GlassSettings,
} from "../../../chat-background/model.ts";
import type {
  AppearanceCustomization,
  CustomizationIssueKind,
  ImportedFont,
} from "../../lib/appearance-customization.ts";
import type { ColorThemeId } from "../../lib/color-themes.ts";
import type { LayoutField, LookField } from "./classification.ts";

export const THEME_FILE_KIND = "unsloth-studio.appearance";
/** A newer version is read best-effort; bump it only for a change older builds would misread. */
export const CURRENT_THEME_SCHEMA_VERSION = 1;
export const THEME_FILE_EXTENSION = ".unsloth-theme.json";
export const MAX_THEME_FILE_CHARS = 8_000_000;
export const MAX_THEME_FILE_CHARS_NO_ASSETS = 262_144;
export const MAX_BACKGROUND_DATA_URL_CHARS = 7_000_000;
export const MAX_THEME_WARNINGS = 50;
export const DEFAULT_THEME_NAME = "Untitled theme";
export const META_LIMITS = { name: 80, author: 80, description: 280 } as const;

export type ThemeMeta = { name: string; author?: string; description?: string };
export type ThemeLook = Pick<AppearanceCustomization, LookField>;
export type ThemeLayout = Pick<AppearanceCustomization, LayoutField>;
export type ThemeBackground = { dataUrl: string; adjustments: BackgroundAdjustments };

export type AppearanceTheme = {
  meta: ThemeMeta;
  palette: ColorThemeId;
  look: ThemeLook;
  glass: GlassSettings;
  layout: ThemeLayout | null;
  fonts: ImportedFont[] | null;
  background: ThemeBackground | null;
};

export type ThemeWarning =
  | { kind: "newerVersion" }
  | { kind: "unknownPalette"; palette: string }
  | { kind: "backgroundSkipped" }
  | { kind: CustomizationIssueKind | "unknown"; path: string }
  | { kind: "more"; count: number };

export type ThemeParseResult =
  | { ok: true; theme: AppearanceTheme; warnings: ThemeWarning[] }
  | { ok: false; error: "tooLarge" | "invalidJson" | "notTheme" };

/** Control, bidi-override and zero-width characters, which could make a name read as another. ZWJ stays for emoji. */
const HIDDEN_CHARS = /[\p{Cc}\u061C\u200B\u200C\u200E\u200F\u202A-\u202E\u2060-\u2069\uFEFF]+/gu;

export function cleanThemeText(value: string, max: number): { text: string; clamped: boolean } {
  // Two UTF-16 units per code point is enough to tell whether the text runs over `max`.
  const chars = Array.from(value.slice(0, max * 2 + 2).replace(HIDDEN_CHARS, " ").trim());
  return { text: chars.slice(0, max).join("").trim(), clamped: chars.length > max };
}
