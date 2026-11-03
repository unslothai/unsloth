// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type BackgroundAdjustments,
  DEFAULT_GLASS_STRENGTH,
  type GlassStrengths,
  type LensMode,
  type WindowMaterial,
} from "../../../chat-background/model.ts";
import type {
  AppearanceCustomization,
  ChatWidthSetting,
  CustomModeColors,
  ImportedFont,
  SentAttachmentsSetting,
  SidebarMenuItemPref,
  SidebarNavItemId,
  SidebarNavItemPref,
} from "../../lib/appearance-customization.ts";
import type { ColorThemeId } from "../../lib/color-themes.ts";
import type { LayoutField, LookField } from "./classification.ts";

export const THEME_FILE_KIND = "unsloth-studio.appearance";
export const CURRENT_THEME_SCHEMA_VERSION = 1;
export const THEME_FILE_EXTENSION = ".unsloth-theme.json";
/** Whole-file cap for a file that carries assets. */
export const MAX_THEME_FILE_CHARS = 8_000_000;
/** A file without assets is settings only, so it stays small. */
export const MAX_THEME_FILE_CHARS_NO_ASSETS = 262_144;
export const MAX_BACKGROUND_DATA_URL_CHARS = 7_000_000;
export const MAX_THEME_NAME_LENGTH = 80;
export const MAX_THEME_AUTHOR_LENGTH = 80;
export const MAX_THEME_DESCRIPTION_LENGTH = 280;
export const MAX_THEME_STUDIO_VERSION_LENGTH = 80;

export type ThemeGlass = {
  material: WindowMaterial;
  strength: GlassStrengths;
  lens: LensMode;
  menus: boolean;
};

export type ThemeLook = Pick<AppearanceCustomization, LookField>;
export type ThemeLayout = Pick<AppearanceCustomization, LayoutField>;

export type ThemeBackground = {
  dataUrl: string;
  adjustments: BackgroundAdjustments;
};

export type ThemeMeta = {
  name: string;
  author?: string;
  description?: string;
  studioVersion?: string;
};

// Mirrors the defaults of useMaterialStore in chat-background/material.ts. That module imports
// zustand and Tauri, so the domain keeps its own copy; change both together.
export const DEFAULT_THEME_GLASS: ThemeGlass = {
  material: "solid",
  strength: { ...DEFAULT_GLASS_STRENGTH },
  lens: "off",
  menus: true,
};

/** On-disk v1 shape, as export writes it. */
export type ThemeFileV1 = {
  $schema?: string;
  kind: typeof THEME_FILE_KIND;
  schemaVersion: 1;
  meta: ThemeMeta;
  base: { palette: ColorThemeId };
  colors: { light: CustomModeColors; dark: CustomModeColors; contrast: number };
  typography: {
    uiFont: string | null;
    headingFont: string | null;
    chatFont: string | null;
    codeFont: string | null;
    uiFontSize: number | null;
    codeFontSize: number | null;
    fontSmoothing: boolean;
  };
  code: { lineNumbers: boolean; wordWrap: boolean; ligatures: boolean };
  chat: { width: ChatWidthSetting; composerOutline: number | null };
  glass: ThemeGlass;
  layout?: {
    sentAttachments: SentAttachmentsSetting;
    sidebarMenu: SidebarMenuItemPref[];
    sidebarNav: SidebarNavItemPref[];
    sidebarNavAuto: SidebarNavItemId[];
  };
  assets?: { fonts?: ImportedFont[]; background?: ThemeBackground };
};

/** Parsed and sanitized, in memory. */
export type AppearanceTheme = {
  meta: ThemeMeta;
  palette: ColorThemeId;
  look: ThemeLook;
  glass: ThemeGlass;
  layout: ThemeLayout | null;
  fonts: ImportedFont[] | null;
  background: ThemeBackground | null;
};

export type ThemeWarning =
  | { kind: "newerVersion" }
  | { kind: "unknownPalette"; palette: string }
  | { kind: "backgroundSkipped" }
  | { kind: "unknown" | "invalid" | "clamped" | "dropped"; path: string };

export type ThemeParseError = "tooLarge" | "invalidJson" | "notTheme";

export type ThemeParseResult =
  | { ok: true; theme: AppearanceTheme; warnings: ThemeWarning[] }
  | { ok: false; error: ThemeParseError };
