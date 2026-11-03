// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sanitizeAdjustments } from "../../../chat-background/model.ts";
import type {
  AppearanceCustomization,
  CustomModeColors,
} from "../../lib/appearance-customization.ts";
import {
  CURRENT_THEME_SCHEMA_VERSION,
  MAX_THEME_AUTHOR_LENGTH,
  MAX_THEME_NAME_LENGTH,
  MAX_THEME_STUDIO_VERSION_LENGTH,
  THEME_FILE_KIND,
  type ThemeFileV1,
  type ThemeGlass,
  type ThemeMeta,
} from "../domain/schema.ts";
import type { AppearanceSnapshot } from "./types.ts";

export type CaptureOptions = {
  name: string;
  author?: string;
  includeLayout: boolean;
  includeAssets: boolean;
  /** Already sized and encoded by the caller; capture only places it in the file. */
  backgroundDataUrl?: string | null;
  studioVersion?: string;
};

const DEFAULT_THEME_NAME = "Untitled theme";

/** Same cleanup the parser applies, so an exported field reads back without a warning. */
function cleanText(value: string, max: number): string {
  return Array.from(value.replace(/\p{Cc}+/gu, " ").trim())
    .slice(0, max)
    .join("")
    .trim();
}

function copyModeColors(colors: CustomModeColors): CustomModeColors {
  return {
    accent: colors.accent,
    background: colors.background,
    foreground: colors.foreground,
  };
}

function copyGlass(glass: ThemeGlass): ThemeGlass {
  return {
    material: glass.material,
    strength: { light: glass.strength.light, dark: glass.strength.dark },
    lens: glass.lens,
    menus: glass.menus,
  };
}

function captureAssets(
  snapshot: AppearanceSnapshot,
  backgroundDataUrl: string | null,
): NonNullable<ThemeFileV1["assets"]> | null {
  const assets: NonNullable<ThemeFileV1["assets"]> = {};
  const fonts = snapshot.customization.importedFonts;
  if (fonts.length > 0) {
    assets.fonts = fonts.map((font) => ({
      name: font.name,
      dataUrl: font.dataUrl,
    }));
  }
  if (backgroundDataUrl) {
    assets.background = {
      dataUrl: backgroundDataUrl,
      adjustments: sanitizeAdjustments(snapshot.background?.adjustments ?? {}),
    };
  }
  return Object.keys(assets).length > 0 ? assets : null;
}

/**
 * Builds the v1 file for the look, and optionally the layout and assets. Personal fields
 * (pointerCursors, reduceMotion) and the theme mode or interface scale never leave the device.
 */
export function captureTheme(
  snapshot: AppearanceSnapshot,
  opts: CaptureOptions,
): ThemeFileV1 {
  const c: AppearanceCustomization = snapshot.customization;
  const meta: ThemeMeta = {
    name: cleanText(opts.name, MAX_THEME_NAME_LENGTH) || DEFAULT_THEME_NAME,
  };
  const author = cleanText(opts.author ?? "", MAX_THEME_AUTHOR_LENGTH);
  if (author) meta.author = author;
  const studioVersion = cleanText(
    opts.studioVersion ?? "",
    MAX_THEME_STUDIO_VERSION_LENGTH,
  );
  if (studioVersion) meta.studioVersion = studioVersion;

  const file: ThemeFileV1 = {
    kind: THEME_FILE_KIND,
    schemaVersion: CURRENT_THEME_SCHEMA_VERSION,
    meta,
    base: { palette: snapshot.palette },
    colors: {
      light: copyModeColors(c.colors.light),
      dark: copyModeColors(c.colors.dark),
      contrast: c.contrast,
    },
    typography: {
      uiFont: c.uiFont,
      headingFont: c.headingFont,
      chatFont: c.chatFont,
      codeFont: c.codeFont,
      uiFontSize: c.uiFontSize,
      codeFontSize: c.codeFontSize,
      fontSmoothing: c.fontSmoothing,
    },
    code: {
      lineNumbers: c.codeLineNumbers,
      wordWrap: c.codeWordWrap,
      ligatures: c.codeLigatures,
    },
    chat: { width: c.chatWidth, composerOutline: c.composerOutline },
    glass: copyGlass(snapshot.glass),
  };

  if (opts.includeLayout) {
    file.layout = {
      sentAttachments: c.sentAttachments,
      sidebarMenu: c.sidebarMenu.map((item) => ({
        id: item.id,
        visible: item.visible,
      })),
      sidebarNav: c.sidebarNav.map((item) => ({
        id: item.id,
        pinned: item.pinned,
      })),
      sidebarNavAuto: [...c.sidebarNavAuto],
    };
  }

  if (opts.includeAssets) {
    const assets = captureAssets(snapshot, opts.backgroundDataUrl ?? null);
    if (assets) file.assets = assets;
  }

  return file;
}
