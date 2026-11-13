// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sectionFields } from "../domain/classification.ts";
import {
  CURRENT_THEME_SCHEMA_VERSION,
  cleanThemeText,
  DEFAULT_THEME_NAME,
  META_LIMITS,
  THEME_FILE_KIND,
  type ThemeMeta,
} from "../domain/schema.ts";
import { type AppearanceSnapshot, FONT_FIELDS } from "./plan-apply.ts";

export type CaptureOptions = {
  name: string;
  author?: string;
  includeLayout: boolean;
  includeAssets: boolean;
  backgroundDataUrl?: string | null;
};

type Dict = Record<string, unknown>;

/** Personal fields, the theme mode and the interface scale never leave the device. */
export function captureTheme(snapshot: AppearanceSnapshot, opts: CaptureOptions): Dict {
  const c = snapshot.customization;
  const meta: ThemeMeta = {
    name: cleanThemeText(opts.name, META_LIMITS.name).text || DEFAULT_THEME_NAME,
  };
  const author = cleanThemeText(opts.author ?? "", META_LIMITS.author).text;
  if (author) meta.author = author;

  const file: Dict = {
    kind: THEME_FILE_KIND,
    schemaVersion: CURRENT_THEME_SCHEMA_VERSION,
    meta,
    base: { palette: snapshot.palette },
  };
  // Without the font files, a name that points at an imported font would mean nothing elsewhere.
  const droppedFonts = new Set(opts.includeAssets ? [] : c.importedFonts.map((font) => font.name));
  const writeSection = (name: string, start: Dict = {}) => {
    for (const [field, key] of sectionFields(name)) {
      const value = c[field];
      if (key) start[key] = (FONT_FIELDS as readonly string[]).includes(field) && droppedFonts.has(value as string) ? null : value;
    }
    file[name] = start;
  };
  writeSection("colors", { light: c.colors.light, dark: c.colors.dark });
  writeSection("typography");
  writeSection("code");
  writeSection("chat");
  const { material, strength, lens, menus } = snapshot.glass;
  file.glass = { material, strength, lens, menus };
  if (opts.includeLayout) writeSection("layout");

  if (opts.includeAssets) {
    const assets: Dict = {};
    if (c.importedFonts.length > 0) assets.fonts = c.importedFonts;
    if (opts.backgroundDataUrl && snapshot.background) {
      assets.background = {
        dataUrl: opts.backgroundDataUrl,
        adjustments: snapshot.background.adjustments,
      };
    }
    if (Object.keys(assets).length > 0) file.assets = assets;
  }
  return file;
}
