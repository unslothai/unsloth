// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  beginBackgroundUpload,
  finishBackgroundUpload,
  hydrateBackground,
  nativeMaterialAvailable,
  resetChatBackground,
  sanitizeAdjustments,
  uploadBackgroundImage,
  useBackgroundStore,
  useMaterialStore,
} from "@/features/chat-background";
import { decodeBase64Bytes } from "@/lib/base64-bytes";
import { blobToDataUrl } from "@/lib/image-fit";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store.ts";
import { getPalette, setPalette } from "../stores/theme-store.ts";
import {
  type AppearanceSnapshot,
  type AppearanceTheme,
  captureTheme,
  FONT_FIELDS,
  fontKey,
  MAX_BACKGROUND_DATA_URL_CHARS,
  MAX_THEME_FILE_CHARS,
  parseThemeFile,
  THEME_FILE_EXTENSION,
  type ThemeBackground,
  type ThemeParseResult,
} from "./theme-file.ts";

export type AppearanceTarget = Omit<AppearanceSnapshot, "background">;

export function readAppearanceSnapshot(): AppearanceSnapshot {
  const { material, style, strength, lens, menus } =
    useMaterialStore.getState();
  const background = useBackgroundStore.getState().value;
  return {
    palette: getPalette(),
    customization: useAppearanceCustomStore.getState().customization,
    glass: { material, style, strength, lens, menus },
    background: background
      ? { adjustments: sanitizeAdjustments(background), blob: background.blob }
      : null,
  };
}

/** After the stored background loads, so Undo and export never miss it. */
export async function readLoadedAppearanceSnapshot(): Promise<AppearanceSnapshot> {
  await hydrateBackground();
  return readAppearanceSnapshot();
}

/** Theme mode and interface scale are never touched; the background is separate. */
export function applyAppearanceSnapshot(next: AppearanceTarget): void {
  useAppearanceCustomStore.getState().replaceAll(next.customization);
  setPalette(next.palette);
  const { material, style, strength, lens, menus } = next.glass;
  useMaterialStore.setState({
    material,
    style,
    strength,
    lens,
    menus,
    nativeFailed: false,
  });
}

/** Off macOS, native-only glass would render solid while its switch reads on, so use Liquid 3D. */
export function portableGlass<T extends AppearanceTarget>(target: T): T {
  if (nativeMaterialAvailable()) return target;
  const portable = <M>(m: M) => (m === "macos" ? "liquid-3d" : m) as M;
  const { material, style } = target.glass;
  return {
    ...target,
    glass: {
      ...target.glass,
      material: portable(material),
      style: portable(style),
    },
  };
}

export function restoreBackground(
  prev: AppearanceSnapshot["background"],
): Promise<boolean> {
  if (!prev) return resetChatBackground();
  return finishBackgroundUpload(beginBackgroundUpload(), {
    ...prev.adjustments,
    blob: prev.blob,
  });
}

/** Goes through the normal upload checks and re-encode; resolves false when it can't. */
export async function applyThemeBackground({
  dataUrl,
  adjustments,
}: ThemeBackground): Promise<boolean> {
  const comma = dataUrl.indexOf(",");
  try {
    const blob = new Blob([decodeBase64Bytes(dataUrl.slice(comma + 1))], {
      type: dataUrl.slice(5, dataUrl.indexOf(";")),
    });
    return await uploadBackgroundImage(blob, sanitizeAdjustments(adjustments));
  } catch {
    return false;
  }
}

/** Drops embedded fonts the browser cannot load (the font picker's own check) and their selections. */
export async function withLoadableFonts(
  theme: AppearanceTheme,
): Promise<[AppearanceTheme, string[]]> {
  if (!theme.fonts?.length) return [theme, []];
  const loads = await Promise.all(
    theme.fonts.map((font) =>
      new FontFace(font.name, `url(${font.dataUrl})`).load().then(
        () => true,
        () => false,
      ),
    ),
  );
  const dropped = theme.fonts
    .filter((_, i) => !loads[i])
    .map((font) => font.name);
  if (dropped.length === 0) return [theme, []];
  const bad = new Set(dropped.map(fontKey));
  const customization = { ...theme.customization };
  for (const field of FONT_FIELDS)
    if (bad.has(fontKey(customization[field]))) customization[field] = null;
  return [
    { ...theme, customization, fonts: theme.fonts.filter((_, i) => loads[i]) },
    dropped,
  ];
}

const slugify = (name: string) =>
  name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 60)
    .replace(/^-+|-+$/g, "") || "theme";

export async function exportThemeFile(opts: {
  name: string;
  includeLayout: boolean;
  includeAssets: boolean;
}): Promise<{ fileName: string; backgroundOmitted: boolean } | null> {
  const snapshot = await readLoadedAppearanceSnapshot();
  const blob = opts.includeAssets ? (snapshot.background?.blob ?? null) : null;
  const serialize = (backgroundDataUrl: string | null) =>
    `${JSON.stringify(captureTheme(snapshot, { ...opts, backgroundDataUrl }), null, 2)}\n`;
  let text = serialize(null);
  // The background only goes in if the file still fits under the import cap.
  const budget = Math.min(
    MAX_BACKGROUND_DATA_URL_CHARS,
    MAX_THEME_FILE_CHARS - text.length - 2048,
  );
  const dataUrl =
    blob && Math.ceil(blob.size / 3) * 4 + 32 <= budget
      ? await blobToDataUrl(blob)
      : null;
  if (dataUrl) text = serialize(dataUrl);
  const fileName = `${slugify(opts.name)}${THEME_FILE_EXTENSION}`;
  try {
    const savedName = await downloadFile(text, fileName, "application/json");
    return {
      fileName: savedName ?? fileName,
      backgroundOmitted: blob !== null && dataUrl === null,
    };
  } catch (error) {
    if (isDownloadCancelled(error)) return null;
    throw error;
  }
}

/** Rejects oversized files before reading them; UTF-8 needs at most 3 bytes per character. */
export async function readThemeFile(file: Blob): Promise<ThemeParseResult> {
  if (file.size > MAX_THEME_FILE_CHARS * 3)
    return { ok: false, error: "tooLarge" };
  return parseThemeFile(await file.text());
}
