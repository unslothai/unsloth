// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redrawImage } from "@/features/chat-background";
import { blobToDataUrl } from "@/lib/image-fit";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { captureTheme } from "./application/capture.ts";
import { parseThemeFile } from "./domain/parse.ts";
import {
  MAX_BACKGROUND_DATA_URL_CHARS,
  MAX_THEME_FILE_CHARS,
  THEME_FILE_EXTENSION,
  type ThemeParseResult,
} from "./domain/schema.ts";
import { readAppearanceSnapshot } from "./snapshot.ts";

export type ExportThemeResult =
  | { status: "saved"; fileName: string; backgroundOmitted: boolean }
  | { status: "cancelled" };

/** Base64 length of `bytes`, plus room for the "data:image/...;base64," prefix. */
const fitsInFile = (bytes: number) => Math.ceil(bytes / 3) * 4 + 32 <= MAX_BACKGROUND_DATA_URL_CHARS;

function slugify(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 60)
    .replace(/^-+|-+$/g, "");
  return slug || "theme";
}

/** The stored background as a data URL that fits the file, re-encoded as JPEG if needed, or null. */
async function backgroundDataUrl(blob: Blob): Promise<string | null> {
  try {
    const source = fitsInFile(blob.size)
      ? blob
      : await redrawImage(blob, { type: "image/jpeg", quality: 0.85, fill: "#ffffff" });
    return fitsInFile(source.size) ? await blobToDataUrl(source) : null;
  } catch {
    return null;
  }
}

/** Saves the current look, and optionally layout and assets, as a theme file. */
export async function exportThemeFile(opts: {
  name: string;
  author?: string;
  includeLayout: boolean;
  includeAssets: boolean;
}): Promise<ExportThemeResult> {
  const snapshot = readAppearanceSnapshot();
  const blob = opts.includeAssets ? (snapshot.background?.blob ?? null) : null;
  const background = blob ? await backgroundDataUrl(blob) : null;
  const serialize = (backgroundDataUrl: string | null) =>
    `${JSON.stringify(captureTheme(snapshot, { ...opts, backgroundDataUrl }), null, 2)}\n`;
  let text = serialize(background);
  // Fonts and background together can pass the import cap; the background gives way first.
  const dropBackground = background !== null && text.length > MAX_THEME_FILE_CHARS;
  if (dropBackground) text = serialize(null);

  const fileName = `${slugify(opts.name)}${THEME_FILE_EXTENSION}`;
  try {
    const savedName = await downloadFile(text, fileName, "application/json");
    return {
      status: "saved",
      fileName: savedName ?? fileName,
      backgroundOmitted: blob !== null && (background === null || dropBackground),
    };
  } catch (error) {
    if (isDownloadCancelled(error)) return { status: "cancelled" };
    throw error;
  }
}

/** Reads a picked file. Oversized files are rejected before any of their text is read. */
export async function readThemeFile(file: Blob): Promise<ThemeParseResult> {
  if (file.size > MAX_THEME_FILE_CHARS) return { ok: false, error: "tooLarge" };
  return parseThemeFile(await file.text());
}
