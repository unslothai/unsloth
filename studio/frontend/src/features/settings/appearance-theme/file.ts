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
import { readLoadedAppearanceSnapshot } from "./snapshot.ts";

export type ExportThemeResult =
  | { status: "saved"; fileName: string; backgroundOmitted: boolean }
  | { status: "cancelled" };

/** Base64 length of `bytes` plus the "data:image/...;base64," prefix. */
const dataUrlLength = (bytes: number) => Math.ceil(bytes / 3) * 4 + 32;
/** Room for the background object's own keys and adjustments in the JSON. */
const BACKGROUND_JSON_OVERHEAD = 2048;

function slugify(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 60)
    .replace(/^-+|-+$/g, "");
  return slug || "theme";
}

/** The stored background as a data URL within `budget`, re-encoded as JPEG if needed, or null. */
async function backgroundDataUrl(blob: Blob, budget: number): Promise<string | null> {
  try {
    const source =
      dataUrlLength(blob.size) <= budget
        ? blob
        : await redrawImage(blob, { type: "image/jpeg", quality: 0.85, fill: "#ffffff" });
    return dataUrlLength(source.size) <= budget ? await blobToDataUrl(source) : null;
  } catch {
    return null;
  }
}

export async function exportThemeFile(opts: {
  name: string;
  author?: string;
  includeLayout: boolean;
  includeAssets: boolean;
}): Promise<ExportThemeResult> {
  // An export right after opening Settings must not miss a background that is still loading.
  const snapshot = await readLoadedAppearanceSnapshot();
  const serialize = (backgroundDataUrl: string | null) =>
    `${JSON.stringify(captureTheme(snapshot, { ...opts, backgroundDataUrl }), null, 2)}\n`;
  let text = serialize(null);
  const blob = opts.includeAssets ? (snapshot.background?.blob ?? null) : null;
  // The background gets whatever room the fonts and settings leave under the import cap.
  const budget = Math.min(
    MAX_BACKGROUND_DATA_URL_CHARS,
    MAX_THEME_FILE_CHARS - text.length - BACKGROUND_JSON_OVERHEAD,
  );
  const background = blob ? await backgroundDataUrl(blob, budget) : null;
  if (background) text = serialize(background);

  const fileName = `${slugify(opts.name)}${THEME_FILE_EXTENSION}`;
  try {
    const savedName = await downloadFile(text, fileName, "application/json");
    return {
      status: "saved",
      fileName: savedName ?? fileName,
      backgroundOmitted: blob !== null && background === null,
    };
  } catch (error) {
    if (isDownloadCancelled(error)) return { status: "cancelled" };
    throw error;
  }
}

/** Reads a picked file. Oversized files are rejected before any of their text is read. */
export async function readThemeFile(file: Blob): Promise<ThemeParseResult> {
  // The cap counts characters; UTF-8 needs at most 3 bytes for each, and the parser checks exactly.
  if (file.size > MAX_THEME_FILE_CHARS * 3) return { ok: false, error: "tooLarge" };
  return parseThemeFile(await file.text());
}
