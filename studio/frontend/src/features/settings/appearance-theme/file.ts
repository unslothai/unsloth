// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { blobToDataUrl, decodeImage } from "@/features/chat-background";
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

export type ExportThemeOptions = {
  name: string;
  author?: string;
  includeLayout: boolean;
  includeAssets: boolean;
};

export type ExportThemeResult =
  | { status: "saved"; fileName: string; backgroundOmitted: boolean }
  | { status: "cancelled" };

const JPEG_QUALITY = 0.85;

function slugify(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/, "");
  return slug || "theme";
}

/** Re-encodes as JPEG on white, so transparent areas do not come out black. */
async function reencodeJpeg(blob: Blob): Promise<Blob> {
  const decoded = await decodeImage(blob);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = decoded.width;
    canvas.height = decoded.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("decode");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(decoded.source, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (out) => (out ? resolve(out) : reject(new Error("decode"))),
        "image/jpeg",
        JPEG_QUALITY,
      ),
    );
  } finally {
    decoded.close();
  }
}

/** The stored background as a data URL small enough for the file, or null when none fits. */
async function backgroundDataUrl(blob: Blob): Promise<string | null> {
  const original = await blobToDataUrl(blob);
  if (original.length <= MAX_BACKGROUND_DATA_URL_CHARS) return original;
  try {
    const jpeg = await blobToDataUrl(await reencodeJpeg(blob));
    return jpeg.length <= MAX_BACKGROUND_DATA_URL_CHARS ? jpeg : null;
  } catch {
    return null;
  }
}

/** Saves the current look, and optionally layout and assets, as a theme file. */
export async function exportThemeFile(
  opts: ExportThemeOptions,
): Promise<ExportThemeResult> {
  const snapshot = readAppearanceSnapshot();
  const base = {
    name: opts.name,
    author: opts.author,
    includeLayout: opts.includeLayout,
    includeAssets: opts.includeAssets,
  };
  const backgroundBlob = opts.includeAssets
    ? (snapshot.background?.blob ?? null)
    : null;
  const background = backgroundBlob
    ? await backgroundDataUrl(backgroundBlob)
    : null;
  let backgroundOmitted = backgroundBlob !== null && background === null;
  const serialize = (dataUrl: string | null) =>
    `${JSON.stringify(
      captureTheme(snapshot, { ...base, backgroundDataUrl: dataUrl }),
      null,
      2,
    )}\n`;
  let text = serialize(background);
  // Fonts and background together can pass the import cap; the background gives way first.
  if (background !== null && text.length > MAX_THEME_FILE_CHARS) {
    backgroundOmitted = true;
    text = serialize(null);
  }

  const fileName = `${slugify(opts.name)}${THEME_FILE_EXTENSION}`;
  try {
    const savedName = await downloadFile(text, fileName, "application/json");
    return {
      status: "saved",
      fileName: savedName ?? fileName,
      backgroundOmitted,
    };
  } catch (error) {
    if (isDownloadCancelled(error)) return { status: "cancelled" };
    throw error;
  }
}

/** Reads a picked file. Oversized files are rejected before any of their text is read. */
export async function readThemeFile(file: Blob): Promise<ThemeParseResult> {
  if (file.size > MAX_THEME_FILE_CHARS) {
    return { ok: false, error: "tooLarge" };
  }
  return parseThemeFile(await file.text());
}
