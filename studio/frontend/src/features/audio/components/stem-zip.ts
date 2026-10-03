// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stored (uncompressed) zip: WAV barely compresses. Free of app imports for the node test runner.

import { Zip, ZipPassThrough } from "fflate";

const INVALID_CHARS = new Set('<>:"/\\|?*');
const MAX_TITLE_CHARS = 120;
const FALLBACK_TITLE = "Separated track";

export function sanitizeFileNamePart(part: string, fallback: string): string {
  let out = "";
  for (const char of part) {
    const code = char.codePointAt(0) ?? 0;
    out += INVALID_CHARS.has(char) || code < 0x20 || code === 0x7f ? "_" : char;
  }
  const cleaned = out
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/, "");
  return cleaned || fallback;
}

/** The title is kept short so the stem label always survives. */
export function stemFileName(title: string, label: string): string {
  const bare = title.replace(/\.(wav|mp3|flac|ogg|m4a|aac|opus|webm)$/i, "");
  const safeTitle = [...sanitizeFileNamePart(bare, FALLBACK_TITLE)]
    .slice(0, MAX_TITLE_CHARS)
    .join("")
    .trim();
  const safeLabel = sanitizeFileNamePart(label, "Stem");
  return `${safeTitle || FALLBACK_TITLE} - ${safeLabel}.wav`;
}

export function stemZipName(title: string): string {
  return stemFileName(title, "stems").replace(/\.wav$/, ".zip");
}

export function uniqueStemNames(names: readonly string[]): string[] {
  const taken = new Set<string>();
  return names.map((name) => {
    let candidate = name;
    const dot = name.lastIndexOf(".");
    const stem = dot > 0 ? name.slice(0, dot) : name;
    const ext = dot > 0 ? name.slice(dot) : "";
    for (let n = 2; taken.has(candidate.toLowerCase()); n += 1)
      candidate = `${stem} (${n})${ext}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  });
}

/** Blobs are streamed in, so no second contiguous copy is held. */
export async function zipStems(
  files: readonly { name: string; blob: Blob }[],
): Promise<Blob> {
  const names = uniqueStemNames(files.map((file) => file.name));
  const chunks: Uint8Array[] = [];
  await new Promise<void>((resolve, reject) => {
    const zip = new Zip((error, data, final) => {
      if (error) {
        reject(error);
        return;
      }
      chunks.push(data);
      if (final) resolve();
    });
    const write = async () => {
      for (const [index, file] of files.entries()) {
        const entry = new ZipPassThrough(names[index]);
        zip.add(entry);
        const reader = file.blob.stream().getReader();
        for (;;) {
          const { done: end, value } = await reader.read();
          if (end) break;
          entry.push(value, false);
        }
        entry.push(new Uint8Array(0), true);
      }
      zip.end();
    };
    write().catch(reject);
  });
  return new Blob(chunks as BlobPart[], { type: "application/zip" });
}
