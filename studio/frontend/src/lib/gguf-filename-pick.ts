// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picking the .gguf a repo-level pick means. No app deps so it stays testable; the request lives in diffusion-gguf-filename.ts.

/** A row of the repo's GGUF listing. Loose so an older response still resolves. */
export interface GgufFilenameCandidate {
  filename?: unknown;
  quant?: unknown;
  downloaded?: unknown;
}

function asName(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

export const isGgufName = (value: string): boolean =>
  value.toLowerCase().endsWith(".gguf");

/** The .gguf to load, given the listing and what the pick carried: a filename, a quant label, or nothing. A load needs a real
 *  filename and a label is not one. Null when the repo is ambiguous, so the caller keeps its prompt. */
export function pickGgufFilename(
  variants: readonly GgufFilenameCandidate[],
  quant?: string | null,
): string | null {
  const listed = variants.flatMap((v) => {
    const filename = asName(v.filename);
    return filename && isGgufName(filename)
      ? [
          {
            filename,
            quant: asName(v.quant),
            downloaded: v.downloaded === true,
          },
        ]
      : [];
  });
  const wanted = quant?.trim() || null;

  // Already a filename. Prefer the listing's spelling, but honour an unlisted one: a failed listing must not lose the caller's.
  if (wanted && isGgufName(wanted)) {
    const match = listed.find(
      (v) => v.filename.toLowerCase() === wanted.toLowerCase(),
    );
    return match?.filename ?? wanted;
  }
  // A label. Downloaded first (a remote sibling can share it), and before the fallbacks so a stale label prompts.
  if (wanted) {
    const byLabel = listed.filter(
      (v) => v.quant?.toLowerCase() === wanted.toLowerCase(),
    );
    return (byLabel.find((v) => v.downloaded) ?? byLabel[0])?.filename ?? null;
  }
  // No label: only a lone file names itself. Downloaded first, so a fully listed remote repo resolves to the quant on disk.
  const downloaded = listed.filter((v) => v.downloaded);
  if (downloaded.length === 1) return downloaded[0].filename;
  if (listed.length === 1) return listed[0].filename;
  return null;
}
