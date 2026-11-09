// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app deps so it stays testable; the request lives in diffusion-gguf-filename.ts.

/** Loose so an older response still resolves. */
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

/** A label is not a filename. Null when ambiguous, so the caller keeps its prompt. */
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

  // Honour an unlisted filename: a failed listing must not lose the caller's.
  if (wanted && isGgufName(wanted)) {
    const match = listed.find(
      (v) => v.filename.toLowerCase() === wanted.toLowerCase(),
    );
    return match?.filename ?? wanted;
  }
  // Downloaded first (a remote sibling can share the label), and before fallbacks so a stale label prompts.
  if (wanted) {
    const byLabel = listed.filter(
      (v) => v.quant?.toLowerCase() === wanted.toLowerCase(),
    );
    return (byLabel.find((v) => v.downloaded) ?? byLabel[0])?.filename ?? null;
  }
  // Only a lone file names itself; downloaded first.
  const downloaded = listed.filter((v) => v.downloaded);
  if (downloaded.length === 1) return downloaded[0].filename;
  if (listed.length === 1) return listed[0].filename;
  return null;
}
