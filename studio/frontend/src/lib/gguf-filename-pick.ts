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

// The backend's `_GGUF_QUANT_RE`, `_select_quant_match` and `_H3_DENOISER_PARTITIONS`, so the
// token a qualified key answers to is derived the way the lister derived the key.
const QUANT_RE =
  /(UD-)?(MXFP\d+(?:_[A-Z0-9]+)*|IQ\d+_[A-Z]+(?:_[A-Z0-9]+)?|TQ\d+_\d+|Q\d+_K_[A-Z]+|Q\d+_\d+|Q\d+_K|BF16|F16|F32)/gi;
const BPW_RE = /^-\d+(?:\.\d+)?bpw/i;
const BPW_TRAILING_RE = /-\d+(?:\.\d+)?bpw(?=\.[A-Za-z0-9]+$|$)/i;
const SPLIT_SUFFIX_RE = /-\d{3,}-of-\d{3,}/i;
const FLOAT_PRECISION = new Set(["BF16", "F16", "F32"]);
const H3_DENOISER_PARTITIONS = ["minimax_h3_fl2va", "minimax_h3_ref2va"];

function selectQuantMatch(text: string): RegExpExecArray | null {
  let fallback: RegExpExecArray | null = null;
  for (const match of text.matchAll(QUANT_RE)) {
    if (FLOAT_PRECISION.has(match[2].toUpperCase())) {
      fallback ??= match;
      continue;
    }
    return match;
  }
  return fallback;
}

/** The complete quant token a qualified key carries, bit-width modifier included: the basename
 *  decides, then parent directories nearest first. Null when nothing in the key names a quant. */
function quantTokenOf(key: string): string | null {
  const parts = key.replace(/\\/g, "/").split("/");
  const stem = (parts.pop() ?? "").replace(SPLIT_SUFFIX_RE, "").trim();
  for (const text of [stem, ...parts.reverse()]) {
    const match = selectQuantMatch(text);
    if (!match) {
      continue;
    }
    const token = `${match[1] ?? ""}${match[2]}`;
    const adjacent = BPW_RE.exec(text.slice(match.index + match[0].length));
    if (adjacent) {
      return `${token}${adjacent[0]}`;
    }
    // Named by a parent directory, the modifier may end the basename instead.
    const trailing = text === stem ? null : BPW_TRAILING_RE.exec(stem);
    return trailing ? `${token}${trailing[0]}` : token;
  }
  return null;
}

/** The backend's `_keys_at_repo_root`: a quant-named parent only repeats how the file was
 *  quantized and leaves the build at the root; any other directory is another checkpoint. */
const atRepoRoot = (key: string): boolean =>
  key
    .replace(/\\/g, "/")
    .split("/")
    .slice(0, -1)
    .every((segment) => !segment || selectQuantMatch(segment) !== null);

/** Whether `quant` is a qualified key whose complete bare quant token is `label`. `Q4_K` is not
 *  the token of `model-Q4_K_M-mtp`, and an H3 denoiser partition never answers to its bare quant
 *  (the backend's `accepts_bare_quant_alias`), since that spelling picks the wrong checkpoint. */
const bareLabelOf = (quant: string, label: string): boolean => {
  if (quant.toLowerCase() === label.toLowerCase()) {
    return false;
  }
  const basename =
    quant.replace(/\\/g, "/").split("/").pop()?.toLowerCase() ?? "";
  if (H3_DENOISER_PARTITIONS.some((p) => basename.startsWith(p))) {
    return false;
  }
  return quantTokenOf(quant)?.toLowerCase() === label.toLowerCase();
};

/** The lone qualified row a bare label is the legacy spelling of, else null. A repo's lone tagged
 *  build is advertised as `model-Q4_K_M-mtp`, which the backend's download and load paths still
 *  accept for the bare label; refusing it here left "Pick a quantization" on a hint that resolves
 *  everywhere else. Root precedence as the backend's shared resolver applies it: the bare quant
 *  is the spelling a root build USED to key under, so a tagged root outranks
 *  `distilled/model-Q4_K_M`, and only two roots still tie. Two rows carrying the label name
 *  neither, and the prompt stays. */
function aliasedFilename(
  listed: readonly {
    filename: string;
    quant: string | null;
    downloaded: boolean;
  }[],
  wanted: string,
): string | null {
  let byAlias = listed.filter((v) => v.quant && bareLabelOf(v.quant, wanted));
  if (byAlias.length > 1) {
    const roots = byAlias.filter((v) => v.quant && atRepoRoot(v.quant));
    byAlias = roots.length > 0 ? roots : byAlias;
  }
  // One IDENTITY, not one row: a downloaded and a remote copy of the same key are one build,
  // as the exact-label branch already reads them, and the downloaded one is the file to load.
  const identities = new Set(byAlias.map((v) => v.quant?.toLowerCase()));
  if (identities.size !== 1) {
    return null;
  }
  return (byAlias.find((v) => v.downloaded) ?? byAlias[0]).filename;
}

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
    if (byLabel.length > 0) {
      return (
        (byLabel.find((v) => v.downloaded) ?? byLabel[0])?.filename ?? null
      );
    }
    return aliasedFilename(listed, wanted);
  }
  // No label: only a lone file names itself. Downloaded first, so a fully listed remote repo resolves to the quant on disk.
  const downloaded = listed.filter((v) => v.downloaded);
  if (downloaded.length === 1) return downloaded[0].filename;
  if (listed.length === 1) return listed[0].filename;
  return null;
}
