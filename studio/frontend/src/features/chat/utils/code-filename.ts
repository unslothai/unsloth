// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EXT_BY_LANGUAGE: Record<string, string> = {
  bash: "sh",
  "c++": "cpp",
  csharp: "cs",
  javascript: "js",
  js: "js",
  json: "json",
  jsx: "jsx",
  markdown: "md",
  md: "md",
  python: "py",
  py: "py",
  ruby: "rb",
  rust: "rs",
  shell: "sh",
  sh: "sh",
  sql: "sql",
  ts: "ts",
  tsx: "tsx",
  typescript: "ts",
  svg: "svg",
  yaml: "yml",
  yml: "yml",
};

/*
 * A filename offered by the model, not a path. Anchored, so a separator anywhere
 * rejects the whole token: the info string is model output, and `download` on an
 * anchor is the one place a traversal-shaped name would be worth attempting. The
 * leading class also rejects a bare ".." and any dotfile, and the length bound
 * keeps a pathological fence from naming a 4KB file.
 */
const OFFERED_FILENAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.[A-Za-z0-9]{1,10}$/;

/**
 * Name the file a fence downloads as.
 *
 * Markdown treats everything after the first word of the info string as metadata,
 * so ```html index.html arrives here whole. A model writing a multi-file answer
 * puts the name it wants there, and it is the only place it can: the download had
 * no other channel for it, and every block landed on `snippet.<ext>` regardless.
 *
 * @param info Raw fence info string ("html", "html index.html", "python startLine=10").
 * @returns The offered filename when the info string carries one, else `snippet.<ext>`
 *   derived from the language token, else `snippet.txt`.
 */
export function getCodeFilename(info: string | null): string {
  const [languageToken, ...metadata] = (info ?? "").trim().split(/\s+/);

  const offered = metadata.find((token) => OFFERED_FILENAME_RE.test(token));
  if (offered) {
    return offered;
  }

  // From the language TOKEN, not the whole info string: a fence carrying metadata
  // used to slug the lot, so ```html index.html downloaded as
  // `snippet.html-index-html`.
  const normalized = languageToken?.toLowerCase();
  const fallbackExt = normalized?.replace(/[^a-z0-9]+/g, "-");
  const ext = normalized
    ? EXT_BY_LANGUAGE[normalized] || fallbackExt || "txt"
    : "txt";
  return `snippet.${ext}`;
}
