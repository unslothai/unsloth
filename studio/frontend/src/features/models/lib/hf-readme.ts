// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ReadmeKind = "model" | "dataset";

const cache = new Map<string, Promise<string | null>>();

async function fetchReadmeOnce(
  repoId: string,
  kind: ReadmeKind,
): Promise<string | null> {
  const prefix = kind === "dataset" ? "datasets/" : "";
  for (const branch of ["main", "master"]) {
    try {
      const url = `https://huggingface.co/${prefix}${repoId}/raw/${branch}/README.md`;
      const res = await fetch(url, { mode: "cors" });
      if (res.ok) {
        return await res.text();
      }
    } catch {
      // try next branch
    }
  }
  return null;
}

export function fetchReadme(
  repoId: string,
  kind: ReadmeKind = "model",
): Promise<string | null> {
  const key = `${kind}::${repoId}`;
  if (!cache.has(key)) {
    cache.set(key, fetchReadmeOnce(repoId, kind));
  }
  return cache.get(key)!;
}

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n?/;

export function stripFrontmatter(markdown: string): {
  body: string;
  frontmatter: string | null;
} {
  const match = FRONTMATTER_RE.exec(markdown);
  if (match) {
    return {
      body: markdown.slice(match[0].length),
      frontmatter: match[1],
    };
  }
  return { body: markdown, frontmatter: null };
}

export function extractDescription(markdown: string): string | null {
  const { body } = stripFrontmatter(markdown);
  const lines = body.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    if (line.startsWith("#")) continue;
    if (/^!\[.*?\]\(/.test(line)) continue;
    if (/^<(img|a|div|p|span)\b/i.test(line)) continue;
    if (/^\[!\[/.test(line)) continue;
    if (line.startsWith(">")) continue;
    if (line.startsWith("|")) continue;
    if (line.startsWith("---")) continue;
    const cleaned = line
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/<[^>]+>/g, "")
      .replace(/[*_`]+/g, "")
      .trim();
    if (cleaned.length >= 24) {
      return cleaned;
    }
  }
  return null;
}

export function extractFrontmatterValue(
  frontmatter: string | null,
  key: string,
): string | null {
  if (!frontmatter) return null;
  const re = new RegExp(`^${key}:\\s*(.+?)\\s*$`, "im");
  const match = re.exec(frontmatter);
  if (!match) return null;
  let value = match[1];
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1);
  }
  return value;
}
