// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LruMap } from "@/features/hub/lib/lru-map";
import { fetchWithTimeout } from "@/features/hub/lib/network";
import { fingerprintToken } from "@/features/hub/lib/token-fingerprint";
import { defaultUrlTransform, type UrlTransform } from "streamdown";

export type ReadmeKind = "model" | "dataset";

export interface FetchedReadme {
  markdown: string;
  branch: "main" | "master";
}

interface ReadmeCacheEntry {
  promise: Promise<FetchedReadme | null>;
  // Epoch ms after which the result is refetched. 404s and transient failures
  // are cached briefly (transient shorter) so absence/outage can recover;
  // positive and pending results never expire within the session.
  staleAt: number;
}

const cache = new LruMap<string, ReadmeCacheEntry>(64);

// Bound each request so a stalled HF connection aborts and surfaces as a
// transient failure, matching the dataset-size and owner-avatar helpers.
const README_FETCH_TIMEOUT_MS = 10_000;
const README_NEGATIVE_TTL_MS = 5 * 60_000;
const README_TRANSIENT_TTL_MS = 30_000;

function readmePrefix(kind: ReadmeKind): string {
  return kind === "dataset" ? "datasets/" : "";
}

export function readmeBaseUrl(
  repoId: string,
  kind: ReadmeKind,
  branch: "main" | "master" = "main",
): string {
  return `https://huggingface.co/${readmePrefix(kind)}${repoId}/resolve/${branch}/`;
}

async function fetchReadmeOnce(
  repoId: string,
  kind: ReadmeKind,
  token: string | null,
): Promise<FetchedReadme | null> {
  const prefix = readmePrefix(kind);
  const headers: HeadersInit = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  let transient = false;
  for (const branch of ["main", "master"] as const) {
    try {
      const url = `https://huggingface.co/${prefix}${repoId}/raw/${branch}/README.md`;
      const res = await fetchWithTimeout(
        url,
        {
          mode: "cors",
          headers,
        },
        README_FETCH_TIMEOUT_MS,
      );
      if (res.ok) {
        return { markdown: await res.text(), branch };
      }
      if (res.status !== 404) transient = true;
    } catch {
      transient = true;
    }
  }
  if (transient) {
    throw new Error(`Failed to fetch README for ${repoId}`);
  }
  return null;
}

export function fetchReadme(
  repoId: string,
  kind: ReadmeKind = "model",
  token: string | null = null,
): Promise<FetchedReadme | null> {
  const key = `${kind}::${repoId}::${fingerprintToken(token)}`;
  const cached = cache.get(key);
  if (cached && Date.now() < cached.staleAt) return cached.promise;

  const entry: ReadmeCacheEntry = {
    promise: Promise.resolve(null),
    staleAt: Number.POSITIVE_INFINITY,
  };
  entry.promise = fetchReadmeOnce(repoId, kind, token)
    .then((result) => {
      if (result === null) entry.staleAt = Date.now() + README_NEGATIVE_TTL_MS;
      return result;
    })
    .catch((err) => {
      entry.staleAt = Date.now() + README_TRANSIENT_TTL_MS;
      throw err;
    });
  cache.set(key, entry);
  return entry.promise;
}

const ABSOLUTE_URL_RE = /^(?:[a-z][a-z0-9+.-]*:|\/\/|#|data:|mailto:|tel:)/i;
const BLOCKED_README_PROTOCOLS = new Set(["javascript:", "vbscript:"]);

function resolveAgainstBase(src: string, baseUrl: string): string {
  if (!src) return src;
  if (ABSOLUTE_URL_RE.test(src)) return src;
  try {
    return new URL(src, baseUrl).toString();
  } catch {
    return src;
  }
}

function readmeUrlProtocol(src: string): string | null {
  try {
    return new URL(src, "https://huggingface.co").protocol.toLowerCase();
  } catch {
    return null;
  }
}

function isBlockedReadmeUrl(
  src: string,
  node: Readonly<{ tagName?: unknown }>,
): boolean {
  const protocol = readmeUrlProtocol(src.trim());
  if (!protocol) return false;
  if (BLOCKED_README_PROTOCOLS.has(protocol)) return true;
  if (protocol !== "data:") return false;
  return String(node.tagName ?? "").toLowerCase() !== "img";
}

// Resolves relative asset URLs against the repo base, then delegates to
// defaultUrlTransform so unsafe schemes (javascript:, etc.) stay stripped.
export function createReadmeUrlTransform(baseUrl: string): UrlTransform {
  return (url, key, node) => {
    const resolved = resolveAgainstBase(url, baseUrl);
    if (isBlockedReadmeUrl(resolved, node)) {
      return undefined;
    }
    return defaultUrlTransform(resolved, key, node);
  };
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

/**
 * Drop heading lines that duplicate the inspector's chrome (`# Model Card …`,
 * standalone `Details` / `Model Card` / `Dataset Card`, etc.), since the page
 * header already shows the repo name and section context.
 */
export function stripChromeHeadings(markdown: string): string {
  const RE_HEADING = /^(#{1,6})\s+(.+?)\s*$/;
  const isChromeTitle = (text: string): boolean => {
    const t = text
      .toLowerCase()
      .replace(/[*_`]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    return (
      t === "details" ||
      t === "model card" ||
      t === "dataset card" ||
      t.startsWith("model card for") ||
      t.startsWith("dataset card for") ||
      t.startsWith("card for")
    );
  };

  const lines = markdown.split(/\r?\n/);
  const out: string[] = [];
  for (const line of lines) {
    const m = RE_HEADING.exec(line);
    if (m && isChromeTitle(m[2])) continue;
    out.push(line);
  }
  return out.join("\n");
}
