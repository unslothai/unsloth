// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SearchEngineId = "duckduckgo" | "google" | "bing" | "brave";

export const SEARCH_ENGINES: Record<SearchEngineId, { label: string; searchUrl: (query: string) => string }> = {
  // Server-rendered, so it works without scripts.
  duckduckgo: {
    label: "DuckDuckGo",
    searchUrl: (query) => `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
  },
  google: {
    label: "Google",
    searchUrl: (query) => `https://www.google.com/search?q=${encodeURIComponent(query)}`,
  },
  bing: {
    label: "Bing",
    searchUrl: (query) => `https://www.bing.com/search?q=${encodeURIComponent(query)}`,
  },
  brave: {
    label: "Brave Search",
    searchUrl: (query) => `https://search.brave.com/search?q=${encodeURIComponent(query)}`,
  },
};

const SCHEME = /^[a-z][a-z\d+.-]*:\/\//i;
// A dotted host (optionally with port and path) and no spaces: "unsloth.ai", "docs.unsloth.ai/get-started".
const HOST_LIKE = /^(?:[\w-]+\.)+[a-z]{2,}(?::\d+)?(?:[/?#]\S*)?$/i;
const IPV4_LIKE = /^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:[/?#]\S*)?$/;
const LOCALHOST = /^localhost(?::\d+)?(?:[/?#]\S*)?$/i;

/** What the address bar navigates to: a URL as typed, a bare host as https, anything else as a search. */
export function resolveAddress(input: string, engine: SearchEngineId): string | null {
  const text = input.trim();
  if (!text) return null;
  if (SCHEME.test(text)) return text;
  if (LOCALHOST.test(text)) return `http://${text}`;
  if (HOST_LIKE.test(text) || IPV4_LIKE.test(text)) return `https://${text}`;
  return SEARCH_ENGINES[engine].searchUrl(text);
}

/** Skip search engines' click-tracking hops, which only exist to redirect. */
export function unwrapRedirect(url: string): string {
  const target = redirectTarget(url);
  // The hop's target is page-supplied; only follow it to another web page.
  return target && isWebUrl(target) ? target : url;
}

function redirectTarget(url: string): string | null {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, "");
    if ((host === "duckduckgo.com" || host.endsWith(".duckduckgo.com")) && parsed.pathname === "/l/") {
      return parsed.searchParams.get("uddg");
    }
    if (/^google\.[a-z.]+$/.test(host) && parsed.pathname === "/url") {
      return parsed.searchParams.get("q") ?? parsed.searchParams.get("url");
    }
  } catch {
    // Not a URL; the fetch reports it.
  }
  return null;
}

export function isWebUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

/** Last path segment, for a document the web served without a title. */
export function fileNameFromUrl(url: string): string {
  try {
    const { pathname, hostname } = new URL(url);
    const last = decodeURIComponent(pathname.split("/").filter(Boolean).pop() ?? "");
    return last || hostname;
  } catch {
    return url;
  }
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
