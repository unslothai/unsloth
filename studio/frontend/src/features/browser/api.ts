// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";

export type BrowserPage =
  | {
      kind: "html";
      url: string;
      base: string;
      html: string;
      refresh: { delay: number; url: string } | null;
    }
  | { kind: "raw"; url: string; blob: Blob; contentType: string };

export type BrowserRequest = { url: string; method?: "GET" | "POST"; body?: string };

/** Fetch a page via the backend, which can load sites that refuse framing. */
export async function fetchBrowserPage(request: BrowserRequest, signal: AbortSignal): Promise<BrowserPage> {
  const response = await authFetch("/api/browser/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: request.url, method: request.method ?? "GET", body: request.body ?? null }),
    signal,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof payload?.detail === "string" ? payload.detail : `HTTP ${response.status}`;
    throw new Error(detail.replace(/^Failed to fetch URL:\s*/, ""));
  }
  if (response.headers.get("X-Unsloth-Browser-Kind") === "html") {
    const page = (await response.json()) as Omit<Extract<BrowserPage, { kind: "html" }>, "kind">;
    return { kind: "html", ...page };
  }
  const finalUrl = response.headers.get("X-Unsloth-Browser-Url");
  const blob = await response.blob();
  return {
    kind: "raw",
    url: finalUrl ?? request.url,
    blob,
    contentType: response.headers.get("Content-Type") ?? blob.type,
  };
}

/** Sandbox shell URL, unique per load. */
export function browserFrameUrl(loadId: string): string {
  return apiUrl(`/api/browser/frame?v=${encodeURIComponent(loadId)}`);
}
