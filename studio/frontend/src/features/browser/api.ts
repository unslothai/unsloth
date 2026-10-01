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

export type BrowserRequest = {
  url: string;
  method?: "GET" | "POST";
  body?: string;
  /** Refuse bodies past this size (favicons); the backend's own cap otherwise. */
  maxBytes?: number;
};

export class BrowserFetchError extends Error {
  /** The site's bot check refused the proxy. */
  readonly botCheck: boolean;

  constructor(message: string, botCheck: boolean) {
    super(message);
    this.botCheck = botCheck;
  }
}

/** Fetch a page via the backend, which can load sites that refuse framing. */
export async function fetchBrowserPage(request: BrowserRequest, signal: AbortSignal): Promise<BrowserPage> {
  const response = await authFetch("/api/browser/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url: request.url,
      method: request.method ?? "GET",
      body: request.body ?? null,
      max_bytes: request.maxBytes ?? null,
    }),
    signal,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = payload?.detail;
    const botCheck = typeof detail === "object" && detail !== null && (detail as { botCheck?: unknown }).botCheck === true;
    const raw = botCheck ? (detail as { message?: unknown }).message : detail;
    const message = typeof raw === "string" ? raw : `HTTP ${response.status}`;
    throw new BrowserFetchError(message.replace(/^Failed to fetch URL:\s*/, ""), botCheck);
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
