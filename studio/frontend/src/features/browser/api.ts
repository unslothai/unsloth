// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import type { PageRequest } from "./frame-message";

export type BrowserPage =
  | {
      kind: "html";
      url: string;
      base: string;
      /** The page declared its <base>, so it stays put when the page's router moves. */
      ownBase?: boolean;
      html: string;
      refresh: { delay: number; url: string } | null;
    }
  | { kind: "raw"; url: string; blob: Blob; contentType: string; /** the server's download name. */ fileName?: string };

export type BrowserRequest = {
  url: string;
  method?: "GET" | "POST";
  body?: string;
  /** refuse bodies past this size (favicons); otherwise use the backend cap. */
  maxBytes?: number;
  /** show the site's HTTP error page; tab loads only. */
  errorPage?: boolean;
};

let annotateCode: Promise<string> | null = null;

/** the page annotate code (`_ANNOTATE_JS`), cached after success so failed fetches can retry. */
export function annotateScript(): Promise<string> {
  annotateCode ??= authFetch("/api/browser/annotate.js")
    .then((response) => {
      if (!response.ok) throw new Error(`annotate code: ${response.status}`);
      return response.text();
    })
    .catch((error: unknown) => {
      annotateCode = null;
      throw error;
    });
  return annotateCode;
}

export class BrowserFetchError extends Error {
  readonly botCheck: boolean;

  constructor(message: string, botCheck: boolean) {
    super(message);
    this.botCheck = botCheck;
  }
}

/** fetch through the backend for sites that refuse framing. */
export async function fetchBrowserPage(request: BrowserRequest, signal: AbortSignal): Promise<BrowserPage> {
  const response = await authFetch("/api/browser/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url: request.url,
      method: request.method ?? "GET",
      body: request.body ?? null,
      max_bytes: request.maxBytes ?? null,
      error_page: request.errorPage ?? false,
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
  let fileName: string | undefined;
  try {
    fileName = decodeURIComponent(response.headers.get("X-Unsloth-Browser-Filename") ?? "") || undefined;
  } catch {
    fileName = undefined;
  }
  return {
    kind: "raw",
    url: finalUrl ?? request.url,
    blob,
    contentType: response.headers.get("Content-Type") ?? blob.type,
    fileName,
  };
}

export type PageResponse = {
  status: number;
  statusText: string;
  url: string;
  headers: Array<[string, string]>;
  body: ArrayBuffer;
  /** A redirect the request said to stop at ("manual"): the page gets an opaque redirect. */
  opaqueRedirect?: boolean;
  redirected?: boolean;
  /** The chain went through another site, so it reads as a CORS response. */
  cors?: boolean;
};

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let at = 0; at < bytes.length; at += 0x8000) binary += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  return btoa(binary);
}

function fromBase64(text: string): ArrayBuffer {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let at = 0; at < binary.length; at++) bytes[at] = binary.charCodeAt(at);
  return bytes.buffer;
}

/** A page's request to its own site, made by the backend for the page at `page`; throws when it fails. */
export async function requestForPage(page: string, request: PageRequest, signal: AbortSignal): Promise<PageResponse> {
  const response = await authFetch("/api/browser/request", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      page,
      url: request.url,
      method: request.method,
      headers: request.headers,
      redirect: request.redirect,
      body: request.body ? toBase64(new Uint8Array(request.body)) : null,
    }),
    signal,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const answer = (await response.json()) as Omit<PageResponse, "body"> & { body: string };
  return { ...answer, body: fromBase64(answer.body) };
}

export function browserFrameUrl(loadId: string): string {
  return apiUrl(`/api/browser/frame?v=${encodeURIComponent(loadId)}`);
}
