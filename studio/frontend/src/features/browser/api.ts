// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import type { PageRequest } from "./frame-message";
import { onPageStorageWrite, pageOrigin, readPageStorage, writePageStorage } from "./page-storage";

export type BrowserPage =
  | {
      kind: "html";
      url: string;
      base: string;
      /** The page declared its <base>, so it stays put when the page's router moves. */
      ownBase?: boolean;
      /** Its site sent Access-Control-Allow-Origin * or null, so the sandbox reads its requests itself. */
      anyOrigin?: boolean;
      /** The backend holds cookies for its site, so the page's own requests go through it. */
      siteCookies?: boolean;
      html: string;
      refresh: { delay: number; url: string } | null;
      cookies?: SiteCookies;
      cookiesVersion?: CookiesVersion;
    }
  | {
      kind: "raw";
      url: string;
      blob: Blob;
      contentType: string;
      fileName?: string;
      attachment?: boolean;
    };

export type BrowserRequest = {
  url: string;
  method?: "GET" | "POST";
  body?: string;
  /** refuse bodies past this size (favicons); otherwise use the backend cap. */
  maxBytes?: number;
  /** show the site's HTTP error page; tab loads only. */
  errorPage?: boolean;
  /** the page a tab navigation came from, for SameSite; none when typed. */
  initiator?: string;
  /** "omit" for a page's subresources (favicons): no cookies sent or kept. */
  credentials?: "include" | "omit";
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

/** The site's script-readable cookies, in the frame's own form (routes/browser_cookies.py). */
export type SiteCookies = Array<[string, string]>;
/** [process epoch, cookie store version] when the snapshot was taken. */
export type CookiesVersion = [string, number];

// Cookie writes and clears run one at a time, so none queued before a clear lands after it.
let lastCookieWrite: Promise<unknown> | null = null;
let cookieClears = 0;

function afterCookieWrites<T>(work: () => Promise<T>): Promise<T> {
  const run = lastCookieWrite ? lastCookieWrite.then(work) : work();
  const settled: Promise<unknown> = run.catch(() => undefined);
  lastCookieWrite = settled;
  void settled.finally(() => {
    if (lastCookieWrite === settled) lastCookieWrite = null;
  });
  return run;
}

// When each of a page's own cookie writes was made, so an answer asked for before it can't undo it.
let cookieWriteCount = 0;
const cookieWrittenAt = new Map<string, number>();

onPageStorageWrite(({ origin, area, key, value, from, sameSite }) => {
  // Only a page's own write: the mirror below writes with no `from`.
  if (area !== "cookie" || key === null || !from) return;
  if (cookieWrittenAt.size >= 4096) cookieWrittenAt.clear();
  cookieWrittenAt.set(`${origin} ${key}`, ++cookieWriteCount);
  const clears = cookieClears;
  void afterCookieWrites(async () => {
    if (clears !== cookieClears) return;
    await authFetch("/api/browser/cookie", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page: `${origin}/`, key, value, ...(sameSite && { same_site: sameSite }) }),
    }).catch(() => undefined);
  });
});

/** A request sent before these land would miss a cookie the page just set. */
async function cookieWritesSent(): Promise<void> {
  if (lastCookieWrite) await lastCookieWrite;
}

// The newest snapshot applied per origin, so an older answer arriving later can't roll it back.
const mirroredVersion = new Map<string, CookiesVersion>();

/** `asked`: the page's write count when the answer was requested; cookies it wrote since are left alone. */
export function mirrorSiteCookies(url: string, cookies: SiteCookies | undefined, asked = cookieWriteCount, version?: CookiesVersion): void {
  const origin = pageOrigin(url);
  if (!origin || !Array.isArray(cookies)) return;
  if (Array.isArray(version)) {
    const applied = mirroredVersion.get(origin);
    if (applied && applied[0] === version[0] && applied[1] > version[1]) return;
    if (mirroredVersion.size >= 4096) mirroredVersion.clear();
    mirroredVersion.set(origin, version);
  }
  const newer = (key: string) => (cookieWrittenAt.get(`${origin} ${key}`) ?? 0) > asked;
  const next = new Map(cookies);
  const held = new Map(readPageStorage(origin, "").cookie);
  for (const key of held.keys()) if (!next.has(key) && !newer(key)) writePageStorage(origin, "", "cookie", key, null);
  for (const [key, value] of next) if (held.get(key) !== value && !newer(key)) writePageStorage(origin, "", "cookie", key, value);
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
  await cookieWritesSent();
  const asked = cookieWriteCount;
  const clears = cookieClears;
  const response = await authFetch("/api/browser/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url: request.url,
      method: request.method ?? "GET",
      body: request.body ?? null,
      max_bytes: request.maxBytes ?? null,
      error_page: request.errorPage ?? false,
      initiator: request.initiator ?? null,
      credentials: request.credentials ?? "include",
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
    if (clears === cookieClears) mirrorSiteCookies(page.url, page.cookies, asked, page.cookiesVersion);
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
    ...(response.headers.get("X-Unsloth-Browser-Attachment") === "1" && { attachment: true }),
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

export async function clearSiteCookies(): Promise<void> {
  cookieClears++;
  cookieWrittenAt.clear();
  const response = await afterCookieWrites(() => authFetch("/api/browser/clear-cookies", { method: "POST" }));
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

/** A page's request to its own site, made by the backend for the page at `page`; throws when it fails. */
export async function requestForPage(page: string, request: PageRequest, signal: AbortSignal): Promise<PageResponse> {
  await cookieWritesSent();
  const asked = cookieWriteCount;
  const clears = cookieClears;
  const response = await authFetch("/api/browser/request", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      page,
      url: request.url,
      method: request.method,
      headers: request.headers,
      redirect: request.redirect,
      integrity: request.integrity,
      credentials: request.credentials,
      body: request.body ? toBase64(new Uint8Array(request.body)) : null,
    }),
    signal,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const answer = (await response.json()) as Omit<PageResponse, "body"> & {
    body: string;
    cookies?: SiteCookies;
    cookiesVersion?: CookiesVersion;
  };
  if (clears === cookieClears) mirrorSiteCookies(page, answer.cookies, asked, answer.cookiesVersion);
  return { ...answer, body: fromBase64(answer.body) };
}

export function browserFrameUrl(loadId: string): string {
  return apiUrl(`/api/browser/frame?v=${encodeURIComponent(loadId)}`);
}
