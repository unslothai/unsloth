// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PageStorageArea } from "./page-storage";

export type FrameMessage =
  | {
      type: "navigate";
      url: string;
      newTab?: boolean;
      background?: boolean;
      replace?: boolean;
      method?: "GET" | "POST";
      body?: string;
    }
  | { type: "external"; url: string }
  | { type: "loaded"; title: string; favicon: string | null }
  | { type: "title"; title: string }
  | { type: "url"; url: string }
  | { type: "reload" }
  | { type: "upload" }
  | { type: "scriptNavigation" }
  /** Find in page: how many matches the query has, and which one is shown (-1: none). */
  | { type: "findResult"; count: number; active: number }
  /** The page's current markup, for printing; null when it couldn't be copied or was too large. */
  | { type: "snapshot"; html: string | null }
  | { type: "shortcut"; key: string; shift: boolean }
  /** A zoom key or Ctrl+wheel in the page: a step in (1), out (-1), or back to 100% (0). */
  | { type: "zoom"; direction: 1 | -1 | 0; wheel: boolean }
  | { type: "annotate"; event: AnnotateEvent }
  /** The browser agent's answer to command `id` (agent/page-agent.ts); the panel checks `result`. */
  | { type: "agent"; id: string; installed: boolean; result: Record<string, unknown> | null }
  /** A fetch or XHR to the page's own site, for the panel to make (routes/browser_network.py);
   *  `request` is null for one the panel won't make (malformed, too large), which then fails. */
  | { type: "request"; id: string; request: PageRequest | null }
  | { type: "requestAbort"; id: string }
  | {
      type: "storage";
      area: PageStorageArea;
      key: string | null;
      value: string | null;
      url: string | null;
      seq: number | null;
    };

export type RedirectMode = "follow" | "error" | "manual";

export type PageRequest = {
  url: string;
  method: string;
  headers: Array<[string, string]>;
  body: ArrayBuffer | null;
  redirect: RedirectMode;
  /** Fetch's integrity metadata, checked against the body. */
  integrity: string;
};

const REDIRECT_MODES = new Set<unknown>(["follow", "error", "manual"]);

export type AnnotateRect = { left: number; top: number; width: number; height: number };

/** What the page reports while annotating (`annotation` in routes/browser.py); it draws its own outlines. */
export type AnnotateEvent =
  | { kind: "ready" }
  | { kind: "up" }
  | { kind: "escape" }
  | { kind: "open"; id: number }
  | {
      kind: "mark";
      id: number;
      rect: AnnotateRect | null;
      quote: string;
      image: boolean;
      alt: string;
      area: boolean;
    }
  | { kind: "rects"; rects: Array<[number, AnnotateRect | null]> };

// Same limits as the fetch endpoint.
const MAX_URL_CHARS = 8192;
const MAX_BODY_CHARS = 1024 * 1024;
// The frame script caps its copy at the same size.
const MAX_SNAPSHOT_CHARS = 8 * 1024 * 1024;
const MAX_TITLE_CHARS = 1024;
const SHORTCUT_KEYS = new Set(["l", "t", "w", "r", "f", "d"]);
const MAX_QUOTE_CHARS = 300;
export const MAX_MARKS = 500;
// Far past any screen, so a page can't make the panel draw something huge.
const MAX_COORD = 1_000_000;
// The request endpoint's limits.
const MAX_REQUEST_HEADERS = 64;
const MAX_HEADER_NAME_CHARS = 256;
const MAX_REQUEST_BODY_BYTES = 1024 * 1024;

const text = (value: unknown, max: number): string | null =>
  typeof value === "string" && value.length <= max ? value : null;

const STORAGE_AREAS = new Set<unknown>(["local", "session", "cookie"]);
// An area's quota (page-storage.ts).
const MAX_STORAGE_CHARS = 128 * 1024;

const title = (value: unknown): string => (typeof value === "string" ? value.slice(0, MAX_TITLE_CHARS) : "");

const coord = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) && Math.abs(value) <= MAX_COORD ? value : null;

function rect(value: unknown): AnnotateRect | null {
  if (!value || typeof value !== "object") return null;
  const box = value as Record<string, unknown>;
  const left = coord(box.left);
  const top = coord(box.top);
  const width = coord(box.width);
  const height = coord(box.height);
  return left === null || top === null || width === null || height === null || width < 0 || height < 0
    ? null
    : { left, top, width, height };
}

function annotateEvent(message: Record<string, unknown>): AnnotateEvent | null {
  switch (message.event) {
    case "ready":
    case "up":
    case "escape":
      return { kind: message.event };
    case "open":
      return typeof message.id === "number" && Number.isSafeInteger(message.id)
        ? { kind: "open", id: message.id }
        : null;
    case "mark": {
      const id = message.id;
      if (typeof id !== "number" || !Number.isSafeInteger(id)) return null;
      return {
        kind: "mark",
        id,
        rect: rect(message.rect),
        quote: typeof message.quote === "string" ? message.quote.slice(0, MAX_QUOTE_CHARS) : "",
        image: message.image === true,
        alt: typeof message.alt === "string" ? message.alt.slice(0, MAX_QUOTE_CHARS) : "",
        area: message.area === true,
      };
    }
    case "rects": {
      if (!Array.isArray(message.rects)) return null;
      const rects = message.rects.slice(0, MAX_MARKS).flatMap((entry): Array<[number, AnnotateRect | null]> => {
        if (!Array.isArray(entry) || typeof entry[0] !== "number") return [];
        return [[entry[0], rect(entry[1])]];
      });
      return { kind: "rects", rects };
    }
    default:
      return null;
  }
}

function headerList(value: unknown): Array<[string, string]> | null {
  if (!Array.isArray(value) || value.length > MAX_REQUEST_HEADERS) return null;
  const headers: Array<[string, string]> = [];
  for (const entry of value) {
    const name = Array.isArray(entry) && entry.length === 2 ? text(entry[0], MAX_HEADER_NAME_CHARS) : null;
    const content = name === null ? null : text(entry[1], MAX_URL_CHARS);
    if (name === null || content === null) return null;
    headers.push([name, content]);
  }
  return headers;
}

function pageRequest(message: Record<string, unknown>): PageRequest | null {
  const url = text(message.url, MAX_URL_CHARS);
  const method = text(message.method, 16);
  const headers = headerList(message.headers);
  const body = message.body;
  if (!url || !method || !/^[A-Za-z]+$/.test(method) || !headers) return null;
  if (body !== null && !(body instanceof ArrayBuffer && body.byteLength <= MAX_REQUEST_BODY_BYTES)) return null;
  const redirect = message.redirect ?? "follow";
  const integrity = text(message.integrity ?? "", 4096);
  if (!REDIRECT_MODES.has(redirect) || integrity === null) return null;
  return { url, method, headers, body, redirect: redirect as RedirectMode, integrity };
}

/** A message from a page, checked field by field: pages can post anything. */
export function parseFrameMessage(data: unknown): FrameMessage | null {
  if (!data || typeof data !== "object") return null;
  const message = data as Record<string, unknown>;
  if (message.source !== "unsloth-browser") return null;
  switch (message.type) {
    case "navigate": {
      const url = text(message.url, MAX_URL_CHARS);
      if (!url) return null;
      const flags = {
        newTab: message.newTab === true,
        background: message.background === true,
        replace: message.replace === true,
      };
      if (message.method !== "POST") return { type: "navigate", url, ...flags };
      const body = text(message.body, MAX_BODY_CHARS);
      return body === null ? null : { type: "navigate", url, method: "POST", body, ...flags };
    }
    case "external":
    case "url": {
      const url = text(message.url, MAX_URL_CHARS);
      return url ? { type: message.type, url } : null;
    }
    case "loaded":
      return { type: "loaded", title: title(message.title), favicon: text(message.favicon, MAX_URL_CHARS) };
    case "title":
      return { type: "title", title: title(message.title) };
    case "reload":
      return { type: "reload" };
    case "upload":
    case "scriptNavigation":
      return { type: message.type };
    case "findResult": {
      const { count, active } = message;
      return Number.isInteger(count) && Number.isInteger(active) && (count as number) >= 0 && (active as number) >= -1 && (active as number) < Math.max(count as number, 1)
        ? { type: "findResult", count: count as number, active: active as number }
        : null;
    }
    case "snapshot":
      return { type: "snapshot", html: typeof message.html === "string" && message.html.length <= MAX_SNAPSHOT_CHARS ? message.html : null };
    case "shortcut":
      return typeof message.key === "string" && SHORTCUT_KEYS.has(message.key)
        ? { type: "shortcut", key: message.key, shift: message.shift === true }
        : null;
    case "zoom":
      return message.direction === 1 || message.direction === -1 || message.direction === 0
        ? { type: "zoom", direction: message.direction, wheel: message.wheel === true }
        : null;
    case "annotate": {
      const event = annotateEvent(message);
      return event ? { type: "annotate", event } : null;
    }
    case "agent": {
      const id = text(message.id, 64);
      if (!id) return null;
      const result =
        message.result && typeof message.result === "object" && !Array.isArray(message.result)
          ? (message.result as Record<string, unknown>)
          : null;
      return { type: "agent", id, installed: message.installed === true, result };
    }
    case "request":
    case "requestAbort": {
      const id = text(message.id, 64);
      if (!id) return null;
      return message.type === "request" ? { type: "request", id, request: pageRequest(message) } : { type: "requestAbort", id };
    }
    case "storage": {
      const { area, key, value } = message;
      if (!STORAGE_AREAS.has(area)) return null;
      const url = text(message.url, MAX_URL_CHARS);
      const seq = Number.isSafeInteger(message.seq) && (message.seq as number) > 0 ? (message.seq as number) : null;
      if (key === null) return { type: "storage", area: area as PageStorageArea, key: null, value: null, url, seq };
      const name = text(key, MAX_STORAGE_CHARS);
      const stored = value === null ? null : text(value, MAX_STORAGE_CHARS);
      return name === null || (value !== null && stored === null)
        ? null
        : { type: "storage", area: area as PageStorageArea, key: name, value: stored, url, seq };
    }
    default:
      return null;
  }
}
