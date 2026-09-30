// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  | { type: "shortcut"; key: string; shift: boolean };

// Same limits as the fetch endpoint.
const MAX_URL_CHARS = 8192;
const MAX_BODY_CHARS = 1024 * 1024;
const MAX_TITLE_CHARS = 1024;
const SHORTCUT_KEYS = new Set(["l", "t", "w", "r"]);

const text = (value: unknown, max: number): string | null =>
  typeof value === "string" && value.length <= max ? value : null;

const title = (value: unknown): string => (typeof value === "string" ? value.slice(0, MAX_TITLE_CHARS) : "");

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
    case "shortcut":
      return typeof message.key === "string" && SHORTCUT_KEYS.has(message.key)
        ? { type: "shortcut", key: message.key, shift: message.shift === true }
        : null;
    default:
      return null;
  }
}
