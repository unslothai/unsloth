// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { fetchImage } from "./api";

type Kind = "character" | "persona";

// Object URLs keyed by revision: a new picture gets a new key, so stale ones are never shown.
const urls = new Map<string, Promise<string | null>>();
const resolved = new Map<string, string | null>();

function cacheKey(kind: Kind, id: string, rev: number): string {
  return `${kind}:${id}:${rev}`;
}

export function loadImageUrl(kind: Kind, id: string, rev: number): Promise<string | null> {
  const key = cacheKey(kind, id, rev);
  const hit = urls.get(key);
  if (hit) return hit;
  const request = fetchImage(kind, id)
    .then((blob) => URL.createObjectURL(blob))
    .catch(() => null)
    .then((url) => {
      if (url) resolved.set(key, url);
      // A failed load is not remembered, so the next look tries again.
      else urls.delete(key);
      return url;
    });
  urls.set(key, request);
  return request;
}

/** The picture's object URL, or null while loading, when there is none, or when it failed. */
export function useImageUrl(
  kind: Kind,
  id: string | null | undefined,
  rev: number,
  hasImage: boolean,
): string | null {
  const key = id && hasImage ? cacheKey(kind, id, rev) : null;
  const [url, setUrl] = useState<string | null>(() => (key ? (resolved.get(key) ?? null) : null));
  useEffect(() => {
    if (!key || !id) {
      setUrl(null);
      return;
    }
    const known = resolved.get(key);
    if (known !== undefined) {
      setUrl(known);
      return;
    }
    let live = true;
    void loadImageUrl(kind, id, rev).then((next) => {
      if (live) setUrl(next);
    });
    return () => {
      live = false;
    };
  }, [key, kind, id, rev]);
  return url;
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    for (const url of resolved.values()) if (url) URL.revokeObjectURL(url);
    urls.clear();
    resolved.clear();
  });
}

const HUES = [12, 28, 160, 190, 215, 250, 280, 330];

/** A stable hue per name, so a portrait-less character keeps its color everywhere. */
export function nameHue(name: string): number {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.codePointAt(0)!) >>> 0;
  return HUES[hash % HUES.length];
}

export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  const first = [...words[0]][0] ?? "";
  const second = words.length > 1 ? ([...words[words.length - 1]][0] ?? "") : "";
  return (first + second).toUpperCase();
}
