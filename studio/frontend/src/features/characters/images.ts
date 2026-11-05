// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { openImageViewer } from "@/components/image-viewer";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { fetchImage } from "./api";
import { isTitle } from "./prompt/speakers";

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
  const request = fetchImage(kind, id, rev)
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

const cutouts = new Map<string, Promise<boolean>>();
const cutoutAnswers = new Map<string, boolean>();

/**
 * Whether a picture is a cut-out: a figure on a transparent background, judged from a small copy's top
 * rows and side columns (the bottom edge often holds the figure). Opaque pictures answer false.
 */
export function isCutout(url: string): Promise<boolean> {
  let known = cutouts.get(url);
  if (!known) {
    known = new Promise<boolean>((resolve) => {
      const image = new Image();
      image.onload = () => {
        try {
          const width = 32;
          const height = 48;
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const context = canvas.getContext("2d", { willReadFrequently: true });
          if (!context) return resolve(false);
          context.drawImage(image, 0, 0, width, height);
          const pixels = context.getImageData(0, 0, width, height).data;
          let edge = 0;
          let clear = 0;
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              if (y > 1 && x > 1 && x < width - 2) continue;
              edge++;
              if (pixels[(y * width + x) * 4 + 3] < 32) clear++;
            }
          }
          resolve(clear / edge > 0.4);
        } catch {
          resolve(false);
        }
      };
      image.onerror = () => resolve(false);
      image.src = url;
    }).then((answer) => {
      cutoutAnswers.set(url, answer);
      return answer;
    });
    cutouts.set(url, known);
  }
  return known;
}

/** isCutout for a picture on screen; false until it is known, and always false when not enabled. */
export function useIsCutout(url: string | null, enabled: boolean): boolean {
  const known = url && enabled ? cutoutAnswers.get(url) : false;
  const [cutout, setCutout] = useState(known ?? false);
  useEffect(() => {
    const answer = url && enabled ? cutoutAnswers.get(url) : false;
    setCutout(answer ?? false);
    if (!url || answer !== undefined) return;
    let live = true;
    void isCutout(url).then((value) => {
      if (live) setCutout(value);
    });
    return () => {
      live = false;
    };
  }, [url, enabled]);
  return cutout;
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    for (const url of resolved.values()) if (url) URL.revokeObjectURL(url);
    urls.clear();
    resolved.clear();
    cutouts.clear();
    cutoutAnswers.clear();
  });
}

const EXTENSIONS: Record<string, string> = { "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif" };

export function viewPicture(kind: Kind, id: string, name: string, rev: number): void {
  const base = name.trim().replace(/[\\/:*?"<>|]+/g, "_") || kind;
  openImageViewer([
    {
      key: cacheKey(kind, id, rev),
      title: name,
      fileName: (contentType) => `${base}.${EXTENSIONS[contentType] ?? "png"}`,
      load: () => fetchImage(kind, id, rev),
    },
  ]);
}

const HUES = [12, 28, 160, 190, 215, 250, 280, 330];

/** A stable hue per name, so a portrait-less character keeps its color everywhere. */
export function nameHue(name: string): number {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.codePointAt(0)!) >>> 0;
  return HUES[hash % HUES.length];
}

export function initials(name: string): string {
  // "Inspector Theo Lindqvist" is TL, not IL.
  const all = name.trim().split(/\s+/).filter(Boolean);
  const start = all.findIndex((word) => !isTitle(word));
  const words = start > 0 ? all.slice(start) : all;
  if (words.length === 0) return "?";
  const first = [...words[0]][0] ?? "";
  const second = words.length > 1 ? ([...words[words.length - 1]][0] ?? "") : "";
  return (first + second).toUpperCase();
}
