// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { openImageViewer } from "@/components/image-viewer";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { fetchAvatar, type AvatarKind } from "./api";
import { AVATAR_TYPES } from "./avatar-upload";

// One URL per avatar: a new revision replaces (and revokes) the old one, so edits never pile up blobs.
const cache = new BlobUrlCache(Number.POSITIVE_INFINITY);
const cachedRevs = new Map<string, number>();
const pending = new Map<string, Promise<string | null>>();
let generation = 0;

function slot(kind: AvatarKind, id: string): string {
  return `${kind}:${id}`;
}

function cachedUrl(key: string, rev: number): string | undefined {
  return cachedRevs.get(key) === rev ? cache.get(key) : undefined;
}

function loadAvatarUrl(kind: AvatarKind, id: string, rev: number): Promise<string | null> {
  const key = slot(kind, id);
  const known = cachedUrl(key, rev);
  if (known) return Promise.resolve(known);
  const pendingKey = `${key}:${rev}`;
  const hit = pending.get(pendingKey);
  if (hit) return hit;
  const started = generation;
  const request = fetchAvatar(kind, id, rev)
    .then((blob) => {
      if (started !== generation) return null;
      // An older revision landing late must not replace a newer picture.
      if ((cachedRevs.get(key) ?? -1) > rev) return cache.get(key) ?? null;
      const old = cache.get(key);
      const url = URL.createObjectURL(blob);
      cache.set(key, url, blob.size);
      cachedRevs.set(key, rev);
      if (old) forgetCutout(old);
      return url;
    })
    // A failed load is not remembered, so the next look tries again.
    .catch(() => null)
    .finally(() => pending.delete(pendingKey));
  pending.set(pendingKey, request);
  return request;
}

/** The avatar's object URL, or null while loading, when there is none, or when it failed. */
export function useAvatarUrl(
  kind: AvatarKind,
  id: string | null | undefined,
  rev: number,
  hasAvatar: boolean,
): string | null {
  const key = id && hasAvatar ? slot(kind, id) : null;
  const [url, setUrl] = useState<string | null>(() => (key ? (cachedUrl(key, rev) ?? null) : null));
  useEffect(() => {
    if (!key || !id) {
      setUrl(null);
      return;
    }
    const known = cachedUrl(key, rev);
    if (known !== undefined) {
      setUrl(known);
      return;
    }
    let live = true;
    void loadAvatarUrl(kind, id, rev).then((next) => {
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
function isCutout(url: string): Promise<boolean> {
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

function forgetCutout(url: string): void {
  cutouts.delete(url);
  cutoutAnswers.delete(url);
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
    generation += 1;
    cache.clear();
    cachedRevs.clear();
    pending.clear();
    cutouts.clear();
    cutoutAnswers.clear();
  });
}

export function viewAvatar(kind: AvatarKind, id: string, name: string, rev: number): void {
  const base = name.trim().replace(/[\\/:*?"<>|]+/g, "_") || kind;
  openImageViewer([
    {
      key: `${slot(kind, id)}:${rev}`,
      title: name,
      fileName: (contentType) => `${base}.${AVATAR_TYPES[contentType]?.[0] ?? "png"}`,
      load: () => fetchAvatar(kind, id, rev),
    },
  ]);
}
