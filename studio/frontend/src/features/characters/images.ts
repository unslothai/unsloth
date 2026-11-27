// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useState } from "react";
import { openImageViewer } from "@/components/image-viewer";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { acquireObjectUrl, clearCachedObjectUrls, peekObjectUrl } from "@/lib/object-url-cache";
import { fetchAvatar, type AvatarKind } from "./api";
import { AVATAR_TYPES } from "./avatar-upload";

function avatarKey(kind: AvatarKind, id: string, rev: number): string {
  return `avatar:${kind}:${id}:${rev}`;
}

/** The avatar's object URL, or null while loading, missing or failed; held while mounted so it is never revoked on screen. */
export function useAvatarUrl(
  kind: AvatarKind,
  id: string | null | undefined,
  rev: number,
  hasAvatar: boolean,
): string | null {
  const key = id && hasAvatar ? avatarKey(kind, id, rev) : null;
  const [state, setState] = useState<{ key: string; url: string | null } | null>(null);
  // Layout effect: an avatar already loaded shows before paint instead of flashing the monogram.
  useLayoutEffect(() => {
    if (!key || !id) return;
    let live = true;
    const { url, release } = acquireObjectUrl(key, () => fetchAvatar(kind, id, rev));
    const ready = peekObjectUrl(key);
    if (ready) setState({ key, url: ready });
    else url.then((next) => live && setState({ key, url: next }), () => live && setState({ key, url: null }));
    return () => {
      live = false;
      release();
    };
  }, [key, kind, id, rev]);
  return key && state?.key === key ? state.url : null;
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
    clearCachedObjectUrls();
    cutouts.clear();
    cutoutAnswers.clear();
  });
}

export function viewAvatar(kind: AvatarKind, id: string, name: string, rev: number): void {
  const base = name.trim().replace(/[\\/:*?"<>|]+/g, "_") || kind;
  openImageViewer([
    {
      key: avatarKey(kind, id, rev),
      title: name,
      fileName: (contentType) => `${base}.${AVATAR_TYPES[contentType]?.[0] ?? "png"}`,
      load: () => fetchAvatar(kind, id, rev),
      closeOnClick: true,
    },
  ]);
}
