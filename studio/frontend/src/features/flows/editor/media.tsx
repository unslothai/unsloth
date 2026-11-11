// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { fetchMediaBlob, mediaPath } from "../api";
import type { MediaValue } from "../types";

const cache = new Map<
  string,
  { url: Promise<string>; refs: number; resolved?: string }
>();

function acquire(path: string): Promise<string> {
  const hit = cache.get(path);
  if (hit) {
    hit.refs += 1;
    return hit.url;
  }
  const entry: { url: Promise<string>; refs: number; resolved?: string } = {
    refs: 1,
    url: Promise.resolve(""),
  };
  entry.url = fetchMediaBlob(path).then((blob) => {
    const url = URL.createObjectURL(blob);
    entry.resolved = url;
    return url;
  });
  entry.url.catch(() => cache.delete(path));
  cache.set(path, entry);
  return entry.url;
}

function release(path: string): void {
  const hit = cache.get(path);
  if (!hit) return;
  hit.refs -= 1;
  if (hit.refs > 0) return;
  cache.delete(path);
  if (hit.resolved) URL.revokeObjectURL(hit.resolved);
  else
    void hit.url.then((url) => URL.revokeObjectURL(url)).catch(() => undefined);
}

export function isMediaValue(value: unknown): value is MediaValue {
  if (!value || typeof value !== "object") return false;
  const v = value as { kind?: unknown; ref?: unknown };
  return (
    (v.kind === "image" || v.kind === "audio" || v.kind === "video") &&
    typeof v.ref === "string"
  );
}

export function useMediaUrl(
  media: MediaValue | null | undefined,
): string | null {
  const path = media ? mediaPath(media) : null;
  const inline =
    path && (path.startsWith("data:") || path.startsWith("blob:"))
      ? path
      : null;
  const [url, setUrl] = useState<string | null>(inline);
  useEffect(() => {
    if (!path || inline) {
      setUrl(inline);
      return;
    }
    let alive = true;
    setUrl(null);
    acquire(path)
      .then((resolved) => {
        if (alive) setUrl(resolved);
      })
      .catch(() => {
        if (alive) setUrl(null);
      });
    return () => {
      alive = false;
      release(path);
    };
  }, [path, inline]);
  return url;
}

/** A small copy of an image, so a zoomed-out canvas full of them pans without decoding full size. */
function useSmallUrl(url: string | null, enabled: boolean): string | null {
  const [small, setSmall] = useState<string | null>(null);
  const want = enabled && !!url && !small;
  useEffect(() => {
    if (!want || !url) return;
    let alive = true;
    let made: string | null = null;
    fetch(url)
      .then((r) => r.blob())
      .then((blob) => createImageBitmap(blob, { resizeWidth: SMALL_PX, resizeQuality: "low" }))
      .then((bitmap) => {
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
        bitmap.close();
        return new Promise<Blob | null>((done) => canvas.toBlob(done, "image/webp", 0.7));
      })
      .then((blob) => {
        if (!blob || !alive) return;
        made = URL.createObjectURL(blob);
        setSmall(made);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [want, url]);
  useEffect(() => () => {
    if (small) URL.revokeObjectURL(small);
  }, [small]);
  useEffect(() => setSmall(null), [url]);
  return small;
}

const SMALL_PX = 192;

/** `low`: the canvas is zoomed far out, so a small copy and no video controls are plenty. */
export function MediaThumb({
  media,
  className,
  low = false,
}: { media: MediaValue; className?: string; low?: boolean }) {
  const url = useMediaUrl(media);
  const small = useSmallUrl(media.kind === "image" ? url : null, low);
  if (media.kind === "image") {
    return url ? (
      <img
        src={(low && small) || url}
        alt={media.prompt ?? ""}
        loading="lazy"
        decoding="async"
        draggable={false}
        className={`flow-media ${className ?? ""}`}
      />
    ) : (
      <div className={`${className ?? ""} animate-pulse bg-muted`} />
    );
  }
  if (media.kind === "audio") {
    return url ? (
      // biome-ignore lint/a11y/useMediaCaption: generated audio has no captions
      <audio
        src={url}
        controls={true}
        preload="none"
        className="nodrag h-8 w-full"
      />
    ) : (
      <div className="h-8 w-full animate-pulse rounded bg-muted" />
    );
  }
  return url ? (
    // biome-ignore lint/a11y/useMediaCaption: generated video has no captions
    <video
      src={url}
      muted={true}
      playsInline={true}
      preload="metadata"
      controls={!low}
      className={`flow-media nodrag ${className ?? ""}`}
    />
  ) : (
    <div className={`${className ?? ""} animate-pulse bg-muted`} />
  );
}
