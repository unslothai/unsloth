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

export function MediaThumb({
  media,
  className,
}: { media: MediaValue; className?: string }) {
  const url = useMediaUrl(media);
  if (media.kind === "image") {
    return url ? (
      <img
        src={url}
        alt={media.prompt ?? ""}
        loading="lazy"
        decoding="async"
        draggable={false}
        className={className}
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
      controls={true}
      className={`nodrag ${className ?? ""}`}
    />
  ) : (
    <div className={`${className ?? ""} animate-pulse bg-muted`} />
  );
}
