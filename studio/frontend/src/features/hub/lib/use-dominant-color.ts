// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

const colorCache = new Map<string, string | null>();
const inflight = new Map<string, Promise<string | null>>();

function computeDominant(data: Uint8ClampedArray): string | null {
  const buckets = new Map<
    number,
    { r: number; g: number; b: number; w: number }
  >();
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) continue;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const light = (max + min) / 510;
    const sat = max === 0 ? 0 : (max - min) / max;
    if (light > 0.93 || light < 0.07) continue;
    if (sat < 0.18) continue;
    const key = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
    const w = sat * (1 - Math.abs(light - 0.5));
    const bucket = buckets.get(key);
    if (bucket) {
      bucket.r += r * w;
      bucket.g += g * w;
      bucket.b += b * w;
      bucket.w += w;
    } else {
      buckets.set(key, { r: r * w, g: g * w, b: b * w, w });
    }
  }
  let best: { r: number; g: number; b: number; w: number } | null = null;
  for (const bucket of buckets.values()) {
    if (!best || bucket.w > best.w) best = bucket;
  }
  if (!best || best.w < 2) return null;
  return `rgb(${Math.round(best.r / best.w)}, ${Math.round(best.g / best.w)}, ${Math.round(best.b / best.w)})`;
}

function extractColor(url: string): Promise<string | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => {
      try {
        const size = 36;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0, size, size);
        resolve(computeDominant(ctx.getImageData(0, 0, size, size).data));
      } catch {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

export function useDominantColor(url: string | null): string | null {
  const [color, setColor] = useState<string | null>(() =>
    url ? (colorCache.get(url) ?? null) : null,
  );
  useEffect(() => {
    if (!url) {
      setColor(null);
      return;
    }
    if (colorCache.has(url)) {
      setColor(colorCache.get(url) ?? null);
      return;
    }
    let alive = true;
    let promise = inflight.get(url);
    if (!promise) {
      promise = extractColor(url);
      inflight.set(url, promise);
    }
    void promise.then((next) => {
      colorCache.set(url, next);
      inflight.delete(url);
      if (alive) setColor(next);
    });
    return () => {
      alive = false;
    };
  }, [url]);
  return color;
}
