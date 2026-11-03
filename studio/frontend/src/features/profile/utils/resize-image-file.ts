// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EDGE_STEP = 32;
const JPEG_QUALITY_START = 0.88;
const WEBP_QUALITY_START = 0.9;
const MIN_QUALITY = 0.45;
const QUALITY_STEP = 0.08;

export type ResizeLimits = {
  maxEdge: number;
  minEdge: number;
  maxBytes: number;
};

const AVATAR_LIMITS: ResizeLimits = {
  maxEdge: 256,
  minEdge: 96,
  maxBytes: 380_000,
};

function maxDataUrlLength(limits: ResizeLimits): number {
  return Math.floor(limits.maxBytes * 1.35);
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load image"));
    };
    img.src = url;
  });
}

function canvasHasTransparency(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): boolean {
  const { data } = ctx.getImageData(0, 0, width, height);
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) return true;
  }
  return false;
}

function encodeCanvasWithinLimit(
  canvas: HTMLCanvasElement,
  mimeType: string,
  startQuality: number,
  maxLength: number,
): string | null {
  let quality = startQuality;
  let dataUrl = canvas.toDataURL(mimeType, quality);
  if (!dataUrl.startsWith(`data:${mimeType}`)) return null;

  while (dataUrl.length > maxLength && quality > MIN_QUALITY) {
    quality -= QUALITY_STEP;
    dataUrl = canvas.toDataURL(mimeType, quality);
  }

  return dataUrl.length <= maxLength ? dataUrl : null;
}

// Safari cannot encode WebP, so PNG is the universal alpha-preserving fallback.
function encodePngWithinLimit(
  canvas: HTMLCanvasElement,
  maxLength: number,
): string | null {
  const dataUrl = canvas.toDataURL("image/png");
  if (!dataUrl.startsWith("data:image/png")) return null;
  return dataUrl.length <= maxLength ? dataUrl : null;
}

function drawScaled(
  img: HTMLImageElement,
  w: number,
  h: number,
  maxEdge: number,
): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  cw: number;
  ch: number;
} {
  const scale = Math.min(1, maxEdge / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));
  const canvas = document.createElement("canvas");
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not available");
  ctx.drawImage(img, 0, 0, cw, ch);
  return { canvas, ctx, cw, ch };
}

export async function resizeImageFileToDataUrl(
  file: File,
  limits: ResizeLimits = AVATAR_LIMITS,
): Promise<string> {
  const img = await loadImage(file);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (!w || !h) throw new Error("Invalid image dimensions");

  const { maxEdge, minEdge } = limits;
  const maxLength = maxDataUrlLength(limits);
  const base = drawScaled(img, w, h, maxEdge);
  const hasTransparency = canvasHasTransparency(base.ctx, base.cw, base.ch);

  if (hasTransparency) {
    // Never JPEG: it would paint a background behind a transparent image.
    for (let edge = maxEdge; edge >= minEdge; edge -= EDGE_STEP) {
      const { canvas } = edge === maxEdge ? base : drawScaled(img, w, h, edge);
      const webpDataUrl = encodeCanvasWithinLimit(
        canvas,
        "image/webp",
        WEBP_QUALITY_START,
        maxLength,
      );
      if (webpDataUrl) return webpDataUrl;
      const pngDataUrl = encodePngWithinLimit(canvas, maxLength);
      if (pngDataUrl) return pngDataUrl;
    }
    throw new Error(
      "Image is still too large after compression. Try a smaller file.",
    );
  }

  const jpegDataUrl = encodeCanvasWithinLimit(
    base.canvas,
    "image/jpeg",
    JPEG_QUALITY_START,
    maxLength,
  );
  if (jpegDataUrl) return jpegDataUrl;

  throw new Error(
    "Image is still too large after compression. Try a smaller file.",
  );
}
