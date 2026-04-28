// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MAX_EDGE = 256;
const MAX_BYTES = 380_000;
const MAX_DATA_URL_LENGTH = Math.floor(MAX_BYTES * 1.35);
const JPEG_QUALITY_START = 0.88;
const WEBP_QUALITY_START = 0.9;
const MIN_QUALITY = 0.45;
const QUALITY_STEP = 0.08;

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

function canvasHasTransparency(ctx: CanvasRenderingContext2D, width: number, height: number): boolean {
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
): string | null {
  let quality = startQuality;
  let dataUrl = canvas.toDataURL(mimeType, quality);
  if (!dataUrl.startsWith(`data:${mimeType}`)) return null;

  while (dataUrl.length > MAX_DATA_URL_LENGTH && quality > MIN_QUALITY) {
    quality -= QUALITY_STEP;
    dataUrl = canvas.toDataURL(mimeType, quality);
  }

  return dataUrl.length <= MAX_DATA_URL_LENGTH ? dataUrl : null;
}

/**
 * Downscale the image and keep transparency when present while staying localStorage-friendly.
 */
export async function resizeImageFileToDataUrl(file: File): Promise<string> {
  const img = await loadImage(file);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (!w || !h) throw new Error("Invalid image dimensions");

  const scale = Math.min(1, MAX_EDGE / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));

  const canvas = document.createElement("canvas");
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not available");
  ctx.drawImage(img, 0, 0, cw, ch);

  const hasTransparency = canvasHasTransparency(ctx, cw, ch);
  if (hasTransparency) {
    const webpDataUrl = encodeCanvasWithinLimit(canvas, "image/webp", WEBP_QUALITY_START);
    if (webpDataUrl) return webpDataUrl;
  }

  const jpegDataUrl = encodeCanvasWithinLimit(canvas, "image/jpeg", JPEG_QUALITY_START);
  if (jpegDataUrl) return jpegDataUrl;

  throw new Error("Image is still too large after compression. Try a smaller file.");
}
