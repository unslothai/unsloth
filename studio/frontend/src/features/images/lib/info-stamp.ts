// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Optional visible info stamp for a downloaded image: `Model | date time | Seed: N` on a shaded
 *  bar along the bottom edge. Drawn only on the exported copy, from the stored original, so the
 *  gallery file, the recipe and any later edit input stay clean and re-exporting never stacks. */

export interface StampSource {
  model: string | null;
  seed: number;
  created_at: number;
}

const BRAND = "Unsloth";
const PATH_SEP = /[\\/]/;
const GGUF_SUFFIX = /(\.gguf|-gguf)$/i;
const FONT_FAMILY =
  'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';

/** Repo or file basename without a trailing -GGUF: `unsloth/FLUX.2-klein-4B-GGUF` -> `FLUX.2-klein-4B`. */
export function stampModelName(model: string | null | undefined): string {
  const base = (model ?? "").trim().split(PATH_SEP).filter(Boolean).pop() ?? "";
  return base.replace(GGUF_SUFFIX, "");
}

export function stampDateTime(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

export function infoStampText(image: StampSource): string {
  const parts = [
    stampModelName(image.model),
    stampDateTime(image.created_at),
    `Seed: ${image.seed}`,
  ];
  return parts.filter(Boolean).join(" | ");
}

/** The subset of CanvasRenderingContext2D the stamp uses, so the layout is testable without a DOM. */
export interface StampContext {
  font: string;
  fillStyle: string | CanvasGradient | CanvasPattern;
  textBaseline: CanvasTextBaseline;
  textAlign: CanvasTextAlign;
  fillRect(x: number, y: number, w: number, h: number): void;
  fillText(text: string, x: number, y: number): void;
  measureText(text: string): { width: number };
}

export interface StampLayout {
  fontPx: number;
  barHeight: number;
  padX: number;
  showBrand: boolean;
}

/** Font scales with the short side; a long line shrinks to fit, dropping the brand first. */
export function layoutInfoStamp(
  ctx: StampContext,
  width: number,
  height: number,
  text: string,
): StampLayout {
  let fontPx = Math.max(10, Math.round(Math.min(width, height) / 48));
  const padX = Math.round(fontPx * 0.75);
  const widthAt = (s: string, px: number) => {
    ctx.font = `${px}px ${FONT_FAMILY}`;
    return ctx.measureText(s).width;
  };
  const room = width - padX * 2;
  const textWidth = widthAt(text, fontPx);
  const showBrand = textWidth + fontPx * 2 + widthAt(BRAND, fontPx) <= room;
  if (textWidth > room) {
    fontPx = Math.max(6, Math.floor((fontPx * room) / textWidth));
  }
  return { fontPx, barHeight: Math.round(fontPx * 1.9), padX, showBrand };
}

export function drawInfoStamp(
  ctx: StampContext,
  width: number,
  height: number,
  text: string,
): StampLayout {
  const layout = layoutInfoStamp(ctx, width, height, text);
  const top = height - layout.barHeight;
  const mid = top + layout.barHeight / 2;
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(0, top, width, layout.barHeight);
  ctx.font = `${layout.fontPx}px ${FONT_FAMILY}`;
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.textAlign = "left";
  ctx.fillText(text, layout.padX, mid);
  if (layout.showBrand) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.textAlign = "right";
    ctx.fillText(BRAND, width - layout.padX, mid);
  }
  return layout;
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const TEXT_CHUNKS = new Set(["tEXt", "iTXt", "zTXt"]);

interface PngChunk {
  type: string;
  start: number;
  end: number;
}

function pngChunks(bytes: Uint8Array): PngChunk[] | null {
  if (bytes.length < 8 || PNG_SIGNATURE.some((b, i) => bytes[i] !== b)) {
    return null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: PngChunk[] = [];
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const length = view.getUint32(offset);
    const end = offset + 12 + length;
    if (end > bytes.length) {
      return null;
    }
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    chunks.push({ type, start: offset, end });
    offset = end;
    if (type === "IEND") {
      return chunks;
    }
  }
  return null;
}

/** `target` with its text chunks replaced by `source`'s (copied verbatim, CRCs intact) right after
 *  IHDR, so a stamped PNG still carries the recipe. Either side unparsable: `target` unchanged. */
export function copyPngTextChunks(
  source: Uint8Array,
  target: Uint8Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> {
  const from = pngChunks(source);
  const into = pngChunks(target);
  if (!from || !into || into[0]?.type !== "IHDR") {
    return target;
  }
  const text = from.filter((c) => TEXT_CHUNKS.has(c.type));
  const parts: Uint8Array[] = [target.subarray(0, into[0].end)];
  for (const c of text) parts.push(source.subarray(c.start, c.end));
  for (const c of into.slice(1)) {
    if (!TEXT_CHUNKS.has(c.type)) parts.push(target.subarray(c.start, c.end));
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
