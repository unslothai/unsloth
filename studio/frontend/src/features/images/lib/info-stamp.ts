// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `Model | date time | Seed: N` bar on the exported copy only, so the stored file stays clean and
 *  re-exporting never stacks. */

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
  /** Shortened from the model, then the date, so the seed is never clipped. */
  text: string;
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
  let fitted = text;
  if (widthAt(fitted, fontPx) > room) {
    const parts = text.split(" | ");
    const seed = parts.pop() ?? "";
    while (parts.length > 0 && widthAt(fitted, fontPx) > room) {
      const first = parts[0];
      if (parts.length > 1 && first.length > 1) {
        parts[0] = `${first.replace(/\u2026$/, "").slice(0, -1)}\u2026`;
        if (parts[0] === "\u2026") parts.shift();
      } else {
        parts.shift();
      }
      fitted = [...parts, seed].join(" | ");
    }
  }
  return {
    text: fitted,
    fontPx,
    barHeight: Math.round(fontPx * 1.9),
    padX,
    showBrand,
  };
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
  ctx.fillText(layout.text, layout.padX, mid);
  if (layout.showBrand) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.textAlign = "right";
    ctx.fillText(BRAND, width - layout.padX, mid);
  }
  return layout;
}
