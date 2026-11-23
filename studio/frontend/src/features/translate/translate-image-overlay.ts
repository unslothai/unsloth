// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node runner can test it directly.

/** A text block Tesseract found, its box `[x0, y0, x1, y1]` as fractions of the image. */
export type ImageTextBlock = {
  text: string;
  lines: number;
  box: [number, number, number, number];
};

const LINE_HEIGHT = 1.2;

/** `sign.jpg` translated into French saves as `sign.fr.png`. */
export function overlayFileName(sourceName: string, target: string): string {
  const base = sourceName.replace(/\.[^.]+$/, "").trim() || "image";
  return `${base}.${target}.png`;
}

export function isImageTextBlock(value: unknown): value is ImageTextBlock {
  const block = value as ImageTextBlock | null;
  return (
    typeof block?.text === "string" &&
    block.text.trim() !== "" &&
    typeof block.lines === "number" &&
    Array.isArray(block.box) &&
    block.box.length === 4 &&
    block.box.every((v) => typeof v === "number" && Number.isFinite(v)) &&
    block.box[2] > block.box[0] &&
    block.box[3] > block.box[1]
  );
}

function median(values: number[]): number {
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

/** Per-channel median of RGBA pixels, skipping transparent ones; null when none is opaque. */
export function medianColor(
  rgba: ArrayLike<number>,
  skip?: (pixel: number) => boolean,
): [number, number, number] | null {
  const channels: [number[], number[], number[]] = [[], [], []];
  for (let i = 0; i + 3 < rgba.length; i += 4) {
    if (rgba[i + 3] < 128 || skip?.(i / 4)) continue;
    channels[0].push(rgba[i]);
    channels[1].push(rgba[i + 1]);
    channels[2].push(rgba[i + 2]);
  }
  if (channels[0].length === 0) return null;
  return [median(channels[0]), median(channels[1]), median(channels[2])];
}

/** Black or white, whichever reads better on `rgb` (WCAG relative luminance). */
export function readableTextColor([r, g, b]: readonly number[]): string {
  const linear = (c: number) => {
    const v = c / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const luminance =
    0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
  return (luminance + 0.05) / 0.05 > 1.05 / (luminance + 0.05)
    ? "#000000"
    : "#ffffff";
}

/** Lines no wider than `width`; a word wider than that (unspaced scripts) breaks between characters. */
export function wrapWords(
  text: string,
  width: number,
  measure: (text: string) => number,
): string[] {
  const lines: string[] = [];
  let line = "";
  const push = (piece: string, joiner: string) => {
    const next = line ? line + joiner + piece : piece;
    if (!line || measure(next) <= width) {
      line = next;
      return;
    }
    lines.push(line);
    line = piece;
  };
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (measure(word) <= width) {
      push(word, " ");
      continue;
    }
    let first = true;
    for (const char of Array.from(word)) {
      push(char, first ? " " : "");
      first = false;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** The largest font size whose wrapped lines fit the box; `measure` takes the font size. */
export function fitText(
  text: string,
  width: number,
  height: number,
  measure: (text: string, size: number) => number,
): { size: number; lines: string[] } {
  let low = 4;
  let high = Math.max(low, Math.floor(height));
  let best = {
    size: low,
    lines: wrapWords(text, width, (s) => measure(s, low)),
  };
  while (low <= high) {
    const size = Math.floor((low + high) / 2);
    const lines = wrapWords(text, width, (s) => measure(s, size));
    const fits =
      lines.length * size * LINE_HEIGHT <= height &&
      lines.every((line) => measure(line, size) <= width);
    if (fits) {
      best = { size, lines };
      low = size + 1;
    } else {
      high = size - 1;
    }
  }
  if (best.lines.length * best.size * LINE_HEIGHT <= height) return best;
  // Too long even at the smallest size: keep the lines that fit and mark the cut.
  const room = Math.max(1, Math.floor(height / (best.size * LINE_HEIGHT)));
  const kept = best.lines.slice(0, room);
  kept[room - 1] = `${kept[room - 1].replace(/\s*\S$/u, "")}\u2026`;
  return { size: best.size, lines: kept };
}

/** Paints each translated block over its box on a canvas already holding the image. */
export function drawTranslatedBlocks(
  ctx: CanvasRenderingContext2D,
  blocks: readonly ImageTextBlock[],
  texts: readonly string[],
  fontFamily: string,
): void {
  const { width, height } = ctx.canvas;
  const boxes = blocks.map(({ box, lines }) => {
    const x0 = Math.floor(box[0] * width);
    const y0 = Math.floor(box[1] * height);
    const x1 = Math.ceil(box[2] * width);
    const y1 = Math.ceil(box[3] * height);
    const pad = Math.max(
      2,
      Math.round(((y1 - y0) / Math.max(1, lines)) * 0.15),
    );
    return {
      x: Math.max(0, x0 - pad),
      y: Math.max(0, y0 - pad),
      w: Math.min(width, x1 + pad) - Math.max(0, x0 - pad),
      h: Math.min(height, y1 + pad) - Math.max(0, y0 - pad),
      pad,
    };
  });
  // Sampled before any painting, so a neighbour's fill never colours another block.
  const fills = boxes.map(({ x, y, w, h, pad }) => {
    const ring = pad * 2;
    const rx = Math.max(0, x - ring);
    const ry = Math.max(0, y - ring);
    const rw = Math.min(width, x + w + ring) - rx;
    const rh = Math.min(height, y + h + ring) - ry;
    const { data } = ctx.getImageData(rx, ry, rw, rh);
    const inside = (pixel: number) => {
      const px = rx + (pixel % rw);
      const py = ry + Math.floor(pixel / rw);
      return px >= x && px < x + w && py >= y && py < y + h;
    };
    return medianColor(data, inside) ?? medianColor(data) ?? [255, 255, 255];
  });
  boxes.forEach(({ x, y, w, h, pad }, index) => {
    const text = texts[index]?.trim();
    if (!text) return;
    const fill = fills[index];
    ctx.fillStyle = `rgb(${fill[0]}, ${fill[1]}, ${fill[2]})`;
    ctx.fillRect(x, y, w, h);
    const innerW = Math.max(1, w - pad * 2);
    const innerH = Math.max(1, h - pad * 2);
    const fitted = fitText(text, innerW, innerH, (line, size) => {
      ctx.font = `${size}px ${fontFamily}`;
      return ctx.measureText(line).width;
    });
    ctx.font = `${fitted.size}px ${fontFamily}`;
    ctx.fillStyle = readableTextColor(fill);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const lineHeight = fitted.size * LINE_HEIGHT;
    const top = y + h / 2 - (fitted.lines.length * lineHeight) / 2;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    fitted.lines.forEach((line, i) => {
      ctx.fillText(line, x + w / 2, top + lineHeight * (i + 0.5));
    });
    ctx.restore();
  });
}
