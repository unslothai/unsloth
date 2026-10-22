// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The labelled grid image: layout after AUTOMATIC1111's images.draw_grid_annotations (font size (w + h) / 25,
 *  row labels in a left band 3/4 of a cell wide, column labels above, margin between cells), plus the Z value as a
 *  title band. Layout is pure (tests pass a fake text measure); drawing needs a 2D canvas. */

export type MeasureText = (text: string, fontSize: number) => number;

export interface GridLayoutInput {
  cols: number;
  rows: number;
  cellWidth: number;
  cellHeight: number;
  colLabels: readonly string[];
  rowLabels: readonly string[];
  title: string;
  legend: boolean;
  margin: number;
}

export interface TextBlock {
  x: number; // centre
  y: number; // top of the first line
  lines: string[];
}

export interface GridLayout {
  width: number;
  height: number;
  scale: number;
  fontSize: number;
  lineHeight: number;
  cellWidth: number;
  cellHeight: number;
  cells: Array<{ x: number; y: number; w: number; h: number }>; // index ix + iy * cols
  colText: TextBlock[];
  rowText: TextBlock[];
  title: TextBlock | null;
}

// Chromium / Safari canvas ceilings: either side above 16384 px, or the area above 2^28, fails to allocate.
export const MAX_CANVAS_SIDE = 16384;
export const MAX_CANVAS_AREA = 268_435_456;

export function wrapText(text: string, maxWidth: number, fontSize: number, measure: MeasureText): string[] {
  if (!text) return [];
  const lines: string[] = [];
  for (const paragraph of text.split("\n")) {
    let line = "";
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      const next = line ? `${line} ${word}` : word;
      if (line && measure(next, fontSize) > maxWidth) {
        lines.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

function layoutAt(input: GridLayoutInput, cellWidth: number, cellHeight: number, measure: MeasureText, scale: number): GridLayout {
  const { cols, rows, legend } = input;
  const margin = Math.round(Math.max(0, input.margin) * scale);
  const fontSize = Math.max(10, Math.round((cellWidth + cellHeight) / 25));
  const lineHeight = Math.round(fontSize * 1.25);
  const gap = Math.round(fontSize / 2);
  const hasCol = legend && input.colLabels.some(Boolean);
  const hasRow = legend && input.rowLabels.some(Boolean);
  const hasTitle = legend && Boolean(input.title);
  const padLeft = hasRow ? Math.floor((cellWidth * 3) / 4) : 0;
  const gridWidth = cols * cellWidth + (cols - 1) * margin;
  const colLines = hasCol ? input.colLabels.map((t) => wrapText(t, cellWidth - gap, fontSize, measure)) : [];
  const rowLines = hasRow ? input.rowLabels.map((t) => wrapText(t, padLeft - gap, fontSize, measure)) : [];
  const titleLines = hasTitle ? wrapText(input.title, padLeft + gridWidth - 2 * gap, fontSize, measure) : [];
  const padTop = hasCol ? Math.max(1, ...colLines.map((l) => l.length)) * lineHeight + gap * 2 : 0;
  const titleHeight = hasTitle ? titleLines.length * lineHeight + gap * 2 : 0;
  const top = titleHeight + padTop;
  const width = padLeft + gridWidth;
  const height = top + rows * cellHeight + (rows - 1) * margin;
  const cells = [];
  for (let iy = 0; iy < rows; iy++) {
    for (let ix = 0; ix < cols; ix++) {
      cells.push({ x: padLeft + ix * (cellWidth + margin), y: top + iy * (cellHeight + margin), w: cellWidth, h: cellHeight });
    }
  }
  const colText = colLines.map((lines, ix) => ({
    x: padLeft + ix * (cellWidth + margin) + cellWidth / 2,
    y: titleHeight + (padTop - lines.length * lineHeight) / 2,
    lines,
  }));
  const rowText = rowLines.map((lines, iy) => ({
    x: padLeft / 2,
    y: top + iy * (cellHeight + margin) + (cellHeight - lines.length * lineHeight) / 2,
    lines,
  }));
  const title = hasTitle ? { x: width / 2, y: gap, lines: titleLines } : null;
  return { width, height, scale, fontSize, lineHeight, cellWidth, cellHeight, cells, colText, rowText, title };
}

/** Cells shrink (never grow) until the whole grid fits a browser canvas. */
export function gridLayout(input: GridLayoutInput, measure: MeasureText): GridLayout {
  const full = layoutAt(input, input.cellWidth, input.cellHeight, measure, 1);
  const scale = Math.min(
    1,
    MAX_CANVAS_SIDE / full.width,
    MAX_CANVAS_SIDE / full.height,
    Math.sqrt(MAX_CANVAS_AREA / (full.width * full.height)),
  );
  if (scale >= 1) return full;
  // A little under, so the re-derived label bands cannot push it back over.
  const s = scale * 0.98;
  return layoutAt(input, Math.floor(input.cellWidth * s), Math.floor(input.cellHeight * s), measure, s);
}

const FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

/** Draw the grid; `images[i]` null = a cell that did not run (stopped or failed), drawn as a grey placeholder. */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  layout: GridLayout,
  images: ReadonlyArray<CanvasImageSource | null>,
): void {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, layout.width, layout.height);
  layout.cells.forEach((cell, i) => {
    const image = images[i];
    if (image) {
      ctx.drawImage(image, cell.x, cell.y, cell.w, cell.h);
      return;
    }
    ctx.fillStyle = "#d4d4d4";
    ctx.fillRect(cell.x, cell.y, cell.w, cell.h);
    ctx.fillStyle = "#525252";
    ctx.font = `${layout.fontSize}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("not generated", cell.x + cell.w / 2, cell.y + cell.h / 2);
  });
  ctx.fillStyle = "#000000";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = `${layout.fontSize}px ${FONT}`;
  const blocks = [...layout.colText, ...layout.rowText];
  for (const block of blocks) {
    block.lines.forEach((line, j) => ctx.fillText(line, block.x, block.y + j * layout.lineHeight));
  }
  if (layout.title) {
    ctx.font = `bold ${layout.fontSize}px ${FONT}`;
    const t = layout.title;
    t.lines.forEach((line, j) => ctx.fillText(line, t.x, t.y + j * layout.lineHeight));
  }
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** `png` with an uncompressed iTXt chunk (UTF-8, so any prompt survives) right after IHDR. Not a PNG: unchanged. */
export function withPngText(png: Uint8Array, keyword: string, text: string): Uint8Array {
  const SIG = [137, 80, 78, 71, 13, 10, 26, 10];
  if (png.length < 33 || SIG.some((b, i) => png[i] !== b)) return png;
  const ihdrEnd = 8 + 12 + 13;
  const enc = new TextEncoder();
  const body = new Uint8Array([...enc.encode(keyword), 0, 0, 0, 0, 0, ...enc.encode(text)]);
  const chunk = new Uint8Array(12 + body.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, body.length);
  chunk.set(enc.encode("iTXt"), 4);
  chunk.set(body, 8);
  view.setUint32(8 + body.length, crc32(chunk.subarray(4, 8 + body.length)));
  const out = new Uint8Array(png.length + chunk.length);
  out.set(png.subarray(0, ihdrEnd), 0);
  out.set(chunk, ihdrEnd);
  out.set(png.subarray(ihdrEnd), ihdrEnd + chunk.length);
  return out;
}

const quote = (v: string) => (/[,:"\n]/.test(v) ? JSON.stringify(v) : v);

/** A1111-style `parameters` for the grid: the shared recipe, then `Script: X/Y/Z plot` and each axis's type and
 *  values (A1111 xyz_grid.py Script.run, the grid infotext). */
export function gridInfotext(input: {
  prompt: string;
  negativePrompt?: string;
  fields: ReadonlyArray<[string, string | number | null | undefined]>;
  axes: ReadonlyArray<{ name: string; type: string; text: string; fixed?: readonly (string | number)[] }>;
}): string {
  const parts: string[] = [];
  for (const [key, value] of input.fields) if (value !== null && value !== undefined && value !== "") parts.push(`${key}: ${quote(String(value))}`);
  parts.push("Script: X/Y/Z plot");
  for (const axis of input.axes) {
    parts.push(`${axis.name} Type: ${quote(axis.type)}`, `${axis.name} Values: ${quote(axis.text)}`);
    if (axis.fixed) parts.push(`Fixed ${axis.name} Values: ${quote(axis.fixed.join(", "))}`);
  }
  const lines = [input.prompt];
  if (input.negativePrompt) lines.push(`Negative prompt: ${input.negativePrompt}`);
  lines.push(parts.join(", "));
  return lines.join("\n");
}
