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

// Chromium / Safari refuse a canvas side above 16384 px. The area is held to a memory budget well under their 2^28
// ceiling: 32 Mpx is 128 MiB of RGBA backing store, before the PNG encode, which mobile browsers can still allocate.
export const MAX_CANVAS_SIDE = 16384;
export const MAX_CANVAS_PIXELS = 32 * 1024 * 1024;

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

function clampLines(lines: string[], max: number): string[] {
  return lines.length <= max ? lines : [...lines.slice(0, max - 1), `${lines[max - 1]}\u2026`];
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
  // A label band is never taller than a cell, so shrinking the cells always shrinks the grid (an overlong label
  // is cut with an ellipsis instead of growing the canvas past its limits).
  const maxLines = Math.max(1, Math.floor(cellHeight / lineHeight));
  const colLines = hasCol ? input.colLabels.map((t) => clampLines(wrapText(t, cellWidth - gap, fontSize, measure), maxLines)) : [];
  const rowLines = hasRow ? input.rowLabels.map((t) => clampLines(wrapText(t, padLeft - gap, fontSize, measure), maxLines)) : [];
  const titleLines = hasTitle
    ? clampLines(wrapText(input.title, padLeft + gridWidth - 2 * gap, fontSize, measure), maxLines)
    : [];
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

const fitScale = (layout: GridLayout): number =>
  Math.min(
    1,
    MAX_CANVAS_SIDE / layout.width,
    MAX_CANVAS_SIDE / layout.height,
    Math.sqrt(MAX_CANVAS_PIXELS / (layout.width * layout.height)),
  );

/** Cells shrink (never grow) until the whole grid fits a browser canvas and the pixel budget. Narrower cells rewrap
 *  their labels onto more lines, so each shrunk layout is checked again until it fits. */
export function gridLayout(input: GridLayoutInput, measure: MeasureText): GridLayout {
  let layout = layoutAt(input, input.cellWidth, input.cellHeight, measure, 1);
  let s = 1;
  for (let i = 0; i < 40 && fitScale(layout) < 1; i++) {
    // A little under, so the re-derived label bands are less likely to push it back over.
    s *= fitScale(layout) * 0.98;
    layout = layoutAt(
      input,
      Math.max(1, Math.floor(input.cellWidth * s)),
      Math.max(1, Math.floor(input.cellHeight * s)),
      measure,
      s,
    );
  }
  // Cannot happen with bounded label bands; refused here rather than failing inside canvas allocation.
  if (fitScale(layout) < 1) throw new Error(`the grid is too large to draw (${layout.width}x${layout.height})`);
  return layout;
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

/** Decode and draw cells strictly one after another, so at most one decoded cell exists at a time. A cell that is
 *  missing or fails to decode keeps the placeholder already drawn. */
export async function drawCellsInTurn<R, B>(
  records: ReadonlyArray<R | null>,
  load: (record: R, index: number) => Promise<B>,
  draw: (bitmap: B, index: number) => void,
): Promise<number> {
  let drawn = 0;
  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    if (!record) continue;
    let bitmap: B;
    try {
      bitmap = await load(record, i);
    } catch {
      continue;
    }
    draw(bitmap, i);
    drawn++;
  }
  return drawn;
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
