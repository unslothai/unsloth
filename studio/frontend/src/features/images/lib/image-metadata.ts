// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Generation settings embedded in exported images, and read back from dropped ones.
 *  PNG: `unsloth` (recipe JSON) and `parameters` (AUTOMATIC1111 text) text chunks, as the backend
 *  writes them. JPEG / WebP: EXIF with `parameters` in UserComment (what AUTOMATIC1111, Forge and
 *  Civitai read) and the recipe JSON in MakerNote. Pure byte work, so it runs in tests without a DOM. */

export interface EmbeddedRecipe {
  recipe?: string;
  parameters?: string;
}

type Bytes = Uint8Array<ArrayBuffer>;

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const TEXT_CHUNKS = new Set(["tEXt", "iTXt", "zTXt"]);
const RECIPE_KEY = "unsloth";
const PARAMETERS_KEY = "parameters";
const MAKER_NOTE_PREFIX = "Unsloth\0";
const SOFTWARE = "Unsloth Studio";

interface Chunk {
  type: string;
  start: number;
  end: number;
  data: number;
}

function concat(parts: Uint8Array[]): Bytes {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return String.fromCharCode(...bytes.subarray(start, end));
}

function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && PNG_SIGNATURE.every((b, i) => bytes[i] === b);
}

function pngChunks(bytes: Uint8Array): Chunk[] | null {
  if (!isPng(bytes)) {
    return null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: Chunk[] = [];
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const end = offset + 12 + view.getUint32(offset);
    if (end > bytes.length) {
      return null;
    }
    const type = ascii(bytes, offset + 4, offset + 8);
    chunks.push({ type, start: offset, end, data: offset + 8 });
    offset = end;
    if (type === "IEND") {
      return chunks;
    }
  }
  return null;
}

/** tEXt (Latin-1) and uncompressed iTXt (UTF-8) chunks by keyword. Compressed chunks are skipped. */
export function pngTextChunks(bytes: Uint8Array): Record<string, string> {
  const out: Record<string, string> = {};
  for (const c of pngChunks(bytes) ?? []) {
    const data = bytes.subarray(c.data, c.end - 4);
    const nul = data.indexOf(0);
    if (nul <= 0 || (c.type !== "tEXt" && c.type !== "iTXt")) {
      continue;
    }
    const key = ascii(data, 0, nul);
    if (c.type === "tEXt") {
      out[key] = new TextDecoder("latin1").decode(data.subarray(nul + 1));
      continue;
    }
    if (data[nul + 1] !== 0) {
      continue;
    }
    const lang = data.indexOf(0, nul + 3);
    const translated = lang < 0 ? -1 : data.indexOf(0, lang + 1);
    if (translated >= 0) {
      out[key] = new TextDecoder().decode(data.subarray(translated + 1));
    }
  }
  return out;
}

/** `target` with its text chunks replaced by `source`'s (copied verbatim, CRCs intact) right after
 *  IHDR. Either side unparsable: `target` unchanged. */
export function copyPngTextChunks(source: Uint8Array, target: Bytes): Bytes {
  const from = pngChunks(source);
  const into = pngChunks(target);
  if (!from || !into || into[0]?.type !== "IHDR") {
    return target;
  }
  const parts: Uint8Array[] = [target.subarray(0, into[0].end)];
  for (const c of from) {
    if (TEXT_CHUNKS.has(c.type)) parts.push(source.subarray(c.start, c.end));
  }
  for (const c of into.slice(1)) {
    if (!TEXT_CHUNKS.has(c.type)) parts.push(target.subarray(c.start, c.end));
  }
  return concat(parts);
}

export function pngRecipe(bytes: Uint8Array): EmbeddedRecipe {
  const text = pngTextChunks(bytes);
  return { recipe: text[RECIPE_KEY], parameters: text[PARAMETERS_KEY] };
}

// ---- EXIF (big-endian TIFF, the layout piexif writes, so AUTOMATIC1111 reads it back) ----

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  value: Uint8Array;
}

const ASCII = 2;
const LONG = 4;
const UNDEFINED = 7;

function asciiValue(text: string): Uint8Array {
  return new Uint8Array([...text].map((c) => c.charCodeAt(0) & 0x7f).concat(0));
}

function userComment(text: string): Uint8Array {
  const out = new Uint8Array(8 + text.length * 2);
  out.set([0x55, 0x4e, 0x49, 0x43, 0x4f, 0x44, 0x45, 0]); // "UNICODE\0", then UTF-16BE
  const view = new DataView(out.buffer);
  for (let i = 0; i < text.length; i++)
    view.setUint16(8 + i * 2, text.charCodeAt(i));
  return out;
}

function ifdSize(entries: IfdEntry[]): number {
  const external = entries.reduce(
    (n, e) =>
      n + (e.value.length > 4 ? e.value.length + (e.value.length & 1) : 0),
    0,
  );
  return 2 + entries.length * 12 + 4 + external;
}

function writeIfd(out: Uint8Array, at: number, entries: IfdEntry[]): void {
  const view = new DataView(out.buffer);
  view.setUint16(at, entries.length);
  let data = at + 2 + entries.length * 12 + 4;
  entries.forEach((e, i) => {
    const slot = at + 2 + i * 12;
    view.setUint16(slot, e.tag);
    view.setUint16(slot + 2, e.type);
    view.setUint32(slot + 4, e.count);
    if (e.value.length <= 4) {
      out.set(e.value, slot + 8);
    } else {
      view.setUint32(slot + 8, data);
      out.set(e.value, data);
      data += e.value.length + (e.value.length & 1);
    }
  });
  view.setUint32(at + 2 + entries.length * 12, 0);
}

/** TIFF bytes (no `Exif\0\0` prefix) carrying the recipe, or null when there is nothing to carry. */
export function buildExif(meta: EmbeddedRecipe): Bytes | null {
  const exifEntries: IfdEntry[] = [];
  if (meta.parameters) {
    const v = userComment(meta.parameters);
    exifEntries.push({
      tag: 0x9286,
      type: UNDEFINED,
      count: v.length,
      value: v,
    });
  }
  if (meta.recipe) {
    const v = concat([
      new TextEncoder().encode(MAKER_NOTE_PREFIX),
      new TextEncoder().encode(meta.recipe),
    ]);
    exifEntries.push({
      tag: 0x927c,
      type: UNDEFINED,
      count: v.length,
      value: v,
    });
  }
  if (exifEntries.length === 0) {
    return null;
  }
  exifEntries.sort((a, b) => a.tag - b.tag);
  const software = asciiValue(SOFTWARE);
  const ifd0: IfdEntry[] = [
    { tag: 0x0131, type: ASCII, count: software.length, value: software },
    { tag: 0x8769, type: LONG, count: 1, value: new Uint8Array(4) },
  ];
  const exifAt = 8 + ifdSize(ifd0);
  new DataView(ifd0[1].value.buffer).setUint32(0, exifAt);
  const out = new Uint8Array(exifAt + ifdSize(exifEntries));
  out.set([0x4d, 0x4d, 0, 0x2a, 0, 0, 0, 8]);
  writeIfd(out, 8, ifd0);
  writeIfd(out, exifAt, exifEntries);
  return out;
}

const EXIF_HEADER = [0x45, 0x78, 0x69, 0x66, 0, 0];
const JPEG_SEGMENT_MAX = 0xffff - 2;

/** `jpeg` with one EXIF APP1 segment (any existing one replaced) after SOI / APP0. A segment too
 *  large for JPEG falls back to the parameters text alone, then to no EXIF. */
export function withJpegExif(jpeg: Bytes, meta: EmbeddedRecipe): Bytes {
  if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8) {
    return jpeg;
  }
  let tiff = buildExif(meta);
  if (tiff && tiff.length + EXIF_HEADER.length > JPEG_SEGMENT_MAX) {
    tiff = buildExif({ parameters: meta.parameters });
  }
  if (tiff && tiff.length + EXIF_HEADER.length > JPEG_SEGMENT_MAX) {
    tiff = null;
  }
  const kept: Uint8Array[] = [];
  let insertAt = 2;
  let at = 2;
  while (at + 4 <= jpeg.length && jpeg[at] === 0xff) {
    const marker = jpeg[at + 1];
    if (marker === 0xda || marker < 0xe0 || marker > 0xef) {
      break;
    }
    const end = at + 2 + ((jpeg[at + 2] << 8) | jpeg[at + 3]);
    const isExif =
      marker === 0xe1 && EXIF_HEADER.every((b, i) => jpeg[at + 4 + i] === b);
    if (!isExif) kept.push(jpeg.subarray(at, end));
    if (marker === 0xe0 && insertAt === at) insertAt = end;
    at = end;
  }
  const head = jpeg.subarray(0, 2);
  const app0 = insertAt > 2 ? [kept.shift() as Uint8Array] : [];
  if (!tiff) {
    return concat([head, ...app0, ...kept, jpeg.subarray(at)]);
  }
  const length = 2 + EXIF_HEADER.length + tiff.length;
  const segment = concat([
    new Uint8Array([0xff, 0xe1, length >> 8, length & 0xff, ...EXIF_HEADER]),
    tiff,
  ]);
  return concat([head, ...app0, segment, ...kept, jpeg.subarray(at)]);
}

function riffChunks(webp: Uint8Array): Chunk[] | null {
  if (
    webp.length < 12 ||
    ascii(webp, 0, 4) !== "RIFF" ||
    ascii(webp, 8, 12) !== "WEBP"
  ) {
    return null;
  }
  const view = new DataView(webp.buffer, webp.byteOffset, webp.byteLength);
  const chunks: Chunk[] = [];
  let at = 12;
  while (at + 8 <= webp.length) {
    const size = view.getUint32(at + 4, true);
    const end = at + 8 + size + (size & 1);
    if (at + 8 + size > webp.length) {
      return null;
    }
    chunks.push({
      type: ascii(webp, at, at + 4),
      start: at,
      end: Math.min(end, webp.length),
      data: at + 8,
    });
    at = end;
  }
  return chunks;
}

function riffChunk(type: string, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(8 + payload.length + (payload.length & 1));
  out.set([...type].map((c) => c.charCodeAt(0)));
  new DataView(out.buffer).setUint32(4, payload.length, true);
  out.set(payload, 8);
  return out;
}

/** `webp` as an extended (VP8X) file with an EXIF chunk. Unparsable input comes back unchanged. */
export function withWebpExif(
  webp: Bytes,
  meta: EmbeddedRecipe,
  width: number,
  height: number,
): Bytes {
  const chunks = riffChunks(webp);
  const tiff = buildExif(meta);
  if (!chunks || chunks.length === 0 || !tiff) {
    return webp;
  }
  const body = chunks.filter((c) => c.type !== "EXIF" && c.type !== "VP8X");
  let vp8x: Uint8Array;
  const existing = chunks.find((c) => c.type === "VP8X");
  if (existing) {
    vp8x = webp.slice(existing.start, existing.end);
    vp8x[8] |= 0x08;
  } else {
    const payload = new Uint8Array(10);
    const lossless = body.find((c) => c.type === "VP8L");
    // VP8L header: signature byte, then 14 + 14 bits of size, then the alpha_is_used bit.
    const alpha = lossless ? (webp[lossless.data + 4] >> 4) & 1 : 0;
    payload[0] = 0x08 | (alpha ? 0x10 : 0);
    const w = width - 1;
    const h = height - 1;
    payload.set(
      [
        w & 0xff,
        (w >> 8) & 0xff,
        (w >> 16) & 0xff,
        h & 0xff,
        (h >> 8) & 0xff,
        (h >> 16) & 0xff,
      ],
      4,
    );
    vp8x = riffChunk("VP8X", payload);
  }
  const parts = [
    vp8x,
    ...body.map((c) => webp.subarray(c.start, c.end)),
    riffChunk("EXIF", tiff),
  ];
  const size = 4 + parts.reduce((n, p) => n + p.length, 0);
  const header = new Uint8Array(12);
  header.set([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]);
  new DataView(header.buffer).setUint32(4, size, true);
  return concat([header, ...parts]);
}

function readTiff(tiff: Uint8Array): EmbeddedRecipe {
  if (tiff.length < 8) {
    return {};
  }
  const little = tiff[0] === 0x49;
  const view = new DataView(tiff.buffer, tiff.byteOffset, tiff.byteLength);
  const u16 = (at: number) => view.getUint16(at, little);
  const u32 = (at: number) => view.getUint32(at, little);
  const entries = (at: number) => {
    const out = new Map<number, Uint8Array>();
    if (at + 2 > tiff.length) return out;
    const n = u16(at);
    for (let i = 0; i < n; i++) {
      const slot = at + 2 + i * 12;
      if (slot + 12 > tiff.length) break;
      const type = u16(slot + 2);
      const count = u32(slot + 4);
      const unit = type === 3 ? 2 : type === LONG ? 4 : 1;
      const len = count * unit;
      const start = len <= 4 ? slot + 8 : u32(slot + 8);
      if (start + len <= tiff.length)
        out.set(u16(slot), tiff.subarray(start, start + len));
    }
    return out;
  };
  const ifd0 = entries(u32(4));
  const pointer = ifd0.get(0x8769);
  if (!pointer || pointer.length < 4) {
    return {};
  }
  const exif = entries(
    new DataView(pointer.buffer, pointer.byteOffset, 4).getUint32(0, little),
  );
  const out: EmbeddedRecipe = {};
  const comment = exif.get(0x9286);
  if (comment && comment.length > 8) {
    const prefix = ascii(comment, 0, 8);
    const body = comment.subarray(8);
    if (prefix === "UNICODE\0") {
      const cv = new DataView(body.buffer, body.byteOffset, body.byteLength);
      let s = "";
      for (let i = 0; i + 1 < body.length; i += 2)
        s += String.fromCharCode(cv.getUint16(i, little));
      out.parameters = s.replace(/\0+$/, "");
    } else {
      out.parameters = new TextDecoder().decode(body).replace(/\0+$/, "");
    }
  }
  const note = exif.get(0x927c);
  if (note && ascii(note, 0, MAKER_NOTE_PREFIX.length) === MAKER_NOTE_PREFIX) {
    out.recipe = new TextDecoder().decode(
      note.subarray(MAKER_NOTE_PREFIX.length),
    );
  }
  return out;
}

function jpegTiff(jpeg: Uint8Array): Uint8Array | null {
  let at = 2;
  while (at + 4 <= jpeg.length && jpeg[at] === 0xff) {
    const marker = jpeg[at + 1];
    if (marker === 0xda) break;
    const end = at + 2 + ((jpeg[at + 2] << 8) | jpeg[at + 3]);
    if (
      marker === 0xe1 &&
      EXIF_HEADER.every((b, i) => jpeg[at + 4 + i] === b)
    ) {
      return jpeg.subarray(
        at + 4 + EXIF_HEADER.length,
        Math.min(end, jpeg.length),
      );
    }
    at = end;
  }
  return null;
}

function webpTiff(webp: Uint8Array): Uint8Array | null {
  const chunk = riffChunks(webp)?.find((c) => c.type === "EXIF");
  if (!chunk) return null;
  const size = new DataView(webp.buffer, webp.byteOffset).getUint32(
    chunk.start + 4,
    true,
  );
  const data = webp.subarray(chunk.data, chunk.data + size);
  return EXIF_HEADER.every((b, i) => data[i] === b)
    ? data.subarray(EXIF_HEADER.length)
    : data;
}

/** Recipe text from a PNG, JPEG or WebP file; empty when it carries none. */
export function readEmbeddedRecipe(bytes: Uint8Array): EmbeddedRecipe {
  if (isPng(bytes)) {
    return pngRecipe(bytes);
  }
  const tiff =
    bytes[0] === 0xff && bytes[1] === 0xd8 ? jpegTiff(bytes) : webpTiff(bytes);
  return tiff ? readTiff(tiff) : {};
}

// ---- turning embedded text back into settings ----

/** Settings read from a file: untrusted, so every field but the prompt stays `unknown` until
 *  `recipeSettings` checks it. */
export interface ImportedRecipe {
  prompt: string;
  [key: string]: unknown;
}

const A1111_FIELD =
  /\s*([A-Za-z][A-Za-z0-9 _-]*):\s*("(?:\\.|[^"\\])*"|[^,]*)(?:,|$)/g;
const SIZE = /^(\d+)\s*x\s*(\d+)$/i;

/** AUTOMATIC1111 `parameters` text: prompt, optional `Negative prompt:` line, then `Key: value, ...`. */
export function parseA1111Parameters(text: string): ImportedRecipe | null {
  const lines = text.replace(/\r\n/g, "\n").trim().split("\n");
  if (lines.length === 0) {
    return null;
  }
  const last = lines[lines.length - 1];
  const hasSettings = /(^|,\s*)Steps:\s*\d+/.test(last);
  const settingsLine = hasSettings ? (lines.pop() ?? "") : "";
  const negativeAt = lines.findIndex((l) => l.startsWith("Negative prompt:"));
  const promptLines = negativeAt >= 0 ? lines.slice(0, negativeAt) : lines;
  const negative =
    negativeAt >= 0
      ? lines
          .slice(negativeAt)
          .join("\n")
          .replace(/^Negative prompt:\s*/, "")
      : null;
  const out: ImportedRecipe = {
    prompt: promptLines.join("\n").trim(),
    negative_prompt: negative,
  };
  for (const m of settingsLine.matchAll(A1111_FIELD)) {
    const key = m[1].trim().toLowerCase();
    const value = m[2].trim();
    const num = Number(value);
    if (key === "steps" && Number.isInteger(num)) out.steps = num;
    else if (key === "cfg scale" && Number.isFinite(num)) out.guidance = num;
    else if (key === "seed" && Number.isSafeInteger(num)) out.seed = num;
    else if (key === "model" && value) out.model = value;
    else if (key === "size") {
      const size = SIZE.exec(value);
      if (size) {
        out.width = Number(size[1]);
        out.height = Number(size[2]);
      }
    }
  }
  return out.prompt || hasSettings ? out : null;
}

function recipeJson(text: string | undefined): ImportedRecipe | null {
  if (!text) {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(text);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return null;
    }
    const r = parsed as Record<string, unknown>;
    return typeof r.prompt === "string" ? (r as ImportedRecipe) : null;
  } catch {
    return null;
  }
}

/** The richest settings the file carries: Unsloth's recipe first, else the AUTOMATIC1111 text. */
export function importedRecipe(
  embedded: EmbeddedRecipe,
): ImportedRecipe | null {
  return (
    recipeJson(embedded.recipe) ??
    (embedded.parameters ? parseA1111Parameters(embedded.parameters) : null)
  );
}

export interface RecipeDefaults {
  width: number;
  height: number;
  steps: number;
  guidance: number;
}

/** The fields the Images form restores, type-checked: a dropped file is untrusted input. A missing
 *  seed stays null so the caller keeps the seed random rather than inventing one. */
export interface RestorableRecipe {
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null;
  batch_seed: number | null;
  batch_size: number;
  model: string | null;
  workflow?: string;
  loras: string[];
  strength?: number;
  upscale?: number;
  reference_image_count?: number;
  reference_resolution?: number;
  localized_edit?: string | null;
}

const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;
const int = (v: unknown): number | undefined => {
  const n = num(v);
  return n !== undefined && Number.isSafeInteger(n) ? n : undefined;
};
const str = (v: unknown): string | undefined =>
  typeof v === "string" ? v : undefined;

export function recipeSettings(
  r: ImportedRecipe,
  d: RecipeDefaults,
): RestorableRecipe {
  const width = int(r.width);
  const height = int(r.height);
  return {
    prompt: str(r.prompt) ?? "",
    negative_prompt: str(r.negative_prompt) ?? null,
    width: width && width > 0 ? width : d.width,
    height: height && height > 0 ? height : d.height,
    steps: int(r.steps) ?? d.steps,
    guidance: num(r.guidance) ?? d.guidance,
    seed: int(r.seed) ?? null,
    batch_seed: int(r.batch_seed) ?? null,
    batch_size: Math.max(1, int(r.batch_size) ?? 1),
    model: str(r.model) ?? null,
    workflow: str(r.workflow),
    loras: Array.isArray(r.loras)
      ? r.loras.filter((x): x is string => typeof x === "string")
      : [],
    strength: num(r.strength),
    upscale: num(r.upscale),
    reference_image_count: int(r.reference_image_count),
    reference_resolution: int(r.reference_resolution),
    localized_edit: str(r.localized_edit) ?? null,
  };
}
