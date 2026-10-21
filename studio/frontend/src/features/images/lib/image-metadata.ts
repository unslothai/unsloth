// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { SAMPLER_LABELS, SCHEDULE_LABELS } from "./sampler-choice.ts";

/** Generation settings in exported images, read back from dropped ones. PNG: `unsloth` + `parameters` text chunks;
 *  JPEG / WebP: EXIF UserComment (`parameters`, read by AUTOMATIC1111 / Forge / Civitai) + MakerNote (recipe). */

export interface EmbeddedRecipe {
  recipe?: string;
  parameters?: string;
  // ComfyUI's API-format graph (its `prompt` PNG chunk), read on import only.
  comfyPrompt?: string;
}

type Bytes = Uint8Array<ArrayBuffer>;

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const TEXT_CHUNKS = new Set(["tEXt", "iTXt", "zTXt"]);
const RECIPE_KEY = "unsloth";
const PARAMETERS_KEY = "parameters";
const MAKER_NOTE_PREFIX = "Unsloth\0";
const SOFTWARE = "Unsloth";
const COMFY_EXIF_PREFIX = "prompt:";

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
  const out: EmbeddedRecipe = {
    recipe: text[RECIPE_KEY],
    parameters: text[PARAMETERS_KEY],
  };
  if (text.prompt?.trimStart().startsWith("{")) {
    out.comfyPrompt = text.prompt;
  }
  return out;
}

// EXIF is big-endian TIFF, the layout piexif writes, so AUTOMATIC1111 reads it back.

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
  const out: EmbeddedRecipe = {};
  // ComfyUI's WebP / AVIF writers put its API graph in IFD0 Model as `prompt:{...}`.
  const model = ifd0.get(0x0110);
  if (model) {
    const text = new TextDecoder().decode(model).replace(/\0+$/, "");
    if (text.startsWith(COMFY_EXIF_PREFIX)) {
      out.comfyPrompt = text.slice(COMFY_EXIF_PREFIX.length);
    }
  }
  const pointer = ifd0.get(0x8769);
  if (!pointer || pointer.length < 4) {
    return out;
  }
  const exif = entries(
    new DataView(pointer.buffer, pointer.byteOffset, 4).getUint32(0, little),
  );
  const comment = exif.get(0x9286);
  if (comment && comment.length > 8) {
    const prefix = ascii(comment, 0, 8);
    const body = comment.subarray(8);
    if (prefix === "UNICODE\0") {
      const cv = new DataView(body.buffer, body.byteOffset, body.byteLength);
      // piexif writes UTF-16BE, others the TIFF order: BOM, else where ASCII's zero bytes sit over the
      // whole text (one leading U+4E00 must not decide), else TIFF order.
      let start = 0;
      let be: boolean;
      if (body[0] === 0xfe && body[1] === 0xff) {
        be = true;
        start = 2;
      } else if (body[0] === 0xff && body[1] === 0xfe) {
        be = false;
        start = 2;
      } else {
        let evenZeros = 0;
        let oddZeros = 0;
        for (let i = 0; i + 1 < body.length; i += 2) {
          if (body[i] === 0 && body[i + 1] !== 0) evenZeros++;
          else if (body[i + 1] === 0 && body[i] !== 0) oddZeros++;
        }
        be = evenZeros === oddZeros ? !little : evenZeros > oddZeros;
      }
      let s = "";
      for (let i = start; i + 1 < body.length; i += 2)
        s += String.fromCharCode(cv.getUint16(i, !be));
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

export function readEmbeddedRecipe(bytes: Uint8Array): EmbeddedRecipe {
  if (isPng(bytes)) {
    return pngRecipe(bytes);
  }
  const tiff =
    bytes[0] === 0xff && bytes[1] === 0xd8 ? jpegTiff(bytes) : webpTiff(bytes);
  return tiff ? readTiff(tiff) : {};
}

/** Settings read from a file: untrusted, so every field but the prompt stays `unknown` until
 *  `recipeSettings` checks it. */
export interface ImportedRecipe {
  prompt: string;
  [key: string]: unknown;
}

const SIZE = /^(\d+)\s*x\s*(\d+)$/i;
const A1111_KEY = /^([A-Za-z][\w \-/]*):\s*/;
// An unknown label stays as written, so the import can name what it could not apply.
const labelId = (labels: Record<string, string>, label: string) =>
  Object.keys(labels).find((id) => labels[id] === label) ?? label;

/** End of a JSON object / list starting at `at` (Civitai writes `Hashes: {...}` and
 *  `Civitai resources: [...]` unquoted, commas inside), or -1 when it never closes. */
function jsonEnd(line: string, at: number): number {
  let depth = 0;
  let inString = false;
  for (let i = at; i < line.length; i++) {
    const c = line[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{" || c === "[") depth++;
    else if (c === "}" || c === "]") {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/** `Key: value, ...` pairs of AUTOMATIC1111's settings line (`re_param`, modules/infotext_utils.py:16):
 *  a quoted value is unquoted and may hold commas (`Lora hashes: "a: 1, b: 2"`); so may a JSON value. */
export function a1111Fields(line: string): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  let i = 0;
  while (i < line.length) {
    while (i < line.length && (line[i] === "," || /\s/.test(line[i]))) i++;
    if (i >= line.length) break;
    const key = A1111_KEY.exec(line.slice(i));
    if (!key) {
      // Not a field (A1111's findall skips it too): resume after the next comma.
      const comma = line.indexOf(",", i);
      if (comma < 0) break;
      i = comma + 1;
      continue;
    }
    i += key[0].length;
    let value: string;
    if (line[i] === '"') {
      let j = i + 1;
      while (j < line.length && line[j] !== '"') j += line[j] === "\\" ? 2 : 1;
      const raw = line.slice(i, Math.min(j + 1, line.length));
      try {
        value = JSON.parse(raw) as string;
      } catch {
        value = raw.replace(/^"|"$/g, "");
      }
      i = j + 1;
    } else {
      const close = line[i] === "{" || line[i] === "[" ? jsonEnd(line, i) : -1;
      const comma = line.indexOf(",", i);
      const stop = close > 0 ? close : comma < 0 ? line.length : comma;
      value = line.slice(i, stop);
      i = stop;
    }
    out.push([key[1].trim(), value.trim()]);
  }
  return out;
}

// Keys that identify a file or the writer and set nothing: read, never reported as not applied.
const A1111_INFO_KEYS = new Set([
  "model hash",
  "vae hash",
  "lora hashes",
  "ti hashes",
  "hashes",
  "version",
  "user",
  "civitai resources",
  "civitai metadata",
  "created date",
  "init image hash",
  "emphasis",
]);
// The names CFG rescale (Lin et al. 2023, diffusers guidance_rescale) is written under: the
// sd-webui-cfg-rescale extension, ComfyUI's RescaleCFG node, Forge's latent modifier field.
const CFG_RESCALE_KEYS = new Set([
  "cfg rescale",
  "cfg rescale phi",
  "rescalecfg",
  "rescale cfg",
  "cfg rescale multiplier",
  "cfgrescale",
  "latent_modifier_rescale_cfg_phi",
]);
// Extra-network tags AUTOMATIC1111 strips from the prompt before encoding (`re_extra_net`,
// modules/extra_networks.py:175); lora / lyco carry a LoRA, the rest Studio cannot load.
const EXTRA_NETWORK = /<(lora|lyco|locon|loha|lokr|hypernet):([^>]+)>/gi;

/** A `<lora:name:weight>` tag: positional `name[:te[:unet]]` or named `te=` / `unet=`
 *  (extensions-builtin/Lora/extra_networks_lora.py:30-36, default weight 1). */
export interface LoraTag {
  name: string;
  weight: number;
  // Set when the U-Net multiplier differs from the text-encoder one: Studio has one weight.
  unet_weight?: number;
}

function loraTag(body: string): LoraTag | null {
  const positional: string[] = [];
  const named: Record<string, string> = {};
  for (const item of body.split(":")) {
    const eq = item.indexOf("=");
    if (eq > 0) named[item.slice(0, eq).trim()] = item.slice(eq + 1).trim();
    else positional.push(item.trim());
  }
  const name = positional[0];
  if (!name) return null;
  const te = Number(named.te ?? positional[1] ?? 1);
  const weight = Number.isFinite(te) ? te : 1;
  const unet = Number(named.unet ?? positional[2] ?? weight);
  return Number.isFinite(unet) && unet !== weight
    ? { name, weight, unet_weight: unet }
    : { name, weight };
}

export function extractExtraNetworks(prompt: string): {
  prompt: string;
  loras: LoraTag[];
  other: string[];
} {
  const loras: LoraTag[] = [];
  const other: string[] = [];
  let found = false;
  const stripped = prompt.replace(
    EXTRA_NETWORK,
    (tag, kind: string, body: string) => {
      found = true;
      const lower = kind.toLowerCase();
      const parsed = lower === "hypernet" ? null : loraTag(body);
      if (parsed) loras.push(parsed);
      else other.push(tag);
      return "";
    },
  );
  if (!found) return { prompt, loras, other };
  const tidy = stripped
    .split("\n")
    .map((line) =>
      line
        .replace(/[ \t]{2,}/g, " ")
        .replace(/\s*,(\s*,)+/g, ",")
        .replace(/^[\s,]+|[\s,]+$/g, ""),
    )
    .join("\n")
    .trim();
  return { prompt: tidy, loras, other };
}

const numberOf = (value: string): number | undefined => {
  const n = Number(value);
  return value !== "" && Number.isFinite(n) ? n : undefined;
};

const SETTINGS_KEYS = new Set([
  "steps",
  "sampler",
  "cfg scale",
  "seed",
  "size",
  "schedule type",
]);

// AUTOMATIC1111 takes the last line as settings when it has 3+ fields (infotext_utils.py:254-257);
// a known key is also required here, so a prompt line like "style: a, mood: b, light: c" stays a prompt.
function isSettingsLine(
  line: string,
  fields: Array<[string, string]>,
): boolean {
  return (
    /(^|,\s*)Steps:\s*\d+/.test(line) ||
    (fields.length >= 3 &&
      fields.some(([k]) => SETTINGS_KEYS.has(k.toLowerCase())))
  );
}

/** Requires A1111's settings line, else any text would parse as a bare prompt. */
export function pastedParameters(text: string): ImportedRecipe | null {
  const trimmed = text.trim();
  if (trimmed.startsWith("{")) return recipeJson(trimmed);
  const last = trimmed.replace(/\r\n?/g, "\n").split("\n").pop() ?? "";
  return isSettingsLine(last, a1111Fields(last))
    ? parseA1111Parameters(trimmed)
    : null;
}

/** A prompt-box paste that is a whole parameters block (Steps line with seed, sampler or CFG). */
export function isGenerationParametersPaste(text: string): boolean {
  const last = text.trim().replace(/\r\n?/g, "\n").split("\n").pop() ?? "";
  const keys = new Set(a1111Fields(last).map(([k]) => k.toLowerCase()));
  return (
    /(^|,\s*)Steps:\s*\d+/.test(last) &&
    (keys.has("seed") || keys.has("sampler") || keys.has("cfg scale")) &&
    pastedParameters(text) !== null
  );
}

/** AUTOMATIC1111 / Forge / Civitai `parameters` text (modules/infotext_utils.py:234). */
export function parseA1111Parameters(text: string): ImportedRecipe | null {
  const lines = text.replace(/\r\n?/g, "\n").trim().split("\n");
  if (lines.length === 0) {
    return null;
  }
  const last = lines[lines.length - 1];
  const fields = a1111Fields(last);
  const hasSettings = isSettingsLine(last, fields);
  if (hasSettings) lines.pop();
  const negativeAt = lines.findIndex((l) => l.startsWith("Negative prompt:"));
  const promptLines = negativeAt >= 0 ? lines.slice(0, negativeAt) : lines;
  const negative =
    negativeAt >= 0
      ? lines
          .slice(negativeAt)
          .join("\n")
          .replace(/^Negative prompt:\s*/, "")
      : null;
  const networks = extractExtraNetworks(promptLines.join("\n").trim());
  const out: ImportedRecipe = {
    prompt: networks.prompt,
    negative_prompt: negative,
  };
  if (networks.loras.length > 0) out.lora_tags = networks.loras;
  const notApplied: string[] = networks.other.slice();
  const hires: Record<string, string> = {};
  let legacySchedule: string | undefined;
  for (const [label, value] of hasSettings ? fields : []) {
    const key = label.toLowerCase();
    const n = numberOf(value);
    if (key === "steps" && n !== undefined && Number.isInteger(n))
      out.steps = n;
    else if (key === "cfg scale" && n !== undefined) out.guidance = n;
    else if (key === "seed" && n !== undefined && Number.isSafeInteger(n))
      out.seed = n;
    else if (key === "model" && value) out.model = value;
    else if (key === "denoising strength" && n !== undefined) out.strength = n;
    // A1111 numbering (1 = last layer); infotext omits it at 1 (modules/processing.py:790).
    else if (key === "clip skip" && n !== undefined && Number.isInteger(n))
      out.clip_skip = n;
    else if (CFG_RESCALE_KEYS.has(key) && n !== undefined) out.cfg_rescale = n;
    else if (
      key === "variation seed" &&
      n !== undefined &&
      Number.isSafeInteger(n)
    )
      out.variation_seed = n;
    else if (key === "variation seed strength" && n !== undefined)
      out.variation_strength = n;
    else if (key === "mask blur" && n !== undefined) out.mask_blur = n;
    else if (key === "inpaint area")
      out.inpaint_only_masked = /only masked/i.test(value);
    else if (
      (key === "masked area padding" || key === "only masked padding") &&
      n !== undefined
    )
      out.inpaint_padding = n;
    else if (key.startsWith("hires")) hires[label] = value;
    else if (key === "sampler") {
      out.sampler = labelId(SAMPLER_LABELS, value);
      // Before AUTOMATIC1111 1.9 the schedule rode on the sampler name: `DPM++ 2M Karras`.
      const suffix = Object.keys(SCHEDULE_LABELS).find(
        (id) => id !== "automatic" && value.endsWith(` ${SCHEDULE_LABELS[id]}`),
      );
      if (out.sampler === value && suffix) {
        out.sampler = labelId(
          SAMPLER_LABELS,
          value.slice(0, -SCHEDULE_LABELS[suffix].length - 1),
        );
        legacySchedule = suffix;
      }
    } else if (key === "schedule type")
      out.schedule_type = labelId(SCHEDULE_LABELS, value);
    else if (key === "size") {
      const size = SIZE.exec(value);
      if (size) {
        out.width = Number(size[1]);
        out.height = Number(size[2]);
      }
    } else if (!A1111_INFO_KEYS.has(key) && value)
      notApplied.push(`${label}: ${value}`);
  }
  out.schedule_type ??= legacySchedule;
  const hiresKeys = Object.keys(hires);
  if (hiresKeys.length > 0) out.hires = hires;
  // txt2img hires fix also writes a denoising strength; without it the strength means img2img,
  // or inpaint when A1111 wrote its mask-only fields.
  if (out.strength !== undefined) {
    if (hiresKeys.length > 0) delete out.strength;
    else
      out.workflow ??=
        out.mask_blur !== undefined ||
        out.inpaint_only_masked !== undefined ||
        out.inpaint_padding !== undefined
          ? "inpaint"
          : "img2img";
  }
  if (notApplied.length > 0) out.parameters_not_applied = notApplied;
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
    (embedded.parameters ? parseA1111Parameters(embedded.parameters) : null) ??
    (embedded.comfyPrompt ? parseComfyPrompt(embedded.comfyPrompt) : null)
  );
}

export interface RecipeDefaults {
  width: number;
  height: number;
  steps: number;
  guidance: number;
}

/** Type-checked form fields (a dropped file is untrusted); a missing seed stays null, i.e. random. */
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
  // Only "connection" matters here: those runs never sent steps, guidance or a seed.
  model_kind?: string;
  sampler?: string;
  schedule_type?: string;
  // The prompt upsampler's caption and outcome; restore decides the Enhance prompt toggle from them.
  enhanced_prompt?: string | null;
  prompt_enhancement?: "applied" | "off" | "unavailable" | "failed" | null;
  mask_blur?: number;
  inpaint_only_masked?: boolean;
  inpaint_padding?: number;
}

export const APPLIED_RECIPE_KEYS = [
  "prompt",
  "negative_prompt",
  "width",
  "height",
  "steps",
  "guidance",
  "seed",
  "batch_seed",
  "batch_size",
  "loras",
  "workflow",
  "strength",
  "upscale",
  "reference_image_count",
  "reference_resolution",
  "localized_edit",
  "sampler",
  "schedule_type",
  "enhanced_prompt",
  "prompt_enhancement",
  "mask_blur",
  "inpaint_only_masked",
  "inpaint_padding",
] as const;

/** Known but not applied: loading a model or a ControlNet image stays the user's call. */
export const INFO_RECIPE_KEYS = [
  "model",
  "model_kind",
  "gguf_filename",
  "transformer_quant",
  "text_encoder_quant",
  "memory_mode",
  "offload_policy",
  "speed_mode",
  "attention_backend",
  "transformer_cache",
  "cpu_offload",
  "baked_loras",
  "controlnet",
  "controlnet_guidance",
  "batch_index",
  "created_at",
  "engine",
  "generator",
  "schema_version",
] as const;

/** Matches RECIPE_SCHEMA_VERSION in core/inference/image_gallery.py. */
export const RECIPE_SCHEMA_VERSION = 1;

/** Read from AUTOMATIC1111 text but not sent by this build. */
export const PENDING_RECIPE_KEYS = [
  "clip_skip",
  "cfg_rescale",
  "variation_seed",
  "variation_strength",
  "hires",
  "lora_tags",
  "parameters_not_applied",
] as const;

const KNOWN_RECIPE_KEYS = new Set<string>([
  ...APPLIED_RECIPE_KEYS,
  ...INFO_RECIPE_KEYS,
  ...PENDING_RECIPE_KEYS,
]);

/** Non-empty keys this build does not know (newer Unsloth, other tools), named instead of dropped. */
export function unknownRecipeKeys(r: ImportedRecipe): string[] {
  return Object.keys(r).filter((key) => {
    if (KNOWN_RECIPE_KEYS.has(key)) {
      return false;
    }
    const v = r[key];
    return !(
      v === null ||
      v === undefined ||
      v === "" ||
      (Array.isArray(v) && v.length === 0)
    );
  });
}

export function importNotApplied(r: ImportedRecipe): string[] {
  const out: string[] = [];
  const clipSkip = num(r.clip_skip);
  // SDXL already encodes from the penultimate layer, which is what A1111 / Forge render for Clip skip
  // 1 and 2 (Forge minimal_clip_skip=2; A1111 skips CLIP-G never, CLIP-L only with sdxl_clip_l_skip).
  if (clipSkip !== undefined && clipSkip > 2) out.push(`Clip skip ${clipSkip}`);
  const rescale = num(r.cfg_rescale);
  if (rescale !== undefined && rescale !== 0)
    out.push(`CFG rescale ${rescale}`);
  const variation = num(r.variation_seed);
  const variationStrength = num(r.variation_strength);
  if (variation !== undefined && variationStrength !== 0) {
    out.push(
      `variation seed ${variation}${variationStrength !== undefined ? ` (strength ${variationStrength})` : ""}`,
    );
  }
  if (r.hires && typeof r.hires === "object" && !Array.isArray(r.hires)) {
    const parts = Object.entries(r.hires as Record<string, unknown>)
      .filter(([, v]) => typeof v === "string" && v !== "")
      .map(([k, v]) => `${k}: ${String(v)}`);
    if (parts.length > 0) out.push(`hires fix (${parts.join(", ")})`);
  }
  if (Array.isArray(r.parameters_not_applied)) {
    for (const item of r.parameters_not_applied)
      if (typeof item === "string") out.push(item);
  }
  return out;
}

export function newerRecipeSchema(r: ImportedRecipe): boolean {
  return (
    typeof r.schema_version === "number" &&
    r.schema_version > RECIPE_SCHEMA_VERSION
  );
}

const clamp = (
  n: number | undefined,
  lo: number,
  hi: number,
): number | undefined =>
  n === undefined ? undefined : Math.min(hi, Math.max(lo, n));
const num = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;
const int = (v: unknown): number | undefined => {
  const n = num(v);
  return n !== undefined && Number.isSafeInteger(n) ? n : undefined;
};
const positive = (n: number | undefined): number | undefined =>
  n !== undefined && n > 0 ? n : undefined;
// The request takes seeds in [0, 2^53 - 1]; anything else leaves the seed random.
const seedOf = (v: unknown): number | null => {
  const n = int(v);
  return n !== undefined && n >= 0 ? n : null;
};
// `id:weight` within the request's LoraSpec bounds (id 1-512 chars, weight 0-2), else dropped.
const loraOf = (v: unknown): string[] => {
  if (typeof v !== "string") return [];
  const at = v.lastIndexOf(":");
  const id = v.slice(0, at).trim();
  const weight = Number(v.slice(at + 1));
  if (at <= 0 || id.length > 512 || !Number.isFinite(weight)) return [];
  return [`${id}:${Math.min(2, Math.max(0, weight))}`];
};
// The request rejects a repeated id: the first entry wins.
const uniqueLoras = (loras: string[]): string[] => {
  const seen = new Set<string>();
  return loras.filter((l) => {
    const id = l.slice(0, l.lastIndexOf(":"));
    return !seen.has(id) && seen.add(id);
  });
};
const str = (v: unknown): string | undefined =>
  typeof v === "string" ? v : undefined;

/** The request's adapter limit (DiffusionGenerateRequest.loras). */
export const MAX_LORAS = 8;

/** The recipe's LoRA ids recipeSettings drops for being past MAX_LORAS. */
export function lorasOverLimit(r: ImportedRecipe): string[] {
  if (!Array.isArray(r.loras)) return [];
  return uniqueLoras(r.loras.flatMap(loraOf))
    .slice(MAX_LORAS)
    .map((l) => l.slice(0, l.lastIndexOf(":")));
}

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
    // Clamped to the request's bounds: other tools allow more (AUTOMATIC1111 goes to 150 steps).
    steps: clamp(int(r.steps), 1, 100) ?? d.steps,
    guidance: clamp(num(r.guidance), 0, 20) ?? d.guidance,
    seed: seedOf(r.seed),
    batch_seed: seedOf(r.batch_seed),
    batch_size: clamp(int(r.batch_size), 1, 32) ?? 1,
    model: str(r.model) ?? null,
    workflow: str(r.workflow),
    loras: Array.isArray(r.loras)
      ? uniqueLoras(r.loras.flatMap(loraOf)).slice(0, MAX_LORAS)
      : [],
    // The request takes strength in (0, 1]: 0 leaves no denoising steps, so it is not restored.
    strength: positive(clamp(num(r.strength), 0, 1)),
    upscale: clamp(num(r.upscale), 1, 4),
    // The request caps reference images at 9; restore builds an array of this length.
    reference_image_count: clamp(int(r.reference_image_count), 0, 9),
    reference_resolution: clamp(int(r.reference_resolution), 256, 2048),
    localized_edit: str(r.localized_edit) ?? null,
    model_kind: r.model_kind === "connection" ? "connection" : undefined,
    sampler: str(r.sampler),
    schedule_type: str(r.schedule_type),
    enhanced_prompt: str(r.enhanced_prompt) ?? null,
    prompt_enhancement:
      r.prompt_enhancement === "applied" ||
      r.prompt_enhancement === "off" ||
      r.prompt_enhancement === "unavailable" ||
      r.prompt_enhancement === "failed"
        ? r.prompt_enhancement
        : null,
    mask_blur: clamp(int(r.mask_blur), 0, 64),
    inpaint_only_masked:
      typeof r.inpaint_only_masked === "boolean"
        ? r.inpaint_only_masked
        : undefined,
    inpaint_padding: clamp(int(r.inpaint_padding), 0, 256),
  };
}

const LORA_EXTENSIONS = /\.(safetensors|gguf|ckpt|pt|bin)$/i;
const loraKey = (name: string) =>
  name
    .trim()
    .replace(LORA_EXTENSIONS, "")
    .split(/[\\/:]/)
    .pop()
    ?.toLowerCase() ?? "";

/** Resolve `<lora:name:weight>` tags against the installed LoRAs by file stem; returns unmatched names. */
export function matchLoraTags(
  tags: unknown,
  available: readonly { id: string; display_name: string; source?: string }[],
): { loras: string[]; unmatched: string[]; notes: string[] } {
  const loras: string[] = [];
  const unmatched: string[] = [];
  const notes: string[] = [];
  if (!Array.isArray(tags)) return { loras, unmatched, notes };
  const local = available.filter(
    (a) => a.source === undefined || a.source === "local",
  );
  for (const tag of tags as unknown[]) {
    if (!tag || typeof tag !== "object") continue;
    const { name, weight, unet_weight } = tag as Partial<LoraTag>;
    if (typeof name !== "string" || typeof weight !== "number") continue;
    const key = loraKey(name);
    const hit = local.find(
      (a) => loraKey(a.id) === key || loraKey(a.display_name) === key,
    );
    if (!hit) {
      unmatched.push(name);
      continue;
    }
    const w = Math.min(2, Math.max(0, weight));
    loras.push(`${hit.id}:${w}`);
    if (w !== weight)
      notes.push(`LoRA ${name} weight ${weight} (clamped to ${w})`);
    if (typeof unet_weight === "number")
      notes.push(`LoRA ${name} U-Net weight ${unet_weight}`);
  }
  return { loras, unmatched, notes };
}

/** matchLoraTags against the catalog a paste can see. Without a LoRA-capable model resident the page holds
 *  no catalog, so the whole local list is fetched rather than reporting every installed adapter missing. */
export async function matchLoraTagsForImport<
  T extends { id: string; display_name: string; source?: string },
>(
  tags: unknown,
  catalogLoaded: boolean,
  loaded: readonly T[],
  fetchAll: () => Promise<readonly T[]>,
): Promise<{ loras: string[]; unmatched: string[]; notes: string[] }> {
  if (catalogLoaded || !Array.isArray(tags) || tags.length === 0)
    return matchLoraTags(tags, loaded);
  try {
    return matchLoraTags(tags, await fetchAll());
  } catch {
    const names = (tags as unknown[])
      .map((t) => (t && typeof t === "object" ? (t as Partial<LoraTag>).name : undefined))
      .filter((n): n is string => typeof n === "string");
    return {
      loras: [],
      unmatched: [],
      notes: names.map((n) => `LoRA ${n} (the installed LoRAs could not be listed)`),
    };
  }
}

type ComfyGraph = Record<
  string,
  { class_type?: unknown; inputs?: Record<string, unknown> }
>;

const COMFY_SAMPLERS = new Set([
  "KSampler",
  "KSamplerAdvanced",
  "SamplerCustomAdvanced",
]);
const COMFY_MODEL_INPUTS = ["ckpt_name", "unet_name", "model_name"];
// ComfyUI core text-encoder prompt inputs (nodes_clip_sdxl.py, nodes_sd3.py, nodes_flux.py, ...), richest first.
const COMFY_TEXT_INPUTS = [
  "text",
  "text_g",
  "t5xxl",
  "clip_g",
  "clip_l",
  "text_l",
  "mt5xl",
  "bert",
];
// ComfyUI sampler / scheduler names (comfy/samplers.py) that have a Studio equivalent.
const COMFY_SAMPLER_IDS: Record<string, string> = {
  euler: "euler",
  euler_ancestral: "euler_a",
  dpmpp_2m: "dpmpp_2m",
  dpmpp_2m_sde: "dpmpp_2m_sde",
};
const COMFY_SCHEDULE_IDS: Record<string, string> = {
  karras: "karras",
  exponential: "exponential",
};

/** Settings from a ComfyUI API graph; wired inputs (`[id, slot]`) are followed a few hops. */
export function parseComfyPrompt(text: string): ImportedRecipe | null {
  let graph: ComfyGraph;
  try {
    const parsed: unknown = JSON.parse(text);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return null;
    }
    graph = parsed as ComfyGraph;
  } catch {
    return null;
  }
  const node = (link: unknown) =>
    Array.isArray(link) &&
    (typeof link[0] === "string" || typeof link[0] === "number")
      ? graph[String(link[0])]
      : undefined;
  const value = (v: unknown, keys: string[], depth = 0): unknown => {
    const linked = node(v);
    if (!linked) {
      return v;
    }
    if (depth > 4) {
      return undefined;
    }
    for (const key of keys) {
      if (linked.inputs && key in linked.inputs) {
        return value(linked.inputs[key], keys, depth + 1);
      }
    }
    return undefined;
  };
  // Flux graphs put FluxGuidance between the text encoder and the sampler.
  let fluxGuidance: unknown;
  const textOf = (link: unknown): string | undefined => {
    let n = node(link);
    for (let hop = 0; n?.class_type === "FluxGuidance" && hop < 4; hop++) {
      fluxGuidance ??= value(n.inputs?.guidance, ["guidance", "value"]);
      n = node(n.inputs?.conditioning);
    }
    if (n?.class_type === "CLIPTextEncodeFlux") {
      fluxGuidance ??= value(n.inputs?.guidance, ["guidance", "value"]);
    }
    for (const key of COMFY_TEXT_INPUTS) {
      const v = value(n?.inputs?.[key], [key, "text", "value", "string"]);
      if (typeof v === "string" && v.trim()) return v;
    }
    const v = value(n?.inputs?.text, ["text", "value", "string"]);
    return typeof v === "string" ? v : undefined;
  };
  const sampler = Object.values(graph).find(
    (n) =>
      typeof n?.class_type === "string" && COMFY_SAMPLERS.has(n.class_type),
  );
  if (!sampler?.inputs) {
    return null;
  }
  let inputs = sampler.inputs;
  if (sampler.class_type === "SamplerCustomAdvanced") {
    const guider = node(inputs.guider)?.inputs ?? {};
    const sigmas = node(inputs.sigmas)?.inputs ?? {};
    inputs = {
      positive: guider.positive ?? guider.conditioning,
      negative: guider.negative,
      cfg: guider.cfg,
      model: guider.model,
      noise_seed: node(inputs.noise)?.inputs?.noise_seed,
      steps: sigmas.steps,
      scheduler: sigmas.scheduler,
      denoise: sigmas.denoise,
      sampler_name: node(inputs.sampler)?.inputs?.sampler_name,
      latent_image: inputs.latent_image,
    };
  }
  const out: ImportedRecipe = { prompt: textOf(inputs.positive) ?? "" };
  const negative = textOf(inputs.negative);
  if (negative !== undefined) out.negative_prompt = negative;
  const seed = value(inputs.seed ?? inputs.noise_seed, [
    "seed",
    "noise_seed",
    "value",
  ]);
  if (typeof seed === "number") out.seed = seed;
  const steps = value(inputs.steps, ["steps", "value"]);
  if (typeof steps === "number") out.steps = steps;
  const cfg = value(inputs.cfg, ["cfg", "value"]);
  if (typeof fluxGuidance === "number") out.guidance = fluxGuidance;
  else if (typeof cfg === "number") out.guidance = cfg;
  // denoise < 1 re-noises an encoded source: img2img at that strength.
  const denoise = value(inputs.denoise, ["denoise", "value"]);
  if (typeof denoise === "number" && denoise > 0 && denoise < 1) {
    out.strength = denoise;
    out.workflow = "img2img";
  }
  const samplerName = value(inputs.sampler_name, ["sampler_name", "value"]);
  const scheduler = value(inputs.scheduler, ["scheduler", "value"]);
  // Unmapped samplers stay named; normal / simple have no exact Studio schedule, so stay automatic.
  if (typeof samplerName === "string" && samplerName)
    out.sampler = Object.hasOwn(COMFY_SAMPLER_IDS, samplerName)
      ? COMFY_SAMPLER_IDS[samplerName]
      : samplerName;
  if (
    typeof scheduler === "string" &&
    Object.hasOwn(COMFY_SCHEDULE_IDS, scheduler)
  )
    out.schedule_type = COMFY_SCHEDULE_IDS[scheduler];
  const latent = node(inputs.latent_image)?.inputs;
  const width = value(latent?.width, ["width", "value"]);
  const height = value(latent?.height, ["height", "value"]);
  if (typeof width === "number" && typeof height === "number") {
    out.width = width;
    out.height = height;
  }
  let link: unknown = inputs.model;
  for (let hop = 0; hop < 8; hop++) {
    const n = node(link);
    const name = COMFY_MODEL_INPUTS.map((k) => n?.inputs?.[k]).find(
      (v) => typeof v === "string",
    );
    if (typeof name === "string") {
      out.model = name;
      break;
    }
    link = n?.inputs?.model;
  }
  return out;
}
