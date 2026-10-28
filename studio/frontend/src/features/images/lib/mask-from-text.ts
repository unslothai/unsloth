// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Mask from text": SAM 3.1 (or SAM 3) finds the objects a short text names and returns their union as a mask the
// inpaint and localized-edit painters load, so it stays editable with the brush. Pure logic only (no "@/" imports)
// so node:test can load it directly.

/** Every segmenter route, in one place. */
export const MASK_SEGMENT_ROUTES = {
  status: "/api/inference/images/segment/status",
  download: "/api/inference/images/segment/download",
  cancel: "/api/inference/images/segment/cancel",
  release: "/api/inference/images/segment/release",
  segment: "/api/inference/images/segment",
} as const;

/** Window event carrying every segmenter status read (detail: SegmenterStatus), so a second reader (the Detailer
 *  control) sees a download finish in the Mask from text panel without polling. */
export const SEGMENTER_STATUS_EVENT = "unsloth:segmenter-status";

export const MASK_FROM_TEXT_LABEL = "Mask from text";
export const MASK_FROM_TEXT_PLACEHOLDER = "sofa, red lamp, pillow:2";
export const MASK_FROM_TEXT_HINT =
  "Name what to mask, separated by commas. Each name selects its best match; add :N to select up to N, as in pillow:3. The mask lands in the painter, so you can still brush or erase it.";

export interface SegmenterDownload {
  downloading: boolean;
  download_id: string | null;
  error: string | null;
  cancelled: boolean;
  bytes_total: number | null;
  bytes_done: number | null;
}

export interface SegmenterStatus {
  supported: boolean;
  reason: string | null;
  downloaded: boolean;
  loaded: boolean;
  device: string | null;
  /** The active checkpoint variant ("sam3.1", "sam3") and its display name. */
  variant: string;
  label: string;
  repo: string;
  revision: string;
  gated: boolean;
  license_url: string;
  size_bytes: number | null;
  /** The weights precision that runs ("bf16", "fp8", "int8"): the loaded model's, else what this host would load or
   *  download. ``size_bytes`` is that download's size until it is on disk. */
  precision: string | null;
  download: SegmenterDownload;
}

export interface SegmentInstance {
  prompt: string;
  score: number;
  box: [number, number, number, number];
}

export interface SegmentPromptResult {
  text: string;
  max_detections: number;
  found: number;
}

export interface SegmentResult {
  mask: string;
  width: number;
  height: number;
  instances: SegmentInstance[];
  prompts: SegmentPromptResult[];
  device: string;
  elapsed_ms: number;
}

export interface SegmentRequest {
  image: string;
  prompt: string;
  threshold?: number;
}

/** What a painter receives: a new ``key`` applies ``dataUrl`` once. */
export interface AppliedMask {
  key: number;
  dataUrl: string;
  additive: boolean;
  /** A Replace that found nothing: the painter clears its canvas and holds no mask (null), not a blank PNG. */
  empty?: boolean;
}

/** Shown when an older backend reports no variant name. */
export const DEFAULT_SEGMENTER_LABEL = "SAM 3";

const IDLE_DOWNLOAD: SegmenterDownload = {
  downloading: false,
  download_id: null,
  error: null,
  cancelled: false,
  bytes_total: null,
  bytes_done: null,
};

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};

/** Tolerant parse of the status body: a missing or malformed field reads as "not there", never a throw. */
export function parseSegmenterStatus(raw: unknown): SegmenterStatus {
  const body = obj(raw);
  const d = obj(body.download);
  return {
    supported: body.supported === true,
    reason: str(body.reason),
    downloaded: body.downloaded === true,
    loaded: body.loaded === true,
    device: str(body.device),
    variant: str(body.variant) ?? "",
    label: str(body.label) || DEFAULT_SEGMENTER_LABEL,
    repo: str(body.repo) ?? "",
    revision: str(body.revision) ?? "",
    gated: body.gated === true,
    license_url: str(body.license_url) ?? "",
    size_bytes: num(body.size_bytes),
    precision: str(body.precision),
    download: {
      ...IDLE_DOWNLOAD,
      downloading: d.downloading === true,
      download_id: str(d.download_id),
      error: str(d.error),
      cancelled: d.cancelled === true,
      bytes_total: num(d.bytes_total),
      bytes_done: num(d.bytes_done),
    },
  };
}

function parseBox(v: unknown): [number, number, number, number] | null {
  if (!Array.isArray(v) || v.length !== 4) return null;
  const out = v.map((x) => (typeof x === "number" && Number.isFinite(x) ? x : NaN));
  return out.some(Number.isNaN) ? null : (out as [number, number, number, number]);
}

/** Parse the segment response. Throws when there is no usable mask, since that is the one thing the painter needs. */
export function parseSegmentResult(raw: unknown): SegmentResult {
  const body = obj(raw);
  const mask = str(body.mask);
  if (!mask || !mask.startsWith("data:image/")) {
    throw new Error("The segmenter returned no mask.");
  }
  const instances: SegmentInstance[] = [];
  for (const item of Array.isArray(body.instances) ? body.instances : []) {
    const it = obj(item);
    const box = parseBox(it.box);
    const score = typeof it.score === "number" && Number.isFinite(it.score) ? it.score : null;
    if (box === null || score === null) continue;
    instances.push({ prompt: str(it.prompt) ?? "", score, box });
  }
  const prompts: SegmentPromptResult[] = [];
  for (const item of Array.isArray(body.prompts) ? body.prompts : []) {
    const it = obj(item);
    const text = str(it.text);
    if (!text) continue;
    prompts.push({
      text,
      max_detections: Math.max(1, Math.round(num(it.max_detections) ?? 1)),
      found: Math.round(num(it.found) ?? 0),
    });
  }
  return {
    mask,
    width: num(body.width) ?? 0,
    height: num(body.height) ?? 0,
    instances,
    prompts,
    device: str(body.device) ?? "",
    elapsed_ms: num(body.elapsed_ms) ?? 0,
  };
}

/** "BF16" / "FP8" / "INT8" for a known precision, else null (an older backend reports none). */
export function segmenterPrecisionLabel(precision: string | null | undefined): string | null {
  return precision === "bf16" || precision === "fp8" || precision === "int8"
    ? precision.toUpperCase()
    : null;
}

/** What the download notice says before a download: the variant's name, the precision and size it would fetch, and
 *  whether its license must be accepted on Hugging Face first (a gated variant only; an ungated one needs no token).
 *  ``size`` is pre-formatted. */
export function segmenterDownloadNotice(
  status: Pick<SegmenterStatus, "label" | "gated"> & { precision?: string | null },
  size: string | null,
): { text: string; needsLicense: boolean } {
  const label = status.label || DEFAULT_SEGMENTER_LABEL;
  const detail = [segmenterPrecisionLabel(status.precision), size].filter(Boolean).join(", ");
  return {
    text: `Mask from text uses ${label}${detail ? ` (${detail})` : ""}, which is not downloaded yet.`,
    needsLicense: status.gated === true,
  };
}

/** The Download control is offered: supported, and the weights are not on disk yet. */
export function segmenterNeedsDownload(status: SegmenterStatus | null): boolean {
  return status?.supported === true && !status.downloaded;
}

/** Find can run: supported, downloaded, and no download in flight. */
export function segmenterReady(status: SegmenterStatus | null): boolean {
  return status?.supported === true && status.downloaded && !status.download.downloading;
}

/** 0..1 download fraction, or null when the total is not known yet. */
export function segmenterDownloadFraction(download: SegmenterDownload): number | null {
  const total = download.bytes_total ?? 0;
  if (!(total > 0)) return null;
  return Math.min(1, Math.max(0, (download.bytes_done ?? 0) / total));
}

/** The detector's query count: the backend never keeps more of one concept. */
export const MAX_DETECTIONS = 200;

/** Python's round(): halves go to the even neighbour, so "pillow:2.5" keeps 2 here as it does in the backend. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** ComfyUI's SAM 3 prompt syntax (comfy/text_encoders/sam3_clip.py ``_parse_prompts``): parentheses dropped, comma
 *  separated concepts, an optional trailing ``:N`` for the most detections of that concept (rounded, at least 1,
 *  default 1). For the hint and validation only: the backend parses the prompt it runs. */
export function parseMaskPrompt(text: string): Array<[string, number]> {
  const cleaned = text.replace(/[()]/g, "");
  const out: Array<[string, number]> = [];
  for (const raw of cleaned.split(",")) {
    const part = raw.trim();
    if (!part) continue;
    const m = /^(.+?)\s*:\s*([\d.]+)\s*$/.exec(part);
    if (m) {
      // Number(), not parseFloat: "2.5.1" is not a count, so the whole part stays the concept (as the backend reads it).
      const value = Number(m[2]);
      if (Number.isFinite(value)) {
        out.push([m[1].trim(), Math.min(MAX_DETECTIONS, Math.max(1, roundHalfEven(value)))]);
        continue;
      }
    }
    out.push([part, 1]);
  }
  return out;
}

/** One line after a run: how many concepts matched and how many objects each gave. */
export function segmentSummary(result: Pick<SegmentResult, "prompts" | "instances">): string {
  const prompts = result.prompts;
  if (prompts.length === 0) {
    return result.instances.length > 0
      ? `Found ${result.instances.length} object${result.instances.length === 1 ? "" : "s"}.`
      : "Nothing found.";
  }
  const matched = prompts.filter((p) => p.found > 0);
  if (matched.length === 0) return `Nothing found for ${prompts.map((p) => p.text).join(", ")}.`;
  const parts = matched.map((p) => (p.max_detections > 1 ? `${p.text} x${p.found}` : p.text));
  const missing = prompts.filter((p) => p.found === 0).map((p) => p.text);
  const head = `Found ${matched.length} of ${prompts.length}: ${parts.join(", ")}.`;
  return missing.length ? `${head} Not found: ${missing.join(", ")}.` : head;
}

/** The overlay colour each painter shows the mask in: red for inpaint and the separate mask, white for paint. */
export const MASK_OVERLAY_RGBA: Readonly<Record<"mask" | "paint", readonly [number, number, number, number]>> = {
  mask: [244, 114, 114, 140],
  paint: [255, 255, 255, 255],
};

/** Turn a white-on-black layer's pixels (RGBA, in place copy) into the display overlay: black transparent, white
 *  the overlay colour scaled by its luminance. */
export function tintMaskPixels(
  layer: Uint8ClampedArray,
  rgba: readonly [number, number, number, number],
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(layer.length);
  for (let i = 0; i < layer.length; i += 4) {
    const v = Math.max(layer[i], layer[i + 1], layer[i + 2]);
    if (v === 0) continue;
    out[i] = rgba[0];
    out[i + 1] = rgba[1];
    out[i + 2] = rgba[2];
    out[i + 3] = Math.round((rgba[3] * v) / 255);
  }
  return out;
}
