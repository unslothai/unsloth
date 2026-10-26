// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Enhance prompt": the FLUX.3 prompt upsampler rewrites a short request into the detailed caption the model was
// trained on. Pure logic only (no "@/" imports) so node:test can load it directly.

/** Every upsampler route, in one place. */
export const PROMPT_UPSAMPLER_ROUTES = {
  status: "/api/inference/images/prompt-upsampler/status",
  download: "/api/inference/images/prompt-upsampler/download",
  cancel: "/api/inference/images/prompt-upsampler/cancel",
  preview: "/api/inference/images/prompt-upsampler/preview",
  warm: "/api/inference/images/prompt-upsampler/warm",
  release: "/api/inference/images/prompt-upsampler/release",
} as const;

export const ENHANCE_PROMPT_LABEL = "Enhance prompt";
export const ENHANCE_PROMPT_HINT =
  "Rewrites your prompt into a detailed caption with FLUX.3's prompt upsampler before generating. Fixes most broken poses and hands. Turn off to use your prompt exactly as typed.";

export type PromptEnhancement = "applied" | "off" | "unavailable" | "failed";

/** Same shape as the speech-to-text model download status (SttDownloadStatus). */
export interface PromptUpsamplerDownload {
  downloading: boolean;
  model: string | null;
  download_id?: string | null;
  completed_download_ids?: string[];
  variant?: string | null;
  error: string | null;
  cancelled?: boolean;
  cancelled_model?: string | null;
  bytes_total: number | null;
  bytes_done: number | null;
}

export interface PromptUpsamplerStatus {
  supported: boolean;
  downloaded: boolean;
  loaded: boolean;
  repo: string;
  filename: string;
  size_bytes: number;
  mmproj_size_bytes: number;
  mmproj_downloaded: boolean;
  // False while an earlier quant serves (downloaded above); an older backend without it reads downloaded && mmproj.
  current_downloaded: boolean;
  current_missing_bytes: number;
  // "" from an older backend.
  current_filename: string;
  // Local files from the env overrides: no repo download is offered beside them.
  override: boolean;
  download: PromptUpsamplerDownload;
}

export interface PromptUpsamplerPreviewRequest {
  prompt: string;
  reference_images?: string[];
  variant?: "distilled" | "base";
}

const IDLE_DOWNLOAD: PromptUpsamplerDownload = {
  downloading: false,
  model: null,
  error: null,
  bytes_total: null,
  bytes_done: null,
};

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

/** Tolerant parse of the status body: a missing or malformed field reads as "not there", never a throw. */
export function parsePromptUpsamplerStatus(raw: unknown): PromptUpsamplerStatus {
  const body = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const d =
    body.download && typeof body.download === "object"
      ? (body.download as Record<string, unknown>)
      : {};
  const download: PromptUpsamplerDownload = {
    ...IDLE_DOWNLOAD,
    downloading: d.downloading === true,
    model: str(d.model),
    error: str(d.error),
    bytes_total: num(d.bytes_total),
    bytes_done: num(d.bytes_done),
  };
  if (typeof d.download_id === "string") download.download_id = d.download_id;
  if (typeof d.cancelled === "boolean") download.cancelled = d.cancelled;
  return {
    supported: body.supported === true,
    downloaded: body.downloaded === true,
    loaded: body.loaded === true,
    repo: str(body.repo) ?? "",
    filename: str(body.filename) ?? "",
    size_bytes: num(body.size_bytes) ?? 0,
    mmproj_size_bytes: num(body.mmproj_size_bytes) ?? 0,
    mmproj_downloaded: body.mmproj_downloaded === true,
    current_downloaded:
      typeof body.current_downloaded === "boolean"
        ? body.current_downloaded
        : body.downloaded === true && body.mmproj_downloaded === true,
    current_missing_bytes: num(body.current_missing_bytes) ?? 0,
    current_filename: str(body.current_filename) ?? "",
    override: body.override === true,
    download,
  };
}

/** The model, its vision projector (edits) or the current default quant is missing. */
export function upsamplerNeedsDownload(status: PromptUpsamplerStatus): boolean {
  return (
    status.supported &&
    !status.override &&
    (!status.downloaded || !status.mmproj_downloaded || !status.current_downloaded)
  );
}

/** An earlier quant serves; a model with no projector counts when it is not the default file (the Download then
 * fetches both, not the projector alone). */
export function upsamplerServesEarlierQuant(status: PromptUpsamplerStatus): boolean {
  if (!status.downloaded || status.current_downloaded) return false;
  return (
    status.mmproj_downloaded ||
    (status.current_filename !== "" && status.filename !== "" && status.filename !== status.current_filename)
  );
}

/** Bytes the Download button fetches: whichever of the model and the vision projector is still missing. */
export function upsamplerDownloadBytes(status: PromptUpsamplerStatus): number {
  if (upsamplerServesEarlierQuant(status)) return status.current_missing_bytes;
  return (
    (status.downloaded ? 0 : status.size_bytes) +
    (status.mmproj_downloaded ? 0 : status.mmproj_size_bytes)
  );
}

/** 0..1 download fraction, or null when the total is not known yet. */
export function upsamplerDownloadFraction(download: PromptUpsamplerDownload): number | null {
  const total = download.bytes_total ?? 0;
  if (!(total > 0)) return null;
  return Math.min(1, Math.max(0, (download.bytes_done ?? 0) / total));
}

/** What the loaded model advertises. Absent on an older backend: unsupported, default off. */
export interface PromptUpsamplerCapability {
  supported: boolean;
  defaultOn: boolean;
}

export function promptUpsamplerCapability(
  status:
    | {
        loaded?: boolean;
        supports_prompt_upsampler?: boolean;
        prompt_upsampler_default?: boolean;
      }
    | null
    | undefined,
  external = false,
): PromptUpsamplerCapability {
  const supported = !external && status?.loaded === true && status.supports_prompt_upsampler === true;
  return { supported, defaultOn: supported && status?.prompt_upsampler_default === true };
}

/** Whether to start the upsampler ahead of the first enhanced run: Enhance on, its files downloaded, no download
 *  running. The backend decides the rest (GPU owner, free memory) and answers idempotently. */
export function shouldWarmUpsampler(
  enabled: boolean,
  status: Pick<PromptUpsamplerStatus, "supported" | "downloaded" | "download"> | null,
): boolean {
  return enabled && status !== null && status.supported && status.downloaded && !status.download.downloading;
}

/** Warm outcomes that clear on their own (a render, a load, training or a GPU handover in progress, a request
 *  holding the server): the panel asks again a little later. Memory, placement and setup outcomes do not. */
export const WARM_RETRY_OUTCOMES: ReadonlySet<string> = new Set([
  "generating",
  "loading",
  "gpu",
  "busy",
  "stopped",
  "cancelled",
]);
export const WARM_RETRY_DELAY_MS = 15_000;
export const WARM_RETRY_LIMIT = 8;

/** The toggle's value: a one-off restore override, else the stored per-family choice, else the family default. */
export function resolveEnhancePrompt(args: {
  capability: PromptUpsamplerCapability;
  stored: boolean | undefined;
  override?: boolean | null;
}): boolean {
  if (!args.capability.supported) return false;
  if (typeof args.override === "boolean") return args.override;
  if (typeof args.stored === "boolean") return args.stored;
  return args.capability.defaultOn;
}

/** The request field. Only a supporting family sends it, so every other request stays byte-identical. */
export function enhancePromptRequestField(
  capability: PromptUpsamplerCapability,
  on: boolean,
): { enhance_prompt?: boolean } {
  return capability.supported ? { enhance_prompt: on } : {};
}

/** The preview variant for the loaded family: base checkpoints get the base caption style. */
export function upsamplerVariantFor(family: string | null | undefined): "distilled" | "base" {
  return (family ?? "").toLowerCase().includes("base") ? "base" : "distilled";
}

const KEY = "unsloth:images:enhance-prompt-by-family";

export function readEnhanceChoices(): Record<string, boolean> {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const out: Record<string, boolean> = {};
    for (const [k, v] of Object.entries(value)) if (typeof v === "boolean") out[k] = v;
    return out;
  } catch {
    return {};
  }
}

/** Store the choice for one family. A choice equal to the family default is dropped, so a changed default reaches
 *  users who never touched the toggle. */
export function withEnhanceChoice(
  choices: Record<string, boolean>,
  familyKey: string,
  on: boolean,
  defaultOn: boolean,
): Record<string, boolean> {
  if (!familyKey) return choices;
  const next = { ...choices };
  if (on === defaultOn) delete next[familyKey];
  else next[familyKey] = on;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // storage can be unavailable in private browsing
  }
  return next;
}

/** What a recipe restore puts back. `exact` reuses the caption that actually ran, with enhancement off. */
export function enhanceRestoreFor(
  image: { prompt: string; enhanced_prompt?: string | null; prompt_enhancement?: string | null },
  exact = false,
): { prompt: string; enhance: boolean | null } {
  const applied = image.prompt_enhancement === "applied";
  const caption = (image.enhanced_prompt ?? "").trim();
  if (exact && applied && caption) return { prompt: caption, enhance: false };
  if (applied) return { prompt: image.prompt, enhance: true };
  // "off" ran the prompt as typed; unavailable / failed did too, but the user had asked for it, so keep asking.
  if (image.prompt_enhancement === "off") return { prompt: image.prompt, enhance: false };
  if (image.prompt_enhancement === "unavailable" || image.prompt_enhancement === "failed") {
    return { prompt: image.prompt, enhance: true };
  }
  // Records from before the upsampler, or another family: leave the toggle alone.
  return { prompt: image.prompt, enhance: null };
}

/** Whether a record carries an enhanced caption worth showing next to the user's prompt. */
export function hasEnhancedPrompt(image: {
  prompt: string;
  enhanced_prompt?: string | null;
  prompt_enhancement?: string | null;
}): boolean {
  const caption = (image.enhanced_prompt ?? "").trim();
  return Boolean(caption) && caption !== image.prompt.trim();
}

/** Two 32-bit hashes (FNV-1a, djb2) over the whole data URL: the same-format headers and trailers of two different
 *  images often match, so the ends alone do not tell them apart. */
export function referenceDigest(src: string): string {
  let fnv = 0x811c9dc5;
  let djb = 5381;
  for (let i = 0; i < src.length; i++) {
    const code = src.charCodeAt(i);
    fnv = Math.imul(fnv ^ code, 0x01000193);
    djb = (Math.imul(djb, 33) + code) | 0;
  }
  return `${src.length}:${(fnv >>> 0).toString(16)}:${(djb >>> 0).toString(16)}`;
}

// The stale check runs on every render: digests of the images the last call named, and only those, so a removed
// image is not kept alive by the cache.
let referenceDigests = new Map<string, string>();

/** Drop the cached digests (and with them the uploads they are keyed on) once no preview is shown. */
export function clearPreviewDigests(): void {
  referenceDigests = new Map();
}

/** What a preview caption was written for: the prompt, the variant and the reference images. */
export function previewSignature(body: PromptUpsamplerPreviewRequest): string {
  const next = new Map<string, string>();
  const refs = (body.reference_images ?? []).map((src) => {
    const digest = next.get(src) ?? referenceDigests.get(src) ?? referenceDigest(src);
    next.set(src, digest);
    return digest;
  });
  referenceDigests = next;
  return JSON.stringify([body.prompt.trim(), body.variant ?? null, refs]);
}

/** Notice when the upsampler failed, else null; "unavailable" stays silent. */
export function enhancementNotice(
  outcome: string | null | undefined,
): { title: string; description: string } | null {
  if (outcome === "failed") {
    return {
      title: "Prompt used as typed",
      description: "The prompt upsampler could not rewrite this prompt, so it ran unchanged.",
    };
  }
  return null;
}

/** The family key of a loaded model or connection with no upsampler. */
export const ENHANCE_UNSUPPORTED_FAMILY = "\u0000unsupported";

/** The one-off Enhance override after the loaded family becomes ``familyKey`` (non-empty): one set before any model
 *  loaded ("") binds to the first FLUX.3 family and survives an unsupported model, one set under another family is
 *  dropped, so it cannot come back on a later switch. */
export function overrideAfterFamilyChange(
  prev: { family: string; on: boolean } | null,
  familyKey: string,
): { family: string; on: boolean } | null {
  if (!prev || !familyKey || prev.family === familyKey) return prev;
  if (prev.family === "") return familyKey === ENHANCE_UNSUPPORTED_FAMILY ? prev : { ...prev, family: familyKey };
  return null;
}

