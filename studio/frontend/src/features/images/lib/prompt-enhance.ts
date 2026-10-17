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
    download,
  };
}

/** Bytes the Download button fetches: the model plus the vision projector (edits) when that is still missing. */
export function upsamplerDownloadBytes(status: PromptUpsamplerStatus): number {
  return status.size_bytes + (status.mmproj_downloaded ? 0 : status.mmproj_size_bytes);
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

/** Non-blocking notice after a run that asked for enhancement and did not get it, else null. */
export function enhancementNotice(
  outcome: string | null | undefined,
): { title: string; description: string } | null {
  if (outcome === "unavailable") {
    return {
      title: "Prompt used as typed",
      description: "The prompt upsampler is not downloaded yet. Download it to enhance prompts.",
    };
  }
  if (outcome === "failed") {
    return {
      title: "Prompt used as typed",
      description: "The prompt upsampler could not rewrite this prompt, so it ran unchanged.",
    };
  }
  return null;
}
