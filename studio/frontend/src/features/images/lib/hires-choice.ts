// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Hires fix: render, upscale, then refine in one request (backend core/inference/diffusion_hires.py).
 *
 * Defaults are ComfyUI's two-pass example (1.5x, denoise 0.5). "Hires steps" is the exact number of refine
 * steps; empty means the base pass's steps, as AUTOMATIC1111 runs `hr_second_pass_steps or steps`.
 * Upscaler labels are what AUTOMATIC1111 writes as `Hires upscaler`, so pasted infotext maps back by name.
 */

export interface HiresUpscalerOption {
  id: string;
  label: string;
  kind: "latent" | "pixel" | "model";
}

export interface HiresCapability {
  upscalers: HiresUpscalerOption[];
  default_upscaler: string;
  defaults: { scale: number; denoise: number };
  max_side: number;
}

export interface HiresRequest {
  scale: number;
  denoise: number;
  steps?: number | null;
  upscaler?: string | null;
  guidance?: number | null;
}

export interface HiresChoice {
  enabled: boolean;
  scale: number;
  denoise: number;
  /** null = the base pass's steps. */
  steps: number | null;
  /** null = the capability's default upscaler. */
  upscaler: string | null;
  /** null = the base guidance. */
  guidance: number | null;
}

export const HIRES_DEFAULT_SCALE = 1.5;
export const HIRES_DEFAULT_DENOISE = 0.5;

export const HIRES_OFF: HiresChoice = {
  enabled: false,
  scale: HIRES_DEFAULT_SCALE,
  denoise: HIRES_DEFAULT_DENOISE,
  steps: null,
  upscaler: null,
  guidance: null,
};

/** Hires fix shows only in Create (text to image) and only where the loaded model reports the capability. */
export function hiresVisible(cap: HiresCapability | null | undefined, workflow: string): cap is HiresCapability {
  return Boolean(cap) && workflow === "create";
}

/** The refine size the backend renders: round(side * scale / 16) * 16, scaled down to fit `maxSide`. */
export function hiresTargetSize(
  width: number,
  height: number,
  scale: number,
  maxSide = 2048,
  multiple = 16,
): { width: number; height: number } {
  const tw = width * scale;
  const th = height * scale;
  const fit = Math.min(1, maxSide / Math.max(tw, th));
  return {
    width: Math.max(multiple, Math.round((tw * fit) / multiple) * multiple),
    height: Math.max(multiple, Math.round((th * fit) / multiple) * multiple),
  };
}

/** The upscaler the request would run: the pick when the model offers it, else the default. */
export function effectiveUpscaler(choice: HiresChoice, cap: HiresCapability): string {
  const picked = choice.upscaler;
  return picked && cap.upscalers.some((u) => u.id === picked) ? picked : cap.default_upscaler;
}

/** The request's `hires` field; undefined when off or not offered (so the request is unchanged). */
export function hiresRequest(
  choice: HiresChoice,
  cap: HiresCapability | null | undefined,
  workflow: string,
): HiresRequest | undefined {
  if (!choice.enabled || !hiresVisible(cap, workflow)) return undefined;
  return {
    scale: choice.scale,
    denoise: choice.denoise,
    steps: choice.steps && choice.steps > 0 ? choice.steps : null,
    upscaler: effectiveUpscaler(choice, cap),
    guidance: choice.guidance,
  };
}

export interface HiresRecord {
  hires_scale?: number | null;
  hires_denoise?: number | null;
  hires_steps?: number | null;
  hires_upscaler?: string | null;
  hires_guidance?: number | null;
  hires_base_width?: number | null;
  hires_base_height?: number | null;
  width?: number;
  height?: number;
}

const finite = (v: unknown): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : undefined;
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

/** The Hires fix state a recipe restores, or HIRES_OFF for an image made without it. */
export function hiresFromRecipe(r: HiresRecord): HiresChoice {
  const scale = finite(r.hires_scale);
  if (scale === undefined) return HIRES_OFF;
  const denoise = finite(r.hires_denoise);
  const steps = finite(r.hires_steps);
  const guidance = finite(r.hires_guidance);
  return {
    enabled: true,
    scale: clamp(scale, 1, 4),
    denoise: denoise === undefined ? HIRES_DEFAULT_DENOISE : clamp(denoise, 0, 1),
    steps: steps !== undefined && Number.isInteger(steps) && steps > 0 ? Math.min(steps, 100) : null,
    upscaler: typeof r.hires_upscaler === "string" && r.hires_upscaler ? r.hires_upscaler : null,
    guidance: guidance === undefined ? null : clamp(guidance, 0, 20),
  };
}

/** The base-pass size a hires recipe restores into the Width / Height controls (the record keeps the output size). */
export function hiresBaseSize(r: HiresRecord): { width: number; height: number } | null {
  const w = finite(r.hires_base_width);
  const h = finite(r.hires_base_height);
  if (finite(r.hires_scale) === undefined || !w || !h) return null;
  return { width: w, height: h };
}

/** Map an upscaler name (an id from a Studio recipe, or AUTOMATIC1111's label) onto the model's options. */
export function matchHiresUpscaler(name: string, cap: HiresCapability): string | null {
  const byId = cap.upscalers.find((u) => u.id === name);
  if (byId) return byId.id;
  const lower = name.trim().toLowerCase();
  const byLabel = cap.upscalers.find((u) => u.label.toLowerCase() === lower);
  if (byLabel) return byLabel.id;
  // AUTOMATIC1111's "Latent", "Latent (bicubic)", ... all upscale the latents; Studio runs nearest-exact.
  if (lower.startsWith("latent")) return cap.upscalers.find((u) => u.kind === "latent")?.id ?? null;
  return null;
}

/** Settings of a restored or pasted Hires fix the loaded model cannot apply. */
export function hiresNotApplied(choice: HiresChoice, cap: HiresCapability | null | undefined): string[] {
  if (!choice.enabled) return [];
  if (!cap) return [`Hires fix (${hiresSummary(choice)})`];
  if (choice.upscaler && matchHiresUpscaler(choice.upscaler, cap) === null)
    return [`Hires upscaler ${choice.upscaler} (not available; using ${labelOf(cap.default_upscaler, cap)})`];
  return [];
}

/** Resolve a restored choice's upscaler name to an id the model offers (null = the default). */
export function resolveRestoredHires(choice: HiresChoice, cap: HiresCapability | null | undefined): HiresChoice {
  if (!choice.enabled || !cap || !choice.upscaler) return choice;
  return { ...choice, upscaler: matchHiresUpscaler(choice.upscaler, cap) };
}

function labelOf(id: string, cap?: HiresCapability | null): string {
  return cap?.upscalers.find((u) => u.id === id)?.label ?? HIRES_UPSCALER_LABELS[id] ?? id.replace(/^local:/, "");
}

/** Labels for ids a recipe can name when no model is loaded (backend diffusion_upscalers). */
export const HIRES_UPSCALER_LABELS: Record<string, string> = {
  latent: "Latent (nearest-exact)",
  lanczos: "Lanczos",
  realesrgan_x4plus: "R-ESRGAN 4x+",
  realesrgan_x4plus_anime_6b: "R-ESRGAN 4x+ Anime6B",
};

/** "1.5x, denoise 0.5, 14 steps, Latent (nearest-exact)" for the Recipe popover. */
export function hiresSummary(choice: HiresChoice, cap?: HiresCapability | null): string {
  const parts = [`${choice.scale}x`, `denoise ${choice.denoise}`];
  parts.push(choice.steps ? `${choice.steps} steps` : "base steps");
  if (choice.upscaler) parts.push(labelOf(choice.upscaler, cap));
  if (choice.guidance !== null) parts.push(`CFG ${choice.guidance}`);
  return parts.join(", ");
}

/**
 * AUTOMATIC1111 hires keys (modules/processing.py) -> recipe fields. `fields` holds every `Hires *` label the
 * infotext carried; the ones mapped here are removed, the rest stay to be named as not applied.
 */
export function parseHiresInfotext(
  fields: Record<string, string>,
  denoise: number | undefined,
  size: { width?: unknown; height?: unknown },
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const take = (label: string): string | undefined => {
    const key = Object.keys(fields).find((k) => k.toLowerCase() === label);
    if (key === undefined) return undefined;
    const value = fields[key];
    delete fields[key];
    return value;
  };
  const num = (v: string | undefined) => {
    if (v === undefined || v.trim() === "") return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  const scale = num(take("hires upscale"));
  const steps = num(take("hires steps"));
  const upscaler = take("hires upscaler");
  const cfg = num(take("hires cfg scale"));
  // Without a factor the pass is sized by `Hires resize`, which Studio does not offer: leave it all unapplied.
  if (scale === undefined) {
    if (steps !== undefined) fields["Hires steps"] = String(steps);
    if (upscaler !== undefined) fields["Hires upscaler"] = upscaler;
    if (cfg !== undefined) fields["Hires CFG Scale"] = String(cfg);
    return out;
  }
  out.hires_scale = scale;
  out.hires_denoise = denoise ?? HIRES_DEFAULT_DENOISE;
  if (steps !== undefined && Number.isInteger(steps) && steps > 0) out.hires_steps = steps;
  if (upscaler) out.hires_upscaler = upscaler;
  if (cfg !== undefined) out.hires_guidance = cfg;
  if (typeof size.width === "number" && typeof size.height === "number") {
    out.hires_base_width = size.width;
    out.hires_base_height = size.height;
  }
  return out;
}
