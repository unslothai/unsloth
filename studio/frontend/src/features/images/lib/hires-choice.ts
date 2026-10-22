// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Hires fix (backend diffusion_hires.py). Upscaler labels match A1111's `Hires upscaler` so infotext maps back. */

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
  /** false = no separate Hires CFG (native engine); absent = supported. */
  supports_guidance?: boolean;
  /** The smallest denoise the engine takes (native: no upscale-only 0); absent = 0. */
  min_denoise?: number;
}

/** A1111's Denoising strength slider and ComfyUI's KSampler denoise both step by 0.01. */
export const HIRES_DENOISE_STEP = 0.01;

export function hiresMinDenoise(cap: HiresCapability): number {
  return finite(cap.min_denoise) ?? 0;
}

/** The smallest denoise for this choice: the latent upscaler always needs a refine pass (the backend refuses 0). */
export function hiresMinDenoiseFor(choice: HiresChoice, cap: HiresCapability): number {
  const latent = cap.upscalers.find((u) => u.id === effectiveUpscaler(choice, cap))?.kind === "latent";
  return Math.max(hiresMinDenoise(cap), latent ? HIRES_DENOISE_STEP : 0);
}

export function hiresGuidanceSupported(cap: HiresCapability): boolean {
  return cap.supports_guidance !== false;
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
  // Snap to the 0.01 grid (a typed 0.004 would otherwise be refused), never below the engine's minimum.
  const denoise = clamp(Math.round(choice.denoise / HIRES_DENOISE_STEP) * HIRES_DENOISE_STEP, hiresMinDenoiseFor(choice, cap), 1);
  return {
    scale: choice.scale,
    denoise: Number(denoise.toFixed(2)),
    steps: choice.steps && choice.steps > 0 ? choice.steps : null,
    upscaler: effectiveUpscaler(choice, cap),
    guidance: hiresGuidanceSupported(cap) ? choice.guidance : null,
  };
}

/** Whether the refine pass's own guidance (Hires CFG) engages a negative prompt the base guidance leaves off. */
export function hiresEngagesNegative(
  choice: HiresChoice,
  cap: HiresCapability | null | undefined,
  workflow: string,
  minGuidance: number | null,
): boolean {
  const req = hiresRequest(choice, cap, workflow);
  if (!req || req.denoise <= 0 || req.guidance === null || req.guidance === undefined) return false;
  return minGuidance === null || req.guidance > minGuidance;
}

/** A recorded image whose Hires CFG ran with CFG on, so its negative prompt shaped the refine pass. */
export function hiresRecordUsedNegative(r: HiresRecord): boolean {
  return finite(r.hires_scale) !== undefined && (finite(r.hires_guidance) ?? 0) > 0;
}

/** The Upscale workflow's `upscale_model`: only an id the status still lists (the selector shows Lanczos otherwise). */
export function upscaleModelRequest(
  id: string | null | undefined,
  models: Array<{ id: string }> | null | undefined,
): string | undefined {
  return id && models?.some((m) => m.id === id) ? id : undefined;
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

export function hiresNotApplied(choice: HiresChoice, cap: HiresCapability | null | undefined): string[] {
  if (!choice.enabled) return [];
  if (!cap) return [`Hires fix (${hiresSummary(choice)})`];
  const notes: string[] = [];
  if (choice.upscaler && matchHiresUpscaler(choice.upscaler, cap) === null)
    notes.push(`Hires upscaler ${choice.upscaler} (not available; using ${labelOf(cap.default_upscaler, cap)})`);
  if (choice.guidance !== null && !hiresGuidanceSupported(cap))
    notes.push(`Hires CFG Scale ${choice.guidance} (not available on this engine; using the base guidance)`);
  if (choice.denoise < hiresMinDenoise(cap))
    notes.push(`Hires denoise ${choice.denoise} (this engine needs at least ${hiresMinDenoise(cap)})`);
  else if (choice.denoise < hiresMinDenoiseFor(resolveRestoredHires(choice, cap), cap))
    notes.push(`Hires denoise ${choice.denoise} (the latent upscaler needs a refine pass; using ${hiresMinDenoiseFor(resolveRestoredHires(choice, cap), cap)})`);
  return notes;
}

export function resolveRestoredHires(choice: HiresChoice, cap: HiresCapability | null | undefined): HiresChoice {
  if (!choice.enabled || !cap || !choice.upscaler) return choice;
  return { ...choice, upscaler: matchHiresUpscaler(choice.upscaler, cap) };
}

function labelOf(id: string, cap?: HiresCapability | null): string {
  return cap?.upscalers.find((u) => u.id === id)?.label ?? HIRES_UPSCALER_LABELS[id] ?? id.replace(/^local:/, "");
}

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

/** A1111 hires keys -> recipe fields; mapped labels are removed from `fields`, the rest reported unapplied. */
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
  let scale = num(take("hires upscale"));
  const resizeKey = Object.keys(fields).find((k) => k.toLowerCase() === "hires resize");
  const steps = num(take("hires steps"));
  const upscaler = take("hires upscaler");
  const cfg = num(take("hires cfg scale"));
  // `Hires resize: WxH` (A1111's explicit size; Studio writes it for a capped target) maps to the factor that
  // reaches it from Size; the backend's own cap lands on the same size again.
  const wh = resizeKey === undefined ? null : fields[resizeKey].trim().match(/^(\d+)\s*x\s*(\d+)$/i);
  if (scale === undefined && wh && typeof size.width === "number" && typeof size.height === "number" && size.width > 0 && size.height > 0) {
    const rw = Number(wh[1]);
    const rh = Number(wh[2]);
    const factor = Math.round(Math.max(rw / size.width, rh / size.height) * 100) / 100;
    // Only a size one factor reproduces (Studio's own export); A1111 crops another aspect, which Studio cannot.
    const reaches = (maxSide: number) => {
      const t = hiresTargetSize(size.width as number, size.height as number, factor, maxSide);
      return t.width === rw && t.height === rh;
    };
    if (factor >= 1 && factor <= 4 && (reaches(2048) || reaches(Math.max(rw, rh)))) {
      scale = factor;
      delete fields[resizeKey as string];
    }
  }
  // Without a factor or a usable size, Studio cannot size the pass: leave it all unapplied.
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
