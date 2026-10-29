// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Detailer (face fix) logic. No "@/" imports so node:test can load it directly.

export const DETAILER_LABEL = "Detailer (face fix)";
export const DETAILER_HINT =
  "After each image, finds every face (or what you name) with SAM 3, redraws it at a higher resolution and pastes it back, like ADetailer or ComfyUI's FaceDetailer. Nothing outside the found areas changes. Each face costs one extra inpaint.";
export const DETAILER_TARGETS_HINT =
  "What to fix, separated by commas, as in Mask from text: face, hand, eyes. Add :N to cap a target (up to 8 each by default).";
export const DETAILER_PROMPT_HINT =
  "Prompt for the redraw. Empty uses the image's prompt; [PROMPT] inserts it; [SEP] gives the next faces their own text.";

export type DetailerGuideSizeFor = "bbox" | "crop_region";

const isGuideSizeFor = (v: unknown): v is DetailerGuideSizeFor => v === "bbox" || v === "crop_region";

export interface DetailerOptions {
  targets: string;
  prompt: string;
  denoise: number;
  confidence: number;
  cropFactor: number;
  guideSize: number;
  guideSizeFor: DetailerGuideSizeFor;
  maxSize: number;
  dilation: number;
  feather: number;
  dropSize: number;
  steps: number | null;
  negativePrompt: string;
  /** null / empty = the image's own. */
  guidance: number | null;
}

/** ComfyUI-Impact-Pack FaceDetailer's defaults (modules/impact/impact_pack.py:743-769). */
export const DETAILER_DEFAULTS: DetailerOptions = {
  targets: "face",
  prompt: "",
  denoise: 0.5,
  confidence: 0.5,
  cropFactor: 3.0,
  guideSize: 512,
  guideSizeFor: "bbox",
  maxSize: 1024,
  dilation: 10,
  feather: 5,
  dropSize: 10,
  steps: null,
  negativePrompt: "",
  guidance: null,
};

export interface DetailerRequest {
  targets: string;
  prompt?: string;
  denoise?: number;
  confidence?: number;
  crop_factor?: number;
  guide_size?: number;
  guide_size_for?: DetailerGuideSizeFor;
  max_size?: number;
  dilation?: number;
  feather?: number;
  drop_size?: number;
  steps?: number;
  negative_prompt?: string;
  guidance?: number;
}

export interface DetailerDetection {
  target?: string;
  score?: number;
  box?: number[];
  crop?: number[];
  size?: number[];
  seed?: number;
  prompt?: string;
}

export interface DetailerRecord {
  targets?: string;
  prompt?: string;
  negative_prompt?: string;
  confidence?: number;
  denoise?: number;
  crop_factor?: number;
  guide_size?: number;
  guide_size_for?: string;
  max_size?: number;
  dilation?: number;
  feather?: number;
  drop_size?: number;
  steps?: number;
  guidance?: number;
  model?: string;
  detections?: DetailerDetection[];
  found?: number;
  error?: string;
  [key: string]: unknown;
}

interface DetailerModelStatus {
  loaded?: boolean;
  workflows?: string[] | null;
}

interface DetailerSegmenterStatus {
  supported: boolean;
  downloaded: boolean;
}

export function detailerModelCapable(status: DetailerModelStatus | null | undefined, external: boolean): boolean {
  return !external && status?.loaded === true && (status.workflows ?? []).includes("inpaint");
}

export function detailerAvailable(
  status: DetailerModelStatus | null | undefined,
  external: boolean,
  segmenter: DetailerSegmenterStatus | null | undefined,
): boolean {
  return detailerModelCapable(status, external) && segmenter?.supported === true && segmenter.downloaded === true;
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

export function detailerRequestField(
  enabled: boolean,
  available: boolean,
  options: DetailerOptions,
): { detailer?: DetailerRequest } {
  if (!enabled || !available) return {};
  const targets = options.targets.trim() || DETAILER_DEFAULTS.targets;
  const out: DetailerRequest = { targets };
  const prompt = options.prompt.trim();
  if (prompt) out.prompt = prompt;
  const numeric: Array<[keyof DetailerRequest, number, number, number, number]> = [
    ["denoise", options.denoise, DETAILER_DEFAULTS.denoise, 0.01, 1],
    ["confidence", options.confidence, DETAILER_DEFAULTS.confidence, 0, 1],
    ["crop_factor", options.cropFactor, DETAILER_DEFAULTS.cropFactor, 1, 10],
    ["guide_size", options.guideSize, DETAILER_DEFAULTS.guideSize, 64, 2048],
    ["max_size", options.maxSize, DETAILER_DEFAULTS.maxSize, 64, 2048],
    ["dilation", options.dilation, DETAILER_DEFAULTS.dilation, -128, 128],
    ["feather", options.feather, DETAILER_DEFAULTS.feather, 0, 100],
    ["drop_size", Math.round(options.dropSize), DETAILER_DEFAULTS.dropSize, 1, 1024],
  ];
  for (const [key, value, fallback, lo, hi] of numeric) {
    if (!Number.isFinite(value)) continue;
    const bounded = clamp(value, lo, hi);
    if (bounded !== fallback) (out as unknown as Record<string, number>)[key] = bounded;
  }
  if (isGuideSizeFor(options.guideSizeFor) && options.guideSizeFor !== DETAILER_DEFAULTS.guideSizeFor)
    out.guide_size_for = options.guideSizeFor;
  if (options.steps !== null && Number.isFinite(options.steps)) out.steps = clamp(Math.round(options.steps), 1, 200);
  const negative = (options.negativePrompt ?? "").trim();
  if (negative) out.negative_prompt = negative;
  if (options.guidance !== null && options.guidance !== undefined && Number.isFinite(options.guidance))
    out.guidance = clamp(options.guidance, 0, 100);
  return { detailer: out };
}

export function detailerTemplateInherits(template: string | undefined): boolean {
  if (!template) return true;
  return template.split(/\s*\[SEP\]\s*/).some((part) => !part.trim() || part.includes("[PROMPT]"));
}

/** The Detailer's CFG can engage the inherited negative prompt even when the base CFG leaves it unused. */
export function detailerEngagesNegative(
  enabled: boolean,
  available: boolean,
  options: DetailerOptions,
  minGuidance: number | null,
): boolean {
  const req = detailerRequestField(enabled, available, options).detailer;
  if (!req || !detailerTemplateInherits(req.negative_prompt) || req.guidance === undefined) return false;
  return minGuidance === null || req.guidance > minGuidance;
}

const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);

export function detailerRecordOf(value: unknown): DetailerRecord | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const v = value as Record<string, unknown>;
  const out: DetailerRecord = {};
  for (const key of ["targets", "prompt", "negative_prompt", "guide_size_for", "model", "error"] as const) {
    const s = str(v[key]);
    if (s !== undefined) out[key] = s;
  }
  for (const key of [
    "confidence",
    "denoise",
    "crop_factor",
    "guide_size",
    "max_size",
    "dilation",
    "feather",
    "drop_size",
    "steps",
    "guidance",
    "found",
  ] as const) {
    const n = num(v[key]);
    if (n !== undefined) out[key] = n;
  }
  if (Array.isArray(v.detections)) out.detections = v.detections.filter((d) => d && typeof d === "object") as DetailerDetection[];
  return Object.keys(out).length > 0 ? out : null;
}

export function restoredDetailer(image: { detailer?: unknown }): {
  enabled: boolean;
  options: Partial<DetailerOptions>;
} {
  const record = detailerRecordOf(image.detailer);
  if (!record) return { enabled: false, options: {} };
  // Missing settings reset to defaults so a restore never keeps a stale value.
  const options: Partial<DetailerOptions> = { ...DETAILER_DEFAULTS };
  if (record.targets) options.targets = record.targets;
  options.prompt = record.prompt ?? "";
  const pairs: Array<[keyof DetailerOptions, number | undefined]> = [
    ["denoise", record.denoise],
    ["confidence", record.confidence],
    ["cropFactor", record.crop_factor],
    ["guideSize", record.guide_size],
    ["maxSize", record.max_size],
    ["dilation", record.dilation],
    ["feather", record.feather],
    ["dropSize", record.drop_size],
  ];
  for (const [key, value] of pairs) {
    if (value !== undefined) (options as Record<string, unknown>)[key] = value;
  }
  options.guideSizeFor = isGuideSizeFor(record.guide_size_for) ? record.guide_size_for : DETAILER_DEFAULTS.guideSizeFor;
  options.steps = record.steps ?? null;
  options.negativePrompt = record.negative_prompt ?? "";
  options.guidance = record.guidance ?? null;
  return { enabled: true, options };
}

/** First-tab `ADetailer ...` infotext fields (lower-cased keys) -> a record. */
export function detailerFromInfotext(fields: Record<string, string>): DetailerRecord | null {
  const get = (key: string): string | undefined => fields[`adetailer ${key}`];
  const model = get("model");
  if (model === undefined || model === "None") return null;
  const n = (key: string): number | undefined => {
    const raw = get(key);
    if (raw === undefined || raw === "") return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
  };
  const out: DetailerRecord = { model };
  // ADetailer's YOLO models carry no text; a face model means faces, a hand model hands.
  const classes = get("model classes");
  out.targets = classes || (/hand/i.test(model) ? "hand" : /person/i.test(model) ? "person" : "face");
  const prompt = get("prompt");
  if (prompt) out.prompt = prompt;
  const negative = get("negative prompt");
  if (negative) out.negative_prompt = negative;
  const confidence = n("confidence");
  if (confidence !== undefined) out.confidence = confidence;
  const dilation = n("dilate erode");
  if (dilation !== undefined) out.dilation = dilation;
  const feather = n("mask blur");
  if (feather !== undefined) out.feather = feather;
  const denoise = n("denoising strength");
  if (denoise !== undefined) out.denoise = denoise;
  if (/^true$/i.test(get("use separate steps") ?? "")) {
    const steps = n("steps");
    if (steps !== undefined) out.steps = steps;
  }
  if (/^true$/i.test(get("use separate cfg scale") ?? "")) {
    const cfg = n("cfg scale");
    if (cfg !== undefined) out.guidance = cfg;
  }
  return out;
}

const INFOTEXT_RESTORED = new Set(
  [
    "model",
    "model classes",
    "prompt",
    "negative prompt",
    "confidence",
    "dilate erode",
    "mask blur",
    "denoising strength",
    "use separate steps",
    "steps",
    "use separate cfg scale",
    "cfg scale",
    "version",
  ].map((key) => `adetailer ${key}`),
);

/** `ADetailer ...` keys an import drops (later tabs, unsupported options). */
export function detailerInfotextNotApplied(fields: Record<string, string>): string[] {
  const restored = detailerFromInfotext(fields) !== null;
  return Object.keys(fields).filter((key) => {
    if (key === "adetailer version" || !fields[key]) return false;
    if (key === "adetailer model") return !restored && fields[key] !== "None";
    if (key === "adetailer inpaint only masked") return !(restored && /^true$/i.test(fields[key] ?? ""));
    return !(restored && INFOTEXT_RESTORED.has(key));
  });
}

export function detailerNotApplied(recorded: boolean, ready: boolean): string[] {
  return recorded && !ready ? ["Detailer (face fix)"] : [];
}

export function detailerSummary(record: DetailerRecord | null | undefined): string | null {
  if (!record) return null;
  if (record.error) return `not applied (${record.error})`;
  const count = record.detections?.length ?? 0;
  const targets = record.targets ?? "face";
  return `${targets}: ${count} fixed, denoise ${record.denoise ?? DETAILER_DEFAULTS.denoise}`;
}

export function detailerErrors(images: Array<{ detailer?: unknown }>): string[] {
  const errors = new Set<string>();
  for (const image of images) {
    const error = detailerRecordOf(image.detailer)?.error;
    if (error && error !== "cancelled") errors.add(error);
  }
  return [...errors];
}

const KEY = "unsloth:images:detailer-options";

export function readDetailerOptions(): DetailerOptions {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return { ...DETAILER_DEFAULTS };
    const out = { ...DETAILER_DEFAULTS };
    for (const key of Object.keys(DETAILER_DEFAULTS) as Array<keyof DetailerOptions>) {
      const v = (value as Record<string, unknown>)[key];
      if (key === "steps" || key === "guidance")
        out[key] = typeof v === "number" && Number.isFinite(v) ? v : null;
      else if (key === "guideSizeFor") out.guideSizeFor = isGuideSizeFor(v) ? v : DETAILER_DEFAULTS.guideSizeFor;
      else if (typeof v === typeof DETAILER_DEFAULTS[key] && (typeof v !== "number" || Number.isFinite(v)))
        (out as Record<string, unknown>)[key] = v;
    }
    return out;
  } catch {
    return { ...DETAILER_DEFAULTS };
  }
}

export function saveDetailerOptions(options: DetailerOptions): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(options));
  } catch {
    // storage can be unavailable in private browsing
  }
}
