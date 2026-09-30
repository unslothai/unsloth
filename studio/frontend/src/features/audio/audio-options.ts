// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model generation options a GGUF audio model declares in its embedded spec. The backend
// reports the schema as `audio_options` on the loaded-model status and takes the chosen values
// back as `audio_options` on a speech or music request, validating them again. Values the user
// set are kept per model id in this browser. Free of app imports so the node test runner can
// load it directly.

export type AudioOptionType = "bool" | "int" | "float" | "string" | "enum";
export type AudioOptionValue = boolean | number | string;
export type AudioOptionValues = Record<string, AudioOptionValue>;

export interface AudioOptionSpec {
  name: string;
  type: AudioOptionType;
  description?: string | null;
  default?: AudioOptionValue | null;
  min?: number | null;
  max?: number | null;
  values?: readonly string[] | null;
  required?: boolean | null;
}

const OPTION_TYPES: ReadonlySet<string> = new Set([
  "bool",
  "int",
  "float",
  "string",
  "enum",
]);

const finite = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/** The status field as a schema, dropping entries this build cannot render. */
export function parseAudioOptions(raw: unknown): AudioOptionSpec[] {
  if (!Array.isArray(raw)) return [];
  const specs: AudioOptionSpec[] = [];
  const seen = new Set<string>();
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const item = entry as Record<string, unknown>;
    const name = typeof item.name === "string" ? item.name.trim() : "";
    const type = typeof item.type === "string" ? item.type : "";
    if (!name || !OPTION_TYPES.has(type) || seen.has(name)) continue;
    const values = Array.isArray(item.values)
      ? item.values.filter((value): value is string => typeof value === "string")
      : null;
    // An enum without choices has nothing to offer.
    if (type === "enum" && !values?.length) continue;
    seen.add(name);
    const spec: AudioOptionSpec = {
      name,
      type: type as AudioOptionType,
      description: typeof item.description === "string" ? item.description : null,
      min: finite(item.min),
      max: finite(item.max),
      values,
      required: item.required === true,
    };
    spec.default = coerceAudioOptionValue(spec, item.default) ?? null;
    specs.push(spec);
  }
  return specs;
}

/** `value` as this option accepts it, or undefined when it cannot be: numbers are clamped to the
 *  declared range and ints rounded, an enum takes only its listed values. */
export function coerceAudioOptionValue(
  spec: AudioOptionSpec,
  value: unknown,
): AudioOptionValue | undefined {
  switch (spec.type) {
    case "bool":
      return typeof value === "boolean" ? value : undefined;
    case "int":
    case "float": {
      const number =
        typeof value === "string" && value.trim() !== "" ? Number(value) : value;
      if (typeof number !== "number" || !Number.isFinite(number)) return undefined;
      let clamped = number;
      if (spec.min != null) clamped = Math.max(spec.min, clamped);
      if (spec.max != null) clamped = Math.min(spec.max, clamped);
      return spec.type === "int" ? Math.round(clamped) : clamped;
    }
    case "enum":
      return typeof value === "string" && spec.values?.includes(value) ? value : undefined;
    case "string":
      return typeof value === "string" ? value : undefined;
  }
}

/** What a control shows: the stored value, else the declared default, else a neutral blank. */
export function audioOptionDisplayValue(
  spec: AudioOptionSpec,
  values: AudioOptionValues,
): AudioOptionValue | undefined {
  const stored = coerceAudioOptionValue(spec, values[spec.name]);
  if (stored !== undefined) return stored;
  return spec.default ?? undefined;
}

/** The values to send: only ones the user set and the schema still accepts. Everything else is
 *  left to the model's own defaults on the server. */
export function audioOptionsForRequest(
  specs: readonly AudioOptionSpec[],
  values: AudioOptionValues,
): AudioOptionValues {
  const request: AudioOptionValues = {};
  for (const spec of specs) {
    const value = coerceAudioOptionValue(spec, values[spec.name]);
    if (value === undefined) continue;
    if (spec.type === "string" && value === "") continue;
    request[spec.name] = value;
  }
  return request;
}

/** Required options with neither a value nor a default, which the model would refuse. */
export function missingRequiredAudioOptions(
  specs: readonly AudioOptionSpec[],
  values: AudioOptionValues,
): AudioOptionSpec[] {
  return specs.filter((spec) => {
    if (!spec.required) return false;
    const value = audioOptionDisplayValue(spec, values);
    return value === undefined || value === "";
  });
}

/** "num_inference_steps" -> "Num inference steps"; a namespaced "yue2.cot" -> "Cot". */
export function audioOptionLabel(name: string): string {
  const leaf = name.split(".").pop() ?? name;
  const words = leaf.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : name;
}

/** A slider step for a float range: about a hundred steps, on a round number. */
export function audioOptionFloatStep(min: number, max: number): number {
  const span = max - min;
  if (!(span > 0)) return 0.01;
  const raw = span / 100;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return nice * magnitude;
}

export const AUDIO_OPTIONS_STORAGE_KEY = "unsloth_audio_model_options";

type StoredAudioOptions = Record<string, AudioOptionValues>;

const storageKey = (modelId: string) => modelId.trim().replace(/\/+$/, "").toLowerCase();

function readAll(): StoredAudioOptions {
  try {
    const raw = globalThis.localStorage?.getItem(AUDIO_OPTIONS_STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as StoredAudioOptions)
      : {};
  } catch {
    return {};
  }
}

/** The values saved for one model, unvalidated: the schema can change between loads, so
 *  callers read them through the helpers above. */
export function readAudioOptionValues(modelId: string | null | undefined): AudioOptionValues {
  if (!modelId?.trim()) return {};
  const entry = readAll()[storageKey(modelId)];
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return {};
  const values: AudioOptionValues = {};
  for (const [name, value] of Object.entries(entry)) {
    if (["boolean", "number", "string"].includes(typeof value)) {
      values[name] = value as AudioOptionValue;
    }
  }
  return values;
}

/** Remember one model's values; an empty set forgets the model. */
export function saveAudioOptionValues(
  modelId: string | null | undefined,
  values: AudioOptionValues,
): void {
  if (!modelId?.trim()) return;
  try {
    const all = readAll();
    const key = storageKey(modelId);
    if (Object.keys(values).length === 0) delete all[key];
    else all[key] = values;
    globalThis.localStorage?.setItem(AUDIO_OPTIONS_STORAGE_KEY, JSON.stringify(all));
  } catch {
    // Storage is best effort: a full or blocked store keeps the values for this session only.
  }
}
