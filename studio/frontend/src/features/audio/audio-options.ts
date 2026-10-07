// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model options a GGUF audio model declares (`audio_options`); the backend re-validates.
// Free of app imports so the node test runner can load it.

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

export function audioOptionDisplayValue(
  spec: AudioOptionSpec,
  values: AudioOptionValues,
): AudioOptionValue | undefined {
  const stored = coerceAudioOptionValue(spec, values[spec.name]);
  if (stored !== undefined) return stored;
  return spec.default ?? undefined;
}

/** Only user-set values the schema still accepts; the rest use server defaults. */
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

// Kokoro voice ids lead with a language letter and a gender letter (hexgrad/Kokoro-82M VOICES.md).
const KOKORO_LANGUAGES: Record<string, string> = {
  a: "American English",
  b: "British English",
  e: "Spanish",
  f: "French",
  h: "Hindi",
  i: "Italian",
  j: "Japanese",
  p: "Brazilian Portuguese",
  z: "Mandarin Chinese",
};

function titleWords(text: string): string {
  return text
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

/** What a built-in voice is called in the picker; the request still sends the id. */
export function audioVoiceLabel(voice: string, family?: string | null): string {
  if (family === "kokoro_tts") {
    const match = /^([a-z])([fm])_([a-z0-9_]+)$/.exec(voice);
    const language = match ? KOKORO_LANGUAGES[match[1]] : undefined;
    if (match && language) {
      return `${titleWords(match[3])} (${language}, ${match[2] === "f" ? "female" : "male"})`;
    }
  }
  return /^[a-z0-9]+(?:[_-][a-z0-9]+)*$/.test(voice) ? titleWords(voice) : voice;
}

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

/** Unvalidated saved values; the schema can change, so read via the helpers above. */
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
