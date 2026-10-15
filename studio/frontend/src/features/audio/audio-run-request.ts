// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app imports: the node test runner loads this directly. Clients send ids only, never paths or audio.

/** Mirrors AUDIO_INPUT_MAX_BYTES in studio/backend/utils/upload_limits.py. */
export const AUDIO_INPUT_MAX_BYTES = 200 * 1024 * 1024;

export const REFERENCE_MAX_SECONDS = 30;

/** Longer is refused, not cut: a cut recording no longer matches its transcript. */
export const EDIT_SOURCE_MAX_SECONDS = 30;
// The server allows the same slack: a take stopped by the 30 s timer can run a frame over.
export const EDIT_SOURCE_SLACK_SECONDS = 0.05;

export type AudioSourceRef =
  | { input_id: string }
  | { clip_id: string }
  | { voice_id: string };

export interface AudioSourceSelection {
  kind: "input" | "clip" | "voice";
  id: string;
  name: string;
  durationS: number | null;
  expiresAt?: string | null;
  transcript?: string | null;
  language?: string | null;
}

export function clipReference(clip: {
  id: string;
  prompt: string;
  duration_s: number | null;
  workflow?: string | null;
  reference_name?: string | null;
}): AudioSourceSelection {
  // Only speech is a transcript, not a Music description, Convert label, or file name.
  const spoken =
    clip.workflow !== "music" &&
    clip.workflow !== "convert" &&
    clip.prompt !== clip.reference_name;
  return {
    kind: "clip",
    id: clip.id,
    name: clip.prompt || "Generated clip",
    durationS: clip.duration_s,
    transcript: (spoken && clip.prompt) || null,
    language: null,
  };
}

export function sourceRefOf(selection: AudioSourceSelection): AudioSourceRef {
  switch (selection.kind) {
    case "input":
      return { input_id: selection.id };
    case "clip":
      return { clip_id: selection.id };
    default:
      return { voice_id: selection.id };
  }
}

export function sourceFileUrl(selection: AudioSourceSelection): string {
  const id = encodeURIComponent(selection.id);
  switch (selection.kind) {
    case "input":
      return `/api/inference/audio/inputs/${id}/file`;
    case "clip":
      return `/api/inference/audio/gallery/${id}/file`;
    default:
      return `/api/inference/audio/voices/${id}/file`;
  }
}

export function transcribeUrl(ref: AudioSourceRef): string {
  if ("input_id" in ref) {
    return `/api/inference/audio/inputs/${encodeURIComponent(ref.input_id)}/transcribe`;
  }
  const query =
    "clip_id" in ref
      ? `clip_id=${encodeURIComponent(ref.clip_id)}`
      : `voice_id=${encodeURIComponent(ref.voice_id)}`;
  return `/api/inference/audio/inputs/source/transcribe?${query}`;
}

export function selectionExpired(
  selection: AudioSourceSelection | null,
  nowMs: number,
): boolean {
  if (!selection || selection.kind !== "input" || !selection.expiresAt)
    return false;
  const expires = Date.parse(selection.expiresAt);
  return Number.isFinite(expires) && expires <= nowMs;
}

export type AudioOptionScalar = boolean | number | string;

export interface AudioRunEditPart {
  mode: "words" | "delivery";
  markup?: string | null;
  instructions?: string[] | null;
  speed?: number | null;
  pitch_steps?: number | null;
}

export interface AudioMusicRunFields {
  mode: "song" | "sfx" | "edit";
  lyrics?: string | null;
  instrumental?: boolean;
  duration_s?: number | null;
  variations?: number | null;
  edit?: {
    action: "repaint" | "extend" | "cover" | "continue" | "inpaint" | "restyle";
    ranges?: { start_s: number; end_s: number }[];
    strength?: number | null;
    extend_s?: number | null;
  } | null;
  source?: AudioSourceRef | null;
}

export type ConvertMode = "speech" | "singing";

export type ConvertStyle = "source" | "target";

export interface AudioConvertParams {
  mode: ConvertMode;
  pitch: number | null;
  pitch_auto: boolean;
  style?: ConvertStyle;
  voice?: string | null;
}

export interface AudioConvertRunRequest {
  workflow: "convert";
  inputs: {
    source: AudioSourceRef;
    target?: AudioSourceRef | null;
    source_text?: string | null;
  };
  convert: AudioConvertParams;
  options?: Record<string, AudioOptionScalar>;
  seed?: number | null;
}

export type AudioRunRequest = AudioTextRunRequest | AudioConvertRunRequest;

export interface AudioTextRunRequest {
  workflow: "clone" | "speak" | "edit" | "music" | "separate";
  music?: AudioMusicRunFields;
  text?: string;
  language?: string | null;
  instructions?: string | null;
  inputs?: {
    reference?: AudioSourceRef | null;
    source?: AudioSourceRef | null;
    reference_text?: string | null;
    emotion?: AudioSourceRef | null;
  };
  edit?: AudioRunEditPart | null;
  options?: Record<string, AudioOptionScalar>;
  speed?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
}

function cleanRef(
  ref: AudioSourceRef | null | undefined,
): Record<string, unknown> | null {
  if (!ref) return null;
  const out: Record<string, unknown> = {};
  for (const key of ["input_id", "clip_id", "voice_id"] as const) {
    const value = (ref as Record<string, unknown>)[key];
    if (typeof value === "string" && value) out[key] = value;
  }
  return Object.keys(out).length === 1 ? out : null;
}

function cleanOptions(
  options: Record<string, AudioOptionScalar> | undefined,
): Record<string, AudioOptionScalar> {
  return Object.fromEntries(
    Object.entries(options ?? {}).filter(
      ([name, value]) =>
        name &&
        (typeof value === "boolean" ||
          typeof value === "string" ||
          (typeof value === "number" && Number.isFinite(value))),
    ),
  );
}

const CONVERT_MODES: ReadonlySet<string> = new Set(["speech", "singing"]);

/** Ids only, never a path: the route resolves them in the caller's account. */
function buildConvertRunBody(
  request: AudioConvertRunRequest,
): Record<string, unknown> {
  const inputs: Record<string, unknown> = {};
  const source = cleanRef(request.inputs?.source);
  if (source) inputs.source = source;
  const target = cleanRef(request.inputs?.target);
  if (target) inputs.target = target;
  const sourceText = request.inputs?.source_text?.trim();
  if (sourceText) inputs.source_text = sourceText;
  const params = request.convert;
  const convert: Record<string, unknown> = {
    mode: CONVERT_MODES.has(params?.mode) ? params.mode : "speech",
    pitch_auto: params?.pitch_auto === true,
    pitch:
      typeof params?.pitch === "number" && Number.isFinite(params.pitch)
        ? Math.min(24, Math.max(-24, Math.round(params.pitch)))
        : null,
  };
  if (params?.style === "source" || params?.style === "target") {
    convert.style = params.style;
  }
  const voice = params?.voice?.trim();
  if (voice) convert.voice = voice;
  const body: Record<string, unknown> = {
    workflow: "convert",
    inputs,
    convert,
  };
  const options = cleanOptions(request.options);
  if (Object.keys(options).length > 0) body.options = options;
  if (typeof request.seed === "number" && Number.isInteger(request.seed))
    body.seed = request.seed;
  return body;
}

/** Allowed keys only, empty values dropped: the route 422s on extra keys. */
export function buildAudioRunBody(
  request: AudioRunRequest,
): Record<string, unknown> {
  if (request.workflow === "convert") return buildConvertRunBody(request);
  const body: Record<string, unknown> = { workflow: request.workflow };
  if (request.workflow !== "separate") body.text = request.text ?? "";
  const language = request.language?.trim();
  if (language) body.language = language;
  const instructions = request.instructions?.trim();
  if (instructions) body.instructions = instructions;
  const inputs: Record<string, unknown> = {};
  const reference = cleanRef(request.inputs?.reference);
  if (reference) inputs.reference = reference;
  const source = cleanRef(request.inputs?.source);
  if (source) inputs.source = source;
  const rawReferenceText = request.inputs?.reference_text;
  // A cleared field is sent blank, so the server does not refill it from a saved voice.
  if (typeof rawReferenceText === "string") {
    inputs.reference_text = rawReferenceText.trim();
  }
  const emotion = cleanRef(request.inputs?.emotion);
  if (emotion) inputs.emotion = emotion;
  if (Object.keys(inputs).length > 0) body.inputs = inputs;
  const edit = cleanEdit(request.edit);
  if (edit) body.edit = edit;
  const options = cleanOptions(request.options);
  if (Object.keys(options).length > 0) body.options = options;
  if (typeof request.speed === "number" && Number.isFinite(request.speed))
    body.speed = request.speed;
  if (typeof request.seed === "number" && Number.isInteger(request.seed))
    body.seed = request.seed;
  if (typeof request.max_tokens === "number" && request.max_tokens > 0)
    body.max_tokens = request.max_tokens;
  if (request.workflow === "music" && request.music) {
    Object.assign(body, musicRunFields(request.music));
  }
  return body;
}

function cleanEdit(
  edit: AudioRunEditPart | null | undefined,
): Record<string, unknown> | null {
  if (!edit || (edit.mode !== "words" && edit.mode !== "delivery")) return null;
  const out: Record<string, unknown> = { mode: edit.mode };
  if (typeof edit.markup === "string" && edit.markup.trim())
    out.markup = edit.markup;
  if (Array.isArray(edit.instructions)) {
    const instructions = edit.instructions.filter(
      (line): line is string => typeof line === "string" && line.length > 0,
    );
    if (instructions.length > 0) out.instructions = instructions;
  }
  if (typeof edit.speed === "number" && Number.isFinite(edit.speed))
    out.speed = edit.speed;
  if (
    typeof edit.pitch_steps === "number" &&
    Number.isInteger(edit.pitch_steps)
  )
    out.pitch_steps = edit.pitch_steps;
  return out;
}

function finiteSeconds(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function musicRunFields(music: AudioMusicRunFields): Record<string, unknown> {
  const out: Record<string, unknown> = { mode: music.mode };
  if (music.mode === "song") {
    const lyrics = music.lyrics?.trim();
    if (lyrics) out.lyrics = lyrics;
    if (music.instrumental) out.instrumental = true;
  }
  if (music.mode !== "edit" || music.edit?.action === "continue") {
    const duration = finiteSeconds(music.duration_s);
    if (duration !== null && duration > 0) out.duration_s = duration;
  }
  if (music.mode !== "edit") {
    const variations = music.variations;
    if (
      typeof variations === "number" &&
      Number.isInteger(variations) &&
      variations > 1
    )
      out.variations = variations;
    return out;
  }
  const source = cleanRef(music.source);
  if (source && !("voice_id" in source)) {
    out.inputs = { source };
  }
  if (music.edit) {
    const edit: Record<string, unknown> = { action: music.edit.action };
    const ranges = (music.edit.ranges ?? [])
      .filter(
        (range) =>
          finiteSeconds(range.start_s) !== null &&
          Number.isFinite(range.end_s) &&
          range.end_s > range.start_s,
      )
      .map((range) => ({ start_s: range.start_s, end_s: range.end_s }));
    if (ranges.length > 0) edit.ranges = ranges;
    const strength = music.edit.strength;
    if (typeof strength === "number" && Number.isFinite(strength))
      edit.strength = Math.min(1, Math.max(0, strength));
    const extend = finiteSeconds(music.edit.extend_s);
    if (music.edit.action === "extend" && extend !== null && extend > 0)
      edit.extend_s = extend;
    out.edit = edit;
  }
  return out;
}

export interface AudioVoiceCreateRequest {
  source: AudioSourceRef;
  name: string;
  transcript?: string | null;
  language?: string | null;
}

export function buildVoiceCreateBody(
  request: AudioVoiceCreateRequest,
): Record<string, unknown> {
  const source = cleanRef(request.source);
  if (!source || "voice_id" in source) {
    throw new Error("A saved voice is made from an upload or a history clip.");
  }
  const body: Record<string, unknown> = {
    source,
    name: request.name.trim().slice(0, 80),
  };
  const transcript = request.transcript?.trim();
  if (transcript) body.transcript = transcript.slice(0, 4000);
  const language = request.language?.trim();
  if (language) body.language = language.slice(0, 64);
  return body;
}
