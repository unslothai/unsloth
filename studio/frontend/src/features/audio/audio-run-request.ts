// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app imports: the node test runner loads this directly. Clients send ids only, never paths or audio.

/** Mirrors AUDIO_INPUT_MAX_BYTES in studio/backend/utils/upload_limits.py. */
export const AUDIO_INPUT_MAX_BYTES = 200 * 1024 * 1024;

export const REFERENCE_MAX_SECONDS = 30;

/** Longer is refused, not cut: a cut recording no longer matches its transcript. */
export const EDIT_SOURCE_MAX_SECONDS = 30;

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

/** A gallery clip as a source: its text doubles as the transcript. */
export function clipReference(clip: {
  id: string;
  prompt: string;
  duration_s: number | null;
  workflow?: string | null;
  reference_name?: string | null;
}): AudioSourceSelection {
  // Only speech is a transcript: not a Music description, nor the file name an untranscribed edit is titled by.
  const spoken =
    clip.workflow !== "music" && clip.prompt !== clip.reference_name;
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

export interface AudioRunRequest {
  workflow: "clone" | "speak" | "edit";
  text: string;
  language?: string | null;
  instructions?: string | null;
  inputs?: {
    reference?: AudioSourceRef | null;
    reference_text?: string | null;
    emotion?: AudioSourceRef | null;
    source?: AudioSourceRef | null;
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

/** Allowed keys only, empty values dropped: the route 422s on extra keys. */
export function buildAudioRunBody(
  request: AudioRunRequest,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    workflow: request.workflow,
    text: request.text,
  };
  const language = request.language?.trim();
  if (language) body.language = language;
  const instructions = request.instructions?.trim();
  if (instructions) body.instructions = instructions;
  const inputs: Record<string, unknown> = {};
  const reference = cleanRef(request.inputs?.reference);
  if (reference) inputs.reference = reference;
  const referenceText = request.inputs?.reference_text?.trim();
  if (referenceText) inputs.reference_text = referenceText;
  const emotion = cleanRef(request.inputs?.emotion);
  if (emotion) inputs.emotion = emotion;
  const source = cleanRef(request.inputs?.source);
  if (source) inputs.source = source;
  if (Object.keys(inputs).length > 0) body.inputs = inputs;
  const edit = cleanEdit(request.edit);
  if (edit) body.edit = edit;
  const options = Object.fromEntries(
    Object.entries(request.options ?? {}).filter(
      ([name, value]) =>
        name &&
        (typeof value === "boolean" ||
          typeof value === "string" ||
          (typeof value === "number" && Number.isFinite(value))),
    ),
  );
  if (Object.keys(options).length > 0) body.options = options;
  if (typeof request.speed === "number" && Number.isFinite(request.speed))
    body.speed = request.speed;
  if (typeof request.seed === "number" && Number.isInteger(request.seed))
    body.seed = request.seed;
  if (typeof request.max_tokens === "number" && request.max_tokens > 0)
    body.max_tokens = request.max_tokens;
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
