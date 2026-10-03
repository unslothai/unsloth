// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The request shapes of /audio/inputs, /audio/run and /audio/voices, and the audio a page picked.
// Free of app imports so the node test runner can load it directly. Clients only ever send ids:
// the server resolves them to its own files, so nothing here may carry a path or the audio itself.

/** Mirrors AUDIO_INPUT_MAX_BYTES in studio/backend/utils/upload_limits.py. */
export const AUDIO_INPUT_MAX_BYTES = 200 * 1024 * 1024;

/** The server clamps references to this many seconds before cloning. */
export const REFERENCE_MAX_SECONDS = 30;

/** One piece of audio the server already holds, by exactly one id. */
export type AudioSourceRef =
  | { input_id: string }
  | { clip_id: string }
  | { voice_id: string };

export interface AudioTrim {
  start_s: number;
  end_s: number;
}

export type AudioSourceKind = "input" | "clip" | "voice";

/** The audio a page picked, as the page keeps it across reloads. */
export interface AudioSourceSelection {
  kind: AudioSourceKind;
  id: string;
  /** The file name, the clip's text, or the voice's name. */
  name: string;
  durationS: number | null;
  /** Uploads expire on the server; history clips and saved voices do not. */
  expiresAt?: string | null;
  /** What the clip says when it is already known: a history clip's text, a voice's transcript. */
  transcript?: string | null;
  language?: string | null;
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

/** Where the server serves a source's audio. */
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

/** The transcribe route takes an upload by path, and a history clip or saved voice by query. */
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

/** Whether an upload is past the server's keep-until time. */
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

/** Music's own request fields (workflow "music"); see PLAN §2a for the server contract. */
export interface AudioMusicRunFields {
  mode: "song" | "sfx" | "edit";
  lyrics?: string | null;
  instrumental?: boolean;
  duration_s?: number | null;
  variations?: number | null;
  edit?: {
    action: "repaint" | "extend" | "cover" | "continue" | "inpaint" | "restyle";
    ranges?: AudioTrim[];
    strength?: number | null;
    extend_s?: number | null;
  } | null;
  /** The clip being edited: an upload or a history clip, never a saved voice. */
  source?: AudioSourceRef | null;
}

export interface AudioRunRequest {
  workflow: "clone" | "speak" | "music";
  /** Only with workflow "music". */
  music?: AudioMusicRunFields;
  text: string;
  language?: string | null;
  instructions?: string | null;
  inputs?: {
    reference?: (AudioSourceRef & { trim?: AudioTrim }) | null;
    reference_text?: string | null;
    emotion?: AudioSourceRef | null;
  };
  options?: Record<string, AudioOptionScalar>;
  speed?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
}

/** A source ref with only its one id and an optional trim, whatever else the caller's object held. */
function cleanRef(
  ref: (AudioSourceRef & { trim?: AudioTrim }) | null | undefined,
  allowTrim = true,
): Record<string, unknown> | null {
  if (!ref) return null;
  const out: Record<string, unknown> = {};
  for (const key of ["input_id", "clip_id", "voice_id"] as const) {
    const value = (ref as Record<string, unknown>)[key];
    if (typeof value === "string" && value) out[key] = value;
  }
  if (Object.keys(out).length !== 1) return null;
  const trim = (ref as { trim?: AudioTrim }).trim;
  if (
    allowTrim &&
    trim &&
    Number.isFinite(trim.start_s) &&
    Number.isFinite(trim.end_s) &&
    trim.end_s > trim.start_s
  ) {
    out.trim = { start_s: trim.start_s, end_s: trim.end_s };
  }
  return out;
}

/** The JSON body of POST /audio/run: the allowed keys only, empty values left out. The route
 *  forbids extra keys, so anything else a caller spread in would be a 422 anyway. */
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
  // Trim belongs to the reference only.
  const emotion = cleanRef(request.inputs?.emotion, false);
  if (emotion) inputs.emotion = emotion;
  if (Object.keys(inputs).length > 0) body.inputs = inputs;
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
  if (request.workflow === "music" && request.music) {
    Object.assign(body, musicRunFields(request.music));
  }
  return body;
}

function finiteSeconds(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

/** Music's keys for the run body: song/sfx length and variations, edit's source and ranges. */
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
  // Edit: the source by id only (no trim, no saved voice) and the action's own values.
  const source = cleanRef(music.source, false);
  if (source && !("voice_id" in source)) {
    // Music never sends a reference, so the source is the only input.
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
  source: AudioSourceRef & { trim?: AudioTrim };
  name: string;
  transcript?: string | null;
  language?: string | null;
}

/** The body of POST /audio/voices, trimmed to the contract's limits. */
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
