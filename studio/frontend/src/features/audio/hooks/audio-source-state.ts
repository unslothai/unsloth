// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What an audio input card is doing, as a pure reducer. Free of app imports so the node test
// runner can load it directly.

import { AUDIO_INPUT_MAX_BYTES } from "../audio-run-request";

export type AudioSourceStatus =
  | { phase: "idle" }
  | { phase: "recording"; startedAt: number }
  /** Progress 0..1, or null while the browser has not reported any. */
  | { phase: "uploading"; name: string; progress: number | null }
  /** Fetching an already-picked source's audio to draw it. */
  | { phase: "loading" }
  | { phase: "ready" }
  | { phase: "error"; message: string }
  | { phase: "expired" };

export interface AudioSourcePreview {
  /** Which selection id this preview draws, or "local" for a file still uploading. */
  key: string | null;
  /** "local" once the upload it was drawing for has an id, so a decode that finishes after the
   *  upload still lands. */
  localKey?: string | null;
  peaks: number[] | null;
  durationS: number | null;
  /** An object URL the card plays from. */
  url: string | null;
}

export interface AudioSourceState {
  status: AudioSourceStatus;
  preview: AudioSourcePreview;
}

export type AudioSourceAction =
  | { type: "record-start"; now: number }
  | { type: "record-stop" }
  | { type: "upload-start"; name: string }
  | { type: "upload-progress"; progress: number | null }
  | { type: "upload-done"; key: string }
  | { type: "load-start"; key: string }
  | { type: "load-done" }
  | {
      type: "preview";
      key: string;
      peaks: number[] | null;
      durationS: number | null;
      url: string | null;
    }
  | { type: "fail"; message: string }
  | { type: "expire" }
  | { type: "reset" };

export const EMPTY_PREVIEW: AudioSourcePreview = {
  key: null,
  peaks: null,
  durationS: null,
  url: null,
};

export const INITIAL_AUDIO_SOURCE_STATE: AudioSourceState = {
  status: { phase: "idle" },
  preview: EMPTY_PREVIEW,
};

export const REFERENCE_EXPIRED_MESSAGE =
  "This reference expired. Add it again.";

export function audioSourceReducer(
  state: AudioSourceState,
  action: AudioSourceAction,
): AudioSourceState {
  switch (action.type) {
    case "record-start":
      return {
        ...state,
        status: { phase: "recording", startedAt: action.now },
      };
    case "record-stop":
      return state.status.phase === "recording"
        ? { ...state, status: { phase: "idle" } }
        : state;
    case "upload-start":
      // A new file replaces whatever the card drew before, and draws as soon as it decodes.
      return {
        status: { phase: "uploading", name: action.name, progress: 0 },
        preview: { ...EMPTY_PREVIEW, key: "local" },
      };
    case "upload-progress":
      if (state.status.phase !== "uploading") return state;
      return {
        ...state,
        status: {
          ...state.status,
          progress:
            action.progress === null
              ? null
              : Math.min(1, Math.max(0, action.progress)),
        },
      };
    case "upload-done":
      return {
        status: { phase: "ready" },
        preview: {
          ...state.preview,
          key: action.key,
          localKey: state.preview.key === "local" ? "local" : null,
        },
      };
    case "load-start":
      return {
        status: { phase: "loading" },
        preview: { ...EMPTY_PREVIEW, key: action.key },
      };
    case "load-done":
      return state.status.phase === "loading"
        ? { ...state, status: { phase: "ready" } }
        : state;
    case "preview":
      // A decode that finishes after the card moved on draws nothing; one for the upload that just
      // got its id still draws it.
      if (
        state.preview.key !== action.key &&
        !(action.key === "local" && state.preview.localKey === "local")
      )
        return state;
      return {
        ...state,
        preview: {
          ...state.preview,
          peaks: action.peaks,
          durationS: action.durationS ?? state.preview.durationS,
          url: action.url ?? state.preview.url,
        },
      };
    case "fail":
      return { ...state, status: { phase: "error", message: action.message } };
    case "expire":
      return { status: { phase: "expired" }, preview: EMPTY_PREVIEW };
    case "reset":
      return INITIAL_AUDIO_SOURCE_STATE;
    default:
      return state;
  }
}

/** Why a picked file cannot be used, before anything is uploaded; null when it can. */
export function audioFileProblem(file: {
  size: number;
  type?: string;
  name?: string;
}): string | null {
  if (file.size <= 0) return "This file is empty.";
  if (file.size > AUDIO_INPUT_MAX_BYTES)
    return "Audio is too large. Keep it under 200 MB.";
  const type = file.type ?? "";
  const name = (file.name ?? "").toLowerCase();
  if (
    type &&
    !type.startsWith("audio/") &&
    !type.startsWith("video/") &&
    type !== "application/octet-stream" &&
    !/\.(wav|mp3|flac|ogg|oga|opus|m4a|aac|webm|mp4)$/.test(name)
  ) {
    return "This is not an audio file. Pick a WAV, MP3, FLAC, OGG or M4A file.";
  }
  return null;
}
