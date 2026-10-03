// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import { AUDIO_INPUT_MAX_BYTES } from "../audio-run-request";

export type AudioSourceStatus =
  | { phase: "idle" }
  | { phase: "recording"; startedAt: number }
  | { phase: "uploading"; name: string; progress: number | null }
  | { phase: "loading" }
  | { phase: "ready" }
  | { phase: "error"; message: string }
  | { phase: "expired" };

interface AudioSourcePreview {
  /** Selection key drawn, or "local:<n>" for a file still uploading. */
  key: string | null;
  /** The local key of the upload that just got an id, so its late decode still lands. */
  localKey?: string | null;
  peaks: number[] | null;
  durationS: number | null;
  url: string | null;
}

interface AudioSourceState {
  status: AudioSourceStatus;
  preview: AudioSourcePreview;
}

type AudioSourceAction =
  | { type: "record-start"; now: number }
  | { type: "record-stop" }
  | { type: "upload-start"; name: string; key: string }
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

const EMPTY_PREVIEW: AudioSourcePreview = {
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
      return {
        status: { phase: "uploading", name: action.name, progress: 0 },
        preview: { ...EMPTY_PREVIEW, key: action.key },
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
          localKey: state.preview.key?.startsWith("local:")
            ? state.preview.key
            : null,
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
      if (
        state.preview.key !== action.key &&
        state.preview.localKey !== action.key
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
