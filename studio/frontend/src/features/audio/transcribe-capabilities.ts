// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so node tests can load it.

export interface SttCapabilities {
  engine: string | null;
  family: string | null;
  /** on_request: only when asked (Qwen3-ASR with its aligner); always: every run has them. */
  timestamps: "on_request" | "always" | "unsupported";
  speakers: boolean;
  aligner: { downloaded: boolean; size_bytes: number } | null;
  cpu_only: boolean;
}

export interface TranscribeSwitch {
  checked: boolean;
  disabled: boolean;
  hint: string;
  suggestSpeakersModel?: boolean;
  always?: boolean;
}

export interface TranscribeSwitches {
  timestamps: TranscribeSwitch;
  speakers: TranscribeSwitch;
  /** Said before a run that costs more than usual (an aligner download, a CPU-only model). */
  notice: string | null;
  request: { timestamps: boolean; speakers: boolean };
}

export const SPEAKERS_MODEL_NAME = "MOSS-Transcribe-Diarize";

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

export function transcribeSwitches(
  caps: SttCapabilities | null,
  prefs: { timestamps: boolean; speakers: boolean },
  state: { loading: boolean; hasModel: boolean },
): TranscribeSwitches {
  if (!caps) {
    const hint = state.loading
      ? "Checking what this model supports…"
      : state.hasModel
        ? "Could not check what this model supports. Transcribing still works."
        : "Pick a speech-to-text model to see what it can add.";
    const off = { checked: false, disabled: true, hint };
    return {
      timestamps: off,
      speakers: off,
      notice: null,
      request: { timestamps: false, speakers: false },
    };
  }

  let timestamps: TranscribeSwitch;
  let notice: string | null = null;
  if (caps.timestamps === "always") {
    timestamps = {
      checked: true,
      disabled: true,
      always: true,
      hint: "This model always adds timestamps.",
    };
  } else if (caps.timestamps === "on_request") {
    const size =
      caps.aligner && !caps.aligner.downloaded
        ? formatSize(caps.aligner.size_bytes)
        : null;
    timestamps = {
      checked: prefs.timestamps,
      disabled: false,
      hint: `Adds the time of each line, using a separate timing aligner.${size ? ` The first run downloads it (${size}).` : ""}`,
    };
    if (prefs.timestamps) {
      notice = size
        ? `Downloads the timing aligner (${size}) and reloads the model.`
        : "Reloads the model with its timing aligner if it is not loaded yet.";
    }
  } else {
    timestamps = {
      checked: false,
      disabled: true,
      hint: `This model returns plain text. Timestamps need Qwen3-ASR (audio.cpp), Parakeet-TDT, ${SPEAKERS_MODEL_NAME} or VibeVoice-ASR.`,
      suggestSpeakersModel: true,
    };
  }

  const speakers: TranscribeSwitch = caps.speakers
    ? {
        checked: prefs.speakers,
        disabled: false,
        hint: "Labels who is speaking. You can rename speakers afterwards.",
      }
    : {
        checked: false,
        disabled: true,
        hint: `Only ${SPEAKERS_MODEL_NAME} and VibeVoice-ASR can tell speakers apart.`,
        suggestSpeakersModel: true,
      };

  if (caps.cpu_only) {
    notice = notice
      ? `${notice} Runs on the CPU.`
      : "This model runs on the CPU.";
  }

  return {
    timestamps,
    speakers,
    notice,
    request: {
      timestamps: caps.timestamps === "on_request" && prefs.timestamps,
      speakers: speakers.checked,
    },
  };
}
