// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which Transcribe switches a model allows, and the plain words for each. Import-free so node
// tests can load it.

/** GET /api/inference/audio/stt/capabilities. */
export interface SttCapabilities {
  engine: string | null;
  family: string | null;
  /** on_request: only when asked (Qwen3-ASR with its aligner); always: every run has them. */
  timestamps: "on_request" | "always" | "unsupported";
  speakers: boolean;
  /** Qwen3-ASR's timing aligner, a separate download. */
  aligner: { downloaded: boolean; size_bytes: number } | null;
  cpu_only: boolean;
}

export interface TranscribeSwitch {
  checked: boolean;
  disabled: boolean;
  hint: string;
  /** Offer the model that can do it, when this one cannot. */
  suggestSpeakersModel?: boolean;
  /** The model always does this; shown as a fact rather than a switch. */
  always?: boolean;
}

export interface TranscribeSwitches {
  timestamps: TranscribeSwitch;
  speakers: TranscribeSwitch;
  /** Said before the run when it costs more than usual (an aligner download, a CPU-only model). */
  notice: string | null;
  /** What the run sends. */
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
    const aligner = caps.aligner;
    const download =
      aligner && !aligner.downloaded
        ? ` The first run downloads it (${formatSize(aligner.size_bytes)}).`
        : "";
    timestamps = {
      checked: prefs.timestamps,
      disabled: false,
      hint: `Adds the time of each line, using a separate timing aligner.${download}`,
    };
    if (prefs.timestamps) {
      notice =
        aligner && !aligner.downloaded
          ? `Downloads the timing aligner (${formatSize(aligner.size_bytes)}) and reloads the model.`
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
