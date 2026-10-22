// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so node tests can load it.

export interface SttCapabilities {
  engine: string | null;
  family: string | null;
  /** on_request: only when asked (Qwen3-ASR with its aligner); always: every run has them. */
  timestamps: "on_request" | "always" | "unsupported";
  speakers: boolean;
  aligner: {
    downloaded: boolean;
    size_bytes: number;
    id?: string;
    variant?: string;
  } | null;
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
  notice: string | null;
  request: { timestamps: boolean; speakers: boolean };
}

export const SPEAKERS_MODEL_NAME = "MOSS-Transcribe-Diarize";

/** The aligner download a timestamped run needs first, staged like any other quant so it shows in
 *  the Downloads panel; null when the run can start now. */
export function alignerDownloadEntry(
  caps: SttCapabilities | null,
  timestamps: boolean,
): {
  repoId: string;
  files: string[];
  bytes: number;
  ggufVariant: string;
} | null {
  const aligner = caps?.aligner;
  if (!timestamps || !aligner || aligner.downloaded) return null;
  if (!aligner.id || !aligner.variant) return null;
  return {
    repoId: aligner.id,
    files: [],
    bytes: aligner.size_bytes,
    ggufVariant: aligner.variant,
  };
}

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

const TRANSCRIBE_MODEL_ID =
  /whisper|transcri|parakeet|voxtral-mini|(?:^|[\W_])(?:asr|stt)(?:[\W_]|$)/i;
// OpenAI's catalog in Studio keeps only chat models.
const OPENAI_TRANSCRIBE_MODELS = [
  "whisper-1",
  "gpt-4o-transcribe",
  "gpt-4o-mini-transcribe",
];

// Azure and other OpenAI-typed endpoints name models by deployment (as speak-connection-models.ts).
function isOpenAIApi(connection: { providerType: string; baseUrl?: string }): boolean {
  if (connection.providerType !== "openai") return false;
  if (!connection.baseUrl) return true;
  try {
    return new URL(connection.baseUrl).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return false;
  }
}

export function connectionTranscribeModels(
  connection: {
    providerType: string;
    baseUrl?: string;
    models: readonly string[];
    availableModels?: readonly string[];
    sharedByOwner?: boolean;
  },
  dictationModel?: string,
): string[] {
  const ids = [
    ...(isOpenAIApi(connection) ? OPENAI_TRANSCRIBE_MODELS : []),
    ...connection.models,
    ...(connection.availableModels ?? []),
  ].filter((id) => TRANSCRIBE_MODEL_ID.test(id) && !/tts/i.test(id));
  if (dictationModel) ids.push(dictationModel);
  // An owner-shared connection refuses any model outside the list the owner allowed (403).
  const allowed = connection.sharedByOwner ? new Set(connection.models) : null;
  return [...new Set(ids)].filter((id) => !allowed || allowed.has(id));
}

const OPENAI_UNTIMED_MODELS = new Set([
  "gpt-4o-transcribe",
  "gpt-4o-mini-transcribe",
]);

export function connectionTranscribeSwitches(
  prefs: { timestamps: boolean },
  model: string,
): TranscribeSwitches {
  const untimed = OPENAI_UNTIMED_MODELS.has(model);
  return {
    timestamps: untimed
      ? {
          checked: false,
          disabled: true,
          hint: "This model returns text only. Use whisper-1 for timestamps.",
        }
      : {
          checked: prefs.timestamps,
          disabled: false,
          hint: "Asks the connection for the time of each line. Not every model returns them.",
        },
    speakers: {
      checked: false,
      disabled: true,
      hint: "A connection returns no speaker labels.",
    },
    notice: null,
    request: { timestamps: !untimed && prefs.timestamps, speakers: false },
  };
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

  const onRequest = caps.timestamps === "on_request";
  const size =
    onRequest && caps.aligner && !caps.aligner.downloaded
      ? formatSize(caps.aligner.size_bytes)
      : null;
  const timestamps: TranscribeSwitch =
    caps.timestamps === "always"
      ? {
          checked: true,
          disabled: true,
          always: true,
          hint: "This model always adds timestamps.",
        }
      : onRequest
        ? {
            checked: prefs.timestamps,
            disabled: false,
            hint: `Adds the time of each line, using a separate timing aligner.${size ? ` The first run downloads it (${size}).` : ""}`,
          }
        : {
            checked: false,
            disabled: true,
            hint: `This model returns plain text. Timestamps need Qwen3-ASR (audio.cpp), Parakeet-TDT, ${SPEAKERS_MODEL_NAME} or VibeVoice-ASR.`,
            suggestSpeakersModel: true,
          };
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
  const request = {
    timestamps: onRequest && prefs.timestamps,
    speakers: speakers.checked,
  };
  const aligner = !request.timestamps
    ? null
    : size
      ? `Downloads the timing aligner (${size}) and reloads the model.`
      : "Reloads the model with its timing aligner if it is not loaded yet.";
  const notice = !caps.cpu_only
    ? aligner
    : aligner
      ? `${aligner} Runs on the CPU.`
      : "This model runs on the CPU.";
  return { timestamps, speakers, notice, request };
}
