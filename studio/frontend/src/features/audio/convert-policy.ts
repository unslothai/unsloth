// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What Convert needs before it can run, which controls a model gets, and when a run reloads the
// model. Free of app imports so the node test runner can load it directly.

import type { AudioConvertCaps } from "@/features/chat/types/api";
import type {
  AudioSourceSelection,
  ConvertMode,
  ConvertStyle,
} from "./audio-run-request";
import type { AudioModelContext } from "./tools/types";

/** Convert's picker order: the general-purpose converters first, then the clone models that also convert. */
export const CONVERT_MODEL_ORDER: string[] = [
  "SeedVC-MLX-GGUF",
  "RVC-GGUF",
  "MeanVC2-GGUF",
  "Chatterbox-GGUF",
  "Vevo2-GGUF",
];

/** The loaded model's Convert caps; null when it does not convert. */
export function convertCaps(
  ctx: Pick<AudioModelContext, "convert">,
): AudioConvertCaps | null {
  return ctx.convert && ctx.convert.modes.length > 0 ? ctx.convert : null;
}

/** Singing always keeps the recording's style: a sung transcript is lyrics, which STT gets wrong. */
export function effectiveConvertStyle(
  caps: AudioConvertCaps | null,
  mode: ConvertMode,
  style: ConvertStyle,
): ConvertStyle {
  return caps?.style && mode === "speech" ? style : "source";
}

/** Whether the Pitch control shows, and whether it offers Auto. RVC shifts by hand only, Seed-VC
 *  only when singing, Vevo2 only while it keeps the recording's style. */
export function convertPitchSupport(
  caps: AudioConvertCaps | null,
  mode: ConvertMode,
  style: ConvertStyle,
): { show: boolean; auto: boolean } {
  const pitch = caps?.modes.includes(mode) ? caps.pitch[mode] : undefined;
  if (!pitch || effectiveConvertStyle(caps, mode, style) === "target") {
    return { show: false, auto: false };
  }
  return { show: true, auto: pitch.auto === true };
}

/** The server task a Convert run in this mode loads, from status `audio_workflow_tasks`
 *  ({"convert": "vc", "convert:singing": "svc"}); null when the model has no such mode. */
export function convertServerTask(
  caps: AudioConvertCaps | null,
  workflowTasks: Readonly<Record<string, string>> | null | undefined,
  mode: ConvertMode,
): string | null {
  if (!caps?.modes.includes(mode) || !workflowTasks) return null;
  return (
    workflowTasks[`convert:${mode}`] ??
    (mode === "speech" ? workflowTasks.convert : undefined) ??
    null
  );
}

export type ConvertBlockerKind =
  | "source"
  | "source-error"
  | "source-expired"
  | "target"
  | "target-error"
  | "target-expired"
  | "mode"
  | "source-text"
  | "panel";

export interface ConvertBlockerInput {
  source: AudioSourceSelection | null;
  /** The source card is uploading or recording. */
  sourceBusy: boolean;
  sourceExpired: boolean;
  sourceError: string | null;
  target: AudioSourceSelection | null;
  targetBusy: boolean;
  targetExpired: boolean;
  targetError: string | null;
  builtinVoice: string;
  caps: AudioConvertCaps | null;
  mode: ConvertMode;
  style: ConvertStyle;
  sourceText: string;
  panelError: string | null;
}

export const CONVERT_SOURCE_MISSING = "Add the recording to convert.";
export const CONVERT_TARGET_MISSING = "Add the target voice.";
export const CONVERT_BUILTIN_MISSING = "Pick a built-in voice.";
export const CONVERT_SINGING_UNSUPPORTED =
  "This model does not convert singing. Switch to Speech.";
export const CONVERT_SOURCE_TEXT_MISSING = "Add what's said in the recording.";
export const CONVERT_INPUTS_MISSING =
  "Add the recording to convert and the target voice.";

type ConvertBlocker = { kind: ConvertBlockerKind; reason: string };

/** An input card's own state: expired, still uploading, failed, or empty. */
function inputBlocker(
  selection: AudioSourceSelection | null,
  busy: boolean,
  expired: boolean,
  error: string | null,
  side: "source" | "target",
): ConvertBlocker | null {
  const what = side === "source" ? "recording" : "target voice";
  if (expired) {
    return {
      kind: `${side}-expired`,
      reason: `The ${what} is no longer on the server.`,
    };
  }
  if (busy) {
    return {
      kind: side,
      reason: `Waiting for the ${what} to finish uploading.`,
    };
  }
  if (error && !selection) return { kind: `${side}-error`, reason: error };
  if (!selection) {
    return {
      kind: side,
      reason:
        side === "source" ? CONVERT_SOURCE_MISSING : CONVERT_TARGET_MISSING,
    };
  }
  return null;
}

/** A built-in voice target (RVC) needs one of the model's voices; any target recording is ignored. */
function builtinBlocker(
  caps: AudioConvertCaps,
  builtinVoice: string,
): ConvertBlocker | null {
  const voice = builtinVoice.trim();
  const known =
    caps.builtin_voices.length === 0 ||
    caps.builtin_voices.some((item) => item.id === voice);
  return voice && known
    ? null
    : { kind: "target", reason: CONVERT_BUILTIN_MISSING };
}

/** The first page input Convert is missing, in rail order (source, target, mode, transcript,
 *  panel); null when it can run. Model blockers (none loaded, cannot convert) come from the host. */
export function convertBlocker(
  input: ConvertBlockerInput,
): ConvertBlocker | null {
  const source = inputBlocker(
    input.source,
    input.sourceBusy,
    input.sourceExpired,
    input.sourceError,
    "source",
  );
  const target =
    input.caps?.target === "builtin"
      ? builtinBlocker(input.caps, input.builtinVoice)
      : inputBlocker(
          input.target,
          input.targetBusy,
          input.targetExpired,
          input.targetError,
          "target",
        );
  // Name everything still missing at once, so nothing new appears after the first fix.
  if (
    source?.kind === "source" &&
    !input.source &&
    !input.sourceBusy &&
    target?.kind === "target" &&
    !input.target &&
    input.caps?.target !== "builtin"
  ) {
    return { kind: "source", reason: CONVERT_INPUTS_MISSING };
  }
  if (source?.kind === "source-expired" && target?.kind === "target-expired") {
    return {
      kind: "source-expired",
      reason: "The recording and target voice are no longer on the server.",
    };
  }
  const blocker = source ?? target;
  if (blocker) return blocker;
  if (input.caps && !input.caps.modes.includes(input.mode)) {
    return { kind: "mode", reason: CONVERT_SINGING_UNSUPPORTED };
  }
  if (
    effectiveConvertStyle(input.caps, input.mode, input.style) === "target" &&
    !input.sourceText.trim()
  ) {
    return { kind: "source-text", reason: CONVERT_SOURCE_TEXT_MISSING };
  }
  return input.panelError ? { kind: "panel", reason: input.panelError } : null;
}

/** Measured reload times into a Convert task (VC spike, GPU). Families missing here get no estimate. */
const CONVERT_RELOAD_SECONDS: Readonly<Record<string, string>> = {
  seed_vc: "about 4–8 s",
  vevo2: "about 6 s",
  chatterbox: "about 2 s",
  meanvc2: "about 2 s",
};

export interface ConvertSwitchInput {
  modelName: string;
  /** The task the server runs the model under now (status `audio_server_task`, or the last run's). */
  loadedTask: string | null;
  /** The task this run needs (convertServerTask). */
  nextTask: string | null;
  /** Seed-VC: the engine differs from the one the model last ran, which also reloads it. */
  routeChange: boolean;
  /** Status `audio_family`, for the measured reload time. */
  family?: string | null;
}

/** The inline notice before and during a run that reloads the model; null when it will not. */
export function convertSwitchNotice(
  input: ConvertSwitchInput,
): { before: string; during: string } | null {
  const taskChange = Boolean(
    input.loadedTask && input.nextTask && input.loadedTask !== input.nextTask,
  );
  if (!taskChange && !input.routeChange) return null;
  const name = input.modelName.trim() || "the model";
  const what = !taskChange
    ? "the new engine"
    : input.nextTask === "svc"
      ? "singing"
      : "Convert";
  const seconds = input.family
    ? CONVERT_RELOAD_SECONDS[input.family]
    : undefined;
  return {
    before: `Reloads ${name} ${taskChange ? "for" : "with"} ${what}${seconds ? `, ${seconds}` : ""}.`,
    during: `Switching ${name} to ${what}…`,
  };
}
