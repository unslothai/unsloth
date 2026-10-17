// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GenerateAction } from "./pages/tts-workspace";
import { useAudioWorkspaceStore } from "./stores/audio-workspace-store.ts";

const RUNTIME_CODES = [
  "runtime_missing",
  "family_unknown_to_runtime",
  "espeak_missing",
];
const AUDIO_FAILURE_CODES = [
  ...RUNTIME_CODES,
  "out_of_memory",
  "gpu_unavailable",
  "file_incomplete",
];

export interface AudioFailure {
  code: string;
  message: string;
  smallerVariants: { variant: string; label: string }[];
}

/** A structured audio.cpp failure from a FastAPI `detail`, or a body / `_deferred_error` wrapping one. */
export function parseAudioFailure(
  detail: unknown,
  depth = 0,
): AudioFailure | null {
  if (!detail || typeof detail !== "object" || depth > 3) return null;
  const body = detail as Record<string, unknown>;
  if (typeof body.code !== "string" || !AUDIO_FAILURE_CODES.includes(body.code))
    return parseAudioFailure(body._deferred_error ?? body.detail, depth + 1);
  const variants = Array.isArray(body.smaller_variants)
    ? (body.smaller_variants as { variant?: unknown; label?: unknown }[])
    : [];
  return {
    code: body.code,
    message: typeof body.message === "string" ? body.message : "",
    smallerVariants: variants.flatMap(({ variant, label } = {}) =>
      typeof variant === "string" && variant
        ? [
            {
              variant,
              label: typeof label === "string" && label ? label : variant,
            },
          ]
        : [],
    ),
  };
}

/** The failure an API error carries (AudioApiError / LoadModelError keep `detail`). */
export function audioFailureOf(error: unknown): AudioFailure | null {
  return error instanceof Error
    ? parseAudioFailure((error as { detail?: unknown }).detail)
    : null;
}

export interface AudioFailureContext {
  deviceIsCpu: boolean;
  /** MiniMax Music 3 needs CUDA, so CPU RAM is never offered for it. */
  cpuUnsupported?: boolean;
  canUpdateRuntime: boolean;
  onUpdateRuntime: () => void;
  onUseCpu: () => void;
  onUseVariant: (variant: string) => void;
  onChooseModel: () => void;
}

/** Actions for a failure, mapped onto existing controls; anything unclassified keeps "Choose a different model". */
export function audioFailureActions(
  failure: AudioFailure | null,
  ctx: AudioFailureContext,
): GenerateAction[] {
  const actions: GenerateAction[] = [];
  const smaller = failure?.smallerVariants[0];
  if (failure && RUNTIME_CODES.includes(failure.code)) {
    if (ctx.canUpdateRuntime)
      actions.push({
        label: "Update the audio runtime",
        onClick: ctx.onUpdateRuntime,
      });
  } else if (
    failure?.code === "out_of_memory" ||
    failure?.code === "gpu_unavailable"
  ) {
    if (failure.code === "out_of_memory" && smaller)
      actions.push({
        label: `Use ${smaller.label}`,
        onClick: () => ctx.onUseVariant(smaller.variant),
      });
    if (!ctx.deviceIsCpu && !ctx.cpuUnsupported)
      actions.push({ label: "Load into CPU RAM", onClick: ctx.onUseCpu });
  }
  return actions.length > 0
    ? actions
    : [{ label: "Choose a different model", onClick: ctx.onChooseModel }];
}

/** Records the failure behind a generation error, so the page can offer its actions while that
 *  error shows. An unclassified error clears it. */
export function reportAudioFailure(error: unknown): AudioFailure | null {
  const failure = audioFailureOf(error);
  useAudioWorkspaceStore
    .getState()
    .setFailure(
      failure && error instanceof Error
        ? { message: error.message, failure }
        : null,
    );
  return failure;
}
