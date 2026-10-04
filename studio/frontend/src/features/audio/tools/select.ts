// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which tool panels a model gets. Free of JSX so the node test runner can load it.

import type { AudioOptionSpec } from "../audio-options";
import {
  type NativeAudioInstructionsKind,
  nativeAudioInstructionsKind,
} from "../audio-page-policy";
import type { AudioWorkflowId } from "../workflows";
import type {
  AnyAudioToolPanel,
  AudioModelContext,
  AudioReferenceTextMode,
  AudioRunPatch,
  AudioToolPanel,
  CoreInputs,
} from "./types";

export function instructionsKindFor(
  ctx: AudioModelContext,
): NativeAudioInstructionsKind | null {
  return ctx.musicGeneration
    ? "music"
    : nativeAudioInstructionsKind(ctx.audioType);
}

export function panelApplies(
  panel: Pick<AudioToolPanel<unknown>, "workflows" | "families" | "appliesTo">,
  workflow: AudioWorkflowId,
  ctx: AudioModelContext,
): boolean {
  if (!panel.workflows.includes(workflow)) return false;
  return panel.appliesTo
    ? panel.appliesTo(ctx)
    : panel.families.includes(ctx.audioFamily ?? "");
}

export function claimedOptionNames(
  panels: readonly Pick<AudioToolPanel<unknown>, "claims">[],
): Set<string> {
  return new Set(panels.flatMap((panel) => panel.claims));
}

const REFERENCE_TEXT_MODES: ReadonlySet<string> = new Set([
  "required",
  "optional",
  "unused",
]);

export function audioModelContextFor(
  status: {
    audio_type?: string | null;
    audio_family?: string | null;
    audio_workflows?: readonly string[] | null;
    audio_required_inputs?: readonly string[] | null;
    audio_reference_text?: string | null;
  } | null,
  page: {
    musicGeneration: boolean;
    cudaMusicGeneration: boolean;
    musicNeedsDescription: boolean;
  },
): AudioModelContext {
  const referenceText = status?.audio_reference_text;
  return {
    audioType: status?.audio_type ?? null,
    audioFamily: status?.audio_family ?? null,
    musicGeneration: page.musicGeneration,
    cudaMusicGeneration: page.cudaMusicGeneration,
    musicNeedsDescription: page.musicNeedsDescription,
    audioWorkflows: Array.isArray(status?.audio_workflows)
      ? status.audio_workflows
      : [],
    requiredInputs: Array.isArray(status?.audio_required_inputs)
      ? status.audio_required_inputs
      : [],
    referenceTextMode:
      typeof referenceText === "string" &&
      REFERENCE_TEXT_MODES.has(referenceText)
        ? (referenceText as AudioReferenceTextMode)
        : null,
  };
}

export function toolValueKey(
  model: string | null | undefined,
  workflow: AudioWorkflowId,
  panelId: string,
): string {
  return `${model ?? ""}:${workflow}:${panelId}`;
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// Kept value over defaults, so a value saved by an older build still has every field.
export function panelValue<V>(
  panel: Pick<AudioToolPanel<V>, "id" | "initial">,
  values: Readonly<Record<string, unknown>>,
  specs: AudioOptionSpec[],
): V {
  const initial = panel.initial(specs);
  const stored = values[panel.id];
  if (stored === undefined) return initial;
  if (isPlainObject(initial) && isPlainObject(stored)) {
    return { ...initial, ...stored } as V;
  }
  return stored as V;
}

export function collectToolRequest(
  panels: readonly AnyAudioToolPanel[],
  values: Readonly<Record<string, unknown>>,
  core: CoreInputs,
  ctx: AudioModelContext,
  specs: AudioOptionSpec[] = [],
): { patch: AudioRunPatch; error: string | null } {
  const patch: AudioRunPatch = {};
  let error: string | null = null;
  for (const panel of panels) {
    const value = panelValue(panel, values, specs);
    error ??= panel.validate?.(value, core, ctx) ?? null;
    const part = panel.toRequest(value);
    if (part.options) patch.options = { ...patch.options, ...part.options };
    if (part.inputs) patch.inputs = { ...patch.inputs, ...part.inputs };
    for (const key of [
      "instructions",
      "language",
      "route",
      "text",
      "speed",
      "referenceTextMode",
    ] as const) {
      if (part[key] !== undefined) {
        (patch as Record<string, unknown>)[key] = part[key];
      }
    }
  }
  return { patch, error };
}
