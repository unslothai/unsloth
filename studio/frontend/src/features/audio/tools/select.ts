// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which tool panels a model gets. Free of JSX so the node test runner can load it.

import type { AudioConvertCaps } from "@/features/chat/types/api";
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

/** Which instruction field the model takes, the same rule the rail always used. */
export function instructionsKindFor(
  ctx: AudioModelContext,
): NativeAudioInstructionsKind | null {
  return ctx.musicGeneration
    ? "music"
    : nativeAudioInstructionsKind(ctx.audioType);
}

/** A panel shows on its own workflows, for the models it matches. */
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

/** Spec options the shown panels render themselves, which Advanced then leaves out. */
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

/** The tool context for the loaded model on one page, read from its status fields. */
export function audioModelContextFor(
  status: {
    audio_type?: string | null;
    audio_family?: string | null;
    audio_workflows?: readonly string[] | null;
    audio_required_inputs?: readonly string[] | null;
    audio_reference_text?: string | null;
    audio_convert?: AudioConvertCaps | null;
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
    convert: convertCapsOf(status?.audio_convert),
  };
}

function convertCapsOf(value: unknown): AudioConvertCaps | null {
  if (!isPlainObject(value)) return null;
  const caps = value as Partial<AudioConvertCaps>;
  const modes = Array.isArray(caps.modes)
    ? caps.modes.filter((mode) => mode === "speech" || mode === "singing")
    : [];
  if (modes.length === 0) return null;
  return {
    modes,
    target: caps.target === "builtin" ? "builtin" : "audio",
    builtin_voices: Array.isArray(caps.builtin_voices)
      ? caps.builtin_voices.filter(
          (voice) =>
            isPlainObject(voice) &&
            typeof voice.id === "string" &&
            typeof voice.label === "string",
        )
      : [],
    pitch: isPlainObject(caps.pitch) ? caps.pitch : {},
    style: caps.style === true,
    route_reloads: caps.route_reloads === true,
    source_max_seconds:
      typeof caps.source_max_seconds === "number" && caps.source_max_seconds > 0
        ? caps.source_max_seconds
        : 300,
  };
}

/** Where a panel's value is kept: per model, page and panel, so models never share settings. */
export function toolValueKey(
  model: string | null | undefined,
  workflow: AudioWorkflowId,
  panelId: string,
): string {
  return `${model ?? ""}:${workflow}:${panelId}`;
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** A panel's value: what was kept for it laid over its defaults, so a value saved by an older
 *  build still has every field the panel reads. */
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

/** Every shown panel's part of the request, merged in rail order, and the first reason one of
 *  them holds Generate back. */
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
    const part = panel.toRequest(value, ctx);
    if (part.options) patch.options = { ...patch.options, ...part.options };
    if (part.inputs) patch.inputs = { ...patch.inputs, ...part.inputs };
    if (part.convert) patch.convert = { ...patch.convert, ...part.convert };
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
