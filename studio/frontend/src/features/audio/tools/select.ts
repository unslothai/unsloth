// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which tool panels a model gets. Free of JSX so the node test runner can load it.

import {
  type NativeAudioInstructionsKind,
  nativeAudioInstructionsKind,
} from "../audio-page-policy";
import type { AudioWorkflowId } from "../workflows";
import type { AudioModelContext, AudioToolPanel } from "./types";

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
