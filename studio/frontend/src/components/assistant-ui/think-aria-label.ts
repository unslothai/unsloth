// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared aria-labels for the Think pill toggle + effort dropdown, so both
// use the same pre-load / unsupported-model wording.

export interface ThinkToggleState {
  reasoningLockedOn: boolean;
  modelLoaded: boolean;
  reasoningDisabled: boolean;
  effectiveReasoningEnabled: boolean;
}

export function thinkToggleAriaLabel(state: ThinkToggleState): string {
  if (state.reasoningLockedOn) return "Thinking is required for this model";
  if (!state.modelLoaded) return "Thinking (model not loaded)";
  if (state.reasoningDisabled)
    return "Thinking (not supported by this model)";
  return state.effectiveReasoningEnabled
    ? "Disable thinking"
    : "Enable thinking";
}

export interface ThinkEffortState {
  modelLoaded: boolean;
  reasoningDisabled: boolean;
  reasoningEffort: string;
}

// Locked-on isn't special-cased: the effort dropdown stays interactive.
export function thinkEffortAriaLabel(state: ThinkEffortState): string {
  if (!state.modelLoaded) return "Thinking (model not loaded)";
  if (state.reasoningDisabled)
    return "Thinking (not supported by this model)";
  return `Reasoning effort: ${state.reasoningEffort}`;
}
