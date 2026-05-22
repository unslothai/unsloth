// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared aria-label helpers for the composer Think pill. Both the on/off
// toggle (used when a model exposes a simple thinking flag) and the
// reasoning-effort dropdown (Claude-style models with effort levels) need
// the same pre-load / unsupported-model fallback wording, so factor it
// once and reuse. Keeping the strings here also means the label set is
// trivially auditable in a single file.

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

// The reasoning-effort dropdown stays interactive in the locked-on case
// (users can still pick an effort level), so locked-on is intentionally
// not a special-case here -- only the truly-disabled states fall back to
// the Thinking (...) wording.
export function thinkEffortAriaLabel(state: ThinkEffortState): string {
  if (!state.modelLoaded) return "Thinking (model not loaded)";
  if (state.reasoningDisabled)
    return "Thinking (not supported by this model)";
  return `Reasoning effort: ${state.reasoningEffort}`;
}
