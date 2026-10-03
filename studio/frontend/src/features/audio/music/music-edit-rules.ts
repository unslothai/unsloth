// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What Music's Edit mode asks for per action, and why Generate is off. Free of app imports so the
// node test runner can load it directly.

import { MIN_RANGE_S } from "../components/waveform-range";
import type {
  MusicEditAction,
  MusicEditDraft,
  MusicModeRule,
} from "./music-types";

export const MUSIC_EDIT_ACTION_LABEL: Record<MusicEditAction, string> = {
  repaint: "Repaint",
  extend: "Extend",
  cover: "Cover",
  continue: "Continue",
  inpaint: "Inpaint",
  restyle: "Restyle",
};

export const MUSIC_EDIT_ACTION_HINT: Record<MusicEditAction, string> = {
  repaint:
    "Regenerate the selected part, keep the rest. Drag past the end to make it longer.",
  extend: "Add new music after the end.",
  cover: "Same structure, new style.",
  continue: "Adds new parts across the whole track.",
  inpaint: "Regenerate the selected parts.",
  restyle: "Re-render the whole clip in a new style.",
};

/** How far past the end a repaint may reach; the part past the end extends the clip. */
export const REPAINT_BEYOND_END_S = 30;

export const EXTEND_MIN_S = 5;
export const EXTEND_MAX_S = 120;

/** "How much to change" when the draft leaves it to the model (null), per action. */
export const MUSIC_EDIT_DEFAULT_STRENGTH: Partial<
  Record<MusicEditAction, number>
> = {
  cover: 0.5,
  restyle: 0.45,
};

export function editUsesRanges(action: MusicEditAction | null): boolean {
  return action === "repaint" || action === "inpaint";
}

export function editUsesStrength(action: MusicEditAction | null): boolean {
  return action === "cover" || action === "restyle";
}

/** Repaint changes one part; inpaint up to the model's limit. */
export function editMaxRanges(
  rule: MusicModeRule,
  action: MusicEditAction | null,
): number {
  if (action === "repaint") return 1;
  if (action === "inpaint") return Math.max(1, rule.max_ranges ?? 1);
  return 0;
}

/** Seconds a part may run past the end: only a repaint, which then extends the clip. */
export function editBeyondEndS(action: MusicEditAction | null): number {
  return action === "repaint" ? REPAINT_BEYOND_END_S : 0;
}

/** The actions the loaded model offers, in the order the status lists them. */
export function editActions(rule: MusicModeRule): MusicEditAction[] {
  return rule.actions ?? [];
}

/** "4 minutes", "1 minute", or "90 s" when it is not whole minutes. */
export function formatEditLimit(seconds: number): string {
  if (seconds >= 60 && seconds % 60 === 0) {
    const minutes = seconds / 60;
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  }
  return `${Math.round(seconds)} s`;
}

/** The clip is longer than the model can edit, said as the card shows it. */
export function editSourceTooLong(
  rule: MusicModeRule,
  sourceDurationS: number | null,
): string | null {
  if (rule.max_source_s === undefined || sourceDurationS === null) return null;
  return sourceDurationS > rule.max_source_s
    ? `Edit clips up to ${formatEditLimit(rule.max_source_s)}. Trim it first.`
    : null;
}

// Times come back rounded to a hundredth; allow that much slack.
const EPSILON = 0.011;

/** What is wrong with the picked parts, or null when they can be sent. */
export function editRangeProblem(
  rule: MusicModeRule,
  draft: Pick<MusicEditDraft, "action" | "ranges">,
  sourceDurationS: number | null,
): string | null {
  const { action, ranges } = draft;
  if (!editUsesRanges(action)) return null;
  if (ranges.length === 0) {
    return action === "inpaint"
      ? "Select the parts to change on the waveform."
      : "Select the part to change on the waveform.";
  }
  const max = editMaxRanges(rule, action);
  if (ranges.length > max) {
    return max === 1
      ? "Select one part to change."
      : `Select up to ${max} parts.`;
  }
  for (const range of ranges) {
    if (
      !(Number.isFinite(range.start_s) && Number.isFinite(range.end_s)) ||
      range.start_s < 0 ||
      range.end_s - range.start_s < MIN_RANGE_S - EPSILON
    ) {
      return `Make each selected part at least ${MIN_RANGE_S} s long.`;
    }
    if (sourceDurationS === null) continue;
    if (range.start_s >= sourceDurationS) {
      return "A selected part starts after the clip ends. Move it inside the clip.";
    }
    if (range.end_s > sourceDurationS + editBeyondEndS(action) + EPSILON) {
      return action === "repaint"
        ? `A repaint can reach at most ${REPAINT_BEYOND_END_S} s past the end.`
        : "A selected part runs past the end of the clip.";
    }
  }
  return null;
}

/** Why Generate is off in Edit mode, as one sentence, or null when it can run. */
export function musicEditProblem(
  rule: MusicModeRule,
  draft: MusicEditDraft,
  sourceDurationS: number | null,
): string | null {
  if (!draft.source) return "Pick a clip to edit.";
  const actions = editActions(rule);
  if (actions.length === 0) return "This model cannot edit clips.";
  if (!(draft.action && actions.includes(draft.action))) {
    return "Pick what to do with the clip.";
  }
  const tooLong = editSourceTooLong(rule, sourceDurationS);
  if (tooLong) return tooLong;
  const rangeProblem = editRangeProblem(rule, draft, sourceDurationS);
  if (rangeProblem) return rangeProblem;
  if (
    draft.action === "extend" &&
    !(draft.extendS >= EXTEND_MIN_S && draft.extendS <= EXTEND_MAX_S)
  ) {
    return `Add between ${EXTEND_MIN_S} and ${EXTEND_MAX_S} seconds.`;
  }
  if (!draft.prompt.trim()) {
    return editUsesStrength(draft.action)
      ? "Describe the new style."
      : "Describe what the new part should sound like.";
  }
  return null;
}

export type SendAction = "edit" | "extend";

/** What a history clip offers for the loaded model: nothing when it cannot edit, Extend only
 *  when its Edit mode extends. */
export function sendActionsFor(
  editActions: readonly string[],
): readonly SendAction[] {
  if (editActions.length === 0) return [];
  return editActions.includes("extend") ? ["edit", "extend"] : ["edit"];
}
