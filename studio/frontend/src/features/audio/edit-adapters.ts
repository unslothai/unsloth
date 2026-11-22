// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// tests/fixtures/audio-edit-requests.json pins the run bodies built here against the backend.

import type { AudioOptionValues } from "./audio-options";
import type {
  AudioRunEditPart,
  AudioTextRunRequest,
  AudioSourceRef,
} from "./audio-run-request";
import {
  EDIT_DIFF_MAX_WORDS,
  type EditChange,
  changesBetween,
  diffWords,
} from "./edit-diff";
import type { AudioModelContext } from "./tools/types";

export const FIRERED_MAX_CHANGES = 5;
/** The route's limit on one FireRedAudio instruction. */
const FIRERED_INSTRUCTION_MAX_CHARS = 300;

export const EDIT_TOO_LONG = `Keep the transcript under ${EDIT_DIFF_MAX_WORDS} words.`;
export const FIRERED_TOO_MANY_CHANGES = `FireRedAudio applies at most ${FIRERED_MAX_CHANGES} changes. Make fewer changes, or use DotTTS Edit.`;
export const FIRERED_INSERT_AT_END =
  "FireRedAudio can only insert before another word. Move the new words, or use DotTTS Edit.";
export const FIRERED_CHANGE_TOO_LONG =
  "Make each change shorter for FireRedAudio, or use DotTTS Edit.";
export const DOTS_BAD_CHARACTERS =
  "Remove quotes and angle brackets from the changed words for DotTTS Edit.";
export const DOTS_ANGLE_BRACKETS =
  "Remove angle brackets from the transcript for DotTTS Edit.";
/** Mirrors `AudioRunEdit.markup` (max_length = 8000) so the cap is a hint here, not a 422. */
export const DOTS_MARKUP_MAX_CHARS = 8000;
export const DOTS_MARKUP_TOO_LONG =
  "Make fewer changes at once for DotTTS Edit, or shorten the transcript.";
export const DELIVERY_NEEDS_FIRERED = "Delivery changes need FireRedAudio.";

export type EditMode = "words" | "delivery";

export interface EditDelivery {
  speed: number;
  pitchSteps: number;
}

export interface EditRunInput {
  source?: AudioSourceRef | null;
  transcript: string;
  edited: string;
  mode: EditMode;
  delivery?: EditDelivery | null;
  advanced?: AudioOptionValues | null;
  sourceName?: string | null;
}

export interface EditAdapter {
  family: string;
  label: string;
  style: "markup" | "sentence" | "instructions";
  /** Spec options the adapter sets itself, which Advanced leaves out. */
  claims: readonly string[];
  maxChanges: number | null;
  /** Speed [min, max, step] and raise-only pitch steps [min, max]; null without delivery control. */
  delivery: {
    speed: readonly [number, number, number];
    pitchSteps: readonly [number, number];
  } | null;
  howItEdits: (changes: number) => string;
  /** Zero changes is not an error here: editBlocker reports it with its own action. */
  validateWords: (original: string, edited: string) => string | null;
  buildEdit: (
    input: Pick<EditRunInput, "transcript" | "edited" | "mode" | "delivery">,
  ) => AudioRunEditPart;
}

/** DotTTS Edit markup: transcript words joined by single spaces, each change wrapped. */
export function markupFor(original: string, edited: string): string | null {
  const ops = diffWords(original, edited);
  if (!ops) return null;
  const pieces: string[] = [];
  let k = 0;
  while (k < ops.length) {
    if (ops[k].kind === "equal") {
      pieces.push(ops[k].word);
      k += 1;
      continue;
    }
    const old: string[] = [];
    const added: string[] = [];
    while (k < ops.length && ops[k].kind !== "equal") {
      (ops[k].kind === "delete" ? old : added).push(ops[k].word);
      k += 1;
    }
    if (old.length && added.length) {
      pieces.push(`<sub targ="${added.join(" ")}">${old.join(" ")}</sub>`);
    } else if (old.length) {
      pieces.push(`<del>${old.join(" ")}</del>`);
    } else {
      pieces.push(`<ins>${added.join(" ")}</ins>`);
    }
  }
  return pieces.join(" ");
}

/** Null for an insert with no word after it: FireRedAudio can only insert before an anchor. */
function fireredInstruction(change: EditChange): string | null {
  const old = change.old.join(" ");
  const added = change.new.join(" ");
  switch (change.kind) {
    case "replace":
      return `Replace '${old}' with '${added}'.`;
    case "delete":
      return `Delete '${old}'.`;
    default:
      return change.before === null
        ? null
        : `Insert '${added}' before '${change.before}'.`;
  }
}

const isUnchangedSpeed = (speed: number | null | undefined): boolean =>
  typeof speed !== "number" ||
  !Number.isFinite(speed) ||
  Math.abs(speed - 1) < 1e-6;

/** Mirrors the backend's `delivery_instructions` (Python `{x:g}` formatting). */
export function deliveryInstructions(
  speed: number | null | undefined,
  pitchSteps: number | null | undefined,
): string[] {
  const lines: string[] = [];
  if (!isUnchangedSpeed(speed))
    lines.push(
      `adjust the speed to ${Number((speed as number).toPrecision(6))}x`,
    );
  if (typeof pitchSteps === "number" && pitchSteps > 0)
    lines.push(`shift the pitch by ${Math.round(pitchSteps)} steps`);
  return lines;
}

const STRUCTURAL_CLAIMS = [
  "template_name",
  "source_text",
  "target_text",
  "instruction",
] as const;

const plural = (count: number, word: string) =>
  `${count} ${word}${count === 1 ? "" : "s"}`;

export const EDIT_ADAPTERS: Readonly<Record<string, EditAdapter>> = {
  dots_tts: {
    family: "dots_tts",
    label: "DotTTS Edit",
    style: "markup",
    claims: STRUCTURAL_CLAIMS,
    maxChanges: null,
    delivery: null,
    howItEdits: (changes) =>
      changes > 1
        ? `Redoes the ${plural(changes, "changed word group")} in one pass and keeps the rest of the recording.`
        : "Redoes only the changed words in one pass and keeps the rest of the recording.",
    validateWords: (original, edited) => {
      const changes = changesBetween(original, edited);
      if (changes === null) return EDIT_TOO_LONG;
      if (
        changes.some((change) =>
          [...change.old, ...change.new].some((word) => /["<>]/.test(word)),
        )
      )
        return DOTS_BAD_CHARACTERS;
      if (/[<>]/.test(original) || /[<>]/.test(edited))
        return DOTS_ANGLE_BRACKETS;
      if ((markupFor(original, edited)?.length ?? 0) > DOTS_MARKUP_MAX_CHARS)
        return DOTS_MARKUP_TOO_LONG;
      return null;
    },
    buildEdit: ({ transcript, edited }) => {
      const changed = (changesBetween(transcript, edited)?.length ?? 0) > 0;
      const markup = changed ? markupFor(transcript, edited) : null;
      return markup ? { mode: "words", markup } : { mode: "words" };
    },
  },
  vevo2: {
    family: "vevo2",
    label: "Vevo2",
    style: "sentence",
    claims: STRUCTURAL_CLAIMS,
    maxChanges: null,
    delivery: null,
    howItEdits: () =>
      "Speaks the whole edited sentence again in the recording's voice, in one pass.",
    validateWords: (original, edited) =>
      changesBetween(original, edited) === null ? EDIT_TOO_LONG : null,
    // Vevo2 reads the edited `text` itself.
    buildEdit: () => ({ mode: "words" }),
  },
  firered_audio: {
    family: "firered_audio",
    label: "FireRedAudio",
    style: "instructions",
    claims: ["template_name", "instruction"],
    maxChanges: FIRERED_MAX_CHANGES,
    delivery: { speed: [0.5, 2, 0.1], pitchSteps: [1, 6] },
    howItEdits: (changes) =>
      changes > 1
        ? `Applies ${plural(changes, "change")}, one pass each, every pass editing the last one's result.`
        : "Applies each change in its own pass, every pass editing the last one's result.",
    validateWords: (original, edited) => {
      const changes = changesBetween(original, edited);
      if (changes === null) return EDIT_TOO_LONG;
      if (changes.length > FIRERED_MAX_CHANGES) return FIRERED_TOO_MANY_CHANGES;
      const lines = changes.map(fireredInstruction);
      if (lines.some((line) => line === null)) return FIRERED_INSERT_AT_END;
      if (
        lines.some(
          (line) => (line?.length ?? 0) > FIRERED_INSTRUCTION_MAX_CHARS,
        )
      )
        return FIRERED_CHANGE_TOO_LONG;
      return null;
    },
    buildEdit: ({ transcript, edited, mode, delivery }) => {
      if (mode === "delivery") {
        const part: AudioRunEditPart = { mode: "delivery" };
        if (!isUnchangedSpeed(delivery?.speed))
          part.speed = Math.round((delivery?.speed as number) * 100) / 100;
        if (delivery && delivery.pitchSteps > 0)
          part.pitch_steps = Math.round(delivery.pitchSteps);
        return part;
      }
      const instructions = (changesBetween(transcript, edited) ?? [])
        .map(fireredInstruction)
        .filter((line): line is string => line !== null);
      return instructions.length
        ? { mode: "words", instructions }
        : { mode: "words" };
    },
  },
};

/** Needs the edit workflow too: DotTTS-MF is dots_tts but cannot edit. */
export function editAdapterFor(
  ctx: Pick<AudioModelContext, "audioFamily" | "audioWorkflows">,
): EditAdapter | null {
  if (!ctx.audioWorkflows?.includes("edit")) return null;
  const family = ctx.audioFamily ?? "";
  return Object.hasOwn(EDIT_ADAPTERS, family) ? EDIT_ADAPTERS[family] : null;
}

/** Does not validate: check editBlocker first. */
export function buildEditRun(
  adapter: EditAdapter,
  input: EditRunInput,
): AudioTextRunRequest {
  const mode: EditMode =
    input.mode === "delivery" && adapter.delivery ? "delivery" : "words";
  const transcript = input.transcript.trim();
  const text =
    mode === "delivery"
      ? transcript || input.sourceName?.trim() || "Recording"
      : input.edited.trim();
  const inputs: NonNullable<AudioTextRunRequest["inputs"]> = {};
  if (input.source) inputs.source = { ...input.source };
  if (transcript) inputs.reference_text = transcript;
  const run: AudioTextRunRequest = {
    workflow: "edit",
    text,
    inputs,
    edit: adapter.buildEdit({
      transcript,
      edited: input.edited,
      mode,
      delivery: input.delivery,
    }),
  };
  const options = Object.fromEntries(
    Object.entries(input.advanced ?? {}).filter(
      ([name]) => name && !adapter.claims.includes(name),
    ),
  );
  if (Object.keys(options).length > 0) run.options = options;
  return run;
}

export function editPhaseLabel(
  adapter: EditAdapter,
  run: AudioTextRunRequest,
): string | null {
  if (adapter.style !== "instructions") return null;
  const delivery = run.edit?.mode === "delivery";
  const passes = delivery
    ? deliveryInstructions(run.edit?.speed, run.edit?.pitch_steps).length
    : (run.edit?.instructions?.length ?? 0);
  if (passes <= 1) return null;
  return delivery
    ? `Applying ${passes} delivery changes, one pass each`
    : `Applying ${passes} changes, one pass each`;
}
