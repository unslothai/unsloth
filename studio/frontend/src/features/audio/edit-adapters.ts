// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How each speech-edit model takes a change, as verified in spike S2:
//   DotTTS Edit   marks the changes inline (<sub targ>, <del>, <ins>) and redoes them in one call.
//   Vevo2         re-speaks the whole edited sentence in the recording's voice.
//   FireRedAudio  takes one plain instruction per change, each call editing the last one's output,
//                 and alone can change the delivery (speed, a raised pitch).
// The run body and the request preview are built here from one diff, and the shared golden
// fixture (tests/fixtures/audio-edit-requests.json) pins both against the backend. Free of app
// imports so the node test runner can load it directly.

import type { AudioOptionValues } from "./audio-options";
import type {
  AudioOptionScalar,
  AudioRunEditPart,
  AudioRunRequest,
  AudioSourceRef,
  AudioTrim,
} from "./audio-run-request";
import {
  EDIT_DIFF_MAX_WORDS,
  type EditChange,
  changesBetween,
  diffWords,
} from "./edit-diff";
import type { AudioModelContext } from "./tools/types";

/** What the preview shows in place of values only the server knows. */
export const MODEL_PLACEHOLDER = "<model>";
export const RECORDING_PLACEHOLDER = "<recording>";
/** The temp file call `call` (1-based) wrote, which the next call edits. */
export function previousResult(call: number): string {
  return `<result of call ${call}>`;
}

export const RUNTIME_TASK_PATH = "/v1/tasks/run";

/** FireRedAudio applies one change per call; more than this is too slow and drifts. */
export const FIRERED_MAX_CHANGES = 5;
/** The route's limit on one FireRedAudio instruction. */
export const FIRERED_INSTRUCTION_MAX_CHARS = 300;

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
export const DELIVERY_NEEDS_FIRERED = "Delivery changes need FireRedAudio.";

export type EditStyle = "markup" | "sentence" | "instructions";
export type EditMode = "words" | "delivery";

export interface EditDelivery {
  /** Playback speed, 1 = unchanged. */
  speed: number;
  /** Steps to raise the pitch, 0 = unchanged. */
  pitchSteps: number;
}

/** The Delivery controls' ranges: speed [min, max, step] and raise-only pitch steps [min, max]. */
export interface EditDeliveryRange {
  speed: readonly [number, number, number];
  pitchSteps: readonly [number, number];
}

/** One request the server posts to the runtime, with server values as placeholders. */
export interface EditRuntimeCall {
  path: string;
  body: Record<string, unknown>;
}

export interface EditRunInput {
  /** The recording, by id. Optional so the preview can be built before one is picked. */
  source?: (AudioSourceRef & { trim?: AudioTrim }) | null;
  /** ① What the recording says. */
  transcript: string;
  /** ② The transcript with the user's changes. */
  edited: string;
  mode: EditMode;
  delivery?: EditDelivery | null;
  /** Advanced options; the adapter's claimed names are dropped. */
  advanced?: AudioOptionValues | null;
  /** The recording's name, the history text of a Delivery run with no transcript. */
  sourceName?: string | null;
}

export interface EditAdapter {
  family: string;
  /** The model's name in messages. */
  label: string;
  style: EditStyle;
  /** Spec options the adapter sets itself, which Advanced leaves out. */
  claims: readonly string[];
  /** Null when every change goes in one call. */
  maxChanges: number | null;
  /** Null when the model cannot change the delivery. */
  delivery: EditDeliveryRange | null;
  /** One plain line on how the model applies `changes` changes. */
  howItEdits: (changes: number) => string;
  /** Why the edit from `original` to `edited` cannot run on this model, or null. Zero changes is
   *  not an error here: editBlocker reports it earlier with its own action. */
  validateWords: (original: string, edited: string) => string | null;
  /** The `edit` part of the run body. */
  buildEdit: (
    input: Pick<EditRunInput, "transcript" | "edited" | "mode" | "delivery">,
  ) => AudioRunEditPart;
  /** The runtime calls the server makes for `run`. Advanced defaults to `run.options`. */
  preview: (
    run: AudioRunRequest,
    advanced?: Readonly<Record<string, AudioOptionScalar>> | null,
  ) => EditRuntimeCall[];
}

/** The markup DotTTS Edit reads: transcript words joined by single spaces, each change wrapped. */
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

/** FireRedAudio's instruction for one change, or null for an insert with no word after it. */
export function fireredInstruction(change: EditChange): string | null {
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

/** One instruction per change, in transcript order; changes with no instruction are left out. */
export function fireredInstructionsFor(
  changes: readonly EditChange[],
): string[] {
  return changes
    .map(fireredInstruction)
    .filter((line): line is string => line !== null);
}

/** Python's `{x:g}` for the speeds the slider makes: 1.5 → "1.5", 2 → "2". */
function formatG(value: number): string {
  return String(Number(value.toPrecision(6)));
}

const isUnchangedSpeed = (speed: number | null | undefined): boolean =>
  typeof speed !== "number" ||
  !Number.isFinite(speed) ||
  Math.abs(speed - 1) < 1e-6;

/** The acoustic_edit instructions the server renders from a Delivery run: speed, then pitch. */
export function deliveryInstructions(
  speed: number | null | undefined,
  pitchSteps: number | null | undefined,
): string[] {
  const lines: string[] = [];
  if (!isUnchangedSpeed(speed))
    lines.push(`adjust the speed to ${formatG(speed as number)}x`);
  if (typeof pitchSteps === "number" && pitchSteps > 0)
    lines.push(`shift the pitch by ${Math.round(pitchSteps)} steps`);
  return lines;
}

/** Option values as the runtime gets them: strings, the backend's `_option_string`. */
function optionStrings(
  options: Readonly<Record<string, AudioOptionScalar>>,
  claims: readonly string[],
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(options)) {
    if (!name || claims.includes(name)) continue;
    if (typeof value === "boolean") out[name] = value ? "true" : "false";
    else if (typeof value === "number") {
      if (Number.isFinite(value)) out[name] = String(value);
    } else if (typeof value === "string") out[name] = value;
  }
  return out;
}

function withoutClaims(
  options: AudioOptionValues | null | undefined,
  claims: readonly string[],
): Record<string, AudioOptionScalar> {
  const out: Record<string, AudioOptionScalar> = {};
  for (const [name, value] of Object.entries(options ?? {})) {
    if (name && !claims.includes(name)) out[name] = value;
  }
  return out;
}

/** The fields every runtime call starts with. */
function baseBody(run: AudioRunRequest): Record<string, unknown> {
  const body: Record<string, unknown> = { model: MODEL_PLACEHOLDER };
  if (typeof run.seed === "number" && Number.isInteger(run.seed))
    body.seed = run.seed;
  return body;
}

/** A chain of calls, each editing the previous call's output. */
function chainedCalls(
  run: AudioRunRequest,
  template: string,
  instructions: readonly string[],
  options: Record<string, string>,
): EditRuntimeCall[] {
  return instructions.map((instruction, index) => ({
    path: RUNTIME_TASK_PATH,
    body: {
      ...baseBody(run),
      audio: index === 0 ? RECORDING_PLACEHOLDER : previousResult(index),
      options: { template_name: template, instruction, ...options },
    },
  }));
}

const STRUCTURAL_CLAIMS = [
  "template_name",
  "source_text",
  "target_text",
  "instruction",
] as const;

function tooLong(original: string, edited: string): string | null {
  return changesBetween(original, edited) === null ? EDIT_TOO_LONG : null;
}

const wordsEdit = (
  extra: Omit<AudioRunEditPart, "mode"> = {},
): AudioRunEditPart => ({ mode: "words", ...extra });

const plural = (count: number, word: string) =>
  `${count} ${word}${count === 1 ? "" : "s"}`;

export const DOTS_EDIT_ADAPTER: EditAdapter = {
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
    return null;
  },
  buildEdit: ({ transcript, edited }) => {
    // Markup with no change in it is not an edit; the route says so if it is ever sent.
    const changed = (changesBetween(transcript, edited)?.length ?? 0) > 0;
    const markup = changed ? markupFor(transcript, edited) : null;
    return wordsEdit(markup ? { markup } : {});
  },
  preview: (run, advanced) => {
    const markup = run.edit?.markup;
    if (!markup) return [];
    const options = optionStrings(
      advanced ?? run.options ?? {},
      DOTS_EDIT_ADAPTER.claims,
    );
    return [
      {
        path: RUNTIME_TASK_PATH,
        body: {
          ...baseBody(run),
          text: markup,
          source_audio: RECORDING_PLACEHOLDER,
          options: { template_name: "edit", ...options },
        },
      },
    ];
  },
};

export const VEVO2_EDIT_ADAPTER: EditAdapter = {
  family: "vevo2",
  label: "Vevo2",
  style: "sentence",
  claims: STRUCTURAL_CLAIMS,
  maxChanges: null,
  delivery: null,
  howItEdits: () =>
    "Speaks the whole edited sentence again in the recording's voice, in one pass.",
  validateWords: tooLong,
  // Vevo2 reads the edited text itself, so the edit part carries only the mode.
  buildEdit: () => wordsEdit(),
  // Vevo2 declares no options, so the server sends none.
  preview: (run) => {
    const body: Record<string, unknown> = {
      ...baseBody(run),
      route: "editing",
      source_audio: RECORDING_PLACEHOLDER,
      target_text: run.text,
    };
    const original = run.inputs?.reference_text?.trim();
    if (original) body.reference_text = original;
    return [{ path: RUNTIME_TASK_PATH, body }];
  },
};

export const FIRERED_EDIT_ADAPTER: EditAdapter = {
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
      lines.some((line) => (line?.length ?? 0) > FIRERED_INSTRUCTION_MAX_CHARS)
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
    const instructions = fireredInstructionsFor(
      changesBetween(transcript, edited) ?? [],
    );
    return wordsEdit(instructions.length ? { instructions } : {});
  },
  preview: (run, advanced) => {
    const options = optionStrings(
      advanced ?? run.options ?? {},
      FIRERED_EDIT_ADAPTER.claims,
    );
    if (run.edit?.mode === "delivery") {
      return chainedCalls(
        run,
        "acoustic_edit",
        deliveryInstructions(run.edit.speed, run.edit.pitch_steps),
        options,
      );
    }
    return chainedCalls(
      run,
      "semantic_edit",
      run.edit?.instructions ?? [],
      options,
    );
  },
};

/** The edit adapters by runtime family. */
export const EDIT_ADAPTERS: Readonly<Record<string, EditAdapter>> = {
  dots_tts: DOTS_EDIT_ADAPTER,
  vevo2: VEVO2_EDIT_ADAPTER,
  firered_audio: FIRERED_EDIT_ADAPTER,
};

/** The loaded model's adapter: its family has one and its status lists the edit workflow (so
 *  DotTTS-MF, family dots_tts but not an edit package, gets none). */
export function editAdapterFor(
  ctx: Pick<AudioModelContext, "audioFamily" | "audioWorkflows">,
): EditAdapter | null {
  if (!ctx.audioWorkflows?.includes("edit")) return null;
  const family = ctx.audioFamily ?? "";
  return Object.hasOwn(EDIT_ADAPTERS, family) ? EDIT_ADAPTERS[family] : null;
}

/** The full /audio/run request for an edit. Check editBlocker first; this does not validate. */
export function buildEditRun(
  adapter: EditAdapter,
  input: EditRunInput,
): AudioRunRequest {
  const mode: EditMode =
    input.mode === "delivery" && adapter.delivery ? "delivery" : "words";
  const transcript = input.transcript.trim();
  const text =
    mode === "delivery"
      ? transcript || input.sourceName?.trim() || "Recording"
      : input.edited.trim();
  const inputs: NonNullable<AudioRunRequest["inputs"]> = {};
  if (input.source) inputs.source = { ...input.source };
  if (transcript) inputs.reference_text = transcript;
  const run: AudioRunRequest = {
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
  const options = withoutClaims(input.advanced, adapter.claims);
  if (Object.keys(options).length > 0) run.options = options;
  return run;
}

/** How many runtime passes the run takes. */
export function editPassCount(
  adapter: EditAdapter,
  run: AudioRunRequest,
): number {
  return adapter.preview(run).length;
}

/** The progress line for a chained run, or null when it is one pass. */
export function editPhaseLabel(
  adapter: EditAdapter,
  run: AudioRunRequest,
): string | null {
  const passes = editPassCount(adapter, run);
  if (passes <= 1) return null;
  return run.edit?.mode === "delivery"
    ? `Applying ${passes} delivery changes, one pass each`
    : `Applying ${passes} changes, one pass each`;
}
