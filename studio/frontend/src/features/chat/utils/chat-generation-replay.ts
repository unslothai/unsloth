// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a recovery follower replays is the run's stored chunk events, and until now it folded them
// back into ONE string: `generationRawContent(stored.content).raw` plus content/reasoning deltas,
// re-parsed whole on every publish. A tool-heavy reply has no string form -- its calls are parts,
// not characters -- so the replay dropped them and a reopened reply lost its pills and their
// output. This module is the follower's half of what `chat-adapter` does live: the same parts, in
// the same order, built from stored frames instead of the live socket, and extended by one delta
// at a time instead of re-parsed from character zero.

import { createSegmentedAssistantText } from "./incremental-assistant-content";
import { extractDeltaText } from "./parse-assistant-content";
import { preferFullToolOutput } from "./tool-output-preference";
import {
  findStreamedToolCallPartIndex,
  mintStreamedToolCallId,
  type StreamedToolCallPart,
} from "../tool-call-id";

type ContentPart = Record<string, unknown> & { type: string };

/** A replayed part, kept with the offset of the reply it was called AT. The live stream stamps
 *  `textCursor` on every part and `buildAssistantContent` cuts the text on those offsets; keeping
 *  them here is what puts a pill between two paragraphs instead of at the end of the message. A
 *  seeded image or source part has no call to answer to, so `toolCallId` stays optional here even
 *  though the slot lookups require one: they read parts as `StreamedToolCallPart`, where an absent
 *  id already means "no slot claimed", and they never rewrite a part they matched by id. */
export type PositionedReplayPart = Omit<StreamedToolCallPart, "toolCallId"> & {
  type: string;
  toolCallId?: string;
  textCursor: number;
  // What a replayed frame writes onto the card it names. They stay `unknown` on purpose: the replay
  // copies them off the frame verbatim, and only the renderer ever reads them.
  toolName?: unknown;
  args?: unknown;
  argsText?: unknown;
  result?: unknown;
  provenance?: unknown;
};

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

export function seededReplayState(content: unknown): {
  raw: string;
  reasoningOpen: boolean;
  parts: PositionedReplayPart[];
} {
  if (typeof content === "string") {
    return { raw: content, reasoningOpen: false, parts: [] };
  }
  if (!Array.isArray(content)) return { raw: "", reasoningOpen: false, parts: [] };
  let raw = "";
  let reasoningOpen = false;
  const parts: PositionedReplayPart[] = [];
  for (const part of content as ContentPart[]) {
    if (!part || typeof part !== "object") continue;
    const text = typeof part.text === "string" ? part.text : "";
    if (part.type === "reasoning") {
      raw += reasoningOpen ? text : `${THINK_OPEN_TAG}${text}`;
      reasoningOpen = true;
    } else if (part.type === "text") {
      raw += reasoningOpen ? `${THINK_CLOSE_TAG}${text}` : text;
      reasoningOpen = false;
    } else {
      parts.push({ ...part, textCursor: raw.length });
    }
  }
  return { raw, reasoningOpen, parts };
}

export type RecoveryReplay = {
  /** Fold one replayed chunk event. Returns whether it changed the reply, which is what decides
   *  whether there is anything worth publishing. */
  applyChunk(chunk: unknown): boolean;
  /** The reply as parts: text and reasoning runs cut at the tool-call offsets, tools interleaved. */
  content(): ContentPart[];
  /** The tagged string, for the prefix comparison a recovery publish makes against the view. */
  rawText(): string;
};

/** Build the follower's copy of a run's reply: `seed` is what storage already holds (the partial a
 *  reload restored, or the request's assistant prefill) and the events replay on top of it. */
export function createRecoveryReplay(seed: unknown): RecoveryReplay {
  const seeded = seededReplayState(seed);
  // The parse of everything replayed so far, extended by each delta rather than redone from
  // character zero: a publish costs one event, not the whole reply.
  const segmented = createSegmentedAssistantText();
  segmented.appendText(seeded.raw);
  const parts: PositionedReplayPart[] = seeded.parts.map((part) => ({ ...part }));
  // Backend call id -> part id. Seeded from what storage holds, whose ids were minted live (a
  // minted id is always `<backend id>:<uuid>`), so a replayed frame naming `call_0` finds the card
  // the first reader already saved instead of opening a second one.
  const idsByBackendId = new Map<string, string>();
  for (const part of parts) {
    const id = typeof part.toolCallId === "string" ? part.toolCallId : "";
    if (id) idsByBackendId.set(id.split(":")[0], id);
  }
  // What the deltas themselves mint. `mintStreamedToolCallId` is deterministic, so a reload
  // replaying the same frames draws the same card and the prefix compare still matches.
  const reserved = new Set<string>(parts.map((part) => part.toolCallId ?? ""));
// The slot lookups read only `toolCallId` and `_delta_index`, neither of which a seeded image or
  // source part carries: an absent id is exactly how such a part says "no slot claimed" to them, so
  // the array is handed over under the narrower type they ask for rather than every part pretending
  // to be a call.
  const slots = parts as unknown as StreamedToolCallPart[];

  const liveOutput = new Map<string, string>();
  let raw = seeded.raw;
  let reasoningOpen = seeded.reasoningOpen;

  /** The one place the reply grows, so a call's boundary is recorded at the character it happened at. */
  const grow = (kind: "text" | "reasoning", text: string): boolean => {
    if (!text) return false;
    let chunk = text;
    if (kind === "reasoning" && !reasoningOpen) chunk = `${THINK_OPEN_TAG}${text}`;
    else if (kind === "text" && reasoningOpen) chunk = `${THINK_CLOSE_TAG}${text}`;
    reasoningOpen = kind === "reasoning";
    raw += chunk;
    segmented.appendText(chunk);
    return true;
  };

  /** The card a frame names. A backend id is only the SPELLING a frame carries; the card answers to
   *  whatever id the run gave it, which for an id-less call is the deterministic `tool_call_<n>` the
   *  deltas minted. Every frame of one call has to land on one part, so a name that already maps to a
   *  card resolves through the map rather than matching on spelling and opening a second one. */
  const cardNamed = (backendId: string | undefined): number => {
    if (!backendId) return -1;
    const id = idsByBackendId.get(backendId);
    return findStreamedToolCallPartIndex(slots, id ?? backendId, undefined);
  };

  /** The card a frame that OPENS a call belongs to: by its id when it has one, else by the slot its
   *  id-less opening fragment drew. A minted card keeps that id, because the backend mints the same
   *  spelling for the same slot and its `tool_start` then reaches this card. */
  const partIdFor = (
    backendId: string | undefined,
    deltaIndex: number | undefined,
  ): string => {
    if (!backendId) return mintStreamedToolCallId(slots, deltaIndex, reserved);
    const existing = idsByBackendId.get(backendId);
    if (existing) return existing;
    const claimed = parts.find(
      (part) => String(part.toolCallId ?? "").split(":")[0] === backendId,
    );
    const id = claimed ? String(claimed.toolCallId) : `${backendId}:${parts.length}`;
    idsByBackendId.set(backendId, id);
    reserved.add(id);
    return id;
  };

  const patchPart = (
    id: string,
    patch: Record<string, unknown>,
    deltaIndex?: number,
  ): boolean => {
    const at = findStreamedToolCallPartIndex(slots, id || undefined, deltaIndex);
    if (at === -1) {
      parts.push({
        type: "tool-call",
        toolCallId: id,
        ...patch,
        textCursor: raw.length,
        ...(deltaIndex !== undefined ? { _delta_index: deltaIndex } : {}),
      });
      return true;
    }
    // A late id claims the card its id-less opening fragment opened, so the rename is the patch.
    parts[at] = { ...parts[at], ...(id ? { toolCallId: id } : {}), ...patch };
    return true;
  };

  const applyToolEvent = (event: Record<string, unknown>): boolean => {
    const type = event.type;
    // Transient store traffic has no part to write to: a status line, a diffusion frame, a
    // container id. It is dropped here on purpose, exactly as the live stream `continue`s past it.
    if (
      type !== "tool_start" &&
      type !== "tool_end" &&
      type !== "tool_args" &&
      type !== "tool_output"
    ) {
      return false;
    }
    const backendId =
      typeof event.tool_call_id === "string" ? event.tool_call_id : "";
    // Resolved through the map, so a frame naming `call_0` finds the card the id-less fragments drew
    // under `tool_call_0` instead of opening a second one.
    const existingIndex = cardNamed(backendId);
    if (type === "tool_output") {
      // Incremental stdout for a call that is still running. With no card there is nothing to
      // append to, which is also what the live path does.
      if (existingIndex === -1) return false;
      const id = String(parts[existingIndex].toolCallId);
      const text = typeof event.text === "string" ? event.text : "";
      if (!text) return false;
      liveOutput.set(id, (liveOutput.get(id) ?? "") + text);
      return true;
    }
    if (type === "tool_args") {
      // The model is still WRITING this call's arguments: a preview that `tool_start` replaces
      // authoritatively. Only an existing card is touched, so a fragment ahead of its call is mute.
      const part = existingIndex === -1 ? undefined : parts[existingIndex];
      const fragment = typeof event.text === "string" ? event.text : "";
      if (!part || !fragment) return false;
      const id = String(part.toolCallId ?? "");
      const argsText =
        String((part.argsText as string | undefined) ?? "") + fragment;
      let args: unknown = part.args;
      try {
        args = JSON.parse(argsText);
      } catch {
        args = { _raw: argsText };
      }
      parts[existingIndex] = { ...part, toolCallId: id, argsText, args };
      return true;
    }
    const id = partIdFor(backendId || undefined, undefined);
    if (type === "tool_start") {
      const at = cardNamed(backendId);
      const existing = at === -1 ? undefined : parts[at];
      const args = (event.arguments ?? {}) as Record<string, unknown>;
      const argsText =
        typeof event.arguments_text === "string" && event.arguments_text
          ? event.arguments_text
          : JSON.stringify(args);
      return patchPart(id, {
        toolName: event.tool_name,
        args: existing ? { ...((existing.args as object) ?? {}), ...args } : args,
        argsText,
        ...(event.provenance && typeof event.provenance === "object"
          ? { provenance: event.provenance }
          : {}),
      });
    }
    // tool_end: the call's result. A longer captured stream beats the model-visible result, which
    // is the live path's rule too, so a reopened card shows what actually ran rather than its tail.
    if (existingIndex === -1) return false;
    const part = parts[existingIndex];
    const streamed = liveOutput.get(String(part.toolCallId ?? id));
    const result = event.result;
    const full =
      streamed !== undefined && typeof result === "string"
        ? preferFullToolOutput(streamed, result)
        : undefined;
    return patchPart(id, {
      ...(full !== undefined ? { result: full } : result !== undefined
        ? { result }
        : {}),
    });
  };

  const applyToolCallDeltas = (calls: unknown): boolean => {
    if (!Array.isArray(calls) || calls.length === 0) return false;
    let changed = false;
    for (const entry of calls) {
      if (!entry || typeof entry !== "object") continue;
      const call = entry as {
        id?: string;
        index?: number;
        function?: { name?: unknown; arguments?: unknown };
      };
      const stableId =
        typeof call.id === "string" && call.id ? call.id : undefined;
      if (stableId) reserved.add(stableId);
      const index = typeof call.index === "number" ? call.index : undefined;
      const id = partIdFor(stableId, index);
      const at = findStreamedToolCallPartIndex(slots, id, index);
      const name =
        typeof call.function?.name === "string" ? call.function.name : "";
      const fragment =
        typeof call.function?.arguments === "string"
          ? (call.function.arguments as string)
          : "";
      if (at === -1) {
        let args: unknown = {};
        try {
          args = fragment ? JSON.parse(fragment) : {};
        } catch {
          args = { _raw: fragment };
        }
        changed =
          patchPart(id, { toolName: name, args, argsText: fragment }, index) ||
          changed;
        continue;
      }
      const existing = parts[at];
      const prevName = String(existing.toolName ?? "");
      // A name fragment continues the call's name; a different one opens the next call.
      const toolName = !name
        ? prevName
        : name.startsWith(prevName) || !prevName
          ? name
          : prevName || name;
      const prevText = String((existing.argsText as string | undefined) ?? "");
      // A fragment repeating what the card already holds is a resend, not more arguments.
      const argsText =
        fragment && fragment === prevText.slice(-fragment.length)
          ? prevText
          : prevText + fragment;
      let args: unknown = existing.args;
      if (argsText) {
        try {
          args = JSON.parse(argsText);
        } catch {
          args = { _raw: argsText };
        }
      }
      parts[at] = { ...existing, toolCallId: id, toolName, argsText, args };
      changed = true;
    }
    return changed;
  };

  return {
    applyChunk(chunk: unknown): boolean {
      const payload = chunk as
        | {
            _toolEvent?: Record<string, unknown>;
            choices?: Array<{
              delta?: {
                content?: unknown;
                reasoning_content?: unknown;
                reasoning_details?: unknown;
                tool_calls?: unknown;
              };
            }>;
          }
        | null
        | undefined;
      const toolEvent = payload?._toolEvent;
      if (toolEvent) return applyToolEvent(toolEvent);
      const delta = payload?.choices?.[0]?.delta;
      const details = Array.isArray(delta?.reasoning_details)
        ? delta.reasoning_details
            .map((part) =>
              part && typeof part === "object"
                ? String((part as { text?: unknown }).text ?? "")
                : "",
            )
            .join("")
        : "";
      const reasoning =
        (typeof delta?.reasoning_content === "string"
          ? delta.reasoning_content
          : "") + details;
      const text = extractDeltaText(delta?.content).text;
      // Reasoning first, then the visible text: a chunk carrying both must close the block before
      // the answer, which is exactly what `grow` does with the tag state it carries.
      let changed = grow("reasoning", reasoning);
      changed = grow("text", text) || changed;
      return applyToolCallDeltas(delta?.tool_calls) || changed;
    },
    content(): ContentPart[] {
      // The same assembly the live stream publishes with: cut the reply at each call's offset, so
      // a pill sits where it was called and only the last run is still growing.
      const positioned = parts.map((part, index) => ({
        part,
        index,
        cursor: Math.min(
          Math.max(Number(part.textCursor ?? 0), 0),
          raw.length,
        ),
      }));
      const boundaries: number[] = [];
      for (const item of positioned) {
        if (boundaries[boundaries.length - 1] !== item.cursor) {
          boundaries.push(item.cursor);
        }
      }
      // Rebuilt from the whole reply whenever a call moved a boundary, which happens once per call.
      const runs = segmented.runs(raw, boundaries) as ContentPart[][];
      const assembled: ContentPart[] = [];
      let next = 0;
      for (let index = 0; index < boundaries.length; index += 1) {
        assembled.push(...runs[index]);
        while (
          next < 
positioned.length &&
          positioned[next].cursor === boundaries[index]
        ) {
          assembled.push(positioned[next].part);
          next += 1;
        }
      }
      assembled.push(...(runs[boundaries.length] ?? []));
      return assembled;
    },
    rawText(): string {
      return raw;
    },
  };
}
