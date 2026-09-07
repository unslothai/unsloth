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
import {
  countReasoningGroups,
  createReasoningDurationTracker,
  lastReasoningGroupTextLength,
} from "./reasoning-duration";
import { preferFullToolOutput } from "./tool-output-preference";
// The frame -> part shaping the live stream applies to a tool result, applied HERE too. A replay that copies
// `event.result` verbatim renders the wire's marker (`__IMAGES__:...`) as content; shaped, a reopened chart card is
// the same object a watched one was.
import { shapeToolResult } from "./tool-result-shape";
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

/** The reply as storage holds it, back in the tagged-string form the parser reads: a stored part is
 *  text with its kind erased, and the tags are what put the kind back. A block left open by the last
 *  stored part stays open here, so the next frame continues it instead of opening a second one. */
function seededReplayState(content: unknown): {
  raw: string;
  parts: PositionedReplayPart[];
  // Whether what storage holds ends inside a thought the replay itself opened with a tag: a stored
  // `reasoning` part carries no tag of its own, so the one re-created here is ours, and ours is the one
  // a following answer chunk has to close. A block a model opened with a tag it carried INSIDE
  // `delta.content` is not ours, and closing that one cuts the thought at the first answer delta.
  reasoningOpen: boolean;
} {
  if (typeof content === "string") return { raw: content, parts: [], reasoningOpen: false };
  if (!Array.isArray(content)) return { raw: "", parts: [], reasoningOpen: false };
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
      // A part that is neither prose nor thought (a call, an image) lands at a boundary, and the live
      // stream had closed the block before it recorded that call. Closing here keeps the close tag at
      // the end of the thought run, where it belongs, instead of at the head of whatever follows it.
      if (reasoningOpen) {
        raw += THINK_CLOSE_TAG;
        reasoningOpen = false;
      }
      parts.push({ ...part, textCursor: raw.length });
    }
  }
  return { raw, parts, reasoningOpen };
}

export type RecoveryReplay = {
  /** Fold one replayed chunk event. `at` is the instant the run WROTE that event, and it is what every
   *  reasoning group on this path is timed against: a follower folds frames written minutes before it
   *  attached, so its own clock would measure the replay, not the thinking. Returns whether it changed
   *  the reply, which is what decides whether there is anything worth publishing. */
  applyChunk(chunk: unknown, at?: number): boolean;
  /** The reply as parts: text and reasoning runs cut at the tool-call offsets, tools interleaved. */
  content(): ContentPart[];
  /** The tagged string, for the prefix comparison a recovery publish makes against the view. */
  rawText(): string;
  /** What each reasoning group has cost so far, in the shape `resolveReasoningGroupDuration` reads.
   *  Empty while nothing has been measured, so a publish cannot overwrite a stored value with nothing. */
  durations(): Record<string, unknown>;
  /** A server-measured summary for the group that most recently opened. Authoritative: it overwrites
   *  that group's slot instead of shifting the groups after it along by one. */
  recordServerDuration(reasoningMs: unknown): boolean;
};

/** Build the follower's copy of a run's reply: `seed` is what storage already holds (the partial a
 *  reload restored, or the request's assistant prefill) and the events replay on top of it.
 *  `seedDurations` is the same message's already-measured group durations: the frames before this
 *  follower's cursor were never folded, so what a previous tab timed is all anyone will ever know about
 *  those groups, and a group this reader does watch has to land in the NEXT slot. */
export function createRecoveryReplay(
  seed: unknown,
  seedDurations?: readonly number[],
  options?: { sandboxSessionId?: string },
): RecoveryReplay {
  // Read lazily, at the frame that needs it: a follower builds its accumulator before the stored run is fetched,
  // then fills this in. A replayed python/terminal card that loses WHICH session ran names a folder from the
  // reader's current scope instead of the run's.
  const sandboxSessionId = () => options?.sandboxSessionId;
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
  // The live stream times a thought against `Date.now()` between two chunks. A replay has no such clock:
  // every frame it folds arrived before this tab existed, so timing them against now collapses every
  // group to "0 seconds". The clock is therefore the frames themselves -- `applyChunk` sets it to each
  // event's own `createdAt` before folding it -- and the groups are counted on the SAME parts array the
  // renderer indexes its `reasoningDurations` by, which is what keeps index N of one the same thought as
  // index N of the other.
  let clockAt = 0;
  const groupTiming = createReasoningDurationTracker(
    () => clockAt,
    seedDurations ? { durations: seedDurations } : undefined,
  );

  /** The one place the reply grows, so a call's boundary is recorded at the character it happened at.
   *  Which tag the next chunk owes is asked of the run being written rather than tracked across the
   *  whole reply: a part landing in between starts a NEW run, which begins outside the block the run
   *  before it ended inside, and a flag carried over from before that boundary keeps closing a block
   *  that is already closed -- the stray `< /think>` then reads as literal text in the part after it.
   *  The live stream cannot hit this (it appends tags into the same string it parses); a replay seeded
   *  from parts can, because the seed has to re-create the tags the parts no longer carry. */
  // Whether the block the run being written sits inside is one THIS replay opened. The live stream
  // closes a thought only when it opened one (`reasoningContentOpen`, set only for a
  // `reasoning_content`/`reasoning_details` frame); a model that carries its own tags inside
  // `delta.content` owns its own block, and its text is appended verbatim so its tags do the
  // classifying. Closing that one instead cut the thought at the first content delta of the answer and
  // left the model's own close tag behind as literal text in the answer part.
  let replayOwnsOpenBlock = seeded.reasoningOpen;
  // A frame that carries its OWN tags classifies itself, so synthesizing a second pair around it is
  // what leaked: only a tagless chunk owes this replay a tag of its own.
  const carriesOwnTags = (text: string): boolean =>
    text.includes(THINK_OPEN_TAG) || text.includes(THINK_CLOSE_TAG);

  const grow = (kind: "text" | "reasoning", text: string): boolean => {
    if (!text) return false;
    const inside = segmented.insideThink();
    let chunk = text;
    if (carriesOwnTags(text)) {
      // Verbatim: its own tag does the classifying, exactly as the live stream appends it. A chunk that
      // carries an OPEN tag of its own is the exception: `extractDeltaText` wraps a structured
      // `thinking` part in a full pair, and an open tag inside a block this replay already opened is not
      // a no-op for the parser -- it survives as literal text inside the thought. A chunk carrying only
      // a CLOSE is NOT: that tag ends the block either way, which is what the model's own close does to
      // a block the seed re-created.
      if (inside && replayOwnsOpenBlock && text.includes(THINK_OPEN_TAG)) {
        chunk = `${THINK_CLOSE_TAG}${text}`;
        replayOwnsOpenBlock = false;
      }
    } else if (kind === "reasoning" && !inside) {
      chunk = `${THINK_OPEN_TAG}${text}`;
      replayOwnsOpenBlock = true;
    } else if (kind === "text" && inside && replayOwnsOpenBlock) {
      chunk = `${THINK_CLOSE_TAG}${text}`;
      replayOwnsOpenBlock = false;
    }
    raw += chunk;
    segmented.appendText(chunk);
    // A close tag the MODEL carried closed the block it opened; nothing is left to close on its behalf.
    if (replayOwnsOpenBlock && !segmented.insideThink()) replayOwnsOpenBlock = false;
    return true;
  };

  /** Close an open thought at a boundary, so the tag lands at the end of the thought run instead of at
   *  the head of whatever part comes next. Only a block this replay opened: a boundary inside a block a
   *  model's own tag opened is the model's business, and closing it strays a tag into the next part. */
  const closeThought = (): void => {
    if (!segmented.insideThink() || !replayOwnsOpenBlock) return;
    raw += THINK_CLOSE_TAG;
    segmented.appendText(THINK_CLOSE_TAG);
    replayOwnsOpenBlock = false;
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
      // The live stream closes an open thought before it records a call, so the boundary sits AFTER
      // the close tag, not inside the block.
      closeThought();
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
    // A longer captured stream beats the model-visible result, which is the live path's rule too.
    const fuller =
      streamed !== undefined && typeof result === "string"
        ? preferFullToolOutput(streamed, result)
        : undefined;
    const chosen = fuller !== undefined ? fuller : result;
    // And what lands on the card is what the live stream would have written there: a marker split into images and
    // files, an MCP envelope unwrapped, an inline base64 image kept off `result`, all under the session that ran.
    const shaped =
      chosen === undefined
        ? undefined
        : shapeToolResult({
            toolName: typeof part.toolName === "string" ? part.toolName : undefined,
            raw: chosen,
            event,
            sandboxSessionId: sandboxSessionId(),
          });
    return patchPart(id, shaped !== undefined ? { result: shaped } : {});
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

  /** The reply as parts: the same assembly the live stream publishes with, cut at each call's offset, so
   *  a pill sits where it was called and only the last run is still growing. */
  const assembledParts = (): ContentPart[] => {
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
  };

  /** The group bookkeeping the live adapter does per chunk, run on the frame's own timestamp. The same
   *  three calls in the same order, and the same rule that the timer stops the moment the block the
   *  chunk left is closed -- which is why a call landing on the reply closes the thought that ran before
   *  it, exactly as `closeReasoningContent` does on the wire. */
  const timeGroups = (): void => {
    const built = assembledParts();
    const groups = countReasoningGroups(built);
    if (groups > groupTiming.groupCount) {
      groupTiming.startGroup(groups - 1);
    }
    if (groups > 0) {
      groupTiming.resumeGroup(groups - 1, lastReasoningGroupTextLength(built));
    }
    if (groupTiming.hasActiveGroup && !segmented.insideThink()) {
      groupTiming.finishGroup();
    }
  };

  return {
    applyChunk(chunk: unknown, at?: number): boolean {
      // Set before the fold, so a group opens at the instant its first frame arrived and not at the
      // instant this tab got round to reading it.
      if (typeof at === "number" && Number.isFinite(at)) clockAt = at;
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
      if (toolEvent) {
        const changed = applyToolEvent(toolEvent);
        // A call landing on the reply is a boundary: it closed the thought before it, and that thought's
        // duration ends at this frame, not at the next one.
        if (changed) timeGroups();
        return changed;
      }
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
      changed = applyToolCallDeltas(delta?.tool_calls) || changed;
      if (changed) timeGroups();
      return changed;
    },
    content: assembledParts,
    durations: () => groupTiming.metadata(),
    recordServerDuration: (reasoningMs: unknown) =>
      groupTiming.recordServerDuration(reasoningMs),
    rawText(): string {
      return raw;
    },
  };
}
