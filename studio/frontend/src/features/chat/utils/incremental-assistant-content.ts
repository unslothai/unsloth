// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

import {
  appendReasoningPart,
  appendTextPart,
} from "./parse-assistant-content.ts";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

/** Length of the longest suffix of `text` that is a proper prefix of `tag`. Those characters
 *  cannot be classified yet: `<thi` at the end of what has arrived is plain text if the reply
 *  stops there and the opening of a reasoning block if `nk>` arrives next, so they are held
 *  back and reclassified once the next arrival settles them. */
function heldBackLength(text: string, tag: string): number {
  const most = Math.min(tag.length - 1, text.length);
  for (let size = most; size > 0; size -= 1) {
    if (text.startsWith(tag.slice(0, size), text.length - size)) {
      return size;
    }
  }
  return 0;
}

/** One run of text between two tool-call boundaries, parsed as it arrives. Reproduces
 *  `parseAssistantContent` over the run exactly, including that it coalesces adjacent parts of
 *  the same kind and that a run ending inside a `<think>` block yields an unclosed reasoning
 *  part. */
class ParsedRun {
  readonly parts: ContentPart[] = [];
  private insideThink = false;
  private held = "";

  append(delta: string): void {
    if (!delta) {
      return;
    }
    const work = this.held ? this.held + delta : delta;
    this.held = "";
    let cursor = 0;
    for (;;) {
      const tag = this.insideThink ? THINK_CLOSE_TAG : THINK_OPEN_TAG;
      const at = work.indexOf(tag, cursor);
      if (at === -1) {
        break;
      }
      this.commit(work.slice(cursor, at));
      cursor = at + tag.length;
      this.insideThink = !this.insideThink;
    }
    const rest = cursor === 0 ? work : work.slice(cursor);
    const hold = heldBackLength(
      rest,
      this.insideThink ? THINK_CLOSE_TAG : THINK_OPEN_TAG,
    );
    if (hold > 0) {
      this.held = rest.slice(rest.length - hold);
      this.commit(rest.slice(0, rest.length - hold));
    } else {
      this.commit(rest);
    }
  }

  /** The run's parts. The held-back characters are folded in as whatever `parseAssistantContent`
   *  would call them if the reply stopped here, so the result is that function's output for
   *  every state the run passes through. */
  view(): ContentPart[] {
    // A copy even when nothing is held back. The retained parts are the state the next arrival
    // extends, so handing the array itself out would let a caller rewrite what has been parsed.
    const out = this.parts.slice();
    if (!this.held) {
      return out;
    }
    if (this.insideThink) {
      appendReasoningPart(out, this.held);
    } else {
      appendTextPart(out, this.held);
    }
    return out;
  }

  private commit(text: string): void {
    if (this.insideThink) {
      appendReasoningPart(this.parts, text);
    } else {
      appendTextPart(this.parts, text);
    }
  }
}

export type SegmentedAssistantText = {
  /** Take the characters an arrival added to the reply. */
  appendText(delta: string): void;
  /** Parsed parts for the run before each boundary, then the run after the last one, so the result
   *  always has `boundaries.length + 1` entries. `rawText` is only read when the retained state
   *  cannot be trusted for it; on the streaming path that is the boundaries changing, once per
   *  tool call rather than once per arrival. */
  runs(rawText: string, boundaries: readonly number[]): ContentPart[][];
};

function sameBoundaries(
  left: readonly number[],
  right: readonly number[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

/** Parse the accumulated reply into content parts without rereading it. The adapter appends to
 *  one string per arrival and then parses the whole thing, which is O(reply) per arrival for
 *  two reasons: the parse itself, and that `text += delta` builds a cons string whose first
 *  read flattens it. This keeps the parse of everything already seen and extends it with the
 *  delta. The retained state describes an append-only reply; anything else shows up as a
 *  length or boundary mismatch and rebuilds from `rawText`, at one full parse. */
export function createSegmentedAssistantText({
  trustAppends = true,
}: { trustAppends?: boolean } = {}): SegmentedAssistantText {
  let runs: ParsedRun[] = [new ParsedRun()];
  let boundaries: number[] = [];
  let length = 0;

  const rebuild = (
    rawText: string,
    nextBoundaries: readonly number[],
  ): void => {
    runs = [];
    boundaries = [...nextBoundaries];
    length = rawText.length;
    let from = 0;
    for (const boundary of boundaries) {
      const run = new ParsedRun();
      run.append(rawText.slice(from, boundary));
      runs.push(run);
      from = boundary;
    }
    const last = new ParsedRun();
    last.append(rawText.slice(from));
    runs.push(last);
  };

  return {
    appendText(delta: string): void {
      if (!delta) {
        return;
      }
      runs[runs.length - 1].append(delta);
      length += delta.length;
    },
    runs(rawText: string, nextBoundaries: readonly number[]): ContentPart[][] {
      if (
        !trustAppends ||
        rawText.length !== length ||
        !sameBoundaries(boundaries, nextBoundaries)
      ) {
        rebuild(rawText, nextBoundaries);
      }
      return runs.map((run) => run.view());
    },
  };
}
