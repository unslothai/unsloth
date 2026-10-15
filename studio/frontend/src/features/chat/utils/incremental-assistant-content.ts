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

/** Trailing chars that may start `tag` are held back until the next arrival settles them. */
function heldBackLength(text: string, tag: string): number {
  const most = Math.min(tag.length - 1, text.length);
  for (let size = most; size > 0; size -= 1) {
    if (text.startsWith(tag.slice(0, size), text.length - size)) {
      return size;
    }
  }
  return 0;
}

/** Must reproduce `parseAssistantContent` over the run exactly. */
class ParsedRun {
  readonly parts: ContentPart[] = [];
  private insideThink = false;
  private held = "";
  private readonly parseThink: boolean;

  constructor(parseThink: boolean) {
    this.parseThink = parseThink;
  }

  append(delta: string): void {
    if (!delta) {
      return;
    }
    if (!this.parseThink) {
      appendTextPart(this.parts, delta);
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

  view(): ContentPart[] {
    // Copy: the retained parts are state the next arrival extends.
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
  appendText(delta: string): void;
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

/** Incremental parse to avoid O(reply) per arrival; non-append changes rebuild from `rawText`. */
export function createSegmentedAssistantText({
  trustAppends = true,
  parseThink = true,
}: {
  trustAppends?: boolean;
  parseThink?: boolean;
} = {}): SegmentedAssistantText {
  let runs: ParsedRun[] = [new ParsedRun(parseThink)];
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
      const run = new ParsedRun(parseThink);
      run.append(rawText.slice(from, boundary));
      runs.push(run);
      from = boundary;
    }
    const last = new ParsedRun(parseThink);
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
