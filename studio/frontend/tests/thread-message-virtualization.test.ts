// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The windowed message list, in the two halves that can be asserted without a DOM.
//
// There is no jsdom here and a virtualizer needs a real scroll element and real layout, so the
// mechanism itself is not mountable. What IS assertable, and is what actually goes wrong:
//   - the keying property, because keying on the index is the failure that parked the block-level
//     windowing work and is invisible until something prepends;
//   - that the flag is off and the flag-off path is still the element it always was, which is the
//     whole claim of this stage.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { THREAD_MESSAGE_VIRTUALIZATION_ENABLED } from "../src/components/assistant-ui/thread-feature-flags.ts";
import {
  THREAD_MESSAGE_ANCHORING,
  THREAD_MESSAGE_ESTIMATE_SIZE_PX,
  THREAD_MESSAGE_OVERSCAN,
  THREAD_MESSAGE_SCROLL_END_THRESHOLD_PX,
  type VirtualizedMessage,
  messageKeyAt,
  scrollMarginFor,
} from "../src/components/assistant-ui/thread-message-virtualizer-policy.ts";
import { cn } from "../src/lib/utils.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const read = (relative: string): string =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const sourceFile = (relative: string): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
};

const THREAD = "../src/components/assistant-ui/thread.tsx";
const AUTOSCROLL =
  "../src/components/assistant-ui/use-intent-aware-autoscroll.tsx";

const messages = (...ids: string[]): VirtualizedMessage[] =>
  ids.map((id) => ({ id }));

// ---------------------------------------------------------------------------
// The flag, and the path it leaves alone
// ---------------------------------------------------------------------------

test("the windowed message list is off", () => {
  // The point of the flag is that the mechanism is exercisable and revertible by config before
  // anything user-visible changes. Turning it on is a separate, measured step.
  assert.equal(THREAD_MESSAGE_VIRTUALIZATION_ENABLED, false);
});

test("the unwindowed path still renders the shared propless slot", () => {
  const source = read(THREAD);
  // Byte-for-byte the element the thread has always rendered, indentation aside. If this ever needs
  // updating, the flag-off DOM has changed and the claim of this stage is void.
  assert.match(
    source,
    /<ThreadPrimitive\.Messages>\s*\{renderThreadMessage\}\s*<\/ThreadPrimitive\.Messages>/,
  );
});

test("the windowed list is the only thing behind the flag in the message slot", () => {
  const source = read(THREAD);
  // The conditional has to be an expression around the two lists, not a mutation of the surrounding
  // JSX: anything else and "off is unchanged" stops being checkable by reading the ternary.
  assert.match(
    source,
    /\{THREAD_MESSAGE_VIRTUALIZATION_ENABLED \? \(\s*<VirtualizedThreadMessages/,
  );
});

test("the anchoring class is applied only under the flag", () => {
  const source = read(THREAD);
  // `cn` drops falsy entries, so with the flag off the viewport's className string is the one it
  // already had. Unconditional `overflow-anchor: none` would change the unwindowed path, which the
  // autoscroll hook is written against; see the rule in index.css.
  assert.match(
    source,
    /THREAD_MESSAGE_VIRTUALIZATION_ENABLED &&\s*"aui-stream-viewport-virtualized"/,
  );
  assert.doesNotMatch(
    read("../src/index.css"),
    /\.aui-stream-viewport\s*\{[^}]*overflow-anchor/,
  );
});

test("with the flag off the viewport's class string is the one it already had", () => {
  // The class list is built with `cn`, and `false && "..."` is dropped by clsx before twMerge sees
  // it. Asserted against the real `cn` and the real base class string lifted out of the source,
  // rather than argued from how clsx ought to behave.
  const source = read(THREAD);
  const base = /"(aui-thread-viewport aui-stream-viewport[^"]*)"/.exec(
    source,
  )?.[1];
  assert.ok(base, "viewport base class string not found in the thread");

  const withFlagOff = cn(
    base,
    THREAD_MESSAGE_VIRTUALIZATION_ENABLED && "aui-stream-viewport-virtualized",
    "pt-4",
  );
  assert.equal(withFlagOff, cn(base, "pt-4"));
  assert.doesNotMatch(withFlagOff, /aui-stream-viewport-virtualized/);
});

test("the message components map is hoisted, not built per render", () => {
  const source = sourceFile(THREAD);
  let componentsProp: string | null = null;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "VirtualizedThreadMessages") {
      for (const property of opening.attributes.properties) {
        if (
          ts.isJsxAttribute(property) &&
          property.name.getText() === "components"
        ) {
          componentsProp = property.initializer?.getText() ?? null;
        }
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);

  assert.ok(
    componentsProp,
    "<VirtualizedThreadMessages> has no components prop",
  );
  // An inline object literal would allocate a fresh map every render, and MessageByIndex is
  // memoized on the identity of each component in it, so every mounted message body would
  // re-render on every Thread render. Same reason renderThreadMessage is hoisted.
  assert.doesNotMatch(componentsProp, /\{\s*\{/);
  assert.match(componentsProp, /^\{virtualizedThreadMessageComponents\}$/);
});

// ---------------------------------------------------------------------------
// Keying: the property that breaks silently
// ---------------------------------------------------------------------------

test("a message is keyed by its id", () => {
  const list = messages("m-a", "m-b", "m-c");
  assert.equal(messageKeyAt(list, 0), "m-a");
  assert.equal(messageKeyAt(list, 1), "m-b");
  assert.equal(messageKeyAt(list, 2), "m-c");
});

test("prepending leaves every existing message's key alone", () => {
  const before = messages("m-a", "m-b", "m-c");
  const keysBefore = new Map(
    before.map((message, index) => [message.id, messageKeyAt(before, index)]),
  );

  // History paging, or a branch switch that reveals earlier turns: every old message moves to a new
  // index. An index key would rename all three, dropping the virtualizer's measurement cache and
  // remounting every message below the insertion point.
  const after = messages("m-x", "m-y", "m-a", "m-b", "m-c");
  for (const [id, key] of keysBefore) {
    const index = after.findIndex((message) => message.id === id);
    assert.equal(messageKeyAt(after, index), key);
  }
});

test("keys are not positional", () => {
  const list = messages("m-a", "m-b", "m-c");
  // Stated directly, because the prepend test above would still pass if keys happened to be
  // stringified indices in a list whose ids were also "0", "1", "2".
  for (let index = 0; index < list.length; index++) {
    assert.notEqual(messageKeyAt(list, index), String(index));
  }
});

test("keys are unique across the thread", () => {
  const list = messages("m-a", "m-b", "m-c", "m-d");
  const keys = list.map((_, index) => messageKeyAt(list, index));
  assert.equal(new Set(keys).size, keys.length);
});

test("an empty thread has no keys to resolve", () => {
  const list = messages();
  assert.equal(list.length, 0);
  // Reached only if count and the array disagree for a render. It must not collide with a real id,
  // so it is not the bare index.
  assert.equal(messageKeyAt(list, 0), "aui-unresolved-message-0");
  assert.notEqual(messageKeyAt(list, 0), "0");
});

test("a single-message thread keys that one message", () => {
  const list = messages("m-only");
  assert.equal(messageKeyAt(list, 0), "m-only");
  assert.equal(messageKeyAt(list, 1), "aui-unresolved-message-1");
});

test("a long thread keys every message by id", () => {
  const list = messages(...Array.from({ length: 5000 }, (_, i) => `m-${i}`));
  assert.equal(messageKeyAt(list, 0), "m-0");
  assert.equal(messageKeyAt(list, 2499), "m-2499");
  assert.equal(messageKeyAt(list, 4999), "m-4999");
  assert.equal(new Set(list.map((_, i) => messageKeyAt(list, i))).size, 5000);
});

// ---------------------------------------------------------------------------
// Sizing policy
// ---------------------------------------------------------------------------

test("the size estimate sits at the low end of the measured range", () => {
  // Measured per-message heights: 574.2 / 566.2 / 564.7 px at 1440x900 across three thread sizes,
  // 536.4 at 1280 wide, 463.1 on the reduced fixture, 461.2 over a first-paint window. Under-
  // estimating makes the virtualizer render more items than it needs; over-estimating paints a
  // blank gap. The failure modes are not symmetric.
  assert.ok(THREAD_MESSAGE_ESTIMATE_SIZE_PX <= 463);
  assert.ok(THREAD_MESSAGE_ESTIMATE_SIZE_PX >= 300);
});

test("the overscan buffers several viewports", () => {
  // At the estimated size, on the 900px-tall viewport the benchmarks use.
  const bufferPx = THREAD_MESSAGE_OVERSCAN * THREAD_MESSAGE_ESTIMATE_SIZE_PX;
  assert.ok(bufferPx >= 3 * 900);
});

test("the end threshold matches the hook's re-attach distance", () => {
  // The virtualizer follows an appended message exactly when isAtEnd(scrollEndThreshold) holds, and
  // the hook re-attaches a scrolled-up user at RE_ATTACH_THRESHOLD_PX. Different numbers means the
  // two disagree about whether the user is following the stream. Read from the source so the pair
  // cannot drift apart unnoticed.
  const match = /RE_ATTACH_THRESHOLD_PX = (\d+)/.exec(read(AUTOSCROLL));
  assert.ok(match, "RE_ATTACH_THRESHOLD_PX not found in the autoscroll hook");
  assert.equal(THREAD_MESSAGE_SCROLL_END_THRESHOLD_PX, Number(match[1]));
});

test("the list anchors to the end and follows appends", () => {
  // virtual-core defaults are anchorTo "start" and followOnAppend false, which is a document, not a
  // chat. Both need >= 3.16.1 to exist at all.
  assert.equal(THREAD_MESSAGE_ANCHORING.anchorTo, "end");
  assert.equal(THREAD_MESSAGE_ANCHORING.followOnAppend, true);
  assert.equal(
    THREAD_MESSAGE_ANCHORING.scrollEndThreshold,
    THREAD_MESSAGE_SCROLL_END_THRESHOLD_PX,
  );
});

// ---------------------------------------------------------------------------
// Scroll margin
// ---------------------------------------------------------------------------

test("the scroll margin is the list's offset inside the scroll element", () => {
  // Viewport top at 100 on screen, scrolled down 500, list starting 260 below the viewport top:
  // the list begins 660px into the scrollable content.
  assert.equal(scrollMarginFor(360, 100, 500), 760);
  // Unscrolled, list flush with the top of the viewport.
  assert.equal(scrollMarginFor(100, 100, 0), 0);
});

test("the scroll margin never goes negative", () => {
  // A thread switch can measure the container while it is detached or mid-layout. A negative margin
  // would push the first message off the top of the list.
  assert.equal(scrollMarginFor(0, 100, 0), 0);
  assert.equal(scrollMarginFor(-5000, 100, 0), 0);
});
