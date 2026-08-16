// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The `components` maps handed to assistant-ui primitives must be referentially
 * stable across renders.
 *
 * `MessagePrimitivePartByIndex` is memoized, and its comparator checks the
 * `components` fields one at a time rather than comparing the object as a
 * whole. Every field the thread passes is a module-level component and so
 * compares equal across renders, with one exception: `tools` was an inline
 * object literal, which meant a fresh identity on every render. That single
 * mismatch failed the comparator and re-rendered every part of the message, so
 * a streaming reply rebuilt all of its already-finished parts on each chunk and
 * the cost grew with the length of the reply.
 *
 * Putting the maps at module scope is a one-line fix that is invisible in
 * review and trivial to undo by accident, which is what this file is for.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const THREAD_SOURCE = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);

test("the assistant part components are not an inline object literal", () => {
  // `components={{` is the exact shape of the regression: an object literal
  // built inline in JSX, which is a new object on every render.
  //
  // Scoped to MessagePrimitive.Parts. ThreadPrimitive.Messages has the same
  // shape but hoisting cannot fix it: the library turns a components map into
  // `() => <ThreadMessageComponent components={...} />`, so the per-message
  // element always carries props and never reaches the propless bail-out in
  // RenderChildrenWithAccessor. #9042 moves that call to the children form,
  // which does reach it, and asserts the opposite of what a hoist would.
  const inline = [
    ...THREAD_SOURCE.matchAll(/<MessagePrimitive\.Parts[^>]*components=\{\{/gs),
  ];
  assert.equal(
    inline.length,
    0,
    `${inline.length} inline components literal(s) in thread.tsx. Each one hands ` +
      "the primitive a fresh object every render and defeats the memo on " +
      "MessagePrimitivePartByIndex. Hoist it to module scope, or wrap it in " +
      "useMemo if it genuinely depends on props or state.",
  );
});

test("the assistant part components are a single module-scope object", () => {
  assert.match(
    THREAD_SOURCE,
    /^const ASSISTANT_PART_COMPONENTS = \{/m,
    "ASSISTANT_PART_COMPONENTS must be declared at module scope, so that every " +
      "render passes the identical object",
  );
  assert.match(
    THREAD_SOURCE,
    /<MessagePrimitive\.Parts components=\{ASSISTANT_PART_COMPONENTS\} \/>/,
    "MessagePrimitive.Parts must be given the hoisted constant by name",
  );
});

test("the upstream memo still compares components.tools by identity", () => {
  // This fix is only worth anything while the comparator behaves this way. If
  // an assistant-ui upgrade starts comparing the map structurally, or stops
  // reading `tools`, this test fails and says to re-measure rather than
  // silently leaving a hoist that no longer buys anything.
  const comparator = readFileSync(
    new URL(
      "../node_modules/@assistant-ui/core/dist/react/primitives/message/MessageParts.js",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    comparator,
    /prev\.components\?\.tools === next\.components\?\.tools/,
    "assistant-ui no longer compares components.tools by identity. The reason " +
      "the maps in thread.tsx are hoisted has changed; re-measure before " +
      "trusting the comment there.",
  );
});
