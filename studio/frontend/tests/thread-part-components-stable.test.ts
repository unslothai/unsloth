// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The `components` maps handed to assistant-ui primitives must be referentially
 * stable: the memo comparator in `MessagePrimitivePartByIndex` checks
 * `components.tools` by identity, so an inline literal failed it and rebuilt
 * every already-finished part of a streaming reply on each chunk.
 *
 * The fix is one line at module scope, invisible in review and easy to undo by
 * accident, which is what this file guards.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const THREAD_SOURCE = readFileSync(
  new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
  "utf8",
);

test("the assistant part components are not an inline object literal", () => {
  // `components={{` is the regression's exact shape: an inline JSX literal, so
  // a new object every render.
  //
  // Scoped to MessagePrimitive.Parts. ThreadPrimitive.Messages looks the same
  // but hoisting cannot fix it: the library wraps a components map as
  // `() => <ThreadMessageComponent components={...} />`, so the element always
  // carries props and never hits the propless bail-out in
  // RenderChildrenWithAccessor. #9042 moves that call to the children form,
  // which does, and asserts the opposite of what a hoist would.
  const inline = [
    ...THREAD_SOURCE.matchAll(/<MessagePrimitive\.Parts[^>]*components=\{\{/gs),
  ];
  assert.equal(
    inline.length,
    0,
    `${inline.length} inline components literal(s) in thread.tsx. Each one hands the primitive a fresh object every render and defeats the memo on MessagePrimitivePartByIndex. Hoist it to module scope, or wrap it in useMemo if it genuinely depends on props or state.`,
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
  // Pins upstream behaviour, so it passes with or without the hoist. The fix is
  // only worth anything while the comparator works this way: if an upgrade
  // compares the map structurally or stops reading `tools`, this fails and says
  // to re-measure instead of leaving a hoist that buys nothing.
  //
  // Read through a named variable so a missing file is distinguishable from a
  // changed comparator; a half-installed node_modules otherwise surfaces as a
  // bare ENOENT that reads like a regression here.
  const comparatorPath = new URL(
    "../node_modules/@assistant-ui/core/dist/react/primitives/message/MessageParts.js",
    import.meta.url,
  );
  let comparator: string;
  try {
    comparator = readFileSync(comparatorPath, "utf8");
  } catch (cause) {
    throw new Error(
      `cannot read the assistant-ui comparator at ${comparatorPath.pathname}. This is an install problem, not a failure of the code under test: reinstall node_modules and re-run before reading anything into it.`,
      { cause },
    );
  }
  assert.match(
    comparator,
    /prev\.components\?\.tools === next\.components\?\.tools/,
    "assistant-ui no longer compares components.tools by identity. The reason " +
      "the maps in thread.tsx are hoisted has changed; re-measure before " +
      "trusting the comment there.",
  );
});
