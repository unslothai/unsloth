// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `components` maps must be referentially stable: MessagePrimitivePartByIndex compares
 * `components.tools` by identity, so an inline literal rebuilds every finished part.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THREAD_SOURCE = readSrc("components/assistant-ui/thread.tsx");

test("the assistant part components are not an inline object literal", () => {
  // Scoped to MessagePrimitive.Parts: ThreadPrimitive.Messages always carries props, so a hoist
  // cannot fix it there; that one uses the children form instead.
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
  // Pins upstream comparator behaviour; if it changes, re-measure whether the hoist still helps.
  // A named path variable distinguishes a missing file from a changed comparator.
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
