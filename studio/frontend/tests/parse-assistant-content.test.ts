// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";

const partsOfType = (raw: string, type: string): string =>
  parseAssistantContent(raw)
    .filter((part) => part.type === type)
    .map((part) => (part as { text: string }).text)
    .join("");

// One case per distinct verdict the literal-close classifier has to reach. The
// unescaped quoted mention and the unequal delimiter runs are dropped here: the
// python contract test drives the same parser through the same two shapes
// (`quoted_literal`, `unequal_runs`) and asserts the whole part list.
const cases: [string, string, string, string][] = [
  // A serialized quotation escapes both quotes, so both are excluded from the
  // parity count and the mention read as the structural close: the drawer shut
  // on the first tag and the rest of the thought was rendered as the answer
  // (#7334). The backend extractor has the same escaped-pair case
  // (_is_literal_think_close).
  [
    "a symmetric escaped pair stays inside the reasoning drawer",
    '<think>serialized \\"</think>\\" still reasoning</think>answer',
    'serialized \\"</think>\\" still reasoning',
    "answer",
  ],
  // The escaped-pair case must not swallow real closes: the checks that already
  // resolve a quoted tag as structural still win.
  ["a bare close tag is still structural", "<think>draft</think>answer", "draft", "answer"],
  [
    "an escaped closing quote running into a word still opens the answer",
    '<think>a \\"</think>\\"The answer is 42.',
    'a \\"',
    '\\"The answer is 42.',
  ],
];

for (const [name, raw, wantReasoning, wantAnswer] of cases) {
  test(name, () => {
    assert.equal(partsOfType(raw, "reasoning"), wantReasoning);
    assert.equal(partsOfType(raw, "text"), wantAnswer);
  });
}
