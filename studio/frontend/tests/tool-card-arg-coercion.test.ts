// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// No DOM renderer here and the cards pull in React, so assert the coercion in
// the source the way artifact-frame-network-access.test.ts does.
const read = (relative: string): string =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const CARDS: ReadonlyArray<readonly [string, string]> = [
  ["../src/components/assistant-ui/tool-ui-python.tsx", "code"],
  ["../src/components/assistant-ui/tool-ui-terminal.tsx", "command"],
  ["../src/components/assistant-ui/tool-ui-knowledge-base.tsx", "query"],
  ["../src/components/assistant-ui/tool-ui-web-search.tsx", "query"],
  ["../src/components/assistant-ui/tool-ui-web-search.tsx", "url"],
];

test("tool cards coerce model-supplied args to strings", () => {
  for (const [path, prop] of CARDS) {
    const source = read(path);
    assert.match(
      source,
      new RegExp(`String\\(\\(args as \\{ ${prop}\\?: unknown \\}\\)\\?\\.${prop} \\?\\? ""\\)`),
      `${path} must coerce ${prop}; a model that emits a number crashes the whole app`,
    );
    assert.doesNotMatch(
      source,
      new RegExp(`const ${prop} = \\(args as`),
      `${path} reads ${prop} without coercion`,
    );
  }
});
