// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A long-output tool end writes toolFullOutput once; only the finished card may re-render for it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type * as FullOutputModule from "../src/components/assistant-ui/use-tool-full-output.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const PANE = "base\u0000\u0000thread-1";
const UNRESOLVED = "base\u0000\u0000";
const key = (scope: string, id: string) => `${scope}\u0000${id}`;

let state: { toolFullOutput: Record<string, string> } = { toolFullOutput: {} };
let isRunning = false;
const selectors: Array<(s: typeof state) => unknown> = [];

// Aliased: the stubs make it a plain function, called here outside any component.
const { useToolFullOutput: readFullOutput } = loadWithStubs<typeof FullOutputModule>(
  new URL("../src/components/assistant-ui/use-tool-full-output.ts", import.meta.url),
  {
    "@assistant-ui/react": {
      useAuiState: (select: (s: { thread: { isRunning: boolean } }) => unknown) =>
        select({ thread: { isRunning } }),
    },
    "@/features/chat": {
      toolOutputKey: key,
      useToolPaneScope: () => PANE,
      useUnresolvedToolPaneScope: () => UNRESOLVED,
      useChatRuntimeStore: (select: (s: typeof state) => unknown) => {
        selectors.push(select);
        return select(state);
      },
    },
  },
);

// The previous whole-map reader, kept as the oracle.
function oracle(map: Record<string, string>, id: string): string {
  const own = map[key(PANE, id)];
  return own ?? (own === undefined && isRunning ? (map[key(UNRESOLVED, id)] ?? "") : "");
}

test("returns what the whole-map reader returned in every scope and run state", () => {
  const cases: Array<Record<string, string>> = [
    {},
    { [key(PANE, "a")]: "own" },
    { [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "a")]: "own", [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "a")]: "", [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "b")]: "other card" },
  ];
  for (const map of cases) {
    for (const running of [false, true]) {
      state = { toolFullOutput: map };
      isRunning = running;
      assert.equal(readFullOutput("a"), oracle(map, "a"), JSON.stringify({ map, running }));
    }
  }
});

test("selects strings, so another card's write leaves this card's selections unchanged", () => {
  selectors.length = 0;
  state = { toolFullOutput: { [key(PANE, "a")]: "own" } };
  readFullOutput("a");
  assert.equal(selectors.length, 2);
  const before = selectors.map((select) => select(state));
  const next = { toolFullOutput: { ...state.toolFullOutput, [key(PANE, "b")]: "x".repeat(50_000) } };
  for (const [i, select] of selectors.entries()) {
    const value = select(next);
    assert.ok(value === undefined || typeof value === "string");
    assert.ok(Object.is(value, before[i]));
  }
});

test("python and terminal cards read their own key and memoise the sanitized output", () => {
  for (const file of ["tool-ui-python.tsx", "tool-ui-terminal.tsx"]) {
    const source = readFileSync(
      new URL(`../src/components/assistant-ui/${file}`, import.meta.url),
      "utf8",
    );
    assert.doesNotMatch(source, /\(s\)\s*=>\s*s\.toolFullOutput\s*\)/, file);
    assert.match(source, /useToolFullOutput\(toolCallId\)/, file);
    assert.match(
      source,
      /useMemo\(\s*\(\)\s*=>\s*preferSanitizedFullToolOutput\(fullOutput, output\),\s*\[fullOutput, output\],?\s*\)/,
      file,
    );
  }
});

test("tool code cells pass Streamdown stable plugins and controls", () => {
  const source = readFileSync(
    new URL("../src/components/assistant-ui/tool-code-cell.tsx", import.meta.url),
    "utf8",
  );
  // Streamdown's memo compares plugins by identity; an inline object re-parses every render.
  assert.doesNotMatch(source, /plugins=\{\{/);
  assert.doesNotMatch(source, /controls=\{\{/);
  assert.match(source, /^const STREAMDOWN_PLUGINS = \{ code: codePlugin \};$/m);
});
