// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One tool's output write must re-render only that tool's card, not every card in the thread.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createStore } from "zustand/vanilla";

import type * as ScopeModule from "../src/features/chat/tool-output-scope.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Field = "toolFullOutput" | "toolLiveOutput";
type State = Record<Field, Record<string, string>>;

const PANE = "base\u0000\u0000thread-1";
const UNRESOLVED = "base\u0000\u0000";
const FIELDS: Field[] = ["toolFullOutput", "toolLiveOutput"];

const store = createStore<State>(() => ({ toolFullOutput: {}, toolLiveOutput: {} }));
let isRunning = false;
let selectors: Array<(s: State) => unknown> = [];

const scope = loadWithStubs<typeof ScopeModule>(
  new URL("../src/features/chat/tool-output-scope.ts", import.meta.url),
  {
    react: { createContext: (v: string) => v, useContext: (v: string) => v },
    "@assistant-ui/react": {
      useAuiState: (select: (s: Record<string, unknown>) => unknown) =>
        select({ thread: { isRunning }, threadListItem: { remoteId: "thread-1" } }),
    },
    "./stores/chat-runtime-store": {
      useChatRuntimeStore: (select: (s: State) => unknown) => {
        selectors.push(select);
        return select(store.getState());
      },
    },
    "./tool-output-result": {},
  },
);
const { toolOutputKey: key, useToolOutput: readOutput } = scope;

// The previous whole-map reader, kept as the oracle.
function oracle(map: Record<string, string>, id: string): string {
  const own = map[key(PANE, id)];
  return own ?? (isRunning ? (map[key(UNRESOLVED, id)] ?? "") : "");
}

test("returns what the whole-map reader returned in every scope and run state", () => {
  const maps: Array<Record<string, string>> = [
    {},
    { [key(PANE, "a")]: "own" },
    { [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "a")]: "own", [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "a")]: "", [key(UNRESOLVED, "a")]: "unresolved" },
    { [key(PANE, "b")]: "other card" },
  ];
  for (const field of FIELDS) {
    for (const map of maps) {
      for (const running of [false, true]) {
        store.setState({ toolFullOutput: {}, toolLiveOutput: {}, [field]: map });
        isRunning = running;
        assert.equal(readOutput(field, "a"), oracle(map, "a"), JSON.stringify({ field, map, running }));
      }
    }
  }
});

test("another card's write leaves this card's selections unchanged; its own write does", () => {
  for (const field of FIELDS) {
    store.setState({ toolFullOutput: {}, toolLiveOutput: {}, [field]: { [key(PANE, "a")]: "own" } });
    selectors = [];
    readOutput(field, "a");
    assert.equal(selectors.length, 2);
    const before = selectors.map((select) => select(store.getState()));
    const write = (id: string, text: string) =>
      store.setState((s) => ({ [field]: { ...s[field], [key(PANE, id)]: text } }));

    write("b", "x".repeat(50_000));
    assert.ok(selectors.every((select, i) => Object.is(select(store.getState()), before[i])), field);
    // The old whole-map selection changes on that same write, which re-rendered every card.
    const wholeMap = store.getState()[field];
    write("c", "y");
    assert.notEqual(store.getState()[field], wholeMap);

    write("a", "own, longer");
    assert.equal(selectors[0](store.getState()), "own, longer");
  }
});

test("tool cards and the live pane read their own keys and memoise the sanitized output", () => {
  const read = (file: string) =>
    readFileSync(new URL(`../src/components/assistant-ui/${file}`, import.meta.url), "utf8");
  for (const file of ["tool-ui-python.tsx", "tool-ui-terminal.tsx"]) {
    const source = read(file);
    assert.match(source, /useToolOutput\("toolFullOutput", toolCallId\)/, file);
    assert.match(
      source,
      /useMemo\(\s*\(\)\s*=>\s*preferSanitizedFullToolOutput\(fullOutput, output\),\s*\[fullOutput, output\],?\s*\)/,
      file,
    );
  }
  assert.match(read("tool-live-output.tsx"), /useToolOutput\("toolLiveOutput", toolCallId\)/);
  for (const file of ["tool-ui-python.tsx", "tool-ui-terminal.tsx", "tool-live-output.tsx"]) {
    assert.doesNotMatch(read(file), /\(s\)\s*=>\s*s\.tool(Full|Live)Output\s*\)/, file);
  }
});

test("tool code cells pass Streamdown stable plugins and controls", () => {
  const source = readFileSync(
    new URL("../src/components/assistant-ui/tool-code-cell.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /plugins=\{\{/);
  assert.doesNotMatch(source, /controls=\{\{/);
  assert.match(source, /^const STREAMDOWN_PLUGINS = \{ code: codePlugin \};$/m);
});
