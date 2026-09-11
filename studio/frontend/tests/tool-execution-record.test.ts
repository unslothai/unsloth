import assert from "node:assert/strict";
import test from "node:test";

import {
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

let running = true;
let remoteId: string | undefined;
let savedMetadata: Record<string, unknown> = {};
const contextValues = new WeakMap<object, unknown>();
const react = {
  createContext(value: unknown) {
    const context = {};
    contextValues.set(context, value);
    return context;
  },
  useContext(context: object) {
    return contextValues.get(context);
  },
};

const scope = loadWithStubs<{
  toolOutputKey: (pane: string, id: string) => string;
  toolPaneScope: (model?: string, pair?: string) => string;
  toolThreadScope: (pane: string, thread?: string) => string;
  useToolOutputFor: (map: Record<string, string>, pane: string, id: string) => string;
  useToolPaneScope: () => string;
  ToolPaneScopeContext: object;
}>(
  new URL("../src/features/chat/tool-output-scope.ts", import.meta.url),
  {
    "@assistant-ui/react": {
      useAuiState: (selector: (state: unknown) => unknown) =>
        selector({ thread: { isRunning: running }, threadListItem: { remoteId } }),
    },
    react,
    "../../lib/strip-ansi": { stripAnsi: (value: string) => value },
    "./tool-output-result": {
      preferFullToolOutput: (_full: string, result: string) => result,
      preferSanitizedFullToolOutput: (_full: string, result: string) => result,
      shouldPreserveFullOutput: () => false,
      toolResultText: (result: unknown) => String(result),
    },
  },
);

function zustandCreate(factory: () => Record<string, unknown>) {
  let state = factory();
  const hook = ((selector: (value: Record<string, unknown>) => unknown) => selector(state)) as {
    (selector: (value: Record<string, unknown>) => unknown): unknown;
    getState: () => Record<string, unknown>;
    setState: (patch: Record<string, unknown>) => void;
  };
  hook.getState = () => state;
  hook.setState = patch => { state = { ...state, ...patch }; };
  return hook;
}

const records = loadWithStubs<{
  recordExecution: (key: string, value: unknown) => void;
  clearExecution: (key: string) => void;
  ToolExecutionDetails: (props: { toolCallId: string }) => unknown;
  useExecutionRecords: { getState: () => { records: Record<string, string> } };
}>(
  new URL("../src/features/chat/tool-execution-record.tsx", import.meta.url),
  {
    "./isolation-labels": { isolationLimitation: (value: string) => value },
    "@assistant-ui/react": {
      useAuiState: (selector: (state: unknown) => unknown) =>
        selector({ message: { metadata: { custom: savedMetadata } } }),
    },
    zustand: { create: zustandCreate },
    react,
    "./tool-output-scope": scope,
    "react/jsx-runtime": stubJsxRuntime(),
  },
);

const execution = {
  os_isolation: true,
  backend: "srt",
  network_policy: "deny",
};

test("execution details survive first-save, settlement, and remount", () => {
  const pane = scope.toolPaneScope();
  const key = scope.toolOutputKey(pane, "call_0:run-a");
  remoteId = undefined;
  running = true;
  records.recordExecution(key, execution);
  assert.notEqual(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
  remoteId = "thread-1";
  assert.notEqual(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
  running = false;
  assert.notEqual(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
  assert.ok(records.useExecutionRecords.getState().records[key]);
  assert.notEqual(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
});

test("another run or pane cannot inherit the previous card's execution record", () => {
  remoteId = "thread-2";
  assert.equal(records.ToolExecutionDetails({ toolCallId: "call_0:run-b" }), null);
  contextValues.set(scope.ToolPaneScopeContext, scope.toolPaneScope("adapter", "compare"));
  assert.equal(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
  contextValues.set(scope.ToolPaneScopeContext, scope.toolPaneScope());
  records.clearExecution(scope.toolOutputKey(scope.toolPaneScope(), "call_0:run-a"));
  assert.equal(records.ToolExecutionDetails({ toolCallId: "call_0:run-a" }), null);
});

test("saved execution labels survive reload with an empty live store", () => {
  assert.deepEqual(records.useExecutionRecords.getState().records, {});
  for (const [record, label] of [
    [execution, "Sandbox · srt"],
    [{ os_isolation: false, backend: "none", network_policy: "unrestricted", effective_mode: "auto" }, "No OS isolation"],
    [{ os_isolation: false, backend: "none", network_policy: "unrestricted", effective_mode: "full" }, "Full access · No OS isolation"],
  ] as const) {
    savedMetadata = JSON.parse(JSON.stringify({ toolExecutions: { "saved-call": record } }));
    assert.ok(JSON.stringify(records.ToolExecutionDetails({ toolCallId: "saved-call" })).includes(label));
    assert.equal(records.ToolExecutionDetails({ toolCallId: "different-call" }), null);
  }
  savedMetadata = {};
  assert.equal(records.ToolExecutionDetails({ toolCallId: "saved-call" }), null);
});

test("malformed persisted metadata cannot establish isolation", () => {
  savedMetadata = { toolExecutions: { "saved-call": { backend: "srt" } } };
  assert.equal(records.ToolExecutionDetails({ toolCallId: "saved-call" }), null);
  savedMetadata = {};
});
