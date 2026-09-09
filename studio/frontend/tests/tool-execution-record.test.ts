import assert from "node:assert/strict";
import test from "node:test";

import {
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

let running = true;
let remoteId: string | undefined;
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
