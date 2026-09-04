// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TOOL_EXECUTION_RECORD_ARG_KEY,
  type ToolExecutionRecord,
  parseToolExecutionRecord,
  toolExecutionRecordFromCard,
  toolExecutionRecordLabel,
} from "../src/features/chat/types/api.ts";

const record = (
  overrides: Partial<ToolExecutionRecord> = {},
): ToolExecutionRecord => ({
  requested_mode: "os_isolation_required",
  effective_mode: "os_isolation_required",
  environment: "native_linux",
  backend: "linux-bubblewrap",
  profile_id: "bubblewrap-v1",
  probe_generation: "generation-1",
  os_isolation: true,
  retained_safeguards: ["process_guard", "sanitized_environment"],
  limitations: [],
  ...overrides,
});

test("execution records are validated before a card trusts them", () => {
  assert.deepEqual(parseToolExecutionRecord(record()), record());
  assert.equal(
    parseToolExecutionRecord({ ...record(), retained_safeguards: [null] }),
    null,
  );
  assert.equal(
    parseToolExecutionRecord({ ...record(), effective_mode: "automatic" }),
    null,
  );
});

test("cards use exact labels from the backend execution record", () => {
  assert.equal(toolExecutionRecordLabel(record()), "Protected · Bubblewrap");
  assert.equal(
    toolExecutionRecordLabel(record({ environment: "wsl2" })),
    "Preview OS isolation · Bubblewrap (WSL2)",
  );
  assert.equal(
    toolExecutionRecordLabel(record({ environment: "linux_unknown" })),
    "Preview OS isolation · Bubblewrap (linux_unknown)",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({ backend: "windows-lpac", environment: "windows" }),
    ),
    "Preview OS isolation · LPAC (Windows)",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({
        backend: "macos-seatbelt",
        environment: "macos",
        limitations: ["detached_descendant_cleanup_unverified"],
      }),
    ),
    "Preview OS isolation · Seatbelt (lifecycle unverified)",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({ effective_mode: "limited", os_isolation: false }),
    ),
    "Limited · no OS isolation",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({ effective_mode: "full", os_isolation: false }),
    ),
    "Full access · security restrictions disabled",
  );
  assert.equal(toolExecutionRecordLabel(null), null);
});

test("the completed record wins over running-card metadata", () => {
  const started = record({ probe_generation: "started" });
  const completed = record({ probe_generation: "completed" });
  assert.equal(
    toolExecutionRecordFromCard(
      { [TOOL_EXECUTION_RECORD_ARG_KEY]: started },
      { execution_record: completed },
    )?.probe_generation,
    "completed",
  );
  assert.deepEqual(
    toolExecutionRecordFromCard(
      { [TOOL_EXECUTION_RECORD_ARG_KEY]: started },
      undefined,
    ),
    started,
  );
});

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const runtimeStore = readFileSync(
  new URL(
    "../src/features/chat/stores/chat-runtime-store.ts",
    import.meta.url,
  ),
  "utf8",
);

test("local Python and Terminal refresh capability and block without downgrade", () => {
  const gateStart = adapter.indexOf("const runsStudioPythonOrTerminal =");
  const requestStart = adapter.indexOf(
    "const buildRequestPayload = async",
    gateStart,
  );
  assert.ok(gateStart > 0 && requestStart > gateStart);
  const gate = adapter.slice(gateStart, requestStart);

  assert.match(gate, /refreshToolIsolationCapability\(\)/);
  assert.match(gate, /mode !== requestedMode/);
  assert.match(gate, /setToolIsolationConsentOpen\(true\)/);
  assert.match(gate, /capability\?\.protection_state !== "protected"/);
  assert.match(gate, /capability\?\.protection_state !== "preview"/);
  assert.doesNotMatch(gate, /setToolExecutionMode\("limited"\)/);
});

test("requests carry a current Limited grant and never attach it to other modes", () => {
  assert.match(adapter, /tool_execution_mode: mode/);
  assert.match(
    adapter,
    /tool_ui_session_id: isolation\.toolIsolationUiSessionId/,
  );
  assert.match(adapter, /mode === "limited" && currentLimitedGrant/);
  assert.match(adapter, /limited_grant: currentLimitedGrant\.grant/);
  assert.match(adapter, /tool_execution_mode: "full"/);
  assert.equal(
    adapter.match(/\.\.\.toolIsolationRequestFields/g)?.length,
    2,
    "both local-model and external-provider Studio-tool requests need the fields",
  );
});

test("token counts carry the same execution mode as their completion", () => {
  const start = adapter.indexOf("export async function buildLocalTokenCountExtras");
  const end = adapter.indexOf("async function resolveUseAdapter", start);
  const builder = adapter.slice(start, end);
  assert.ok(start > 0 && end > start);
  assert.match(builder, /toolExecutionMode/);
  assert.match(builder, /tool_execution_mode: toolExecutionMode/);
});

test("auth-session changes discard Limited grants and rotate their page binding", () => {
  assert.match(runtimeStore, /AUTH_SESSION_CLEARED_EVENT/);
  assert.match(runtimeStore, /AUTH_SESSION_STORED_EVENT/);
  assert.match(runtimeStore, /clearToolIsolationGrantForAuthSession/);
  assert.match(runtimeStore, /toolIsolationUiSessionId: createToolIsolationUiSessionId\(\)/);
  assert.match(runtimeStore, /limitedToolGrant: null/);
  assert.match(runtimeStore, /state\.toolExecutionMode === "limited"/);
});

test("only the started event paints a running record and tool_end persists it", () => {
  assert.match(
    adapter,
    /toolEvent\.execution_state === "started"\s*\? parseToolExecutionRecord\(toolEvent\.execution_record\)/,
  );
  assert.match(
    adapter,
    /\[TOOL_EXECUTION_RECORD_ARG_KEY\]:\s*cardExecutionRecord/,
  );
  assert.match(adapter, /execution_record: executionRecord/);
  assert.match(
    adapter,
    /key !== TOOL_EXECUTION_RECORD_ARG_KEY/,
    "card-only metadata must not become model-visible tool arguments",
  );
});

for (const component of ["tool-ui-python.tsx", "tool-ui-terminal.tsx"]) {
  test(`${component} renders only a parsed execution record`, () => {
    const source = readFileSync(
      new URL(`../src/components/assistant-ui/${component}`, import.meta.url),
      "utf8",
    );
    assert.match(source, /toolExecutionRecordFromCard\(args, result\)/);
    assert.match(source, /data-slot="tool-execution-protection"/);
  });
}
