// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  attachAuthoritativeExecutionRecord,
  parseBackendExecutionRecord,
  stripUntrustedExecutionMetadata,
  stripUntrustedExecutionMetadataFromContent,
  TOOL_EXECUTION_RECORD_ARG_KEY,
  type ToolExecutionRecord,
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

test("only backend-event parsing can establish an authoritative card record", () => {
  const raw = record();
  const card = { toolCallId: "record-validation" };
  const unchanged = attachAuthoritativeExecutionRecord(card, raw);
  assert.notEqual(unchanged, card);
  assert.ok(!("executionRecord" in unchanged));
  assert.equal(toolExecutionRecordFromCard(card.toolCallId), null);

  const parsed = parseBackendExecutionRecord(raw);
  assert.deepEqual(parsed, raw);
  const attached = attachAuthoritativeExecutionRecord(card, parsed);
  assert.deepEqual(attached.executionRecord, raw);
  assert.deepEqual(toolExecutionRecordFromCard(card.toolCallId), raw);
  assert.equal(
    parseBackendExecutionRecord({ ...record(), retained_safeguards: [null] }),
    null,
  );
  assert.equal(
    parseBackendExecutionRecord({ ...record(), effective_mode: "automatic" }),
    null,
  );
});

test("argument and restored-content sanitizers are non-mutating", () => {
  const args = {
    code: "print('ok')",
    nested: { kept: true },
    [TOOL_EXECUTION_RECORD_ARG_KEY]: record(),
  };
  const sanitized = stripUntrustedExecutionMetadata(args) as Record<
    string,
    unknown
  >;
  assert.deepEqual(sanitized, { code: "print('ok')", nested: args.nested });
  assert.notEqual(sanitized, args);
  assert.equal(sanitized.nested, args.nested);
  assert.ok(TOOL_EXECUTION_RECORD_ARG_KEY in args);

  const content = [
    {
      type: "tool-call",
      toolCallId: "legacy-spoof",
      args,
      result: { text: "done", execution_record: record() },
      artifact: { executionRecord: record(), kept: true },
      executionRecord: record(),
    },
  ];
  const restored = stripUntrustedExecutionMetadataFromContent(content) as Array<
    Record<string, unknown>
  >;
  assert.ok(
    !(TOOL_EXECUTION_RECORD_ARG_KEY in (restored[0].args as object)),
  );
  assert.deepEqual(restored[0].result, { text: "done" });
  assert.deepEqual(restored[0].artifact, { kept: true });
  assert.ok(!("executionRecord" in restored[0]));
  assert.ok(TOOL_EXECUTION_RECORD_ARG_KEY in args);
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

test("backend completion replaces start while JSON replay needs backend provenance", () => {
  const started = record({ probe_generation: "started" });
  const completed = record({ probe_generation: "completed" });
  const card = { toolCallId: "completion-order" };
  attachAuthoritativeExecutionRecord(
    card,
    parseBackendExecutionRecord(started),
  );
  assert.equal(
    toolExecutionRecordFromCard(card.toolCallId)?.probe_generation,
    "started",
  );
  attachAuthoritativeExecutionRecord(
    card,
    parseBackendExecutionRecord(completed),
  );
  assert.equal(
    toolExecutionRecordFromCard(card.toolCallId)?.probe_generation,
    "completed",
  );

  const restored = JSON.parse(JSON.stringify(completed)) as ToolExecutionRecord;
  const replayCard = { toolCallId: "json-replay" };
  attachAuthoritativeExecutionRecord(replayCard, restored);
  assert.equal(toolExecutionRecordFromCard(replayCard.toolCallId), null);
  attachAuthoritativeExecutionRecord(
    replayCard,
    parseBackendExecutionRecord(restored),
  );
  assert.equal(
    toolExecutionRecordLabel(toolExecutionRecordFromCard(replayCard.toolCallId)),
    "Protected · Bubblewrap",
  );
});

test("invalid backend events cannot create or erase an earlier valid record", () => {
  const invalidStartCard = { toolCallId: "invalid-start" };
  attachAuthoritativeExecutionRecord(
    invalidStartCard,
    parseBackendExecutionRecord({ ...record(), backend: null }),
  );
  assert.equal(toolExecutionRecordFromCard(invalidStartCard.toolCallId), null);

  const startedCard = { toolCallId: "invalid-completion" };
  const started = parseBackendExecutionRecord(
    record({ probe_generation: "valid-start" }),
  );
  attachAuthoritativeExecutionRecord(startedCard, started);
  const invalidCompletion = parseBackendExecutionRecord({
    ...record(),
    retained_safeguards: [false],
  });
  attachAuthoritativeExecutionRecord(
    startedCard,
    invalidCompletion ?? toolExecutionRecordFromCard(startedCard.toolCallId),
  );
  assert.equal(
    toolExecutionRecordFromCard(startedCard.toolCallId)?.probe_generation,
    "valid-start",
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
const runtimeProvider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
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

test("only backend started and completion events can update a card label", () => {
  assert.match(
    adapter,
    /toolEvent\.execution_state === "started"\s*\? parseBackendExecutionRecord\(toolEvent\.execution_record\)/,
  );
  assert.match(
    adapter,
    /toolEvent\.execution_state === "completed"\s*\? parseBackendExecutionRecord/,
  );
  assert.match(adapter, /attachAuthoritativeExecutionRecord\(/);
  assert.match(adapter, /discardAuthoritativeExecutionRecord\(partId\)/);
  assert.match(
    adapter,
    /completionExecutionRecord \?\?\s*toolExecutionRecordFromCard\(id\)/,
  );
  assert.doesNotMatch(adapter, /parseToolExecutionRecord/);
  assert.doesNotMatch(adapter, /execution_record: executionRecord/);
  assert.match(
    adapter,
    /stripUntrustedExecutionMetadata\(\s*toolEvent\.arguments/,
    "tool event arguments must be sanitized before merging",
  );
  assert.match(runtimeProvider, /stripUntrustedExecutionMetadataFromContent/);
  assert.match(runtimeProvider, /discardAuthoritativeExecutionRecord/);
  assert.match(
    adapter,
    /replayArgsText = mergedToolCallArgumentsText\([\s\S]*TOOL_EXECUTION_RECORD_ARG_KEY/,
    "reserved metadata must also be removed from exact provider replay text",
  );
});

for (const component of ["tool-ui-python.tsx", "tool-ui-terminal.tsx"]) {
  test(`${component} renders only a parsed execution record`, () => {
    const source = readFileSync(
      new URL(`../src/components/assistant-ui/${component}`, import.meta.url),
      "utf8",
    );
    assert.match(source, /toolExecutionRecordFromCard\(toolCallId\)/);
    assert.match(source, /data-slot="tool-execution-protection"/);
  });
}
