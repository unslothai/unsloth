// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TOOL_EXECUTION_RECORD_ARG_KEY,
  type ToolExecutionRecord,
  attachAuthoritativeExecutionRecord,
  authoritativeExecutionRecordCount,
  discardAuthoritativeExecutionRecord,
  parseBackendExecutionRecord,
  stripUntrustedExecutionMetadata,
  stripUntrustedExecutionMetadataFromContent,
  toolExecutionRecordFromCard,
  toolExecutionRecordLabel,
} from "../src/features/chat/types/api.ts";
import { snapshotQueuedChatRunSettings } from "../src/features/chat/utils/queued-chat-run-settings.ts";
import { protectedIsolationDefaults } from "../src/features/chat/utils/tool-isolation-defaults.ts";
import { queuedIsolationDecisionIsCurrent } from "../src/features/chat/utils/queued-isolation-gate.ts";

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
  const restored = stripUntrustedExecutionMetadataFromContent(content) as Record<
    string,
    unknown
  >[];
  assert.ok(!(TOOL_EXECUTION_RECORD_ARG_KEY in (restored[0].args as object)));
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
  // The plain AppContainer fallback is a weaker profile and is never called LPAC.
  assert.equal(
    toolExecutionRecordLabel(
      record({
        backend: "windows-lpac",
        environment: "windows",
        profile_id: "windows-appcontainer-preview-v1",
        limitations: ["all_application_packages_ambient_read"],
      }),
    ),
    "Preview OS isolation · AppContainer (Windows)",
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

test("records carry the network policy and the card label says when the allowlist was open", () => {
  // Older backends omit both fields; the parsed record then has neither and the label is unchanged.
  const legacy = parseBackendExecutionRecord(record());
  assert.ok(legacy);
  assert.ok(!("network_policy" in legacy));
  assert.equal(toolExecutionRecordLabel(legacy), "Protected · Bubblewrap");

  const allowlisted = parseBackendExecutionRecord(
    record({
      network_policy: "allowlist",
      network_allowlist: ["pypi.org", "huggingface.co"],
    }),
  );
  assert.ok(allowlisted);
  assert.equal(allowlisted.network_policy, "allowlist");
  assert.deepEqual(allowlisted.network_allowlist, ["pypi.org", "huggingface.co"]);
  assert.equal(
    toolExecutionRecordLabel(allowlisted),
    "Protected · Bubblewrap · network allowlist",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({ network_policy: "deny", network_allowlist: [] }),
    ),
    "Protected · Bubblewrap",
  );
  // A Limited or Full launch never had a sandbox to open, so the suffix never appears there
  // even if a backend echoed the field.
  assert.equal(
    toolExecutionRecordLabel(
      record({ effective_mode: "full", os_isolation: false, network_policy: "allowlist" }),
    ),
    "Full access · security restrictions disabled",
  );
  // Full access and Limited launches report "unrestricted": the backend emits the field on
  // every record, so the value must parse or every Full and Limited card would lose its label.
  const fullRecord = parseBackendExecutionRecord(
    record({
      effective_mode: "full",
      os_isolation: false,
      backend: "none",
      network_policy: "unrestricted",
    }),
  );
  assert.ok(fullRecord);
  assert.equal(fullRecord.network_policy, "unrestricted");
  assert.equal(toolExecutionRecordLabel(fullRecord), "Full access · security restrictions disabled");
  const tokenRecord = parseBackendExecutionRecord(
    record({
      effective_mode: "limited",
      os_isolation: false,
      backend: "windows-restricted-token",
      profile_id: "windows-restricted-token-write-isolation-v1",
      network_policy: "unrestricted",
    }),
  );
  assert.ok(tokenRecord);
  assert.equal(toolExecutionRecordLabel(tokenRecord), "Limited · restricted token (Windows)");
  // Unknown policies and malformed host lists invalidate the record rather than degrading.
  assert.equal(
    parseBackendExecutionRecord({ ...record(), network_policy: "open" }),
    null,
  );
  assert.equal(
    parseBackendExecutionRecord({ ...record(), network_allowlist: "pypi.org" }),
    null,
  );
});

test("a Limited launch under the Windows restricted token is labelled as such", () => {
  assert.equal(
    toolExecutionRecordLabel(
      record({
        effective_mode: "limited",
        os_isolation: false,
        backend: "windows-restricted-token",
        profile_id: "windows-restricted-token-write-isolation-v1",
        environment: "windows",
        limitations: ["user_profile_readable", "network_unrestricted"],
      }),
    ),
    "Limited · restricted token (Windows)",
  );
  assert.equal(
    toolExecutionRecordLabel(
      record({ effective_mode: "limited", os_isolation: false, backend: "none" }),
    ),
    "Limited · no OS isolation",
  );
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
    toolExecutionRecordLabel(
      toolExecutionRecordFromCard(replayCard.toolCallId),
    ),
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


test("returning to protected defaults drops Full and Limited for every persisted level", () => {
  for (const level of ["ask", "auto", "off"] as const) {
    const next = protectedIsolationDefaults(level);
    assert.equal(next.toolExecutionMode, "os_isolation_required");
    assert.equal(next.toolNetworkPolicy, "deny");
    assert.equal(next.limitedToolGrant, null);
    assert.equal(next.bypassPermissions, false);
    assert.equal(next.toolIsolationConsentOpen, false);
    assert.equal(next.permissionMode, level);
    assert.equal(next.confirmToolCalls, level !== "off");
  }
  // Full is a session decision and never a level to fall back to.
  const clamped = protectedIsolationDefaults("full");
  assert.equal(clamped.permissionMode, "auto");
  assert.equal(clamped.toolExecutionMode, "os_isolation_required");
  assert.equal(clamped.bypassPermissions, false);
});

test("a queued send snapshots the isolation decision alongside the permission level", () => {
  const grant = {
    grant: "opaque",
    expires_at: 1,
    probe_generation: "generation-1",
    ui_session_id: "ui-1",
  };
  const state = {
    params: { checkpoint: "m" },
    permissionMode: "ask",
    bypassPermissions: false,
    confirmToolCalls: true,
    toolExecutionMode: "limited",
    toolNetworkPolicy: "allowlist",
    limitedToolGrant: grant,
    toolIsolationUiSessionId: "ui-1",
  } as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0];
  const snapshot = snapshotQueuedChatRunSettings(state);
  assert.equal(snapshot.toolExecutionMode, "limited");
  assert.equal(snapshot.toolNetworkPolicy, "allowlist");
  assert.equal(snapshot.limitedToolGrant, grant);
  assert.equal(snapshot.toolIsolationUiSessionId, "ui-1");
  // Later store changes do not reach a snapshot already taken.
  (state as { toolExecutionMode: string }).toolExecutionMode = "full";
  assert.equal(snapshot.toolExecutionMode, "limited");
});

test("execution records are filed per pane and thread scope, never by bare call id", () => {
  const id = "call_0";
  const protectedRecord = parseBackendExecutionRecord(record());
  const fullRecord = parseBackendExecutionRecord(
    record({
      requested_mode: "full",
      effective_mode: "full",
      backend: "none",
      profile_id: "full-access",
      os_isolation: false,
    }),
  );
  attachAuthoritativeExecutionRecord({ toolCallId: id }, protectedRecord, "pane-a\u0000thread-1");
  attachAuthoritativeExecutionRecord({ toolCallId: id }, fullRecord, "pane-a\u0000thread-2");
  assert.equal(
    toolExecutionRecordFromCard(id, "pane-a\u0000thread-1")?.effective_mode,
    "os_isolation_required",
  );
  assert.equal(toolExecutionRecordFromCard(id, "pane-a\u0000thread-2")?.effective_mode, "full");
  assert.equal(toolExecutionRecordFromCard(id, "pane-a\u0000thread-3"), null);
  assert.equal(toolExecutionRecordFromCard(id), null, "the legacy namespace stays separate");

  discardAuthoritativeExecutionRecord(id, "pane-a\u0000thread-2");
  assert.equal(toolExecutionRecordFromCard(id, "pane-a\u0000thread-2"), null);
  assert.ok(toolExecutionRecordFromCard(id, "pane-a\u0000thread-1"));

  // An unscoped discard is the legacy namespace only: hydrating one conversation must not
  // erase a record another pane or thread filed under the same repeating id.
  attachAuthoritativeExecutionRecord({ toolCallId: id }, fullRecord, "pane-b\u0000thread-9");
  attachAuthoritativeExecutionRecord({ toolCallId: id }, fullRecord);
  discardAuthoritativeExecutionRecord(id);
  assert.equal(toolExecutionRecordFromCard(id), null);
  assert.ok(toolExecutionRecordFromCard(id, "pane-a\u0000thread-1"));
  assert.ok(toolExecutionRecordFromCard(id, "pane-b\u0000thread-9"));
  discardAuthoritativeExecutionRecord(id, "pane-a\u0000thread-1");
  discardAuthoritativeExecutionRecord(id, "pane-b\u0000thread-9");
});

test("execution records are filed per assistant message, so a repeated turn id cannot relabel an earlier card", () => {
  const id = "tool_call_0";
  // toolPaneScope / toolThreadScope / toolExecutionRecordScope spelled out (tool-output-scope.ts
  // imports React, which node --test cannot load): pane, pair, thread, then assistant message.
  const thread = "base\u0000\u0000thread-1";
  const turnOne = `${thread}\u0000msg-1`;
  const turnTwo = `${thread}\u0000msg-2`;
  const fullRecord = parseBackendExecutionRecord(
    record({
      requested_mode: "full",
      effective_mode: "full",
      backend: "none",
      profile_id: "full-access",
      os_isolation: false,
    }),
  );
  attachAuthoritativeExecutionRecord({ toolCallId: id }, fullRecord, turnOne);
  // The next turn mints tool_call_0 again and is Required this time.
  discardAuthoritativeExecutionRecord(id, turnTwo);
  attachAuthoritativeExecutionRecord({ toolCallId: id }, parseBackendExecutionRecord(record()), turnTwo);
  assert.equal(toolExecutionRecordFromCard(id, turnOne)?.effective_mode, "full");
  assert.equal(toolExecutionRecordFromCard(id, turnTwo)?.effective_mode, "os_isolation_required");
  // Hydrating message 1 of this thread drops exactly its record and nothing else.
  discardAuthoritativeExecutionRecord(id, turnOne);
  assert.equal(toolExecutionRecordFromCard(id, turnOne), null);
  assert.ok(toolExecutionRecordFromCard(id, turnTwo));
  discardAuthoritativeExecutionRecord(id, turnTwo);
});

test("a queued Full send is fenced to the UI session that authorized it", () => {
  // Session A chooses Full and queues a send while the model loads.
  const sessionA = { toolExecutionMode: "full" as const, toolIsolationUiSessionId: "ui-a" };
  const snapshot = { ...sessionA };
  assert.ok(queuedIsolationDecisionIsCurrent(snapshot, sessionA));
  // Authentication rotates: the store returns to protected defaults under a new session id.
  const rotated = {
    ...protectedIsolationDefaults("auto"),
    toolIsolationUiSessionId: "ui-b",
  };
  assert.equal(rotated.toolExecutionMode, "os_isolation_required");
  assert.ok(!queuedIsolationDecisionIsCurrent(snapshot, rotated));
  // Session B independently selects Full: A's queued request still does not qualify.
  const sessionB = { toolExecutionMode: "full" as const, toolIsolationUiSessionId: "ui-b" };
  assert.ok(!queuedIsolationDecisionIsCurrent(snapshot, sessionB));
  // Turning Full off in the same session is rejected as before.
  assert.ok(
    !queuedIsolationDecisionIsCurrent(snapshot, {
      toolExecutionMode: "os_isolation_required",
      toolIsolationUiSessionId: "ui-a",
    }),
  );
});

test("the record map is bounded and evicts the oldest entry first", () => {
  const parsed = parseBackendExecutionRecord(record());
  const scope = "cap-test";
  for (let index = 0; index < 2100; index += 1) {
    attachAuthoritativeExecutionRecord({ toolCallId: `bulk-${index}` }, parsed, scope);
  }
  assert.ok(authoritativeExecutionRecordCount() <= 2048);
  assert.equal(toolExecutionRecordFromCard("bulk-0", scope), null);
  assert.ok(toolExecutionRecordFromCard("bulk-2099", scope));
  for (let index = 0; index < 2100; index += 1) {
    discardAuthoritativeExecutionRecord(`bulk-${index}`, scope);
  }
});

test("the store and adapter route every exit from Full through the shared transition", () => {
  const runtimeStore = readFileSync(
    new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
    "utf8",
  );
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // Deep Research and auth rotation each apply the helper. (A thread switch deliberately keeps
  // Full: it was accepted through the warning dialog, and the store keeps permissionMode,
  // bypassPermissions and toolExecutionMode together, so no reset is needed there.)
  assert.ok(
    (runtimeStore.match(/protectedIsolationDefaults\(/g)?.length ?? 0) >= 2,
    "setDeepResearchEnabled and the auth-session reset must both use protectedIsolationDefaults",
  );
  // The auth reset applies the transition unconditionally instead of demoting only Limited.
  const authResetStart = runtimeStore.indexOf(
    "function clearToolIsolationGrantForAuthSession(): void {",
  );
  const authResetEnd = runtimeStore.indexOf("\n}\n", authResetStart);
  assert.ok(authResetStart > 0 && authResetEnd > authResetStart);
  const authReset = runtimeStore.slice(authResetStart, authResetEnd);
  assert.match(authReset, /protectedIsolationDefaults\(/);
  assert.doesNotMatch(authReset, /=== "limited"/);
  // The send path reads the run snapshot, not the live store, for the isolation decision.
  assert.match(adapter, /const requestedMode = runtime\.toolExecutionMode;/);
  assert.match(adapter, /const requestedGrant = runtime\.limitedToolGrant;/);
  // The queued snapshot is clamped by the live store: a withdrawn allowlist wins.
  assert.match(
    adapter,
    /const requestedNetworkPolicy = queuedToolNetworkPolicy\(\s*runtime\.toolNetworkPolicy,\s*useChatRuntimeStore\.getState\(\)\.toolNetworkPolicy,\s*\);/,
  );
  // Every network field on the wire goes through the capability-gated helper, never raw.
  assert.doesNotMatch(adapter, /tool_network_policy: requestedNetworkPolicy/);
  assert.doesNotMatch(adapter, /tool_network_policy: runtime\.toolNetworkPolicy/);
  assert.doesNotMatch(adapter, /tool_network_policy: toolNetworkPolicy\b/);
  assert.match(adapter, /isolation\.toolIsolationUiSessionId !== requestedUiSessionId/);
  // The Full branch applies the same session fence through the shared helper.
  const fullBranch = adapter.slice(
    adapter.indexOf('if (requestedMode === "full") {'),
    adapter.indexOf('toolIsolationRequestFields = { tool_execution_mode: "full" };'),
  );
  assert.match(fullBranch, /queuedIsolationDecisionIsCurrent\(/);
  assert.match(fullBranch, /toolIsolationUiSessionId: requestedUiSessionId/);
  // Launch records are attached and discarded under the message-scoped key, and hydration
  // discards in that exact scope rather than sweeping every scope for the id.
  assert.match(adapter, /toolExecutionRecordScope\(\s*toolOutputPaneScope,\s*unstable_assistantMessageId,?\s*\)/);
  assert.doesNotMatch(adapter, /discardAuthoritativeExecutionRecord\([^)]*toolOutputPaneScope\)/);
  const provider = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(provider, /discardAuthoritativeExecutionRecord\(part\.toolCallId\)/);
  assert.match(provider, /discardAuthoritativeExecutionRecord\(part\.toolCallId, recordScope\)/);
});

test("every Full entry and exit and the Limited grant close the network allowlist", () => {
  // The allowlist is a per-decision grant. It must not survive a trip through Full (where it is
  // hidden and ignored) or a Limited grant (which cannot enforce it) and resurface later.
  const runtimeStore = readFileSync(
    new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
    "utf8",
  );
  const fullEntries = runtimeStore.match(
    /toolExecutionMode: "full" as ToolExecutionMode,\n\s*toolNetworkPolicy: "deny" as ToolNetworkPolicy,/g,
  );
  assert.equal(fullEntries?.length ?? 0, 2, "setPermissionMode(full) and setBypassPermissions(true)");
  assert.match(
    runtimeStore,
    /leavingFullAccess\n\s*\? \{\n\s*toolExecutionMode:\n\s*"os_isolation_required" as ToolExecutionMode,\n(\s*\/\/.*\n)*\s*toolNetworkPolicy: "deny" as ToolNetworkPolicy,/,
  );
  assert.match(
    runtimeStore,
    /limitedToolGrant: grant,\n\s*toolExecutionMode: "limited",\n(\s*\/\/.*\n)*\s*toolNetworkPolicy: "deny" as ToolNetworkPolicy,/,
  );
  // setBypassPermissions(false) and setToolExecutionMode are the other two ways in and
  // out of Full; both close the network too.
  assert.equal(
    (runtimeStore.match(/toolNetworkPolicy: "deny" as ToolNetworkPolicy,/g) ?? []).length,
    7,
  );
});
