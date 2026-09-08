// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Production request building, dispatch and auth retries with controlled I/O.
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { runInNewContext } from "node:vm";
import { queuedIsolationDecisionIsCurrent } from "../src/features/chat/utils/queued-isolation-gate.ts";
import { queuedToolNetworkPolicy } from "../src/features/chat/utils/tool-network-policy.ts";
const source = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const api = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);
const guard = source.slice(
  source.indexOf("      function currentToolIsolationRequestFields()"),
  source.indexOf(
    "      if (selectedImageEditReference && !imageGenerationEnabledForThisTurn)",
  ),
);
const localStart = source.indexOf(
  "          return {\n            model: params.checkpoint,",
  source.indexOf("const buildRequestPayload"),
);
assert.ok(localStart > 0);
const localBody = source.slice(
  localStart,
  source.indexOf(
    "\n        };\n\n        let retriedWithRefreshedKey",
    localStart,
  ),
);
const dispatchStart = source.indexOf(
  "            let requestPayload: OpenAIChatCompletionsRequest;",
  localStart,
);
const dispatch = source.slice(
  dispatchStart,
  source.indexOf("            // Per run, not per module", dispatchStart),
);
const sendStart = api.indexOf("export async function* streamChatCompletions(");
const sendPrefix = `${api
  .slice(sendStart, api.indexOf("\n  if (!response.ok)", sendStart))
  .replace("export ", "")}\n yield response;\n}`;
const auth = readFileSync(
  new URL("../src/features/auth/api.ts", import.meta.url),
  "utf8",
);
const isolation = readFileSync(
  new URL("../src/features/chat/tool-isolation.ts", import.meta.url),
  "utf8",
);
const isLimitedGrantCurrent = runInNewContext(
  `${stripTypeScriptTypes(
    isolation.slice(
      isolation.indexOf("export function isLimitedGrantCurrent("),
      isolation.indexOf("export function toolIsolationPresentation("),
    ),
  ).replace("export ", "")}\nisLimitedGrantCurrent;`,
);
const authFunctions = [
  "fetchWithTauriNetworkRetry",
  "retryWithCurrentToken",
  "authFetch",
]
  .map((name) => {
    const start = auth.indexOf(`async function ${name}(`);
    assert.ok(start >= 0);
    return auth.slice(start, auth.indexOf("\n}", start) + 2);
  })
  .join("\n");
for (const scenario of [
  "unchanged",
  "revoke-before-build",
  "revoke-during-first-save",
  "revoke-network-during-first-save",
  "session-during-first-save",
  "roundtrip-during-first-save",
  "grant-during-first-save",
  "capability-during-first-save",
  "network-auth-refresh",
  "network-transport-retry",
  "unchanged-auth-refresh",
  "unchanged-grant",
  "expired-grant-during-first-save",
  "network-encryption-wait",
]) {
  test(scenario, async () => {
    let resume!: () => void;
    const pending = new Promise<void>((resolve) => {
      resume = resolve;
    });
    const network =
      scenario.includes("network") || scenario.includes("capability");
    let live = {
      toolExecutionMode: network ? "os_isolation_required" : "full",
      toolIsolationDecisionEpoch: 1,
      toolIsolationUiSessionId: "page-a",
      toolNetworkPolicy: network ? "allowlist" : "deny",
      toolIsolationCapability: {
        probe_generation: "generation",
        protection_state: "preview",
        network_policies: ["deny", "allowlist"],
      },
      limitedToolGrant: {
        grant: "grant",
        expires_at: Date.now() + 60_000,
        probe_generation: "generation",
      } as {
        grant: string;
        expires_at: number;
        probe_generation: string;
      } | null,
    };
    if (scenario.includes("grant")) live.toolExecutionMode = "limited";
    const runtime = { ...live, maxToolCallsPerMessage: 5, toolCallTimeout: 1 };
    const wire: { url: string; payload: Record<string, unknown> }[] = [];
    const revoke = () => {
      if (scenario.includes("session"))
        live = { ...live, toolIsolationUiSessionId: "page-b" };
      else if (scenario.includes("roundtrip"))
        live = { ...live, toolIsolationDecisionEpoch: 3 };
      else if (scenario.includes("expired")) {
        assert.ok(live.limitedToolGrant);
        live = {
          ...live,
          limitedToolGrant: { ...live.limitedToolGrant, expires_at: 1 },
        };
      } else if (scenario.includes("grant"))
        live = { ...live, limitedToolGrant: null };
      else if (scenario.includes("capability"))
        live = {
          ...live,
          toolIsolationCapability: {
            probe_generation: "generation",
            protection_state: "unavailable",
            network_policies: ["deny"],
          },
        };
      else
        live = {
          ...live,
          toolExecutionMode: "os_isolation_required",
          toolNetworkPolicy: "deny",
          toolIsolationDecisionEpoch: 2,
        };
    };
    const context = {
      queuedIsolationDecisionIsCurrent,
      queuedToolNetworkPolicy,
      runtime,
      runsStudioPythonOrTerminal: true,
      isLimitedGrantCurrent,
      supportsStudioToolsForThisTurn: true,
      useChatRuntimeStore: { getState: () => live },
      toolIsolationRequestFields: {
        tool_execution_mode: runtime.toolExecutionMode,
        tool_network_policy: runtime.toolNetworkPolicy,
        limited_grant: scenario.includes("grant") ? "grant" : undefined,
      },
      permissionMode: network ? "off" : "full",
      bypassPermissions: !network,
      params: { checkpoint: "fixture", maxTokens: 10, maxSeqLength: 1024 },
      outboundMessages: [],
      survivingMessages: [],
      continuation: false,
      studioToolHistoryRequestFieldsAfterReplay: () => ({}),
      ggufCompactionRequestFields: () => ({}),
      activeModel: { isGguf: true },
      imageBase64: undefined,
      audioBase64: undefined,
      videoBase64: undefined,
      cancelId: "cancel",
      sandboxSessionId: "sandbox",
      resolvedThreadId: "new-thread",
      useAdapter: undefined,
      supportsReasoning: false,
      supportsPreserveThinking: false,
      deepResearchArmed: false,
      supportsTools: true,
      toolsEnabled: false,
      codeToolsEnabled: true,
      renderHtmlToolEnabledForThisTurn: false,
      mcpEnabledForChat: false,
      ragEnabled: false,
      projectRagEnabled: false,
      retriedWithRefreshedKey: false,
      clearSelectedImageEditReference() {},
      requestedMaxTokens: 0,
      ThreadAutosaveHandle: { awaitFirstSave: () => pending },
      encryptionWait: async () => {
        if (scenario.includes("encryption")) revoke();
      },
      generationDecision: "pending",
      runSignal: new AbortController().signal,
      isExternalRequest: false,
      Headers,
      Response,
      apiUrl: (value: string) => value,
      addBrowserTimezoneHeaders() {},
      getAuthToken: () => "access",
      getRefreshToken: () => "refresh",
      isTauri: scenario.includes("transport"),
      TAURI_FETCH_RETRY_DELAYS_MS: [0],
      asTransportFailure: (error: unknown) => error,
      isPasswordChangeRequiredResponse: async () => false,
      mustChangePassword: () => false,
      refreshSession: async () => {
        if (!scenario.startsWith("unchanged")) revoke();
        return true;
      },
      wait: async () => {
        revoke();
      },
      fetch: async (url: string, init: RequestInit) => {
        wire.push({ url, payload: JSON.parse(String(init.body)) });
        if (scenario.includes("transport") && wire.length === 1)
          throw new TypeError("transport");
        return new Response("", {
          status:
            scenario.includes("auth-refresh") && wire.length === 1 ? 401 : 200,
        });
      },
      TypeError,
    };
    if (scenario === "revoke-before-build") revoke();
    const run = runInNewContext(
      stripTypeScriptTypes(
        `${authFunctions}\n${guard}\n${sendPrefix}\nasync function buildRequestPayload(){const payload = await (async () => {${localBody}})(); await encryptionWait(); return payload;}\nasync function probe(){${dispatch}\nfor await(const item of stream){}\n} probe();`,
      ),
      context,
    );
    await Promise.resolve();
    await Promise.resolve();
    if (scenario.includes("during")) revoke();
    resume();
    if (!scenario.startsWith("unchanged")) {
      await assert.rejects(run, /Tool permissions changed/);
      assert.equal(
        wire.length,
        scenario.includes("auth-refresh") || scenario.includes("transport")
          ? 1
          : 0,
      );
    } else {
      await run;
      assert.equal(wire.length, scenario.includes("auth-refresh") ? 2 : 1);
      assert.equal(
        wire.at(-1)?.payload.tool_execution_mode,
        scenario.includes("grant") ? "limited" : "full",
      );
    }
  });
}
