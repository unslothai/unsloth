// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import {
  effectiveToolNetworkPolicy,
  queuedToolNetworkPolicy,
} from "../src/features/chat/utils/tool-network-policy.ts";
import { queuedIsolationDecisionIsCurrent } from "../src/features/chat/utils/queued-isolation-gate.ts";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");
const adapter = read("../src/features/chat/api/chat-adapter.ts");
const store = read("../src/features/chat/stores/chat-runtime-store.ts");
const isolation = read("../src/features/chat/tool-isolation.ts");
const validator = isolation.slice(
  isolation.indexOf("export function isLimitedGrantCurrent("),
  isolation.indexOf("export function toolIsolationPresentation("),
);
const isLimitedGrantCurrent = runInNewContext(
  stripTypeScriptTypes(validator).replace(/^export /gm, "") +
    "\nisLimitedGrantCurrent;",
);
const gate = adapter.slice(
  adapter.indexOf("      let toolIsolationRequestFields: Pick<"),
  adapter.indexOf(
    "      if (selectedImageEditReference && !imageGenerationEnabledForThisTurn)",
  ),
);
const grantStart = store.indexOf("  requestLimitedToolGrant: async () => {");
const grantMethod = store.slice(
  grantStart,
  store.indexOf("  clearLimitedToolGrant: () =>", grantStart),
);
assert.ok(
  gate.includes("await useChatRuntimeStore") &&
    grantMethod.includes("await fetchLimitedToolGrant"),
);

function deferred() {
  let resolve!: (value?: unknown) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<unknown>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

for (const scenario of [
  "unchanged",
  "revoke-before",
  "revoke-during",
  "enable-during",
]) {
  test(`actual adapter capability await: ${scenario}`, async () => {
    const pending = deferred();
    const live = {
      toolExecutionMode: "os_isolation_required",
      toolIsolationUiSessionId: "session-a",
      toolNetworkPolicy: scenario === "enable-during" ? "deny" : "allowlist",
      toolIsolationCapability: {
        protection_state: "protected",
        network_policies: ["deny", "allowlist"],
      },
      refreshToolIsolationCapability: () => pending.promise,
    };
    const runtime = { ...live, limitedToolGrant: null };
    if (scenario === "revoke-before") live.toolNetworkPolicy = "deny";
    const result = runInNewContext(
      stripTypeScriptTypes(
        `async function run() { ${gate}\nreturn toolIsolationRequestFields; } run();`,
      ),
      {
        runtime,
        useChatRuntimeStore: { getState: () => live },
        supportsStudioToolsForThisTurn: true,
        runsStudioPythonOrTerminal: true,
        isLimitedGrantCurrent,
        effectiveToolNetworkPolicy,
        queuedToolNetworkPolicy,
        queuedIsolationDecisionIsCurrent,
        clearSelectedImageEditReference() {},
      },
    );
    if (scenario === "revoke-during") live.toolNetworkPolicy = "deny";
    if (scenario === "enable-during") live.toolNetworkPolicy = "allowlist";
    pending.resolve();
    assert.equal(
      (await result).tool_network_policy,
      scenario === "unchanged" ? "allowlist" : "deny",
    );
  });
}

type GrantState = {
  toolIsolationUiSessionId: string;
  toolExecutionMode: string;
  toolIsolationCapability: {
    protection_state: string;
    probe_generation: string;
  };
  queuedSettingsEpoch: number;
  limitedToolGrant: ReturnType<typeof grant> | null;
  toolIsolationGrantLoading: boolean;
};

function grantHarness() {
  let state: GrantState = {
    toolIsolationUiSessionId: "old-session",
    toolExecutionMode: "os_isolation_required",
    toolIsolationCapability: {
      protection_state: "unavailable",
      probe_generation: "g",
    },
    queuedSettingsEpoch: 0,
    limitedToolGrant: null,
    toolIsolationGrantLoading: false,
  };
  const requests: ReturnType<typeof deferred>[] = [];
  const method = runInNewContext(
    stripTypeScriptTypes(
      `let limitedGrantRequestId = 0; const methods = { ${grantMethod} }; methods.requestLimitedToolGrant;`,
    ),
    {
      get: () => state,
      set: (update: (state: GrantState) => Partial<GrantState>) => {
        state = { ...state, ...update(state) };
      },
      fetchLimitedToolGrant: () => {
        const d = deferred();
        requests.push(d);
        return d.promise;
      },
      isLimitedGrantCurrent,
    },
  );
  return {
    method,
    requests,
    get: () => state,
    change: (value: Partial<GrantState>) => {
      state = { ...state, ...value };
    },
  };
}
const grant = (value: string) => ({
  grant: value,
  probe_generation: "g",
  expires_at: Date.now() + 300000,
});

for (const failure of [false, true]) {
  test(`auth rotation discards pending grant ${failure ? "failure" : "success"}`, async () => {
    const h = grantHarness();
    const pending = h.method();
    const rejected = assert.rejects(pending);
    h.change({
      toolIsolationUiSessionId: "new-session",
      toolIsolationGrantLoading: false,
    });
    const expected = { ...h.get() };
    if (failure) h.requests[0].reject(new Error("old failure"));
    else h.requests[0].resolve(grant("old"));
    await rejected;
    assert.deepEqual(h.get(), expected);
  });
}

test("an older response cannot overwrite the successor's consent", async () => {
  const h = grantHarness();
  const first = h.method();
  const rejected = assert.rejects(first);
  const second = h.method();
  h.requests[1].resolve(grant("second"));
  await second;
  const expected = { ...h.get() };
  h.requests[0].resolve(grant("first"));
  await rejected;
  assert.deepEqual(h.get(), expected);
  assert.equal(h.get().limitedToolGrant?.grant, "second");
});

test("a refused successor clears loading without letting the old response restore consent", async () => {
  const h = grantHarness();
  const pending = h.method();
  const rejected = assert.rejects(pending);
  h.change({
    toolIsolationCapability: {
      protection_state: "protected",
      probe_generation: "g2",
    },
  });
  await assert.rejects(
    h.method(),
    /only available when OS isolation is unavailable/,
  );
  assert.equal(h.get().toolIsolationGrantLoading, false);
  assert.equal(h.requests.length, 1);
  const expected = { ...h.get() };
  h.requests[0].resolve(grant("obsolete"));
  await rejected;
  assert.deepEqual(h.get(), expected);
});

for (const change of [
  { toolExecutionMode: "full", queuedSettingsEpoch: 1 },
  { toolIsolationGrantLoading: false },
  { queuedSettingsEpoch: 2 },
]) {
  test(`changed consent cannot be overwritten: ${JSON.stringify(change)}`, async () => {
    const h = grantHarness();
    const pending = h.method();
    const rejected = assert.rejects(pending);
    h.change(change);
    const mode = h.get().toolExecutionMode;
    h.requests[0].resolve(grant("obsolete"));
    await rejected;
    assert.equal(h.get().toolExecutionMode, mode);
    assert.equal(h.get().limitedToolGrant, null);
    assert.equal(h.get().toolIsolationGrantLoading, false);
  });
}

test("both final request objects narrow network permission again at serialization", () => {
  const fields = adapter.match(
    /tool_network_policy: queuedToolNetworkPolicy\(\s*toolIsolationRequestFields\.tool_network_policy \?\? "deny",\s*useChatRuntimeStore\.getState\(\)\.toolNetworkPolicy,\s*\)/g,
  );
  assert.equal(fields?.length, 2);
  for (const text of fields ?? []) {
    const expression = text.slice(text.indexOf(":") + 1);
    assert.equal(
      runInNewContext(expression, {
        queuedToolNetworkPolicy,
        toolIsolationRequestFields: { tool_network_policy: "allowlist" },
        useChatRuntimeStore: {
          getState: () => ({ toolNetworkPolicy: "deny" }),
        },
      }),
      "deny",
    );
  }
});
