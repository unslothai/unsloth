// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import ts from "typescript";

const source = stripTypeScriptTypes(
  readFileSync(
    new URL("../src/features/chat/tool-isolation.ts", import.meta.url),
    "utf8",
  ),
)
  .replace(/^import .*;\r?\n/gm, "")
  .replace(/^export /gm, "");

function client(authFetch: (...args: any[]) => Promise<any>) {
  return runInNewContext(
    source + "\n({ fetchNestedToolGrant, isNestedGrantCurrent });",
    {
      authFetch,
      Date,
      Error,
    },
  );
}

test("nested consent uses its own HTTP endpoint and never submits a command", async () => {
  const requests: any[] = [];
  const api = client(async (...args) => {
    requests.push(args);
    return {
      ok: true,
      json: async () => ({
        grant: "opaque",
        expires_at: Date.now() + 60000,
        probe_generation: "g1",
      }),
    };
  });
  const grant = await api.fetchNestedToolGrant("page", "g1");
  assert.equal(requests.length, 1);
  assert.equal(requests[0][0], "/api/inference/tool-isolation/nested-grant");
  assert.equal(requests[0][1].method, "POST");
  assert.deepEqual(JSON.parse(requests[0][1].body), {
    ui_session_id: "page",
    probe_generation: "g1",
  });
  assert.equal(grant.mode, "container_isolation");
  const cap = {
    protection_state: "unavailable",
    nested_eligible: true,
    nested_profile_id: "nested-v1",
    probe_generation: "g1",
  };
  assert.equal(api.isNestedGrantCurrent(grant, cap), true);
  for (const patch of [
    { nested_eligible: false },
    { nested_profile_id: null },
    { probe_generation: "g2" },
    { protection_state: "protected" },
  ]) {
    assert.equal(api.isNestedGrantCurrent(grant, { ...cap, ...patch }), false);
  }
  assert.equal(
    api.isNestedGrantCurrent({ ...grant, mode: "limited" }, cap),
    false,
  );
  assert.equal(
    api.isNestedGrantCurrent({ ...grant, expires_at: 1 }, cap),
    false,
  );
});

test("failed nested consent retains only bounded diagnostic fields", async () => {
  const api = client(async () => ({
    ok: false,
    status: 409,
    json: async () => ({
      detail: {
        message: "The isolation check timed out.",
        diagnostic: {
          code: "probe_timeout",
          stage: "enforcement",
          secret: "NOT_FOR_UI",
        },
      },
    }),
  }));
  await assert.rejects(api.fetchNestedToolGrant("page", "g1"), (error: any) => {
    assert.equal(error.message, "The isolation check timed out.");
    assert.deepEqual(JSON.parse(JSON.stringify(error.diagnostic)), {
      code: "probe_timeout",
      stage: "enforcement",
      dependency: null,
    });
    assert.doesNotMatch(JSON.stringify(error), /NOT_FOR_UI/);
    return true;
  });
});

test("failed or malformed nested consent rejects without retry or fallback", async () => {
  for (const response of [
    { ok: false, status: 409, json: async () => ({}) },
    { ok: true, json: async () => ({ grant: "bad" }) },
  ]) {
    let calls = 0;
    const api = client(async () => {
      calls++;
      return response;
    });
    await assert.rejects(
      api.fetchNestedToolGrant("page", "g1"),
      /[Cc]ontainer-compatible/,
    );
    assert.equal(calls, 1);
  }
});

const storeSource = readFileSync(
  new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
  "utf8",
);

test("nested dialog uses backend disclosure, checks only on confirmation, and cancels pending consent", async () => {
  const ui = readFileSync(
    new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
    "utf8",
  );
  const component = ui.slice(
    ui.indexOf("function LimitedModeConfirmDialog("),
    ui.indexOf("/** Page-root dialog"),
  );
  const compiled = ts.transpileModule(component, {
    compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  let requested = 0;
  let cancelled = 0;
  let closed = false;
  let loading = false;
  let failure = false;
  const context: Record<string, unknown> = {
    React: {
      createElement: (type: any, props: any, ...children: any[]) => ({
        type,
        props,
        children,
      }),
    },
    useChatRuntimeStore: (select: (s: any) => any) =>
      select({
        toolIsolationCapability: {
          nested_eligible: true,
          nested_disclosure: "BACKEND shared /proc disclosure",
        },
        toolIsolationGrantLoading: loading,
        toolIsolationError: failure ? "The isolation check timed out." : null,
        toolIsolationErrorDiagnostic: failure
          ? { code: "probe_timeout", stage: "enforcement" }
          : null,
        requestNestedToolGrant: async () => {
          requested++;
        },
        clearNestedToolGrant: () => {
          cancelled++;
        },
        requestLimitedToolGrant: () => {
          throw new Error("wrong consent namespace");
        },
        clearLimitedToolGrant: () => {
          throw new Error("wrong consent namespace");
        },
      }),
  };
  for (const name of [
    "AlertDialog",
    "AlertDialogContent",
    "AlertDialogHeader",
    "AlertDialogTitle",
    "AlertDialogDescription",
    "AlertDialogFooter",
    "AlertDialogCancel",
    "AlertDialogAction",
  ])
    context[name] = name;
  const render = runInNewContext(
    compiled + "\nLimitedModeConfirmDialog;",
    context,
  );
  const find = (node: any, type: string): any =>
    node?.type === type
      ? node
      : node?.children?.map((child: any) => find(child, type)).find(Boolean);
  const props = {
    open: true,
    variant: "nested",
    onOpenChange: (next: boolean) => {
      closed = !next;
    },
  };
  const dialog = render(props);
  assert.deepEqual(find(dialog, "AlertDialogDescription").children, [
    "BACKEND shared /proc disclosure",
  ]);
  assert.equal(requested, 0);
  dialog.props.onOpenChange(false);
  assert.equal(requested, 0);
  assert.equal(closed, true);
  closed = false;
  find(dialog, "AlertDialogAction").props.onClick({ preventDefault() {} });
  await Promise.resolve();
  assert.equal(requested, 1);
  assert.equal(closed, true);
  loading = true;
  render(props).props.onOpenChange(false);
  assert.equal(cancelled, 1);
  failure = true;
  const failedDialog = render(props);
  assert.deepEqual(find(failedDialog, "summary").children, [
    "Diagnostic details",
  ]);
  assert.match(JSON.stringify(find(failedDialog, "details")), /probe_timeout/);
  assert.match(JSON.stringify(find(failedDialog, "details")), /enforcement/);
});
const requestMethod = storeSource.slice(
  storeSource.indexOf("  requestNestedToolGrant: async () => {"),
  storeSource.indexOf("  requestLimitedToolGrant: async () => {"),
);
test("a new permission refusal replaces stale probe diagnostics", () => {
  const setters = storeSource.slice(
    storeSource.indexOf("  setToolExecutionMode: (toolExecutionMode) =>"),
    storeSource.indexOf("  refreshToolIsolationCapability: async () =>"),
  );
  for (const mode of ["container_isolation", "limited", "full", "allowlist"]) {
    let state: any = {
      permissionMode: "auto",
      toolIsolationCapability: null,
      toolIsolationError: "Previous probe timed out",
      toolIsolationErrorDiagnostic: {
        code: "probe_timeout",
        stage: "enforcement",
      },
    };
    const methods = runInNewContext(stripTypeScriptTypes(`({${setters}})`), {
      set: (update: (s: any) => any) => {
        state = { ...state, ...update(state) };
      },
      isNestedGrantCurrent: () => false,
      isLimitedGrantCurrent: () => false,
      capabilityOffersNetworkAllowlist: () => false,
    });
    if (mode === "allowlist") methods.setToolNetworkPolicy(mode);
    else methods.setToolExecutionMode(mode);
    assert.notEqual(state.toolIsolationError, "Previous probe timed out");
    assert.equal(state.toolIsolationErrorDiagnostic, null, mode);
  }
});
const clearStart = storeSource.indexOf("  clearNestedToolGrant: () => {");
const clearMethod = storeSource.slice(
  clearStart,
  storeSource.indexOf("  clearLimitedToolGrant: () =>", clearStart),
);

test("nested probe failure reaches consent state and is cleared on cancellation", async () => {
  let posts = 0;
  const api = client(async () => {
    posts++;
    return {
      ok: false,
      status: 409,
      json: async () => ({
        detail: {
          message: "The isolation check timed out.",
          diagnostic: { code: "probe_timeout", stage: "enforcement" },
        },
      }),
    };
  });
  let state: any = {
    toolIsolationUiSessionId: "page",
    queuedSettingsEpoch: 1,
    toolIsolationDecisionEpoch: 1,
    toolExecutionMode: "os_isolation_required",
    toolIsolationCapability: {
      protection_state: "unavailable",
      nested_eligible: true,
      nested_profile_id: "nested-v1",
      probe_generation: "g1",
    },
  };
  const methods = runInNewContext(
    stripTypeScriptTypes(
      `let nestedGrantRequestId=0; let limitedGrantRequestId=0; ({${requestMethod}${clearMethod}});`,
    ),
    {
      Error,
      get: () => state,
      set: (update: (s: any) => any) => {
        state = { ...state, ...update(state) };
      },
      fetchNestedToolGrant: api.fetchNestedToolGrant,
      isNestedGrantCurrent: api.isNestedGrantCurrent,
    },
  );
  await assert.rejects(methods.requestNestedToolGrant(), /timed out/);
  assert.equal(posts, 1);
  assert.equal(state.toolExecutionMode, "os_isolation_required");
  assert.equal(state.nestedToolGrant, null);
  assert.equal(state.toolIsolationErrorDiagnostic.code, "probe_timeout");
  assert.equal(state.toolIsolationErrorDiagnostic.stage, "enforcement");
  methods.clearNestedToolGrant();
  assert.equal(state.toolIsolationErrorDiagnostic, null);
});

for (const change of [
  "unchanged",
  "cancel",
  "session",
  "settings",
  "generation",
  "eligibility",
  "mode",
] as const) {
  test(`nested consent completion after ${change}`, async () => {
    let resolve!: (grant: any) => void;
    const pending = new Promise((done) => {
      resolve = done;
    });
    let state: any = {
      toolIsolationUiSessionId: "page",
      queuedSettingsEpoch: 1,
      toolIsolationDecisionEpoch: 2,
      toolExecutionMode: "os_isolation_required",
      toolNetworkPolicy: "allowlist",
      nestedToolGrant: null,
      limitedToolGrant: { grant: "old-limited" },
      toolIsolationCapability: {
        protection_state: "unavailable",
        nested_eligible: true,
        nested_profile_id: "nested-v1",
        probe_generation: "g1",
      },
    };
    const api = client(async () => {
      throw new Error("unexpected HTTP");
    });
    const methods = runInNewContext(
      stripTypeScriptTypes(
        `let nestedGrantRequestId = 0; let limitedGrantRequestId = 0; const methods = { ${requestMethod} ${clearMethod} }; methods;`,
      ),
      {
        get: () => state,
        set: (update: (value: any) => any) => {
          state = { ...state, ...update(state) };
        },
        fetchNestedToolGrant: () => pending,
        isNestedGrantCurrent: api.isNestedGrantCurrent,
      },
    );
    const result = methods.requestNestedToolGrant();
    if (change === "cancel") methods.clearNestedToolGrant();
    if (change === "session")
      state = { ...state, toolIsolationUiSessionId: "new-page" };
    if (change === "settings") state = { ...state, queuedSettingsEpoch: 2 };
    if (change === "generation")
      state = {
        ...state,
        toolIsolationCapability: {
          ...state.toolIsolationCapability,
          probe_generation: "g2",
        },
      };
    if (change === "eligibility")
      state = {
        ...state,
        toolIsolationCapability: {
          ...state.toolIsolationCapability,
          nested_eligible: false,
        },
      };
    if (change === "mode") state = { ...state, toolExecutionMode: "limited" };
    resolve({
      mode: "container_isolation",
      grant: "nested-token",
      probe_generation: "g1",
      expires_at: Date.now() + 60000,
    });
    if (change === "unchanged") {
      await result;
      assert.equal(state.toolExecutionMode, "container_isolation");
      assert.equal(state.limitedToolGrant, null);
      assert.equal(state.toolNetworkPolicy, "deny");
      assert.equal(state.toolIsolationDecisionEpoch, 3);
    } else {
      await assert.rejects(result, /changed/);
      assert.equal(state.nestedToolGrant, null);
      assert.notEqual(state.toolExecutionMode, "container_isolation");
    }
  });
}
