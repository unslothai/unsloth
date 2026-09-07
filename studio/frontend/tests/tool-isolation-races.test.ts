// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import ts from "typescript";
import { protectedIsolationDefaults } from "../src/features/chat/utils/tool-isolation-defaults.ts";
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
const permissionSelect = read("../src/features/chat/permission-mode-select.tsx");
for (const persisted of ["ask", "auto", "off", "full"] as const) {
  test(`Deep Research revokes elevated execution using current chat level ${persisted}`, () => {
    const start = store.indexOf(
      "  setDeepResearchEnabled: (deepResearchEnabled) =>",
    );
    const method = store.slice(
      start,
      store.indexOf("  setResearchWebsitePolicy:", start),
    );
    for (const mode of ["full", "limited"]) {
      let state: Record<string, unknown> = {
        permissionMode: "full",
        toolExecutionMode: mode,
        limitedToolGrant: { grant: "old" },
        bypassPermissions: true,
        queuedSettingsEpoch: 4,
      };
      const context: Record<string, unknown> = {
        set: (
          update: (state: Record<string, unknown>) => Record<string, unknown>,
        ) => {
          state = { ...state, ...update(state) };
        },
        saveBool() {},
        threadScopedOverride: () => persisted,
        loadPermissionMode: () => {
          throw Error("chat override must take precedence");
        },
        protectedIsolationDefaults,
      };
      for (const key of method.matchAll(/\bCHAT_[A-Z_]+_KEY\b/g))
        context[key[0]] = key[0];
      const change = runInNewContext(
        stripTypeScriptTypes(
          `const methods = { ${method} }; methods.setDeepResearchEnabled;`,
        ),
        context,
      );
      change(true);
      assert.equal(
        state.permissionMode,
        persisted === "full" ? "auto" : persisted,
      );
      assert.equal(state.toolExecutionMode, "os_isolation_required");
      assert.equal(state.limitedToolGrant, null);
      assert.equal(state.bypassPermissions, false);
      assert.equal(state.toolNetworkPolicy, "deny");
      assert.equal(state.codeToolsEnabled, false);
      assert.equal(state.queuedSettingsEpoch, 5);
    }
  });
}

test("composer hides inactive code isolation but retains Full and Limited warnings", () => {
  const component = permissionSelect.slice(
    permissionSelect.indexOf("export function PermissionModeComposerPill("),
  );
  const compiled = ts.transpileModule(
    component.replace("export function", "function"),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.React,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  for (const [codeToolsEnabled, mode, shown] of [
    [false, "os_isolation_required", false],
    [true, "os_isolation_required", true],
    [false, "full", true],
    [false, "limited", true],
  ] as const) {
    const state = {
      codeToolsEnabled,
      toolExecutionMode: mode,
      permissionMode: mode === "full" ? "full" : "auto",
    };
    const context: Record<string, unknown> = {
      React: {
        createElement: (
          type: unknown,
          props: unknown,
          ...children: unknown[]
        ) => ({ type, props, children }),
      },
      useToolIsolationCapabilityRefresh() {},
      useState: () => [false, () => {}],
      useChatRuntimeStore: (select: (state: unknown) => unknown) =>
        select(state),
      permissionModeOption: () => ({
        label: "Approve for me",
        description: "Approval description",
        icon: "Icon",
      }),
      toolIsolationPresentation: () => ({ label: "Isolation status" }),
    };
    for (const name of [
      "DropdownMenu",
      "DropdownMenuTrigger",
      "HugeiconsIcon",
      "ChevronDownStandardIcon",
      "DropdownMenuContent",
      "DropdownMenuLabel",
      "PermissionModeMenuItems",
      "ToolIsolationMenuSection",
      "LimitedModeConfirmDialog",
    ])
      context[name] = name;
    const tree = runInNewContext(
      `${compiled}\nPermissionModeComposerPill;`,
      context,
    )();
    const button = tree.children[0].children[0].children[0];
    assert.equal(
      button.props.title,
      shown
        ? "Approve for me: Approval description. Isolation status."
        : "Approve for me: Approval description",
    );
    assert.equal(Boolean(button.children[2]), shown);
  }
});
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

for (const mode of ["full", "limited", "os_isolation_required"]) {
  test(`a queued ${mode} decision cannot survive a consent round trip`, async () => {
    const live = {
      toolExecutionMode: mode,
      toolIsolationUiSessionId: "same-session",
      toolIsolationDecisionEpoch: 3,
      toolNetworkPolicy: "deny",
      limitedToolGrant: grant("replacement"),
      toolIsolationCapability: { protection_state: "unavailable", probe_generation: "g" },
      refreshToolIsolationCapability: async () => {},
      setToolIsolationConsentOpen: () => {},
    };
    const runtime = { ...live, toolIsolationDecisionEpoch: 1, limitedToolGrant: grant("earlier") };
    const result = runInNewContext(
      stripTypeScriptTypes(`async function run() { ${gate}\nreturn toolIsolationRequestFields; } run();`),
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
    await assert.rejects(result, /authorization changed|capability changed/);
  });
}

type GrantState = {
  toolIsolationDecisionEpoch: number;
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

test("permission revocation after preparation is checked again at serialization", async () => {
  const live = { toolExecutionMode: "full", toolIsolationUiSessionId: "session", toolIsolationDecisionEpoch: 1, toolNetworkPolicy: "deny" };
  const runtime = { ...live, limitedToolGrant: null };
  const validate = await runInNewContext(
    stripTypeScriptTypes(`async function run() { ${gate}\nreturn currentToolIsolationRequestFields; } run();`),
    { runtime, useChatRuntimeStore: { getState: () => live }, supportsStudioToolsForThisTurn: true, runsStudioPythonOrTerminal: true, queuedIsolationDecisionIsCurrent, queuedToolNetworkPolicy },
  );
  assert.equal(validate().tool_execution_mode, "full");
  live.toolIsolationDecisionEpoch += 2;
  assert.throws(validate, /permissions changed/);
  assert.equal((adapter.match(/\.\.\.currentToolIsolationRequestFields\(\)/g) ?? []).length, 2);
});

function grantHarness() {
  let state: GrantState = {
    toolIsolationDecisionEpoch: 0,
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

test("closing the actual Limited dialog rejects its pending grant response", async () => {
  const h = grantHarness();
  const pending = h.method();
  const rejected = assert.rejects(pending, /permissions changed/);
  const clearStart = store.indexOf("  clearLimitedToolGrant: () =>", grantStart);
  const clearMethod = store.slice(clearStart, store.indexOf("  setToolIsolationConsentOpen:", clearStart));
  const clear = runInNewContext(
    stripTypeScriptTypes(`const methods = { ${clearMethod} }; methods.clearLimitedToolGrant;`),
    { set: (update: (state: GrantState) => Partial<GrantState>) => h.change(update(h.get())) },
  );
  const start = permissionSelect.indexOf("export function LimitedModeConfirmDialog(");
  const component = permissionSelect.slice(start, permissionSelect.indexOf("/** Page-root dialog", start));
  const compiled = ts.transpileModule(component.replace("export function", "function"), {
    compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  let closed = false;
  const context: Record<string, unknown> = {
    React: { createElement: (type: unknown, props: unknown, ...children: unknown[]) => ({ type, props, children }) },
    useChatRuntimeStore: (select: (state: unknown) => unknown) => select({ ...h.get(), requestLimitedToolGrant: h.method, clearLimitedToolGrant: clear }),
    limitedModeWarning: () => "warning",
  };
  for (const name of ["AlertDialog", "AlertDialogContent", "AlertDialogHeader", "AlertDialogTitle", "AlertDialogDescription", "AlertDialogFooter", "AlertDialogCancel", "AlertDialogAction"]) context[name] = name;
  const render = runInNewContext(`${compiled}\nLimitedModeConfirmDialog;`, context);
  const dialog = render({ open: true, onOpenChange: (open: boolean) => { closed = !open; } });
  const cancel = dialog.children[0].children[2].children[0];
  assert.equal(cancel.type, "AlertDialogCancel");
  assert.notEqual(cancel.props?.disabled, true);
  dialog.props.onOpenChange(false);
  assert.equal(closed, true);
  h.requests[0].resolve(grant("dismissed"));
  await rejected;
  assert.equal(h.get().toolExecutionMode, "os_isolation_required");
  assert.equal(h.get().limitedToolGrant, null);
  assert.equal(h.get().toolIsolationGrantLoading, false);
});
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
