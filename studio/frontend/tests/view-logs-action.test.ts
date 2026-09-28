// SPDX-License-Identifier: AGPL-3.0-only Copyright 2026-present the Unsloth AI Inc.

/** The "View logs" route a failure offers, and the request that carries it. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  OWNER_ONLY_SETTINGS_TABS,
  resolveSettingsTab,
} from "../src/features/settings/settings-tab-visibility.ts";
import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";
import { en } from "../src/i18n/locales/en.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const RUNNER_LOG =
  "/home/u/.unsloth/studio/logs/llama-server-20260921-141500.log";
const ACTION_URL = new URL(
  "../src/features/settings/lib/view-logs-action.ts",
  import.meta.url,
);

const store = useSettingsDialogStore;

function reset() {
  store.setState({
    open: false,
    activeTab: "general",
    scrollTarget: null,
    archivedRequested: null,
    logFamilyRequested: null,
    logSourcePathRequested: null,
    connectionRequested: null,
  });
}

test("the action opens Logs on the family that failed", () => {
  reset();
  const translated: string[] = [];
  const { viewLogsAction } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": {
      translate: (key: string) => {
        translated.push(key);
        return `t(${key})`;
      },
    },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });

  const action = viewLogsAction("llama-server");
  // The owner gets one; the non-owner case has its own test below.
  assert.ok(action, "an owner must be offered the action");
  assert.equal(action.label, "t(settings.debugging.viewLogs)");
  assert.deepEqual(translated, ["settings.debugging.viewLogs"]);
  // The key has to exist in the shipped catalogue, or the label renders as the raw key.
  const shipped = "settings.debugging.viewLogs"
    .split(".")
    .reduce<unknown>(
      (node, part) => (node as Record<string, unknown> | undefined)?.[part],
      en as unknown,
    );
  assert.equal(
    typeof shipped,
    "string",
    "no English message for the View logs label",
  );

  assert.equal(store.getState().logFamilyRequested, null);
  action.onClick();
  const after = store.getState();
  assert.equal(after.open, true);
  assert.equal(after.activeTab, "debugging");
  assert.equal(after.logFamilyRequested, "llama-server");
});

test("a generation failure asks for the server log, a load failure for the runner's", () => {
  reset();
  const { viewLogsAction } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });
  for (const family of ["llama-server", "server"] as const) {
    reset();
    const action = viewLogsAction(family);
    assert.ok(action, family);
    action.onClick();
    assert.equal(store.getState().logFamilyRequested, family, family);
  }
});

test("a GGUF diffusion load asks for the diffusion runner's log, not the LLM one", () => {
  reset();
  const { viewLogsAction } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });
  const diffusionAction = viewLogsAction("diffusion-server");
  assert.ok(diffusionAction);
  diffusionAction.onClick();
  assert.equal(store.getState().logFamilyRequested, "diffusion-server");
});

test("only a GGUF load is explained by a runner log; everything else is in the server log", () => {
  const { loadFailureLogFamily } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });

  assert.equal(loadFailureLogFamily(true, false, RUNNER_LOG), "llama-server");
  assert.equal(
    loadFailureLogFamily(true, true, RUNNER_LOG),
    "diffusion-server",
  );
  // A Transformers or MLX load has no runner at all.
  for (const notGguf of [false, undefined] as const) {
    for (const diffusion of [true, false, undefined] as const) {
      assert.equal(
        loadFailureLogFamily(notGguf, diffusion, RUNNER_LOG),
        "server",
        `isGguf=${notGguf} isDiffusion=${diffusion}`,
      );
    }
  }
});

test("a load whose diagnostic names no runner log has none to open", () => {
  const { loadFailureLogFamily } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });
  for (const gguf of [true, false, undefined] as const) {
    for (const diffusion of [true, false, undefined] as const) {
      for (const noPath of [null, undefined, ""] as const) {
        assert.equal(
          loadFailureLogFamily(gguf, diffusion, noPath),
          "server",
          `isGguf=${gguf} isDiffusion=${diffusion} path=${String(noPath)}`,
        );
      }
    }
  }
  assert.equal(loadFailureLogFamily(true, false, RUNNER_LOG), "llama-server");
  assert.equal(
    loadFailureLogFamily(true, true, RUNNER_LOG),
    "diffusion-server",
  );
});

test("the family and the path are read from the same diagnostic", () => {
  const src = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.ok(
    src.includes("const runnerLogPath = failureLogPath(message);"),
    "the hook no longer derives the runner log path from the diagnostic",
  );
  assert.ok(
    src.includes("loadFailureLogFamily(isGguf, isDiffusion, runnerLogPath),"),
    "the family no longer comes from the diagnostic's own path",
  );
  assert.match(
    src,
    /viewLogsAction\(\s*loadFailureLogFamily\(isGguf, isDiffusion, runnerLogPath\),\s*runnerLogPath,\s*\)/,
    "the family and the path handed to the action are no longer the same value",
  );
  assert.ok(
    !/loadFailureLogFamily\([^)]*loadRequestIssued/.test(src),
    "the hook still equates issuing the request with a runner having started",
  );
  assert.match(
    src,
    /runnerLogPath \|\| loadRequestIssued\s*\?\s*viewLogsAction\(/,
    "a failure before the request was sent still offers the server log",
  );
});

test("the exact log the diagnostic named is carried, and wins over family recency", () => {
  reset();
  const { viewLogsAction, failureLogPath } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => true },
  });

  // The wording llama_cpp.py appends, and treats as a diagnostics marker.
  const diagnostic =
    "llama-server failed to start\n\nllama-server output:\n  ggml_abort\n\n" +
    "Full log: /home/u/.unsloth/studio/logs/llama-server/llama-1765000000-port-8080.log";
  const path = failureLogPath(diagnostic);
  assert.equal(
    path,
    "/home/u/.unsloth/studio/logs/llama-server/llama-1765000000-port-8080.log",
  );
  assert.equal(failureLogPath("Failed to load model"), null);
  assert.equal(failureLogPath("Full log: "), null);

  const pathAction = viewLogsAction("llama-server", path);
  assert.ok(pathAction);
  pathAction.onClick();
  assert.equal(store.getState().logSourcePathRequested, path);
  // Cleared with the family, so a later visit cannot land on a stale file.
  store.getState().consumeLogFamilyRequest();
  assert.equal(store.getState().logSourcePathRequested, null);
});

test("the Logs panel clears the request once it has consumed it", () => {
  reset();
  store.getState().openLogs("server");
  assert.equal(store.getState().logFamilyRequested, "server");
  store.getState().consumeLogFamilyRequest();
  assert.equal(store.getState().logFamilyRequested, null);
  // Still on the tab: consuming the request must not close or navigate the dialog.
  assert.equal(store.getState().activeTab, "debugging");
  assert.equal(store.getState().open, true);
});

test("an unconsumed request survives landing back on Logs and is dropped by any other tab", () => {
  reset();
  store.getState().openLogs("llama-server");
  // Reselecting the tab that reads it keeps it: the panel may not have mounted yet.
  store.getState().openDialog("debugging");
  assert.equal(store.getState().logFamilyRequested, "llama-server");
  // Anything else abandons it, so it cannot replay on a later visit.
  store.getState().openDialog("general");
  assert.equal(store.getState().logFamilyRequested, null);

  reset();
  store.getState().openLogs("llama-server");
  store.getState().setActiveTab("about");
  assert.equal(store.getState().logFamilyRequested, null);
});

test("closing the dialog drops the request, like every other pending one", () => {
  reset();
  store.getState().openLogs("server");
  store.getState().closeDialog();
  const after = store.getState();
  assert.equal(after.open, false);
  assert.equal(after.logFamilyRequested, null);
  assert.equal(after.archivedRequested, null);
  assert.equal(after.connectionRequested, null);
});

test("the sibling openers do not leave a stale log request behind", () => {
  for (const open of [
    () => store.getState().openArchivedChats(),
    () => store.getState().openArchivedMedia("images"),
    () => store.getState().openConnectionSettings("openai"),
  ]) {
    reset();
    store.getState().openLogs("server");
    open();
    assert.equal(store.getState().logFamilyRequested, null);
  }
});

test("an account that cannot open Logs is offered no action at all", () => {
  reset();
  const { viewLogsAction } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    "@/i18n": { translate: (k: string) => k },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
    "@/features/auth/account-session": { isAccountOwner: () => false },
  });
  for (const family of [
    "llama-server",
    "diffusion-server",
    "server",
  ] as const) {
    assert.equal(viewLogsAction(family), undefined, family);
    assert.equal(viewLogsAction(family, "/some/log.log"), undefined, family);
  }
  // And nothing was requested as a side effect of asking.
  assert.equal(store.getState().logFamilyRequested, null);
  assert.equal(store.getState().open, false);
});

test("the owner-only tab list is what the gate is gating on", () => {
  assert.equal(
    OWNER_ONLY_SETTINGS_TABS.has("debugging"),
    true,
    "Logs is no longer owner-only, so the action no longer needs gating",
  );
  assert.equal(resolveSettingsTab("debugging", false), "general");
  assert.equal(resolveSettingsTab("debugging", true), "debugging");
});

test("a request arriving while Logs is already open is visible to a subscriber", async () => {
  const { pendingLogRequestKey, NO_PENDING_LOG_REQUEST } = await import(
    "../src/features/settings/stores/settings-dialog-store.ts"
  );
  reset();
  assert.equal(
    pendingLogRequestKey(store.getState()),
    NO_PENDING_LOG_REQUEST,
    "an idle store must read as nothing pending, or the panel refreshes on every change",
  );

  store.getState().openLogs("llama-server", "/logs/llama-a.log");
  const first = pendingLogRequestKey(store.getState());
  assert.notEqual(first, NO_PENDING_LOG_REQUEST);

  store.getState().openLogs("llama-server", "/logs/llama-b.log");
  assert.notEqual(pendingLogRequestKey(store.getState()), first);

  // And consuming it settles, so the arrival fires the panel once rather than looping.
  store.getState().consumeLogFamilyRequest();
  assert.equal(pendingLogRequestKey(store.getState()), NO_PENDING_LOG_REQUEST);

  // A family-only request still registers: not every diagnostic carries a path.
  store.getState().openLogs("server");
  assert.notEqual(
    pendingLogRequestKey(store.getState()),
    NO_PENDING_LOG_REQUEST,
  );
});

test("an older in-flight refresh does not consume a newer request", async () => {
  const { pendingLogRequestKey, NO_PENDING_LOG_REQUEST } = await import(
    "../src/features/settings/stores/settings-dialog-store.ts"
  );
  reset();

  // What the older fetch captured: nothing pending.
  const capturedByOlderFetch = pendingLogRequestKey(store.getState());
  assert.equal(capturedByOlderFetch, NO_PENDING_LOG_REQUEST);

  // The click lands while that fetch is still out.
  store.getState().openLogs("llama-server", "/logs/llama-failed.log");
  const nowPending = pendingLogRequestKey(store.getState());
  assert.notEqual(
    nowPending,
    capturedByOlderFetch,
    "the older fetch would be unable to tell it was answering someone else's request",
  );

  const capturedByNewerFetch = pendingLogRequestKey(store.getState());
  assert.equal(capturedByNewerFetch, nowPending);
  store.getState().consumeLogFamilyRequest();
  assert.equal(pendingLogRequestKey(store.getState()), NO_PENDING_LOG_REQUEST);

  const tab = await readFile(
    new URL("../src/features/settings/tabs/debugging-tab.tsx", import.meta.url),
    "utf8",
  );
  for (const needle of [
    "const requestedFor = pendingLogRequestKey(",
    "pendingLogRequestKey(dialog) === requestedFor",
    "if (fromFailure && stillTheSameRequest)",
    "if (fromFailure && !stillTheSameRequest) return;",
  ]) {
    assert.ok(
      tab.includes(needle),
      `the panel no longer guards consumption on the request it fetched for: ${needle}`,
    );
  }
});
