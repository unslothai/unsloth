// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The "View logs" route a failure offers, and the request that carries it.
 *
 * A failure toast is the only thing pointing at Settings > Logs, so if the action or the
 * request it sets stops working the reported experience -- a failure with no reason and no
 * route to one -- comes straight back, silently. Three call sites raise it (a GGUF load, an
 * image generation, a video generation) and none of them had a test.
 *
 * The request's LIFETIME is asserted here too. Only the panel that performs the jump clears
 * it, and panels are fetched on first view, so a request has to survive a navigation that
 * lands back on its own tab and be dropped by anything else. Held wider it replays on a
 * later visit; held narrower a deep link still in flight is lost.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

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
    connectionRequested: null,
  });
}

test("the action opens Logs on the family that failed", () => {
  reset();
  const translated: string[] = [];
  const { viewLogsAction } = loadWithStubs<
    typeof import("../src/features/settings/lib/view-logs-action.ts")
  >(ACTION_URL, {
    // Not the useT hook: two of the three call sites raise their toast from a callback
    // outside a component body, so the action has to translate without one.
    "@/i18n": {
      translate: (key: string) => {
        translated.push(key);
        return `t(${key})`;
      },
    },
    "../stores/settings-dialog-store": { useSettingsDialogStore: store },
  });

  const action = viewLogsAction("llama-server");
  assert.equal(action.label, "t(settings.debugging.viewLogs)");
  assert.deepEqual(translated, ["settings.debugging.viewLogs"]);
  // The key has to exist in the shipped catalogue, or the label renders as the raw key.
  const shipped = "settings.debugging.viewLogs"
    .split(".")
    .reduce<unknown>(
      (node, part) => (node as Record<string, unknown> | undefined)?.[part],
      en as unknown,
    );
  assert.equal(typeof shipped, "string", "no English message for the View logs label");

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
  });
  // The runner writes its own file per attempt, so a load's reason is there rather than in
  // the server log; the diffusion runners log through the backend's own stream.
  for (const family of ["llama-server", "server"] as const) {
    reset();
    viewLogsAction(family).onClick();
    assert.equal(store.getState().logFamilyRequested, family, family);
  }
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
  // Each opener nulls every request it does not set, so arriving at Data or Connections
  // cannot carry a log family into a panel that never reads it.
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
