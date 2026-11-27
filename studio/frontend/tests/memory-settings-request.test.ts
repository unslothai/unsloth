// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import { useSettingsDialogStore as store } from "../src/features/settings/stores/settings-dialog-store.ts";

test("Manage memory selects the requested scope, including personal after project", () => {
  store.getState().openDialog("chat", {
    scrollTarget: "chat-memory",
    memoryScope: { projectId: "project-a" },
  });
  assert.equal(store.getState().activeTab, "chat");
  assert.deepEqual(store.getState().memoryScopeRequested, {
    projectId: "project-a",
  });
  store.getState().consumeMemoryScopeRequest();
  assert.equal(store.getState().memoryScopeRequested, null);
  store.getState().openDialog("chat", {
    scrollTarget: "chat-memory",
    memoryScope: { projectId: null },
  });
  assert.deepEqual(store.getState().memoryScopeRequested, { projectId: null });
  store.getState().closeDialog();
  assert.equal(store.getState().memoryScopeRequested, null);
});

test("a memory request survives lazy loading but not leaving Chat settings", () => {
  const open = () =>
    store.getState().openDialog("chat", {
      scrollTarget: "chat-memory",
      memoryScope: { projectId: "project-a" },
    });
  open();
  store.getState().setActiveTab("chat");
  assert.deepEqual(store.getState().memoryScopeRequested, {
    projectId: "project-a",
  });
  store.getState().setActiveTab("general");
  assert.equal(store.getState().memoryScopeRequested, null);
  open();
  store.getState().openDialog("general");
  assert.equal(store.getState().memoryScopeRequested, null);
  open();
  store.getState().openAudioApi({ workflow: "speak", model: null });
  assert.equal(store.getState().memoryScopeRequested, null);
});
