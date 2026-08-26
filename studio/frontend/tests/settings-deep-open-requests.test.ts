// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `archivedRequested` and `scrollTarget` are one-shot deep-link requests, cleared only by
// the panel that performs the jump. Panels load on first view, so a navigation can move
// before the chunk arrives: clearing too widely loses a deep-link still being served,
// too narrowly replays a stale one. Both live while the dialog is open on their tab.

import assert from "node:assert/strict";
import test from "node:test";

import { useSettingsDialogStore } from "../src/features/settings/stores/settings-dialog-store.ts";

const store = useSettingsDialogStore;

function reset(): void {
  store.setState({
    open: false,
    activeTab: "general",
    scrollTarget: null,
    opener: null,
    archivedRequested: null,
  });
}

test("an archive deep-open survives until the panel it targets can read it", () => {
  reset();
  store.getState().openArchivedChats();
  assert.equal(store.getState().open, true);
  assert.equal(store.getState().activeTab, "data");
  assert.equal(store.getState().archivedRequested, "chats");
});

test("closing the dialog drops a deep-open the panel never reached", () => {
  reset();
  store.getState().openArchivedChats();
  store.getState().closeDialog();
  assert.equal(store.getState().archivedRequested, null);
});

test("leaving Data drops a deep-open the panel never reached", () => {
  reset();
  store.getState().openArchivedMedia("images");
  store.getState().setActiveTab("about");
  assert.equal(store.getState().archivedRequested, null);
});

test("an ordinary open does not inherit an abandoned deep-open", () => {
  reset();
  store.getState().openArchivedMedia("videos");
  store.getState().openDialog("voice");
  assert.equal(store.getState().archivedRequested, null);
});

test("reselecting Data navigates nowhere, so the deep-open still stands", () => {
  // The nav button fires on the active tab too, and the panel is still on the wire.
  reset();
  store.getState().openArchivedChats();
  store.getState().setActiveTab("data");
  assert.equal(store.getState().archivedRequested, "chats");
});

test("reopening on Data does not drop a deep-open still in flight", () => {
  reset();
  store.getState().openArchivedMedia("images");
  store.getState().openDialog();
  assert.equal(store.getState().archivedRequested, "images");
  store.getState().openDialog("data");
  assert.equal(store.getState().archivedRequested, "images");
});

test("consuming it clears it, so a later visit to Data is an ordinary one", () => {
  reset();
  store.getState().openArchivedChats();
  store.getState().consumeArchivedChatsRequest();
  assert.equal(store.getState().archivedRequested, null);
});

test("a scroll target survives until the panel it targets can read it", () => {
  reset();
  store.getState().openDialog("about", { scrollTarget: "about-updates" });
  assert.equal(store.getState().activeTab, "about");
  assert.equal(store.getState().scrollTarget, "about-updates");
});

test("reselecting the target's own tab keeps the scroll target", () => {
  reset();
  store.getState().openDialog("about", { scrollTarget: "about-updates" });
  store.getState().setActiveTab("about");
  assert.equal(store.getState().scrollTarget, "about-updates");
  store.getState().openDialog();
  assert.equal(store.getState().scrollTarget, "about-updates");
});

test("leaving the target's tab, or closing, drops the scroll target", () => {
  reset();
  store
    .getState()
    .openDialog("appearance", { scrollTarget: "appearance-sidebar-nav" });
  store.getState().setActiveTab("about");
  assert.equal(store.getState().scrollTarget, null);

  reset();
  store
    .getState()
    .openDialog("appearance", { scrollTarget: "appearance-sidebar-nav" });
  store.getState().closeDialog();
  assert.equal(store.getState().scrollTarget, null);
});

test("an open that names its own target replaces one still pending", () => {
  reset();
  store.getState().openDialog("about", { scrollTarget: "about-updates" });
  store
    .getState()
    .openDialog("appearance", { scrollTarget: "appearance-sidebar-nav" });
  assert.equal(store.getState().scrollTarget, "appearance-sidebar-nav");
});

test("consuming a scroll target clears it", () => {
  reset();
  store.getState().openDialog("about", { scrollTarget: "about-updates" });
  store.getState().consumeScrollTarget("about-updates");
  assert.equal(store.getState().scrollTarget, null);
});
