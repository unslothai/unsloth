// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `archivedRequested` is a one-shot deep-open request that only DataTab clears. The panel
// is loaded on first view, so a dialog closed or a tab switched while the chunk is still
// in flight leaves the request set with nothing left to consume it, and the next visit to
// Data opens an archive listing nobody asked for. `scrollTarget` is the same kind of
// request and is already dropped by every navigation away; this holds the two together.

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

test("consuming it clears it, so a later visit to Data is an ordinary one", () => {
  reset();
  store.getState().openArchivedChats();
  store.getState().consumeArchivedChatsRequest();
  assert.equal(store.getState().archivedRequested, null);
});
