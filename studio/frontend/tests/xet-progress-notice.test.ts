// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Xet writes chunks out of order, so progress reads 0% and then completes at
// once, which looks like a hang. The first few model downloads now say so.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store, storage } = installLocalStorageFake();
registerBundlerResolver();

const {
  XET_NOTICE_DESCRIPTION,
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
  XET_NOTICE_LIMIT,
  XET_NOTICE_STORAGE_KEY,
  XET_NOTICE_TITLE,
  recordXetNoticeShown,
  shouldShowXetNotice,
  xetNoticesShown,
} = await import("../src/features/hub/download-manager/xet-progress-notice.ts");

const XET_MODEL = {
  kind: "model",
  transport: "xet",
  attached: false,
  live: true,
} as const;

test("the notice is for Xet model downloads and nothing else", () => {
  assert.ok(shouldShowXetNotice({ ...XET_MODEL, shown: 0 }));
  // HTTP reports steady progress, so the explanation would be wrong there.
  assert.ok(
    !shouldShowXetNotice({ ...XET_MODEL, transport: "http", shown: 0 }),
  );
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, kind: "dataset", shown: 0 }));
});

test("attaching to someone else's job shows nothing", () => {
  // The backend accepts the start and reports that job's transport, but this
  // client began no transfer, so it must not spend one of the three either.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, attached: true, shown: 0 }));
});

test("a start that is already stopping shows nothing", () => {
  // Cancelling while the start POST is in flight still returns an accepted
  // start. Promising a running download there would be a lie, and it would
  // spend one of the three on it.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, live: false, shown: 0 }));
});

test("it stops after the first three", () => {
  assert.equal(XET_NOTICE_LIMIT, 3);
  assert.ok(shouldShowXetNotice({ ...XET_MODEL, shown: XET_NOTICE_LIMIT - 1 }));
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, shown: XET_NOTICE_LIMIT }));
  // A count already past the limit (an older, larger value) still stops.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, shown: 99 }));
});

// Before anything is recorded, so the session count cannot mask the read.
test("a missing or junk stored count reads as none shown", () => {
  store.clear();
  assert.equal(xetNoticesShown(), 0);
  store.set(XET_NOTICE_STORAGE_KEY, "not a number");
  assert.equal(xetNoticesShown(), 0);
  store.set(XET_NOTICE_STORAGE_KEY, "-4");
  assert.equal(xetNoticesShown(), 0);
});

test("the count persists across sessions", () => {
  store.clear();
  recordXetNoticeShown();
  assert.equal(store.get(XET_NOTICE_STORAGE_KEY), "1");
  recordXetNoticeShown();
  assert.equal(store.get(XET_NOTICE_STORAGE_KEY), "2");
  assert.equal(xetNoticesShown(), 2);

  // What a later session reads back, including one past this session's count.
  store.set(XET_NOTICE_STORAGE_KEY, "3");
  assert.equal(xetNoticesShown(), 3);
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, shown: xetNoticesShown() }));
});

test("a storage that refuses writes still advances the count", () => {
  // Private mode: setItem throws. Without the in-memory carry, every download
  // would show the toast again.
  store.clear();
  const before = xetNoticesShown();
  const setItem = storage.setItem;
  storage.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    recordXetNoticeShown();
  } finally {
    storage.setItem = setItem;
  }
  assert.equal(xetNoticesShown(), before + 1);
  assert.equal(store.get(XET_NOTICE_STORAGE_KEY), undefined);
});

test("the copy says it is running and where to switch", () => {
  // The reassurance is the point of the toast, so it leads.
  assert.match(XET_NOTICE_TITLE, /still running/);
  assert.match(XET_NOTICE_DESCRIPTION, /actively downloading/);
  // The control is the transport toggle in Model Hub, labelled HTTP.
  assert.match(
    XET_NOTICE_DESCRIPTION,
    /'Model Hub' and switch transport to HTTP/,
  );
});

test("the closing advice renders as its own paragraph", () => {
  // A blank line needs pre-line, and the per-toast class replaces the
  // Toaster's rather than merging.
  assert.match(XET_NOTICE_DESCRIPTION, /\n\nFor smoother progress/);
  assert.match(XET_NOTICE_DESCRIPTION_CLASS, /whitespace-pre-line/);
  assert.match(XET_NOTICE_DESCRIPTION_CLASS, /text-muted-foreground/);
});

test("it stays up longer than the Toaster default", () => {
  // The copy does not fit in the 5s every other toast gets.
  assert.ok(XET_NOTICE_DURATION_MS > 5000);
});
