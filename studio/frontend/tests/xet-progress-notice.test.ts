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

test("the copy reassures, in plain words", () => {
  // The reassurance is the whole point of the toast, so it leads and it is the
  // only thing here. Explaining chunking, out-of-order writes and batched
  // commits is what made the first version 330 characters.
  assert.match(XET_NOTICE_TITLE, /running/);
  assert.match(XET_NOTICE_DESCRIPTION, /Nothing is stuck/);
});

test("the copy stays short enough to clear the hub toolbar", () => {
  // This test IS the #9293 revert, written down.
  //
  // The first version of this notice ran 62 characters of title and 330 of
  // description. Sonner rendered that 235px tall in the top-right corner,
  // which is where the Model hub keeps its own toolbar, so while the toast
  // was up the capability filter, the sort dropdown, the Models and Datasets
  // tabs and the repo action icons were underneath it. Hit testing each
  // control's own centre point put 4 to 6 of them inside the toast, meaning
  // unclickable, three times per install at 8s each.
  //
  // The budgets below are not guesses. At 149 characters the toast measured
  // 114.5px tall, bottom edge y=126.5, against a filter row whose centre is
  // y=127: clickable by half a pixel, which would not have survived a longer
  // translation or a zoom level. At 101 it ends near y=100 and does not reach
  // the row at all. So 110 is the real ceiling, not the point where it starts
  // to break. Nothing else enforces this, and the failure is invisible to unit
  // tests and to a screenshot taken at the wrong viewport, which is exactly how
  // it shipped the first time.
  assert.ok(
    XET_NOTICE_TITLE.length <= 32,
    `title is ${XET_NOTICE_TITLE.length} chars, budget 32`,
  );
  assert.ok(
    XET_NOTICE_DESCRIPTION.length <= 110,
    `description is ${XET_NOTICE_DESCRIPTION.length} chars, budget 110`,
  );
  // A newline costs a whole line and brings back the pre-line class the first
  // version needed. One paragraph only.
  assert.ok(!XET_NOTICE_DESCRIPTION.includes("\n"));
  assert.match(XET_NOTICE_DESCRIPTION_CLASS, /text-muted-foreground/);
});

test("it stays up longer than the Toaster default", () => {
  // Shorter than the first version, but still two sentences a user has to
  // notice while looking at a progress bar, and it appears at most 3 times
  // ever. 5s is enough to miss entirely.
  assert.ok(XET_NOTICE_DURATION_MS > 5000);
});
