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

installLocalStorageFake();
registerBundlerResolver();

const noticeModule = await import(
  "../src/features/hub/download-manager/xet-progress-notice.ts"
);
const {
  XET_NOTICE_DESCRIPTION,
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
  XET_NOTICE_TITLE,
  composeNoticeDescription,
  shouldShowXetNotice,
} = noticeModule;

const XET_MODEL = {
  kind: "model",
  transport: "xet",
  attached: false,
  live: true,
} as const;

test("the notice is for Xet model downloads and nothing else", () => {
  assert.ok(shouldShowXetNotice({ ...XET_MODEL }));
  // HTTP reports steady progress, so the explanation would be wrong there.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, transport: "http" }));
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, kind: "dataset" }));
});

test("attaching to someone else's job shows nothing", () => {
  // Accepted, and it reports that job's transport, but this user started nothing.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, attached: true }));
});

test("a start that is already stopping shows nothing", () => {
  // Nothing to reassure the user about if the download is on its way out.
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, live: false }));
});

test("the predicate does not decide the cap", () => {
  // The limit lives on the server (utils/xet_notice_settings.py); a copy here could
  // only drift out of step with what is enforced. The predicate answers "is this
  // worth explaining", and the reservation is the only thing that counts.
  const notice = noticeModule as Record<string, unknown>;
  assert.equal(notice.XET_NOTICE_LIMIT, undefined);
  assert.equal(notice.XET_NOTICE_STORAGE_KEY, undefined);
  assert.equal(notice.xetNoticesShown, undefined);
  assert.equal(notice.recordXetNoticeShown, undefined);
});

test("the copy reassures, in plain words", () => {
  // The reassurance is the point. Explaining chunking and batched commits is what
  // made the first version 330 characters.
  assert.match(XET_NOTICE_TITLE, /running/);
  assert.match(XET_NOTICE_DESCRIPTION, /Nothing is stuck/);
});

test("the copy stays short enough to clear the hub toolbar", () => {
  // This test IS the #9293 revert, written down. At 62 + 330 chars the toast rendered
  // 235px tall over the Model hub toolbar, leaving 4 to 6 controls unclickable for 8s.
  //
  // The budgets are measured, not guessed. At 149 chars the bottom edge sat at y=126.5
  // against a filter row centred at y=127: clickable by half a pixel. At 101 it ends
  // near y=100 and never reaches the row, so 110 is the ceiling rather than the point
  // it breaks. Applies to the BASE description only; the composed form is chat-only
  // and has no toolbar beneath it.
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

test("the caller's line is folded in rather than raised as a second toast", () => {
  // Reported on main: chat produced the Xet notice AND its own "Downloading model",
  // stacked. Suppressing one loses information, so the sentence joins the notice.
  const composed = composeNoticeDescription({
    description: "It'll load automatically once the download finishes.",
  });
  assert.ok(composed.startsWith(XET_NOTICE_DESCRIPTION));
  assert.match(composed, /load automatically once the download finishes\.$/);
  // Exactly one space at the seam, not a double space or a missing one.
  assert.ok(
    composed.includes(`${XET_NOTICE_DESCRIPTION} It'll`),
    `bad seam: ${composed}`,
  );
});

test("a caller with nothing to add leaves the notice alone", () => {
  // The Hub passes none: nothing auto-loads there, so the sentence would be false.
  assert.equal(composeNoticeDescription(), XET_NOTICE_DESCRIPTION);
  assert.equal(composeNoticeDescription(null), XET_NOTICE_DESCRIPTION);
  assert.equal(
    composeNoticeDescription({ description: "   " }),
    XET_NOTICE_DESCRIPTION,
  );
});

test("it stays up longer than the Toaster default", () => {
  // Two sentences to notice while watching a progress bar, at most 3 times ever, so
  // 5s is enough to miss. An upper bound only: it is dismissed when the transfer ends.
  assert.ok(XET_NOTICE_DURATION_MS > 5000);
});
