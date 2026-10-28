// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Xet writes chunks out of order, so progress reads 0% then jumps, which looks like a hang.

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
  RESTART_NOTICE_DESCRIPTION,
  RESTART_NOTICE_TITLE,
  RESTART_XET_NOTICE_DESCRIPTION,
  XET_NOTICE_DESCRIPTION,
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
  XET_NOTICE_TITLE,
  composeNoticeDescription,
  composeRestartNoticeDescription,
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
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, transport: "http" }));
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, kind: "dataset" }));
});

test("attaching to someone else's job shows nothing", () => {
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, attached: true }));
});

test("a start that is already stopping shows nothing", () => {
  assert.ok(!shouldShowXetNotice({ ...XET_MODEL, live: false }));
});

test("the predicate does not decide the cap", () => {
  // The limit lives on the server (utils/xet_notice_settings.py); a copy here could only drift.
  const notice = noticeModule as Record<string, unknown>;
  assert.equal(notice.XET_NOTICE_LIMIT, undefined);
  assert.equal(notice.XET_NOTICE_STORAGE_KEY, undefined);
  assert.equal(notice.xetNoticesShown, undefined);
  assert.equal(notice.recordXetNoticeShown, undefined);
});

test("the copy reassures, in plain words", () => {
  assert.match(XET_NOTICE_TITLE, /running/);
  assert.match(XET_NOTICE_DESCRIPTION, /Nothing is stuck/);
});

test("the copy stays short enough to clear the hub toolbar", () => {
  // Measured: a longer description covered the Model hub filter row; 110 chars keeps it clear.
  // Applies to the BASE description only; the composed form is chat-only.
  assert.ok(
    XET_NOTICE_TITLE.length <= 32,
    `title is ${XET_NOTICE_TITLE.length} chars, budget 32`,
  );
  assert.ok(
    XET_NOTICE_DESCRIPTION.length <= 110,
    `description is ${XET_NOTICE_DESCRIPTION.length} chars, budget 110`,
  );
  // A newline costs a whole line and needs a pre-line class.
  assert.ok(!XET_NOTICE_DESCRIPTION.includes("\n"));
  assert.match(XET_NOTICE_DESCRIPTION_CLASS, /text-muted-foreground/);
});

test("the caller's line is folded in rather than raised as a second toast", () => {
  const composed = composeNoticeDescription({
    description: "It'll load automatically once the download finishes.",
  });
  assert.ok(composed.startsWith(XET_NOTICE_DESCRIPTION));
  assert.match(composed, /load automatically once the download finishes\.$/);
  assert.ok(
    composed.includes(`${XET_NOTICE_DESCRIPTION} It'll`),
    `bad seam: ${composed}`,
  );
});

test("a caller with nothing to add leaves the notice alone", () => {
  // The Hub passes none: nothing auto-loads there.
  assert.equal(composeNoticeDescription(), XET_NOTICE_DESCRIPTION);
  assert.equal(composeNoticeDescription(null), XET_NOTICE_DESCRIPTION);
  assert.equal(
    composeNoticeDescription({ description: "   " }),
    XET_NOTICE_DESCRIPTION,
  );
});

test("restart and Xet facts compose into one short notice", () => {
  const composed = composeRestartNoticeDescription({ xet: true });
  assert.equal(RESTART_NOTICE_TITLE, "Restarting this download");
  assert.equal(composed, RESTART_XET_NOTICE_DESCRIPTION);
  assert.match(composed, /can't be resumed/);
  assert.match(composed, /0%/);
  assert.match(composed, /jump to done/);
  assert.ok(RESTART_NOTICE_TITLE.length <= 32);
  assert.ok(composed.length <= 110);
  assert.ok(!composed.includes("\n"));
});

test("restart disclosure survives a spent Xet allowance", () => {
  assert.equal(
    composeRestartNoticeDescription({ xet: false }),
    RESTART_NOTICE_DESCRIPTION,
  );
  assert.match(RESTART_NOTICE_DESCRIPTION, /starting over/);
  assert.ok(RESTART_NOTICE_DESCRIPTION.length <= 110);
});

test("chat auto-load copy folds into the restart notice", () => {
  const composed = composeRestartNoticeDescription({
    xet: true,
    callerToast: {
      description: "It'll load automatically once the download finishes.",
    },
  });
  assert.ok(composed.startsWith(RESTART_XET_NOTICE_DESCRIPTION));
  assert.match(composed, /load automatically once the download finishes\.$/);
});

test("it stays up longer than the Toaster default", () => {
  assert.ok(XET_NOTICE_DURATION_MS > 5000);
});
