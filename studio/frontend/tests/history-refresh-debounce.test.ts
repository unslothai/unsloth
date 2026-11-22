// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Projects page reloads open project rows on history events. A streamed reply's chunk saves
// wait for a lull, but must not push back a rename or delete that lands mid-stream.

import assert from "node:assert/strict";
import test, { mock } from "node:test";

const { createHistoryRefreshDebounce } = await import(
  "../src/features/chat/utils/chat-history-revision.ts"
);

const chunk = () => new CustomEvent("x", { detail: { coalesce: true } });

test("chunk saves refresh once on a lull, not per chunk", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let runs = 0;
    const debounce = createHistoryRefreshDebounce(
      () => {
        runs += 1;
      },
      300,
      2000,
    );
    for (let i = 0; i < 20; i += 1) {
      debounce.onEvent(chunk());
      mock.timers.tick(500);
    }
    assert.equal(runs, 0, "chunks spaced under the lull keep postponing");
    mock.timers.tick(2000);
    assert.equal(runs, 1);
  } finally {
    mock.timers.reset();
  }
});

test("a structural change during a stream refreshes on its own deadline", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let runs = 0;
    const debounce = createHistoryRefreshDebounce(
      () => {
        runs += 1;
      },
      300,
      2000,
    );
    debounce.onEvent(chunk());
    mock.timers.tick(100);
    debounce.onEvent(new Event("x"));
    for (let i = 0; i < 3; i += 1) {
      mock.timers.tick(100);
      debounce.onEvent(chunk());
    }
    assert.equal(runs, 1, "a rename or delete waited for the stream to end");
    mock.timers.tick(1999);
    assert.equal(runs, 1);
    mock.timers.tick(1);
    assert.equal(
      runs,
      2,
      "chunks after the structural run still get their lull",
    );
  } finally {
    mock.timers.reset();
  }
});

test("cancel drops a pending refresh", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let runs = 0;
    const debounce = createHistoryRefreshDebounce(
      () => {
        runs += 1;
      },
      300,
      2000,
    );
    debounce.onEvent(new Event("x"));
    debounce.cancel();
    mock.timers.tick(5000);
    assert.equal(runs, 0);
  } finally {
    mock.timers.reset();
  }
});
