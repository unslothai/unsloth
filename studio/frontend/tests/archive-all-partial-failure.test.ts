// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Archive All sends every PATCH with { notify: false } and relies on one batch notification
// afterwards. Promise.all rejects on the first failure, so without a catch that notification
// never runs: whatever did archive stays listed here and in every other tab until some
// unrelated history change.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./sidebar-items-resolver.mjs", import.meta.url);

const { recorder, resetRecorder } = await import(
  "./helpers/store-stubs/sidebar-items-deps.ts"
);
const { archiveAllChatItems } = await import(
  "../src/features/chat/hooks/use-chat-sidebar-items.ts"
);

const threads = [{ id: "a" }, { id: "b" }, { id: "c" }];

test("a partially failed Archive All still announces what did archive", async () => {
  resetRecorder(threads, ["b"]);

  await assert.rejects(() => archiveAllChatItems(), /PATCH failed for b/);

  assert.deepEqual(
    recorder.patched.sort(),
    ["a", "c"],
    "the other two threads should still have been archived",
  );
  assert.equal(
    recorder.notifications,
    1,
    "the batch must announce itself even though the batch threw",
  );
});

test("a fully successful Archive All announces exactly once", async () => {
  resetRecorder(threads);

  const archived = await archiveAllChatItems();

  assert.equal(archived, 3);
  assert.deepEqual(recorder.patched.sort(), ["a", "b", "c"]);
  assert.equal(
    recorder.notifications,
    1,
    "the per-thread updates stay silent, so exactly one batch notification",
  );
});

test("Archive All with nothing to archive does not notify", async () => {
  resetRecorder([{ id: "a", archived: true }]);

  assert.equal(await archiveAllChatItems(), 0);
  assert.equal(recorder.notifications, 0);
  assert.deepEqual(recorder.patched, []);
});
