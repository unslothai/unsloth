// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readAfterUnload } from "../src/lib/read-after-unload.ts";

test("a probe answered mid-unload is read again once the unload settles", async () => {
  const pending: { current: Promise<unknown> | null } = { current: null };
  let finish!: () => void;
  const answers = [{ active: true }, { active: true }];
  let reads = 0;
  const read = async () => {
    reads += 1;
    if (reads === 1) {
      pending.current = new Promise<void>((resolve) => {
        finish = () => {
          pending.current = null;
          resolve();
        };
      });
    }
    return answers[reads - 1];
  };
  const result = readAfterUnload(pending, read);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(reads, 1);
  finish();
  assert.deepEqual(await result, { active: true });
  assert.equal(reads, 2);
});

test("a rejected unload still leads to a fresh read", async () => {
  const pending: { current: Promise<unknown> | null } = { current: null };
  let reads = 0;
  const value = await readAfterUnload(pending, async () => {
    reads += 1;
    if (reads === 1) {
      pending.current = new Promise((_, reject) => {
        setImmediate(() => {
          pending.current = null;
          reject(new Error("unload failed"));
        });
      });
    }
    return reads;
  });
  assert.equal(value, 2);
});
