// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { claimUntitledForkTitle } = await import(
  "../src/features/chat/utils/untitled-fork-title.ts"
);

test("assistant-ui titles only a thread it created, never one opened as regular", () => {
  const shipped = readFileSync(
    new URL(
      "../node_modules/@assistant-ui/core/dist/react/runtimes/RemoteThreadListHookInstanceManager.js",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    shipped,
    /if \(state\.status === "new"\) \{[\s\S]{0,400}generateTitle\(\)/,
    "assistant-ui changed when it auto-titles; recheck whether the fork trigger is still needed",
  );
});

test("an untitled fork is claimed once", () => {
  const fork = { id: "fork-a", title: "New Chat", forkedFromThreadId: "src" };
  assert.equal(claimUntitledForkTitle(fork), true);
  assert.equal(claimUntitledForkTitle(fork), false);
});

test("titled forks and chats that are not forks are left alone", () => {
  assert.equal(
    claimUntitledForkTitle({
      id: "fork-b",
      title: "Notes (1)",
      forkedFromThreadId: "src",
    }),
    false,
  );
  assert.equal(
    claimUntitledForkTitle({
      id: "plain",
      title: "New Chat",
      forkedFromThreadId: null,
    }),
    false,
  );
  assert.equal(claimUntitledForkTitle(undefined), false);
});

test("the autosave asks for a title when a run ends in an untitled fork", () => {
  const source = readSrc("features/chat/runtime-provider.tsx");
  const runEnd = source.slice(source.indexOf('useAuiEvent("thread.runEnd"'));
  assert.match(runEnd.slice(0, 300), /nameUntitledFork\(threadId\)/);
  const helper = source.slice(source.indexOf("const nameUntitledFork"));
  assert.match(helper, /status !== "regular"/);
  assert.match(helper, /claimUntitledForkTitle\(record\)/);
  assert.match(helper, /item\.generateTitle\(\)/);
});
