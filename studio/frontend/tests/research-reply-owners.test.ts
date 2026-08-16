// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every user message's action bar asks whether that message owns a research reply, and the
// answer needs the whole repository rather than the visible message list, because a reply can
// sit on a branch the current view is not showing. Asking it per message meant one full export
// per user message on every thread change: quadratic in thread length, and paid on every token
// of a generation as well as on every delete.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ExportedReplyItem,
  researchReplyOwners,
} from "../src/components/assistant-ui/research-reply-owners.ts";

const research = (id: string) => ({ custom: { researchRunId: id } });

function items(): ExportedReplyItem[] {
  return [
    { parentId: null, message: { metadata: {} } },
    { parentId: "prompt-1", message: { metadata: research("run-a") } },
    { parentId: "prompt-2", message: { metadata: {} } },
    // A second reply under the same prompt, on another branch. The visible list holds one of
    // them at most, which is why this cannot be answered from thread.messages.
    { parentId: "prompt-1", message: { metadata: research("run-b") } },
  ];
}

const isResearch = (metadata: unknown) =>
  typeof (metadata as { custom?: { researchRunId?: unknown } } | undefined)
    ?.custom?.researchRunId === "string";

test("collects the parents of research replies and nothing else", () => {
  const owners = researchReplyOwners({}, items, isResearch);

  assert.ok(owners.has("prompt-1"));
  // Has a reply, but not a research one.
  assert.ok(!owners.has("prompt-2"));
  assert.equal(owners.size, 1);
});

test("a rootless research reply names no owner", () => {
  const owners = researchReplyOwners(
    {},
    () => [{ parentId: null, message: { metadata: research("run-a") } }],
    isResearch,
  );

  assert.equal(owners.size, 0);
});

test("one export serves every message at the same revision", () => {
  const revision = {};
  let exports = 0;
  const read = () => {
    exports += 1;
    return items();
  };

  // Stands in for the action bars of a thread: each asks about its own message, at one revision.
  for (const messageId of ["prompt-1", "prompt-2", "prompt-3"]) {
    researchReplyOwners(revision, read, isResearch).has(messageId);
  }

  assert.equal(exports, 1);
});

test("a new revision is exported again, and sees the change", () => {
  const before = researchReplyOwners({}, items, isResearch);
  assert.ok(before.has("prompt-1"));

  let exports = 0;
  const after = researchReplyOwners(
    {},
    () => {
      exports += 1;
      // The research reply has been deleted.
      return items().filter(({ parentId }) => parentId !== "prompt-1");
    },
    isResearch,
  );

  assert.equal(exports, 1);
  assert.ok(!after.has("prompt-1"));
});
