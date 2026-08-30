// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Reproduction harness for https://github.com/unslothai/unsloth/issues/9984
 *
 * The reported bug: Regenerate inserts a new user row (same parent_id, cloned
 * attachments) instead of reusing the existing user message. This file checks
 * whether assistant-ui's standard Reload path does that, and documents the
 * persistence shape the autosave would write when it does.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { MessageRepository } from "@assistant-ui/core/internal";
import type { ThreadMessage } from "@assistant-ui/react";
import { dedupeIdenticalUserSiblings } from "../src/features/chat/utils/dedupe-identical-user-siblings.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";

const priorId = "cOfdER0";
const userId = "SaKf868";

function mkUser(id: string): ThreadMessage {
  return {
    id,
    role: "user",
    content: [{ type: "text", text: "Good, let me share document v3.1" }],
    attachments: [
      {
        type: "document",
        name: "doc.md",
        contentType: "text/markdown",
        content: "x".repeat(1000),
      },
    ],
    status: { type: "complete", reason: "unknown" },
    createdAt: new Date(),
    metadata: { custom: {} },
  } as ThreadMessage;
}

function mkAssistant(id: string): ThreadMessage {
  return {
    id,
    role: "assistant",
    content: [{ type: "text", text: "Here is my response" }],
    status: { type: "complete", reason: "unknown" },
    createdAt: new Date(),
    metadata: { custom: {} },
  } as ThreadMessage;
}

function userSiblingsUnder(
  repo: MessageRepository,
  parentId: string,
): string[] {
  const exp = repo.export();
  return exp.messages
    .filter(
      ({ message, parentId: pid }) =>
        message.role === "user" && pid === parentId,
    )
    .map(({ message }) => message.id);
}

test("assistant-ui Reload adds an assistant sibling only, not a duplicate user", () => {
  const repo = new MessageRepository();
  repo.addOrUpdateMessage(null, mkAssistant(priorId));
  repo.addOrUpdateMessage(priorId, mkUser(userId));
  repo.addOrUpdateMessage(userId, mkAssistant("a1"));

  assert.deepEqual(userSiblingsUnder(repo, priorId), [userId]);

  // ActionBarPrimitive.Reload -> message.reload() -> startRun({ parentId: userId })
  repo.addOrUpdateMessage(userId, {
    id: "a1b",
    role: "assistant",
    content: [],
    status: { type: "running" },
    createdAt: new Date(),
    metadata: { custom: {} },
  } as ThreadMessage);
  repo.resetHead("a1b");

  assert.deepEqual(
    userSiblingsUnder(repo, priorId),
    [userId],
    "standard Reload must not insert another user row",
  );
});

test("the issue's DB shape is what autosave would persist if a duplicate user were appended", () => {
  const repo = new MessageRepository();
  repo.addOrUpdateMessage(null, mkAssistant(priorId));
  repo.addOrUpdateMessage(priorId, mkUser(userId));
  repo.addOrUpdateMessage(userId, mkAssistant("a1"));

  // Bug pattern from #9984: a second user sibling under the same parent.
  const dupId = "oHXbD51";
  repo.addOrUpdateMessage(priorId, mkUser(dupId));
  repo.addOrUpdateMessage(dupId, mkAssistant("a2"));

  const siblings = userSiblingsUnder(repo, priorId);
  assert.equal(siblings.length, 2);
  assert.deepEqual(siblings.sort(), [dupId, userId].sort());

  const users = repo
    .export()
    .messages.filter(({ message }) => message.role === "user");
  assert.equal(users.length, 2);
  assert.ok(
    users.every(({ parentId }) => parentId === priorId),
    "both user rows share the same parent_id, matching the issue report",
  );
  assert.ok(
    users.every(
      ({ message }) =>
        message.role === "user" && (message.attachments?.length ?? 0) > 0,
    ),
    "attachments are cloned onto each duplicate user row",
  );
});

test("autosave dedupe collapses the issue's duplicate user siblings before sync", () => {
  const repo = new MessageRepository();
  repo.addOrUpdateMessage(null, mkAssistant(priorId));
  repo.addOrUpdateMessage(priorId, mkUser(userId));
  repo.addOrUpdateMessage(userId, mkAssistant("a1"));

  const dupId = "oHXbD51";
  repo.addOrUpdateMessage(priorId, mkUser(dupId));
  repo.addOrUpdateMessage(dupId, mkAssistant("a2"));

  const toRecord = (
    parentId: string | null,
    message: ThreadMessage,
  ): MessageRecord => ({
    id: message.id,
    threadId: "thread-1",
    parentId,
    role: message.role,
    content: message.content,
    ...(message.role === "user" &&
      message.attachments &&
      message.attachments.length > 0 && { attachments: [...message.attachments] }),
    createdAt: message.createdAt?.getTime?.() ?? Date.now(),
  });

  const records = repo
    .export()
    .messages.map(({ message, parentId }) => toRecord(parentId, message));
  const { records: deduped, collapsedIds } =
    dedupeIdenticalUserSiblings(records);

  assert.equal(collapsedIds.length, 1);
  assert.ok(collapsedIds[0] === userId || collapsedIds[0] === dupId);
  const survivingUser = deduped.find((record) => record.role === "user");
  assert.ok(survivingUser);
  assert.equal(deduped.filter((record) => record.role === "user").length, 1);
  assert.equal(
    deduped.find((record) => record.id === "a2")?.parentId,
    survivingUser.id,
  );
});
