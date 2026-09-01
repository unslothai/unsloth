// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Regeneration reuses its user id; equal-content user ids remain distinct branches. */

import assert from "node:assert/strict";
import { test } from "node:test";
import { MessageRepository } from "@assistant-ui/core/internal";
import type { ThreadMessage } from "@assistant-ui/react";

const priorId = "cOfdER0";
const userId = "SaKf868";

const messageMetadata = {
  unstable_state: null,
  unstable_annotations: [],
  unstable_data: [],
  steps: [],
  custom: {},
} as const;

function mkUser(id: string, text = "Good, let me share document v3.1"): ThreadMessage {
  return {
    id,
    role: "user",
    content: [{ type: "text", text }],
    attachments: [
      {
        id: `att-${id}`,
        type: "document",
        name: "doc.md",
        contentType: "text/markdown",
        content: [{ type: "text", text: "x".repeat(1000) }],
        status: { type: "complete" },
      },
    ],
    createdAt: new Date(),
    metadata: messageMetadata,
  } as unknown as ThreadMessage;
}

function mkAssistant(id: string): ThreadMessage {
  return {
    id,
    role: "assistant",
    content: [{ type: "text", text: "Here is my response" }],
    status: { type: "complete", reason: "unknown" },
    createdAt: new Date(),
    metadata: messageMetadata,
  } as unknown as ThreadMessage;
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
    metadata: messageMetadata,
  } as unknown as ThreadMessage);
  repo.resetHead("a1b");

  assert.deepEqual(
    userSiblingsUnder(repo, priorId),
    [userId],
    "standard Reload must not insert another user row",
  );
});

test("two user ids that share text stay distinct branches", () => {
  const repo = new MessageRepository();
  repo.addOrUpdateMessage(null, mkAssistant(priorId));
  repo.addOrUpdateMessage(priorId, mkUser(userId));
  repo.addOrUpdateMessage(userId, mkAssistant("a1"));

  const otherId = "oHXbD51";
  repo.addOrUpdateMessage(priorId, mkUser(otherId));
  repo.addOrUpdateMessage(otherId, mkAssistant("a2"));

  const siblings = userSiblingsUnder(repo, priorId);
  assert.equal(siblings.length, 2);
  assert.deepEqual(siblings.sort(), [otherId, userId].sort());
});
