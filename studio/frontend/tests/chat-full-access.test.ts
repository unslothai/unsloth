// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasChatFullAccess,
  revokeChatFullAccess,
  withChatFullAccess,
  type ChatFullAccessGrants,
} from "../src/features/chat/utils/chat-full-access.ts";

const grants: ChatFullAccessGrants = {
  "run-a": { runId: "run-a", threadId: "chat-a", sessionId: "shared-project" },
};

function base(permissionMode: "auto" | "ask" | "off" = "auto") {
  return {
    permissionMode: permissionMode as "auto" | "ask" | "off" | "full",
    bypassPermissions: false,
    confirmToolCalls: permissionMode !== "off",
    deepResearchEnabled: false,
    chatFullAccessPreviousMode: null as "auto" | "ask" | "off" | null,
  };
}

test("a chat grant follows later turns without granting other chats in the same project", () => {
  assert.equal(hasChatFullAccess(grants, "chat-a", "later-run"), true);
  assert.equal(hasChatFullAccess(grants, "chat-b", "run-b"), false);
  assert.equal(hasChatFullAccess(grants, "chat-b", "run-a"), false);
  assert.equal(hasChatFullAccess(grants, null), false);
});

test("an unsaved chat grant belongs only to its unique run", () => {
  const pending = {
    "run-a": { runId: "run-a", threadId: "", sessionId: "shared-project" },
  };
  assert.equal(hasChatFullAccess(pending, null, "run-a"), true);
  assert.equal(hasChatFullAccess(pending, null, "run-b"), false);
  assert.equal(hasChatFullAccess(pending, null), false);
  assert.equal(hasChatFullAccess(pending, "chat-b"), false);
});

test("the confirmed grant updates all permission flags for its chat", () => {
  const before = base("ask");
  const full = withChatFullAccess(before, grants, "chat-a");
  assert.equal(full.permissionMode, "full");
  assert.equal(full.bypassPermissions, true);
  assert.equal(full.confirmToolCalls, false);
  assert.equal(full.chatFullAccessPreviousMode, "ask");
  assert.equal(full.deepResearchEnabled, false);
  assert.equal(before.permissionMode, "ask");
});

test("switching away restores the underlying mode without inheriting the bypass", () => {
  const full = withChatFullAccess(base("auto"), grants, "chat-a");
  const other = withChatFullAccess(full, grants, "chat-b");
  assert.equal(other.permissionMode, "auto");
  assert.equal(other.bypassPermissions, false);
  assert.equal(other.confirmToolCalls, true);
  assert.equal(other.chatFullAccessPreviousMode, null);
  assert.equal(withChatFullAccess(other, grants, "chat-a").permissionMode, "full");
});

test("an incoming queued chat keeps its own non-full permission mode", () => {
  const queued = {
    ...base("ask"),
    chatFullAccessPreviousMode: "auto" as const,
  };
  const other = withChatFullAccess(queued, grants, "chat-b");
  assert.equal(other.permissionMode, "ask");
  assert.equal(other.bypassPermissions, false);
  assert.equal(other.chatFullAccessPreviousMode, null);
});

test("a queued granted chat remembers its own underlying mode", () => {
  const queued = { ...base("off"), chatFullAccessPreviousMode: "auto" as const };
  const full = withChatFullAccess(queued, grants, "chat-a");
  assert.equal(full.permissionMode, "full");
  assert.equal(full.chatFullAccessPreviousMode, "off");
  assert.equal(withChatFullAccess(full, {}, "chat-a").permissionMode, "off");
});

test("revoking a chat removes all its runs and preserves other chat grants", () => {
  const both: ChatFullAccessGrants = {
    ...grants,
    "run-a2": { runId: "run-a2", threadId: "chat-a", sessionId: "shared-project" },
    "run-b": { runId: "run-b", threadId: "chat-b", sessionId: "shared-project" },
  };
  const revoked = revokeChatFullAccess(both, "chat-a");
  assert.equal(hasChatFullAccess(revoked, "chat-a"), false);
  assert.equal(hasChatFullAccess(revoked, "chat-b"), true);
  assert.equal(hasChatFullAccess(both, "chat-a"), true);
});

test("revoking an unsaved chat targets one run rather than every unsaved chat", () => {
  const pending: ChatFullAccessGrants = {
    "run-a": { runId: "run-a", threadId: "", sessionId: "shared-project" },
    "run-b": { runId: "run-b", threadId: "", sessionId: "shared-project" },
  };
  const revoked = revokeChatFullAccess(pending, null, "run-a");
  assert.equal(hasChatFullAccess(revoked, null, "run-a"), false);
  assert.equal(hasChatFullAccess(revoked, null, "run-b"), true);
});

test("an unrelated blank composer cannot revoke grants belonging to unsaved runs", () => {
  const pending: ChatFullAccessGrants = {
    "run-a": { runId: "run-a", threadId: "", sessionId: "shared-project" },
  };
  assert.deepEqual(revokeChatFullAccess(pending, null), pending);
});

test("the existing session-wide dropdown grant keeps its behavior", () => {
  const global = {
    ...base("ask"), permissionMode: "full" as const,
    bypassPermissions: true, confirmToolCalls: false,
  };
  assert.equal(withChatFullAccess(global, {}, "chat-b"), global);
  assert.equal(withChatFullAccess(global, grants, "chat-a"), global);
});
