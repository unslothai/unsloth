// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The settings row decides what to say from new_chats, which crosses the wire
// in snake_case: a missed rename reads as undefined and turns every import into
// "already up to date".

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// The settings API modules reach authFetch through the auth barrel, which
// re-exports login-page.tsx. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

let calls: string[] = [];
let nextStatus = 200;
let nextBody: unknown = {};

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(
    typeof input === "string" ? input : (input as Request).url,
  );
  calls.push(`${init?.method ?? "GET"} ${url}`);
  return new Response(JSON.stringify(nextBody), {
    status: nextStatus,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { importClaudeChats, loadClaudeImportStatus } = await import(
  "../src/features/settings/api/claude-import.ts"
);

test("the status probe reports what Claude Code has", async () => {
  calls = [];
  nextStatus = 200;
  nextBody = { available: true, projects: 3, chats: 42 };

  const status = await loadClaudeImportStatus();

  assert.deepEqual(calls, ["GET /api/import/claude/status"]);
  assert.deepEqual(status, { available: true, projects: 3, chats: 42 });
});

test("the import posts once and names the count of new conversations", async () => {
  calls = [];
  nextStatus = 200;
  nextBody = {
    projects: 2,
    chats: 10,
    new_chats: 4,
    messages: 120,
    skipped: 1,
    warnings: [],
  };

  const result = await importClaudeChats();

  assert.deepEqual(calls, ["POST /api/import/claude"]);
  assert.equal(result.newChats, 4);
  assert.equal(result.chats, 10);
  assert.equal(result.messages, 120);
  assert.equal(result.skipped, 1);
});

test("a failed import rejects rather than reporting an empty run", async () => {
  calls = [];
  nextStatus = 500;
  nextBody = { detail: "Could not read Claude Code's conversations." };

  await assert.rejects(importClaudeChats(), /Could not read Claude Code/);
});
