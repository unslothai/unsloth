// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// new_chats is snake_case on the wire; a missed rename reads as "already up to date".

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// See helpers/auth-stub.mjs.
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

const { importExternalChats, loadExternalImportStatus } = await import(
  "../src/features/settings/api/external-import.ts"
);

test("the client hits each source's routes and maps new_chats", async () => {
  calls = [];
  nextStatus = 200;
  nextBody = { available: true, chats: 42 };
  assert.deepEqual(await loadExternalImportStatus("cursor"), nextBody);

  nextBody = {
    projects: 2,
    chats: 10,
    new_chats: 4,
    messages: 120,
    skipped: 1,
    warnings: [],
  };
  assert.deepEqual(await importExternalChats("claude"), {
    newChats: 4,
    messages: 120,
    warnings: [],
  });
  assert.deepEqual(calls, [
    "GET /api/import/cursor/status",
    "POST /api/import/claude",
  ]);
});

test("a failed import rejects rather than reporting an empty run", async () => {
  nextStatus = 500;
  nextBody = { detail: "Could not read Claude Code's conversations." };
  await assert.rejects(
    importExternalChats("claude"),
    /Could not read Claude Code/,
  );
});
