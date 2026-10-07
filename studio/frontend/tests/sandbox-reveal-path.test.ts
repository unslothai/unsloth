// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

// sandbox-reveal imports the auth barrel, which re-exports login-page.tsx; see auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);

const { sandboxRevealPath, revealSandbox, sandboxHasFiles } = await import(
  "../src/components/assistant-ui/sandbox-reveal.ts"
);

test("a path-safe session id reveals through its own path segment", () => {
  assert.equal(
    sandboxRevealPath("thread-1"),
    "/api/inference/sandbox/thread-1/reveal",
  );
});

test("an id the router cannot carry moves to the query, after the verb", () => {
  // ASGI decodes %2F before matching; the suffix must land before the query.
  assert.equal(
    sandboxRevealPath("thread/with/slashes"),
    "/api/inference/sandbox/_/reveal?session=thread%2Fwith%2Fslashes",
  );
});

test("a project workspace id is path-safe and stays in the segment", () => {
  assert.equal(
    sandboxRevealPath("project-p1"),
    "/api/inference/sandbox/project-p1/reveal",
  );
});

test("an id past the path-safe length falls back to the query rather than truncating", () => {
  const long = "a".repeat(65);
  assert.equal(
    sandboxRevealPath(long),
    `/api/inference/sandbox/_/reveal?session=${long}`,
  );
});

function respond(status: number, body: string, ok = false): Response {
  return {
    ok,
    status,
    json: async () => JSON.parse(body) as unknown,
  } as unknown as Response;
}

test("the backend's reason is what the user is shown", async () => {
  globalThis.fetch = (async () =>
    respond(
      404,
      JSON.stringify({ detail: "This chat has no folder yet" }),
    )) as typeof fetch;
  await assert.rejects(revealSandbox("thread-1"), {
    message: "This chat has no folder yet",
  });
});

test("a body that is not JSON leaves the status as the only thing to report", async () => {
  globalThis.fetch = (async () =>
    respond(500, "<html>502</html>")) as typeof fetch;
  await assert.rejects(revealSandbox("thread-1"), {
    message: "Request failed (500)",
  });
});

test("an old backend with no reveal route rejects rather than resolving silently", async () => {
  // An updated desktop bundle can meet an older backend, which answers 405 here.
  globalThis.fetch = (async () =>
    respond(
      405,
      JSON.stringify({ detail: "Method Not Allowed" }),
    )) as typeof fetch;
  await assert.rejects(revealSandbox("thread-1"), {
    message: "Method Not Allowed",
  });
});

test("a successful reveal resolves without reading the body", async () => {
  function refuse(): Promise<unknown> {
    return Promise.reject(new Error("the body must not be read on success"));
  }
  globalThis.fetch = (async () =>
    ({
      ok: true,
      status: 200,
      json: refuse,
    }) as unknown as Response) as typeof fetch;
  await revealSandbox("thread-1");
});

const SIDEBAR = readFileSync(
  fileURLToPath(new URL("../src/components/app-sidebar.tsx", import.meta.url)),
  "utf-8",
);
const ROW_MENU = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/components/chat-row-menu.ts", import.meta.url),
  ),
  "utf-8",
);
const OPEN_CHAT_FOLDER = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/components/open-chat-folder-item.tsx", import.meta.url),
  ),
  "utf-8",
);
const PROJECTS_PAGE = readFileSync(
  fileURLToPath(new URL("../src/features/chat/projects-page.tsx", import.meta.url)),
  "utf-8",
);

test("a failed history read is reported, not mistaken for a chat that ran no tools", () => {
  // A per-pane catch would make a failed read look like "never ran a tool".
  const start = ROW_MENU.indexOf("async function recordedSandboxSessionIds");
  const end = ROW_MENU.indexOf("\n}", start);
  assert.ok(start !== -1 && end > start, "the read block moved");
  const block = ROW_MENU.slice(start, end);
  assert.ok(
    block.includes("allRecordedSandboxSessionIds"),
    "the read block moved",
  );
  assert.ok(
    !block.includes(".catch("),
    "a per-pane catch turns a failed read into a wrong folder",
  );
});

test("one thread that outlived a move counts as two folders, not one", () => {
  // A chat can name two sandboxes after moving between projects.
  const start = ROW_MENU.indexOf("async function recordedSandboxSessionIds");
  const end = ROW_MENU.indexOf("\n}", start);
  const block = ROW_MENU.slice(start, end);
  assert.match(
    block,
    /recorded\.push\(\n\s*\.\.\.allRecordedSandboxSessionIds\(await listStoredChatMessages\(threadId\)\),\n\s*\);/,
  );
  assert.equal(SIDEBAR.split("distinct.length > 1").length - 1, 1);
  assert.equal(OPEN_CHAT_FOLDER.split("distinct.length > 1").length - 1, 1);
  assert.equal(PROJECTS_PAGE.split("distinct.length > 1").length - 1, 0);
});

test("a sandbox holding files is told apart from one that was never written", async () => {
  // A missing sandbox lists as 200 with an empty array, not an error.
  globalThis.fetch = (async () =>
    ({
      ok: true,
      status: 200,
      json: async () => ({ path: "/s/thread-1", files: [{ name: "a.csv" }] }),
    }) as unknown as Response) as typeof fetch;
  assert.equal(await sandboxHasFiles("thread-1"), true);

  globalThis.fetch = (async () =>
    ({
      ok: true,
      status: 200,
      json: async () => ({ path: "/s/thread-1", files: [] }),
    }) as unknown as Response) as typeof fetch;
  assert.equal(await sandboxHasFiles("thread-1"), false);
});

test("a probe that could not be answered is reported, not read as an empty folder", async () => {
  // A non-OK is a storage failure; treating it as "no files" would open another workspace.
  globalThis.fetch = (async () =>
    ({
      ok: false,
      status: 500,
      json: async () => ({}),
    }) as unknown as Response) as typeof fetch;
  await assert.rejects(sandboxHasFiles("thread-1"), {
    message: "Could not read the chat's folder (500)",
  });
});

test("the legacy probe runs whichever project the chat sits in now", () => {
  // The thread folder is probed whatever project the chat is in now; the shared workspace is not.
  const start = ROW_MENU.indexOf("async function sandboxSessionIdsHolding");
  assert.notEqual(start, -1, "the legacy probe moved");
  const block = ROW_MENU.slice(start, ROW_MENU.indexOf("\n}", start));
  assert.ok(!block.includes("if (!item.projectId) return recorded;"));
  assert.ok(block.includes("sandboxHasFiles(candidate)"));
  assert.ok(!block.includes("sandboxSessionIdFor("));
});

test("a recorded session does not hide a legacy folder beside it", () => {
  const start = ROW_MENU.indexOf("async function sandboxSessionIdsHolding");
  const block = ROW_MENU.slice(start, ROW_MENU.indexOf("\n}", start));
  // A union, so a recorded id cannot short-circuit the probe.
  assert.match(block, /return \[\.\.\.new Set\(\[\.\.\.recorded, \.\.\.held\]\)\];/);
  assert.ok(
    !block.includes("recorded.length > 0"),
    "an early return on any recorded id is what hid the legacy folder",
  );
  assert.match(block, /if \(recorded\.includes\(candidate\)\) continue;/);
});

test("both the folder and the session id are answered from the same probe", () => {
  const callers = SIDEBAR.match(/await sandboxSessionIdsHolding\(/g) ?? [];
  assert.equal(callers.length, 1);
  assert.notEqual(SIDEBAR.indexOf("copyChatSessionId"), -1, "copyChatSessionId moved");
  assert.equal(
    (OPEN_CHAT_FOLDER.match(/await sandboxSessionIdsHolding\(/g) ?? []).length,
    1,
  );
  assert.notEqual(OPEN_CHAT_FOLDER.indexOf('t("library.chats.folder.openChat")'), -1);
  assert.equal(
    (PROJECTS_PAGE.match(/await sandboxSessionIdsHolding\(/g) ?? []).length,
    0,
  );
  assert.match(PROJECTS_PAGE, /<OpenChatFolderItem item=\{chat\} \/>/);
  const copyAt = SIDEBAR.indexOf("async function copyChatSessionId");
  const copy = SIDEBAR.slice(copyAt, SIDEBAR.indexOf("\n  }\n", copyAt));
  assert.ok(!copy.includes("await recordedSandboxSessionIds("));
});

test("a sandbox tool result is wrapped even when it carries no envelope", () => {
  // chat-adapter.ts reaches JSX barrels, so assert on source. __FILES__ can be suppressed.
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf-8",
  );
  const branch = adapter.slice(
    adapter.indexOf(
      "} else if (\n                      createdFiles.length > 0 ||",
    ),
    adapter.indexOf("// Merge tool_end args first"),
  );
  assert.ok(branch.length > 0, "the sandbox result branch moved");
  assert.ok(
    branch.includes("SANDBOX_FILE_TOOLS.has(toolCallParts[idx].toolName"),
    "python and terminal results must be wrapped without an envelope too",
  );
  assert.ok(branch.includes("sessionId: sandboxSessionId"));
});

test("the sandbox reads stay off Promise.all, as the export contract requires", () => {
  // Batches end in native save dialogs; concurrent ones race and lose cancellation.
  assert.ok(!SIDEBAR.includes("await Promise.all("));
});
