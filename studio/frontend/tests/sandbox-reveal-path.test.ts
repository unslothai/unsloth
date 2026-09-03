// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

// sandbox-reveal reaches authFetch through the auth barrel, which re-exports
// login-page.tsx. See helpers/auth-stub.mjs.
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
  // ASGI decodes %2F before matching, so a slashed id cannot ride in the path.
  // The suffix must land BEFORE the query or the backend answers the listing.
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
  // A self-updating desktop bundle can meet a backend predating this route.
  // It answers 405: the download route claims the same path for GET/HEAD.
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

test("a failed history read is reported, not mistaken for a chat that ran no tools", () => {
  // No React renderer here, so this asserts on source, like ~50 sibling tests.
  // A per-pane catch makes a failed read look like "never ran a tool", and the
  // fallback is project membership, the answer the recorded id overrides.
  // Both "Open chat folder" and "Copy session id" read through this helper.
  const start = SIDEBAR.indexOf("async function recordedSandboxSessionIds");
  const end = SIDEBAR.indexOf("\n  }", start);
  assert.ok(start !== -1 && end > start, "the read block moved");
  const block = SIDEBAR.slice(start, end);
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
  // A chat can name two sandboxes on its own: ran a tool, moved between
  // projects, ran another. Taking one id per thread would leave the refusal
  // blind to that and hand out the newer id as though it were the only one.
  const start = SIDEBAR.indexOf("async function recordedSandboxSessionIds");
  const end = SIDEBAR.indexOf("\n  }", start);
  const block = SIDEBAR.slice(start, end);
  assert.match(
    block,
    /recorded\.push\(\n\s*\.\.\.allRecordedSandboxSessionIds\(await listStoredChatMessages\(threadId\)\),\n\s*\);/,
  );
  // Both actions refuse on more than one, rather than picking a folder.
  assert.equal(SIDEBAR.split("distinct.length > 1").length - 1, 2);
});

test("a sandbox holding files is told apart from one that was never written", async () => {
  // The legacy-folder probe. A missing sandbox lists as 200 with an empty
  // array rather than an error, so both cases must read as "not this one".
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
  // "No sandbox" is already a 200 with an empty list, so a non-OK is a backend
  // or storage failure. Reading it as "no files" would fall through to the
  // current project scope and open a different workspace, silently.
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
  // This used to skip a chat outside a project, on the grounds that one moved
  // OUT wrote under project-<id> and nothing retains which one. A chat can join
  // a project, record that session, and move back out, and skipping the probe
  // there reported one folder while its older files stayed hidden. The thread
  // folder is probed whichever project the chat sits in now; the project
  // workspace is not, since every chat in the project writes to it.
  const start = SIDEBAR.indexOf("async function sandboxSessionIdsHolding");
  assert.notEqual(start, -1, "the legacy probe moved");
  const block = SIDEBAR.slice(start, SIDEBAR.indexOf("\n  }", start));
  assert.ok(!block.includes("if (!item.projectId) return recorded;"));
  assert.ok(block.includes("sandboxHasFiles(candidate)"));
  // Not the project workspace: it is shared by every chat in the project, so
  // files there say nothing about this one.
  assert.ok(!block.includes("sandboxSessionIdFor("));
});

test("a recorded session does not hide a legacy folder beside it", () => {
  // The mixed history: a tool ran before results carried a session, the chat
  // moved into a project, another tool ran and recorded one. Treating the
  // recorded id as proof that nothing else holds files answers for one folder
  // while the older keeps the rest.
  const start = SIDEBAR.indexOf("async function sandboxSessionIdsHolding");
  const block = SIDEBAR.slice(start, SIDEBAR.indexOf("\n  }", start));
  // A union, so a recorded id cannot short-circuit the probe.
  assert.match(block, /return \[\.\.\.new Set\(\[\.\.\.recorded, \.\.\.held\]\)\];/);
  assert.ok(
    !block.includes("recorded.length > 0"),
    "an early return on any recorded id is what hid the legacy folder",
  );
  // Probing an id that is already named would only cost a request.
  assert.match(block, /if \(recorded\.includes\(candidate\)\) continue;/);
});

test("both the folder and the session id are answered from the same probe", () => {
  // They drifted once: only the reveal path fell back to the legacy folder, so
  // Copy session id answered a legacy chat that had since joined a project with
  // project-<id>, a folder it had never written to, and called it a success.
  const callers = SIDEBAR.match(/await sandboxSessionIdsHolding\(/g) ?? [];
  assert.equal(callers.length, 2);
  for (const action of ["copyChatSessionId", "Open chat folder"]) {
    const at = SIDEBAR.indexOf(action);
    assert.notEqual(at, -1, `${action} moved`);
  }
  // Neither may reach past it to the recorded ids alone.
  const copyAt = SIDEBAR.indexOf("async function copyChatSessionId");
  const copy = SIDEBAR.slice(copyAt, SIDEBAR.indexOf("\n  }\n", copyAt));
  assert.ok(!copy.includes("await recordedSandboxSessionIds("));
});

test("a sandbox tool result is wrapped even when it carries no envelope", () => {
  // chat-adapter.ts reaches JSX barrels and cannot be imported, so this asserts
  // on source. The session id is the only record of WHERE a call ran, and the
  // backend suppresses __FILES__ when a concurrent call shared the directory,
  // so a run that did write files can arrive bare and a moved chat would then
  // name a folder from its current scope that it never wrote to.
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
  // test_desktop_reliability_frontend_contract.py forbids `await Promise.all(`
  // here: every batch ends in a native save dialog, and concurrent ones race
  // and lose cancellation. Asserted from this side so the reason travels with
  // the code, not only with the Python guard that fails the build.
  assert.ok(!SIDEBAR.includes("await Promise.all("));
});
