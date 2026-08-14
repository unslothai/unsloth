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
  const start = SIDEBAR.indexOf("const recorded: (string | undefined)[] = [];");
  const end = SIDEBAR.indexOf("let distinct = ");
  assert.ok(start !== -1 && end > start, "the read block moved");
  const block = SIDEBAR.slice(start, end);
  assert.ok(block.includes("recordedSandboxSessionId"), "the read block moved");
  assert.ok(
    !block.includes(".catch("),
    "a per-pane catch turns a failed read into a wrong folder",
  );
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

test("the legacy probe runs only for a chat now inside a project", () => {
  // A chat moved OUT of a project wrote under project-<id>, and nothing retains
  // which one, so there is no candidate to probe in that direction.
  const block = SIDEBAR.slice(
    SIDEBAR.indexOf("let distinct = [...new Set(recorded.filter(Boolean))];"),
    SIDEBAR.indexOf("if (distinct.length > 1)"),
  );
  assert.ok(block.includes("distinct.length === 0 && item.projectId"));
  assert.ok(block.includes("sandboxHasFiles(threadId)"));
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
