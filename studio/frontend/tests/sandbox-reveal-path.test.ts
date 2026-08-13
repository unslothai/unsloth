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

const { sandboxRevealPath, revealSandbox } = await import(
  "../src/components/assistant-ui/sandbox-reveal.ts"
);

test("a path-safe session id reveals through its own path segment", () => {
  assert.equal(
    sandboxRevealPath("thread-1"),
    "/api/inference/sandbox/thread-1/reveal",
  );
});

test("an id the router cannot carry moves to the query, after the verb", () => {
  // ASGI decodes %2F before a route matches, so an id with a slash cannot ride
  // in the path. The suffix has to land BEFORE the query or the backend sees
  // /sandbox/_?session=...%2Freveal and answers the listing route instead.
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
  // The desktop app updates on its own schedule, so a new bundle can meet a
  // backend that predates this route. It answers 405, because the download
  // route claims the same path for GET/HEAD.
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
  // Catching per pane would make a read failure indistinguishable from "never
  // ran a tool", and the handler falls back to current project membership --
  // which is the answer the recorded session id exists to override.
  const block = SIDEBAR.slice(
    SIDEBAR.indexOf("const recorded: (string | undefined)[] = [];"),
    SIDEBAR.indexOf("const distinct = "),
  );
  assert.ok(block.includes("recordedSandboxSessionId"), "the read block moved");
  assert.ok(
    !block.includes(".catch("),
    "a per-pane catch turns a failed read into a wrong folder",
  );
});

test("the sandbox reads stay off Promise.all, as the export contract requires", () => {
  // tests/studio/test_desktop_reliability_frontend_contract.py forbids
  // `await Promise.all(` anywhere in this file: every batch here ends in a
  // native save dialog, and concurrent ones race each other and lose
  // cancellation. Asserted from this side too, so the reason travels with the
  // code rather than only with the Python guard that fails the build.
  assert.ok(!SIDEBAR.includes("await Promise.all("));
});
