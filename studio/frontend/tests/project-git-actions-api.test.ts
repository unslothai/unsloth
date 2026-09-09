// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();
const api = await import("../src/features/chat/api/project-git-actions-api.ts");
test.afterEach(() => setAuthFetchHandler(null));

test("a reviewed mutation retains its project, revision, token and exact JSON body", async () => {
  const payload = {
    confirmationToken: "one-use-token",
    expectedRequestDigest: "digest",
  };
  setAuthFetchHandler((url, init) => {
    assert.equal(
      url,
      "/api/agent/projects/project%20one/review/pull-request-handoff/preview/confirm?workspaceRevision=9",
    );
    assert.equal(init?.method, "POST");
    assert.equal(init?.cache, "no-store");
    assert.deepEqual(JSON.parse(String(init?.body)), payload);
    return new Response(JSON.stringify({ submitted: true }), { status: 200 });
  });
  assert.deepEqual(
    await api.gitAction(
      "project one",
      9,
      "review/pull-request-handoff/preview/confirm",
      "POST",
      payload,
    ),
    { submitted: true },
  );
});

test("uncertain submissions expose the backend error and never retry", async () => {
  let calls = 0;
  setAuthFetchHandler(() => {
    calls += 1;
    return new Response(
      JSON.stringify({
        detail:
          "GitHub handoff outcome is unknown. Check GitHub before retrying.",
      }),
      { status: 502 },
    );
  });
  await assert.rejects(
    api.gitAction(
      "project",
      0,
      "review/pull-request-handoff/id/confirm",
      "POST",
      {},
    ),
    /outcome is unknown/,
  );
  assert.equal(calls, 1);
});

test("confirmation keys change with project, workspace, selected files, message or destination", () => {
  const input = {
    paths: ["a.ts"],
    message: "Reviewed",
    owner: "owner",
    repository: "repo",
  };
  const original = api.gitPreviewKey("one", 2, input);
  assert.equal(api.gitPreviewKey("one", 2, { ...input }), original);
  for (const value of [
    api.gitPreviewKey("two", 2, input),
    api.gitPreviewKey("one", 3, input),
    api.gitPreviewKey("one", 2, { ...input, paths: ["b.ts"] }),
    api.gitPreviewKey("one", 2, { ...input, message: "Changed" }),
    api.gitPreviewKey("one", 2, { ...input, owner: "another" }),
    api.gitPreviewKey("one", 2, { ...input, repository: "another" }),
  ])
    assert.notEqual(value, original);
});
