// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// sandbox-upload reaches authFetch through the auth barrel. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);

const { uploadToSandbox } = await import("../src/components/assistant-ui/sandbox-upload.ts");

type Sent = { url: string; form: FormData };

function capture(status: number, body: unknown): Sent[] {
  const sent: Sent[] = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    sent.push({ url, form: init?.body as FormData });
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as unknown as Response;
  }) as typeof fetch;
  return sent;
}

test("a file goes to the session's upload route, refusing to replace by default", async () => {
  const sent = capture(200, { name: "report.xlsx", path: "/x/report.xlsx", replaced: false });
  const result = await uploadToSandbox("project-p1", new File(["x"], "report.xlsx"));
  assert.equal(result.name, "report.xlsx");
  assert.match(sent[0].url, /\/api\/inference\/sandbox\/project-p1\/upload$/);
  assert.equal(sent[0].form.get("overwrite"), "false");
  assert.equal((sent[0].form.get("file") as File).name, "report.xlsx");
});

test("an id the router cannot carry rides in the query, after the verb", async () => {
  const sent = capture(200, { name: "a.csv", path: "/x/a.csv", replaced: true });
  await uploadToSandbox("thread/with/slashes", new File(["x"], "a.csv"), true);
  assert.match(sent[0].url, /\/api\/inference\/sandbox\/_\/upload\?session=thread%2Fwith%2Fslashes$/);
  assert.equal(sent[0].form.get("overwrite"), "true");
});

test("a taken name rejects with its status, so the caller can offer to replace it", async () => {
  capture(409, { detail: "report.xlsx already exists" });
  await assert.rejects(uploadToSandbox("thread-1", new File(["x"], "report.xlsx")), {
    message: "report.xlsx already exists",
    status: 409,
  });
});
