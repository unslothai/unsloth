// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";
import type { ProjectTask } from "../src/features/chat/api/project-tasks-api.ts";

registerStoreStubResolver();
const api = await import("../src/features/chat/api/project-tasks-api.ts");
test.afterEach(() => setAuthFetchHandler(null));

test("submission retains model selection and limits without accepting workspace authority", async () => {
  setAuthFetchHandler((url, init) => {
    assert.equal(url, "/api/agent/projects/project%20one/tasks");
    assert.equal(init?.method, "POST");
    assert.equal(init?.cache, "no-store");
    assert.deepEqual(JSON.parse(String(init?.body)), {
      instruction: "Fix the parser", kind: "provider", providerId: "saved", model: "model",
      maxOutputTokens: 8192, childLimit: 2, childBudget: 8192, timeout: 900, allowCommands: false,
    });
    return new Response(JSON.stringify({ id: "task" }), { status: 202 });
  });
  assert.equal((await api.submitProjectTask("project one", "Fix the parser", { kind: "provider", providerId: "saved", model: "model" })).id, "task");
});

test("cancel and retry keep both project and attempt scope", async () => {
  const urls: string[] = [];
  setAuthFetchHandler((url, init) => {
    urls.push(String(url));
    assert.equal(init?.method, "POST");
    return new Response("{}");
  });
  await api.cancelProjectTask("p/one", "t/one");
  await api.retryProjectTask("p/one", "t/one");
  assert.deepEqual(urls, ["/api/agent/projects/p%2Fone/tasks/t%2Fone/cancel", "/api/agent/projects/p%2Fone/tasks/t%2Fone/retry"]);
});

test("an uncertain submission is never sent again automatically", async () => {
  let calls = 0;
  setAuthFetchHandler(() => { calls++; throw new TypeError("connection lost"); });
  await assert.rejects(api.submitProjectTask("p", "Fix", { kind: "local", model: "model" }), /connection lost/);
  assert.equal(calls, 1);
});

test("unsupported runtime errors are shown as readable text", async () => {
  setAuthFetchHandler(() => new Response(JSON.stringify({ detail: "Load the selected GGUF model." }), { status: 409 }));
  await assert.rejects(api.listProjectTasks("p"), /Load the selected GGUF model/);
});

test("stale or closed panels abort their active read", async () => {
  const controller = new AbortController();
  let requestSignal: AbortSignal | null = null;
  setAuthFetchHandler((_url, init) => {
    requestSignal = init?.signal ?? null;
    controller.abort();
    assert.equal(requestSignal?.aborted, true);
    return new Response("[]");
  });
  await api.listProjectTasks("p", controller.signal);
});

test("retry controls hide completed, running, child, exhausted and superseded attempts", () => {
  const task = { id: "one", status: "failed", parentId: null, attempt: 1 } as ProjectTask;
  assert.equal(api.taskCanRetry(task, [task]), true);
  for (const status of ["queued", "running", "cancelling", "completed"] as const) {
    assert.equal(api.taskCanRetry({ ...task, status }, []), false);
  }
  assert.equal(api.taskCanRetry({ ...task, parentId: "parent" }, []), false);
  assert.equal(api.taskCanRetry({ ...task, attempt: 3 }, []), false);
  assert.equal(api.taskCanRetry(task, [{ ...task, id: "two", retryOf: "one" }]), false);
  assert.equal(api.taskCanCancel({ ...task, status: "running" }), true);
  assert.equal(api.taskCanCancel({ ...task, status: "cancelling" }), false);
});


test("commands require explicit opt-in on the submitted attempt", async () => {
  setAuthFetchHandler((_url, init) => {
    const body = JSON.parse(String(init?.body));
    assert.equal(body.allowCommands, true);
    assert.equal(body.root, undefined);
    assert.equal(body.env, undefined);
    return new Response("{}");
  });
  await api.submitProjectTask("p", "Run tests", { kind: "local", model: "m" }, undefined, true);
});

test("command evidence reads retain project, task and command scope", async () => {
  const urls: string[] = [];
  setAuthFetchHandler((url, init) => {
    urls.push(String(url));
    assert.equal(init?.method, "GET");
    return new Response("{}");
  });
  await api.getTaskCapabilities("p/one");
  await api.listTaskCommands("p/one", "t/one");
  await api.getTaskCommand("p/one", "t/one", "c/one");
  assert.deepEqual(urls, ["/api/agent/projects/p%2Fone/tasks/capabilities", "/api/agent/projects/p%2Fone/tasks/t%2Fone/commands", "/api/agent/projects/p%2Fone/tasks/t%2Fone/commands/c%2Fone"]);
});

test("interrupted and quarantined commands never render as passed", () => {
  assert.equal(api.commandStatusLabel("passed"), "Passed");
  assert.equal(api.commandStatusLabel("interrupted"), "Outcome unconfirmed");
  assert.equal(api.commandStatusLabel("containment_pending"), "Cleanup unconfirmed");
});
