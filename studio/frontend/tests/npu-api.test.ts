// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as NpuApi from "../src/features/npu/api.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof NpuApi>(
    new URL("../src/features/npu/api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  return { api, requests };
}

/** An SSE body delivered in the given chunks, so frames can straddle reads. */
function sse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

test("recognizes NPU model paths only", () => {
  const { api } = client(() => Response.json({}));
  assert.equal(api.isNpuModelId("lemonade:qwen3-0.6b-FLM"), true);
  assert.equal(api.isNpuModelId("unsloth/Qwen3-0.6B-GGUF"), false);
  assert.equal(api.isNpuModelId(null), false);
});

test("download reports progress across split frames and resolves on complete", async () => {
  const c = client(() =>
    sse([
      'data: {"event":"progress","percent":10}\n\ndata: {"event":"pro',
      'gress","percent":55}\n\n',
      'data: {"event":"complete","percent":100}\n\n',
    ]),
  );
  const seen: number[] = [];
  await c.api.downloadNpuModel("qwen3-0.6b-FLM", (event) => {
    if (typeof event.percent === "number") seen.push(event.percent);
  });
  assert.deepEqual(seen, [10, 55, 100]);
  assert.equal(c.requests[0].path, "/api/npu/models/qwen3-0.6b-FLM/download");
  assert.equal(c.requests[0].init?.method, "POST");
});

test("download rejects on an error event", async () => {
  const c = client(() =>
    sse(['data: {"event":"error","error":"disk full"}\n\n']),
  );
  await assert.rejects(
    c.api.downloadNpuModel("qwen3-0.6b-FLM", () => {}),
    /disk full/,
  );
});

test("download rejects a stream that ends before completing", async () => {
  const c = client(() => sse(['data: {"event":"progress","percent":5}\n\n']));
  await assert.rejects(
    c.api.downloadNpuModel("qwen3-0.6b-FLM", () => {}),
    /ended before it completed/,
  );
});

test("enable surfaces the backend's reason", async () => {
  const c = client(() =>
    Response.json(
      { detail: "The locked-memory limit is too low." },
      { status: 400 },
    ),
  );
  await assert.rejects(c.api.enableNpu(), /locked-memory limit/);
});

test("a runtime left running after a failed setup is not ready", () => {
  const { api } = client(() => new Response(null));
  const base = {
    supported: true,
    hardware: { present: true, supported: true },
    runtime_installed: true,
    error: null,
    validation: null,
    help_url: null,
    loaded_model: null,
    context_length: null,
    loading_model: null,
  };
  const ready = (state: string, runtimeRunning: boolean) =>
    api.isNpuRuntimeReady({ ...base, state, runtime_running: runtimeRunning });
  assert.equal(ready("failed", true), false);
  assert.equal(ready("validating", true), false);
  assert.equal(ready("idle", false), false);
  assert.equal(ready("ready", true), true);
  // Started by a catalog request after a restart.
  assert.equal(ready("idle", true), true);
});
