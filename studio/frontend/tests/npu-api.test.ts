// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { matchesFormatFilter } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import type * as NpuApi from "../src/features/npu/api.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as sseJsonEvents from "../src/lib/sse-json-events.ts";
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
      "@/lib/sse-json-events": sseJsonEvents,
    },
  );
  return { api, requests };
}

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

test("npuRuntimeLabel names the installed FastFlowLM and Lemonade versions", () => {
  const { api } = client(() => Response.json({}));
  assert.equal(
    api.npuRuntimeLabel({ fastflowlm: "1.0.12", lemonade: "2026.44.1" }),
    "FastFlowLM v1.0.12 · Lemonade v2026.44.1",
  );
  assert.equal(
    api.npuRuntimeLabel({ fastflowlm: "v1.0.12", lemonade: null }),
    "FastFlowLM v1.0.12",
  );
  assert.equal(api.npuRuntimeLabel({ fastflowlm: null, lemonade: null }), null);
  assert.equal(api.npuRuntimeLabel(undefined), null);
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

test("following a download reads the running pull and never starts one", async () => {
  const c = client(() =>
    sse([
      'data: {"event":"progress","percent":30}\n\n',
      'data: {"event":"complete"}\n\n',
    ]),
  );
  const seen: number[] = [];
  await c.api.followNpuModelDownload("gemma3-4b-FLM", (event) => {
    if (typeof event.percent === "number") seen.push(event.percent);
  });
  assert.deepEqual(seen, [30]);
  assert.equal(c.requests[0].path, "/api/npu/models/gemma3-4b-FLM/download");
  assert.equal(c.requests[0].init?.method, undefined);
});

test("following a pull that already ended resolves without an error", async () => {
  const c = client(() =>
    Response.json(
      { detail: "gemma3-4b-FLM is not downloading." },
      { status: 404 },
    ),
  );
  await c.api.followNpuModelDownload("gemma3-4b-FLM", () => {
    throw new Error("no progress expected");
  });
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

test("NPU rows: downloaded ones on device, all of them to browse, by query", () => {
  const { api } = client(() => new Response(null));
  const model = (id: string, downloaded: boolean) => ({
    id,
    model_path: `lemonade:${id}`,
    checkpoint: id.replace("-FLM", ""),
    size_gb: 1,
    downloaded,
    labels: [],
    supports_vision: false,
    supports_reasoning: false,
    supports_tools: false,
    max_context_length: null,
    resume_percent: null,
  });
  const models = [model("qwen3-0.6b-FLM", true), model("gemma3-4b-FLM", false)];
  const ids = (onDevice: boolean, query: string) =>
    api.npuRowsFor(models, { onDevice, query }).map((m) => m.id);
  assert.deepEqual(ids(true, ""), ["qwen3-0.6b-FLM"]);
  assert.deepEqual(ids(false, ""), ["qwen3-0.6b-FLM", "gemma3-4b-FLM"]);
  assert.deepEqual(ids(false, " GEMMA "), ["gemma3-4b-FLM"]);
  assert.deepEqual(ids(true, "gemma"), []);
  assert.deepEqual(api.npuRowsFor(null, { onDevice: false, query: "" }), []);
});

test("an interrupted NPU download says how much it kept", () => {
  const { api } = client(() => new Response(null));
  const model = (resume_percent: number | null) =>
    ({ resume_percent }) as Parameters<typeof api.npuResumeLabel>[0];
  assert.equal(api.npuResumeLabel(model(37)), "37% downloaded");
  assert.equal(api.npuResumeLabel(model(null)), null);
});

test("a download waiting for the backend says it is reconnecting", () => {
  const { api } = client(() => new Response(null));
  assert.equal(api.npuDownloadLabel(40), "Downloading 40%");
  assert.equal(api.npuDownloadLabel(40, true), "Reconnecting 40%");
  assert.equal(api.npuDownloadLabel(null), "Downloading");
  assert.equal(api.npuDownloadLabel(null, true), "Reconnecting");
});

test("a rehash and Lemonade's last step get their own labels, not a frozen percent", () => {
  const { api } = client(() => Response.json({}));
  assert.equal(api.npuDownloadLabel(40, false, "verifying"), "Verifying 40%");
  assert.equal(api.npuDownloadLabel(99, false, "finishing"), "Finishing");
  assert.equal(api.npuDownloadLabel(99, true, "finishing"), "Reconnecting 99%");
});

test("a cancelled download rejects with NpuDownloadCancelled", async () => {
  const c = client(() =>
    sse([
      'data: {"event":"progress","percent":10}\n\n',
      'data: {"event":"cancelled","model":"qwen3-0.6b-FLM"}\n\n',
    ]),
  );
  await assert.rejects(
    c.api.downloadNpuModel("qwen3-0.6b-FLM", () => {}),
    (error: unknown) => error instanceof c.api.NpuDownloadCancelled,
  );
});

test("cancel posts to the pull's cancel route and a pull already over is fine", async () => {
  const c = client(() => new Response(null, { status: 404 }));
  await c.api.cancelNpuModelDownload("qwen3 4b");
  assert.equal(c.requests[0].path, "/api/npu/models/qwen3%204b/download/cancel");
  assert.equal(c.requests[0].init?.method, "POST");
  const failing = client(() =>
    Response.json({ detail: "nope" }, { status: 500 }),
  );
  await assert.rejects(failing.api.cancelNpuModelDownload("qwen3-4b-FLM"), /nope/);
});

test("the picker's NPU format holds no Hub repo", () => {
  for (const [id, gguf] of [
    ["unsloth/Qwen3-0.6B-GGUF", true],
    ["mlx-community/Qwen3-8B-4bit", false],
    ["unsloth/Qwen3-0.6B", false],
  ] as const) {
    assert.equal(matchesFormatFilter(id, gguf, "npu"), false);
    assert.equal(matchesFormatFilter(id, gguf, "all"), true);
  }
});
