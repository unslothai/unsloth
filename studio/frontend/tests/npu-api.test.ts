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

test("a PC whose GPU outruns its NPU gets one note, and the NPU stays offered", () => {
  const { api } = client(() => new Response(null));
  const status = (gpu_recommended?: boolean) =>
    ({ gpu_recommended }) as Parameters<typeof api.npuGpuHint>[0];
  const hint = api.npuGpuHint(status(true));
  assert.match(hint ?? "", /GPU .* several times faster/);
  assert.match(hint ?? "", /NPU is still available/);
  assert.equal(api.npuGpuHint(status(false)), null);
  // An older server sends no flag.
  assert.equal(api.npuGpuHint(status()), null);
  assert.equal(api.npuGpuHint(null), null);
});

test("a status read before the GPU scan answered is provisional, an answer is not", () => {
  const { api } = client(() => new Response(null));
  const status = (gpu_recommended?: boolean | null) =>
    ({ gpu_recommended }) as Parameters<typeof api.npuStatusIsProvisional>[0];
  assert.equal(api.npuStatusIsProvisional(status(null)), true);
  assert.equal(api.npuGpuHint(status(null)), null);
  assert.equal(api.npuStatusIsProvisional(status(true)), false);
  assert.equal(api.npuStatusIsProvisional(status(false)), false);
  // An older server sends no flag and never will.
  assert.equal(api.npuStatusIsProvisional(status()), false);
  assert.equal(api.npuStatusIsProvisional(null), false);
});

test("a pending status is re-read for longer than the slowest GPU scan, then left alone", () => {
  const { api } = client(() => new Response(null));
  const status = (gpu_recommended?: boolean | null) =>
    ({ gpu_recommended }) as Parameters<typeof api.npuProvisionalRetryMs>[0];
  let waited = 0;
  let attempt = 0;
  for (;;) {
    const wait = api.npuProvisionalRetryMs(status(null), attempt);
    if (wait == null) break;
    waited += wait;
    attempt += 1;
    assert.ok(attempt < 1000, "the re-read never stops");
  }
  // nvidia-smi's 10 s timeout, then a 10 s AMD probe, with room to spare.
  assert.ok(waited >= 30_000, `only ${waited} ms of re-reads`);
  assert.equal(api.npuProvisionalRetryMs(status(true), 0), null);
  assert.equal(api.npuProvisionalRetryMs(status(false), 0), null);
  assert.equal(api.npuProvisionalRetryMs(status(), 0), null);
  assert.equal(api.npuProvisionalRetryMs(null, 0), null);
});

test("only a pending GPU scan that came back yes asks for a new listing", () => {
  const { api } = client(() => new Response(null));
  assert.equal(api.npuGpuHintArrived(null, true), true);
  assert.equal(api.npuGpuHintArrived(null, false), false);
  assert.equal(api.npuGpuHintArrived(null, null), false);
  assert.equal(api.npuGpuHintArrived(undefined, true), false);
  assert.equal(api.npuGpuHintArrived(true, true), false);
});

test("a model with a known FastFlowLM issue on this PC is tagged", () => {
  const { api } = client(() => new Response(null));
  const model = (known_issue_url?: string | null) =>
    ({ known_issue_url }) as Parameters<typeof api.npuKnownIssueLabel>[0];
  assert.equal(
    api.npuKnownIssueLabel(
      model("https://github.com/ROCm/FastFlowLM/issues/705"),
    ),
    "Known FastFlowLM issue on this PC",
  );
  assert.equal(api.npuKnownIssueLabel(model(null)), null);
  assert.equal(api.npuKnownIssueLabel(model()), null);
});

test("a download waiting for the backend says it is reconnecting", () => {
  const { api } = client(() => new Response(null));
  assert.equal(api.npuDownloadLabel(40), "Downloading 40%");
  assert.equal(api.npuDownloadLabel(40, true), "Reconnecting 40%");
  assert.equal(api.npuDownloadLabel(null), "Downloading");
  assert.equal(api.npuDownloadLabel(null, true), "Reconnecting");
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
