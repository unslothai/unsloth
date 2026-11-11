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
      "@/features/auth/api": {
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

  // Lemonade's order, with real FastFlowLM checkpoints.
  const backendOrder = [
    ["d3-edge-FLM", "d3:edge"],
    ["gemma3-4b-FLM", "gemma3:4b"],
    ["lfm3-1.7b-FLM", "lfm3:1.7b"],
    ["qwen3-4b-FLM", "qwen3:4b"],
    ["qwen3.5-0.8b-FLM", "qwen3.5:0.8b"],
    ["qwen3.5-4b-FLM", "qwen3.5:4b"],
    ["qwen3.5-9b-FLM", "qwen3.5:9b"],
    ["qwen3.6-moe-35b-a3b-FLM", "qwen3.6-moe:35b-a3b"],
    ["qwen3.8-mtp-27b-FLM", "qwen3.8-mtp:27b"],
  ].map(([id, checkpoint]) => ({ ...model(id, false), checkpoint }));
  assert.deepEqual(
    api
      .npuRowsFor(backendOrder, { onDevice: false, query: "" })
      .map((m) => m.id),
    [
      "qwen3.5-9b-FLM",
      "qwen3.5-4b-FLM",
      "lfm3-1.7b-FLM",
      "qwen3.5-0.8b-FLM",
      "d3-edge-FLM",
      "gemma3-4b-FLM",
      "qwen3-4b-FLM",
      "qwen3.6-moe-35b-a3b-FLM",
      "qwen3.8-mtp-27b-FLM",
    ],
  );
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

test("NPU models carry FastFlowLM's model card repo, absent from older servers", async () => {
  const rows = [
    {
      id: "qwen3.5-9b-FLM",
      hf_repo: "FastFlowLM/Qwen3.5-9B-NPU2",
      hf_revision: "flm_q4k_high_precision",
    },
    { id: "custom-FLM", hf_repo: null, hf_revision: null },
    { id: "older-FLM" },
  ];
  const c = client(() => Response.json({ models: rows }));
  const models = await c.api.listNpuModels();
  assert.equal(c.requests[0].path, "/api/npu/models");
  assert.equal(models[0].hf_repo, "FastFlowLM/Qwen3.5-9B-NPU2");
  assert.equal(models[0].hf_revision, "flm_q4k_high_precision");
  assert.equal(models[1].hf_repo, null);
  assert.equal(models[2].hf_repo ?? null, null);
});

test("NPU capability tags and context label", () => {
  const { api } = client(() => new Response(null));
  const model = (vision: boolean, reasoning: boolean, tools: boolean) =>
    ({
      supports_vision: vision,
      supports_reasoning: reasoning,
      supports_tools: tools,
    }) as Parameters<typeof api.npuCapabilities>[0];
  assert.deepEqual(api.npuCapabilities(model(true, true, true)), [
    { key: "vision", label: "Vision" },
    { key: "reasoning", label: "Reasoning" },
    { key: "tools", label: "Tools" },
  ]);
  assert.deepEqual(api.npuCapabilities(model(false, true, false)), [
    { key: "reasoning", label: "Reasoning" },
  ]);
  assert.deepEqual(api.npuCapabilities(model(false, false, false)), []);
  assert.equal(api.npuContextLabel(32768), "32K context");
  assert.equal(api.npuContextLabel(262144), "256K context");
  assert.equal(api.npuContextLabel(10000), "10,000 context");
  assert.equal(api.npuContextLabel(null), null);
  assert.equal(api.npuContextLabel(0), null);
});

// model_info.json's oids are per-file blob hashes, not a commit, so a branch-pinned card can drift.
test("an NPU card read from a branch says it may be newer than the installed files", () => {
  const { api } = client(() => new Response(null));
  const model = (hf_repo: string | null, hf_revision: string | null) =>
    ({ hf_repo, hf_revision }) as Parameters<typeof api.npuReadmeNote>[0];
  assert.equal(
    api.npuReadmeNote(model("FastFlowLM/Qwen3-0.6B-NPU2", null)),
    "Latest on Hugging Face (may be newer than the installed files)",
  );
  // A pinned branch moves too; only a commit revision is the card the files came from.
  assert.equal(
    api.npuReadmeNote(
      model("FastFlowLM/Qwen3.5-9B-NPU2", "flm_q4k_high_precision"),
    ),
    "Latest on Hugging Face (may be newer than the installed files)",
  );
  assert.equal(
    api.npuReadmeNote(
      model("FastFlowLM/Qwen3.5-9B-NPU2", "0123456789abcdef0123456789abcdef01234567"),
    ),
    null,
  );
  assert.equal(api.npuReadmeNote(model(null, null)), null);
});

test("a model larger than the NPU memory window gets a warning naming both figures", () => {
  const { api } = client(() => Response.json({}));
  const model = (
    fits_memory: boolean | null | undefined,
    memory_needed_gb: number | null = 19,
  ) =>
    ({ fits_memory, memory_needed_gb }) as Parameters<
      typeof api.npuMemoryWarning
    >[0];
  assert.equal(
    api.npuMemoryWarning(model(false), { memory_window_gb: 15.6 }),
    "Needs about 19 GB of NPU memory; this PC can give the NPU about 16 GB.",
  );
  assert.equal(api.npuMemoryTag(model(false)), "May not fit NPU memory");
  // Flagged with a figure missing still warns, without inventing a number.
  assert.match(
    api.npuMemoryWarning(model(false, null), { memory_window_gb: 16 }) ?? "",
    /may need more NPU memory/,
  );
  // Fits, unknown, or a backend that predates the field: no warning.
  for (const fits of [true, null, undefined]) {
    assert.equal(
      api.npuMemoryWarning(model(fits), { memory_window_gb: 16 }),
      null,
    );
    assert.equal(api.npuMemoryTag(model(fits)), null);
  }
});

test("Storage deletes an NPU folder by its checkpoint, which the backend resolves", async () => {
  const { api, requests } = client(() =>
    Response.json({ status: "deleted", model: "qwen3-0.6b-FLM" }),
  );
  assert.equal(
    await api.deleteNpuModelByCheckpoint("qwen3:0.6b"),
    "qwen3-0.6b-FLM",
  );
  assert.deepEqual(
    requests.map((request) => [request.path, request.init?.method]),
    [["/api/npu/checkpoints/qwen3%3A0.6b", "DELETE"]],
  );
});

test("a refused Storage delete surfaces the backend's reason", async () => {
  const { api } = client(() =>
    Response.json(
      { detail: "Lemonade has no FastFlowLM model for gone:1b." },
      { status: 409 },
    ),
  );
  await assert.rejects(
    api.deleteNpuModelByCheckpoint("gone:1b"),
    /no FastFlowLM model for gone:1b/,
  );
});
