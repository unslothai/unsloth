// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_flows_editor.py on an in-memory backend; `?flow=perf&nodes=1000` opens the perf graph.

import "@/index.css";
import { TooltipProvider } from "@/components/ui/tooltip";
import { setFlowsTransport } from "@/features/flows/api";
import catalogSnapshot from "./tests/fixtures/flows-catalog.json";
import { FlowsEditorPage } from "@/features/flows/editor/flows-editor-page";
import { FlowsGalleryPage } from "@/features/flows/gallery/flows-gallery-page";
import { templateDocument } from "@/features/flows/gallery/templates";
import { useDocStore } from "@/features/flows/stores/doc-store";
import { useRuntimeStore } from "@/features/flows/stores/runtime-store";
import { useUiStore } from "@/features/flows/stores/ui-store";
import type {
  FlowDocument,
  FlowEdge,
  FlowNode,
  NodeCatalog,
  RunEvent,
} from "@/features/flows/types";
import {
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { Profiler, type ProfilerOnRenderCallback } from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";

const catalog = catalogSnapshot as unknown as NodeCatalog;
const specs = Object.fromEntries(catalog.nodes.map((s) => [s.type, s]));

interface StoredFlow {
  doc: FlowDocument;
  createdAt: number;
  updatedAt: number;
}

const flows = new Map<string, StoredFlow>();
const runs = new Map<
  string,
  {
    flowId: string;
    events: RunEvent[];
    done: boolean;
    cancelled: boolean;
    listeners: Set<() => void>;
  }
>();
let runCounter = 0;

function put(doc: FlowDocument): StoredFlow {
  const now = Date.now();
  const prior = flows.get(doc.id);
  const entry = {
    doc,
    createdAt: prior?.createdAt ?? now,
    updatedAt: Math.max(now, (prior?.updatedAt ?? 0) + 1),
  };
  flows.set(doc.id, entry);
  return entry;
}

function record(entry: StoredFlow) {
  return {
    id: entry.doc.id,
    name: entry.doc.name,
    doc: entry.doc,
    createdAt: entry.createdAt,
    updatedAt: entry.updatedAt,
  };
}

function summary(entry: StoredFlow, example = false) {
  return {
    id: entry.doc.id,
    name: entry.doc.name,
    kind: entry.doc.kind,
    description: entry.doc.description,
    updatedAt: entry.updatedAt,
    createdAt: entry.createdAt,
    cover: null,
    nodeCount: entry.doc.nodes.length,
    example,
    ...(example
      ? {
          tags: { story: ["story"], world: ["world", "npc"], pipeline: ["image", "comic"] }[entry.doc.kind],
          needs: { story: ["image"], world: ["chat", "image"], pipeline: ["chat", "image"] }[entry.doc.kind],
        }
      : {}),
  };
}

function perfDocument(count: number): FlowDocument {
  const doc = templateDocument("pipeline");
  doc.id = "perf";
  doc.name = `Perf ${count} nodes`;
  const types = [
    "input.text",
    "llm.generate",
    "text.template",
    "image.generate",
    "json.get",
    "logic.random_pick",
    "audio.speak",
    "output.preview",
  ];
  const nodes: FlowNode[] = [];
  const cols = Math.ceil(Math.sqrt(count) * 1.4);
  for (let i = 0; i < count; i++) {
    const type = types[i % types.length];
    nodes.push({
      id: `n${i}`,
      type,
      v: 1,
      params: Object.fromEntries(
        (specs[type]?.params ?? []).map((p) => [p.id, p.default]),
      ),
      pos: [(i % cols) * 320, Math.floor(i / cols) * 260],
    });
  }
  const target = Math.round(count * 1.5);
  const edges: FlowEdge[] = [];
  for (const back of [1, cols, 2, 3, 5, 7, 11]) {
    for (let i = back; i < count && edges.length < target; i++) {
      const s = specs[nodes[i - back].type];
      const t = specs[nodes[i].type];
      const out = s?.outputs[0];
      const inp = t?.inputs.find(
        (p) => p.type === "any" || (p.type === out?.type && p.type !== "list"),
      );
      if (!out || !inp) continue;
      edges.push({
        id: `e${edges.length}`,
        source: `n${i - back}`,
        source_port: out.id,
        target: `n${i}`,
        target_port: inp.id,
        kind: "data",
      });
    }
  }
  doc.nodes = nodes;
  doc.edges = edges;
  return doc;
}

const svgCache = new Map<string, Blob>();
function mockImage(ref: string): Blob {
  const hit = svgCache.get(ref);
  if (hit) return hit;
  let h = 0;
  for (const ch of ref) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const hue = h % 360;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="hsl(${hue},70%,55%)"/><stop offset="1" stop-color="hsl(${(hue + 80) % 360},70%,25%)"/></linearGradient></defs><rect width="512" height="512" fill="url(#g)"/><circle cx="${150 + (h % 200)}" cy="${180 + (h % 150)}" r="90" fill="rgba(255,255,255,0.25)"/><text x="24" y="488" font-family="sans-serif" font-size="28" fill="white">${ref.slice(-10)}</text></svg>`;
  const blob = new Blob([svg], { type: "image/svg+xml" });
  svgCache.set(ref, blob);
  return blob;
}

function sseResponse(runId: string, after: number): Response {
  const run = runs.get(runId);
  if (!run)
    return new Response(JSON.stringify({ detail: "Run not found" }), {
      status: 404,
    });
  const encoder = new TextEncoder();
  let cursor = 0;
  let listener: (() => void) | null = null;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: RunEvent) =>
        controller.enqueue(
          encoder.encode(
            `id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`,
          ),
        );
      send({
        seq: after,
        run_id: runId,
        type: "snapshot",
        ts: Date.now() / 1000,
        data: {
          run: { id: runId, status: run.done ? "completed" : "running" },
          nodes: {},
        },
      });
      const pump = () => {
        while (cursor < run.events.length) {
          const event = run.events[cursor++];
          if (event.seq > after) send(event);
        }
        if (run.done) {
          if (listener) run.listeners.delete(listener);
          controller.close();
        }
      };
      listener = pump;
      run.listeners.add(pump);
      pump();
    },
    cancel() {
      if (listener) run.listeners.delete(listener);
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function simulateRun(
  flowId: string,
  doc: FlowDocument,
  targets?: string[],
): string {
  const runId = `run${++runCounter}`;
  const run = {
    flowId,
    events: [] as RunEvent[],
    done: false,
    cancelled: false,
    listeners: new Set<() => void>(),
  };
  runs.set(runId, run);
  let seq = 0;
  const emit = (
    type: RunEvent["type"],
    nodeId: string | null,
    data: Record<string, unknown> = {},
  ) => {
    run.events.push({
      seq: ++seq,
      run_id: runId,
      type,
      node_id: nodeId,
      ts: Date.now() / 1000,
      data,
    });
    for (const l of run.listeners) l();
  };
  const indeg = new Map(doc.nodes.map((n) => [n.id, 0]));
  for (const e of doc.edges)
    if (e.kind === "data") indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
  const order: FlowNode[] = [];
  const queue = doc.nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0);
  while (queue.length) {
    const n = queue.shift() as FlowNode;
    order.push(n);
    for (const e of doc.edges) {
      if (e.kind !== "data" || e.source !== n.id) continue;
      indeg.set(e.target, (indeg.get(e.target) ?? 1) - 1);
      if (indeg.get(e.target) === 0)
        queue.push(doc.nodes.find((x) => x.id === e.target) as FlowNode);
    }
  }
  const runnable = order
    .filter(
      (n) =>
        !specs[n.type]?.player && (!targets || targets.length === 0 || true),
    )
    .slice(0, 60);
  emit("run.started", null);
  for (const n of runnable) emit("node.status", n.id, { status: "queued" });
  let i = 0;
  const step = () => {
    if (run.cancelled) {
      emit("run.cancelled", null);
      run.done = true;
      for (const l of run.listeners) l();
      return;
    }
    const node = runnable[i++];
    if (!node) {
      emit("run.completed", null);
      run.done = true;
      for (const l of run.listeners) l();
      return;
    }
    const spec = specs[node.type];
    emit("node.status", node.id, {
      status: spec?.lane === "gpu" ? "loading_model" : "running",
    });
    let p = 0;
    const tick = () => {
      p += 0.25;
      if (p < 1) {
        emit("node.status", node.id, { status: "running" });
        emit("node.progress", node.id, {
          fraction: p,
          phase: `step ${Math.round(p * 4)}/4`,
        });
        if (
          node.type.startsWith("llm") ||
          node.type.startsWith("story.generator")
        ) {
          emit("node.text_delta", node.id, {
            port: "text",
            delta: "Once upon a time, a tiny sloth ",
          });
        }
        setTimeout(tick, 90);
        return;
      }
      const outputs: Record<string, unknown> = {};
      for (const port of spec?.outputs ?? []) {
        if (port.type === "image")
          outputs[port.id] = {
            kind: "image",
            ref: `flowasset:${node.id}-${runId}`,
            url: `/api/flows/assets/flowasset:${node.id}-${runId}`,
          };
        else if (port.type === "text")
          outputs[port.id] =
            `Generated text for ${node.label ?? spec?.title}: a sloth astronaut waves from a strawberry moon.`;
        else if (port.type === "number") outputs[port.id] = 42;
        else if (port.type === "list")
          outputs[port.id] = ["one", "two", "three"];
        else if (port.type === "json")
          outputs[port.id] = { title: "Mock", items: [1, 2, 3] };
      }
      emit("node.output", node.id, { outputs, cached: false });
      emit("node.log", node.id, {
        level: "info",
        message: `${spec?.title ?? node.type} finished`,
      });
      emit("node.status", node.id, { status: "done" });
      setTimeout(step, 40);
    };
    setTimeout(tick, 60);
  };
  setTimeout(step, 50);
  return runId;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

async function bodyOf(init?: RequestInit): Promise<unknown> {
  if (!init?.body) return {};
  if (typeof init.body === "string") return JSON.parse(init.body);
  if (init.body instanceof Blob) return JSON.parse(await init.body.text());
  return {};
}

setFlowsTransport(async (url, init) => {
  await new Promise((r) => setTimeout(r, 5));
  const u = new URL(url, location.origin);
  const path = u.pathname.replace(/^\/api\/flows/, "");
  const method = (init?.method ?? "GET").toUpperCase();
  if (path === "/nodes") return json(catalog);
  if (path === "/dataset-rows") {
    const { offset = 0, limit = 20 } = (await bodyOf(init)) as {
      offset?: number;
      limit?: number;
    };
    const topics = ["churn", "cashflow", "inventory", "payroll", "returns"];
    const rows = Array.from({ length: 20 }, (_, i) => ({
      domain: ["SaaS", "Finance", "Retail", "HR"][i % 4],
      topic: topics[i % topics.length],
      sql_prompt: `Write a SQL query that reports monthly ${topics[i % topics.length]} for the last year, grouped by region, and explain which months stand out (${i + 1}).`,
      sql: `-- SCHEMA\nCREATE TABLE ${topics[i % topics.length]} (\n  id INT PRIMARY KEY,\n  region TEXT,\n  month DATE,\n  amount DECIMAL(10,2)\n);\n\n-- QUERY\nSELECT region, month, SUM(amount)\nFROM ${topics[i % topics.length]}\nGROUP BY region, month\nORDER BY month;`,
      score: Math.round((0.55 + ((i * 7) % 10) / 22) * 100) / 100,
    }));
    return json({ rows: rows.slice(offset, offset + limit), total: rows.length, offset, view_rows: 1000 });
  }
  if (path === "" && method === "GET")
    return json({
      flows: [...flows.values()]
        .filter((f) => f.doc.id !== "perf")
        .map((f) => summary(f)),
    });
  if (path === "" && method === "POST")
    return json(
      record(put(((await bodyOf(init)) as { doc: FlowDocument }).doc)),
    );
  if (path === "/examples") {
    const examples = (["story", "world", "pipeline"] as const).map((k) => {
      const d = templateDocument(k);
      d.id = `example-${k}`;
      d.name = {
        story: "The Sloth Who Lost the Moon",
        world: "Pocket Village",
        pipeline: "Comic Strip Machine",
      }[k];
      d.description = "A bundled example.";
      return summary({ doc: d, createdAt: 0, updatedAt: 0 }, true);
    });
    return json({ examples });
  }
  const copy = /^\/examples\/example-(\w+)\/copy$/.exec(path);
  if (copy) {
    const d = templateDocument(copy[1] as "story");
    return json(record(put(d)));
  }
  if (path.startsWith("/assets/"))
    return new Response(mockImage(decodeURIComponent(path.slice(8))), {
      status: 200,
    });
  if (path === "/import") {
    const body = (await bodyOf(init)) as FlowDocument;
    const doc = { ...body, id: `imported${Date.now()}` };
    return json({
      flow: record(put(doc)),
      report: {
        missing_models: [
          {
            kind: "image",
            repo: "unsloth/FLUX.2-klein-4B-GGUF",
            variant: "Q4_K_M",
            size_bytes: 2.6e9,
            downloaded: false,
          },
        ],
        warnings: [],
        baked_nodes: [],
      },
    });
  }
  const events = /^\/runs\/([^/]+)\/events$/.exec(path);
  if (events)
    return sseResponse(events[1], Number(u.searchParams.get("after") ?? 0));
  const cancel = /^\/runs\/([^/]+)\/cancel$/.exec(path);
  if (cancel) {
    const run = runs.get(cancel[1]);
    if (run) run.cancelled = true;
    return json({ status: "cancelling" });
  }
  const flowMatch = /^\/([^/]+)(\/.*)?$/.exec(path);
  if (flowMatch) {
    const id = decodeURIComponent(flowMatch[1]);
    const rest = flowMatch[2] ?? "";
    const entry = flows.get(id);
    if (!entry) return json({ detail: "Flow not found" }, 404);
    if (rest === "" && method === "GET") return json(record(entry));
    if (rest === "" && method === "PUT") {
      const body = (await bodyOf(init)) as {
        doc: FlowDocument;
        baseUpdatedAt: number | null;
      };
      if (body.baseUpdatedAt !== null && body.baseUpdatedAt !== entry.updatedAt)
        return json({ detail: "Conflict" }, 409);
      return json(record(put(body.doc)));
    }
    if (rest === "" && method === "DELETE") {
      flows.delete(id);
      return new Response(null, { status: 204 });
    }
    if (rest === "/runs" && method === "POST") {
      const body = (await bodyOf(init)) as {
        doc?: FlowDocument;
        target_nodes?: string[];
      };
      return json({
        run_id: simulateRun(id, body.doc ?? entry.doc, body.target_nodes),
      });
    }
    if (rest === "/runs") return json({ runs: [] });
    if (rest === "/latest-outputs") return json({ nodes: {} });
    if (rest === "/plan") {
      return json({
        models: [
          {
            kind: "chat",
            repo: "unsloth/Qwen3-4B-GGUF",
            variant: "Q4_K_M",
            size_bytes: 2.5e9,
            downloaded: true,
            loaded: true,
          },
          {
            kind: "image",
            repo: "unsloth/FLUX.2-klein-4B-GGUF",
            variant: "Q4_K_M",
            size_bytes: 2.6e9,
            downloaded: false,
            fits: true,
          },
        ],
        gpu_ops: entry.doc.nodes.filter((n) => specs[n.type]?.lane === "gpu")
          .length,
        cached_nodes: 0,
        errors: [],
        warnings: [],
      });
    }
    if (rest.startsWith("/export"))
      return new Response(new Blob([JSON.stringify(entry.doc, null, 2)]), {
        status: 200,
      });
  }
  return json({ detail: `mock: no route for ${method} ${path}` }, 404);
});

const params = new URLSearchParams(location.search);
const story = templateDocument("story");
story.id = "story";
put(story);
const pipeline = templateDocument("pipeline");
pipeline.id = "pipeline";
put(pipeline);
const nodeCount = Number(params.get("nodes") ?? 0);
if (nodeCount > 0) put(perfDocument(nodeCount));

const commits: {
  id: string;
  phase: string;
  actualDuration: number;
  at: number;
}[] = [];
const onRender: ProfilerOnRenderCallback = (id, phase, actualDuration) => {
  commits.push({ id, phase, actualDuration, at: performance.now() });
  if (commits.length > 5000) commits.splice(0, 1000);
};

declare global {
  interface Window {
    __flowsSmoke?: {
      doc: typeof useDocStore;
      runtime: typeof useRuntimeStore;
      ui: typeof useUiStore;
      commits: typeof commits;
    };
  }
}
window.__flowsSmoke = {
  doc: useDocStore,
  runtime: useRuntimeStore,
  ui: useUiStore,
  commits,
};

const rootRoute = createRootRoute({
  component: () => (
    <TooltipProvider>
      <div className="dark h-screen w-screen overflow-auto bg-background text-foreground">
        <Outlet />
      </div>
      <Toaster theme="dark" position="bottom-right" />
    </TooltipProvider>
  ),
});
const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows",
  component: FlowsGalleryPage,
});
const editorRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows/$flowId",
  component: function Editor() {
    const { flowId } = editorRoute.useParams();
    return (
      <div className="h-screen">
        <Profiler id="editor" onRender={onRender}>
          <FlowsEditorPage flowId={flowId} />
        </Profiler>
      </div>
    );
  },
});
const playRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows/$flowId/play",
  component: () => (
    <div className="p-8 text-sm">Player (Fork C) would open here.</div>
  ),
});

const start = params.get("flow");
const router = createRouter({
  routeTree: rootRoute.addChildren([galleryRoute, editorRoute, playRoute]),
  history: createMemoryHistory({
    initialEntries: [start ? `/flows/${start}` : "/flows"],
  }),
});

const container = document.getElementById("root");
if (container) createRoot(container).render(<RouterProvider router={router} />);
