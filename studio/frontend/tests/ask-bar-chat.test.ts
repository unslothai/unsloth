// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Ask bar's client against a loopback server: what counts as a whole answer, and which model it uses.

import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const paddedResponse = await import("../src/features/chat/api/padded-response.ts");

type Chat = {
  AskError: new (kind: string) => Error & { kind: string };
  adoptBackendPort: () => boolean;
  resolveModel: (signal: AbortSignal, onLoading: (model: string) => void) => Promise<string>;
  streamAnswer: (
    model: string,
    messages: Array<{ role: string; content: string }>,
    signal: AbortSignal,
  ) => AsyncGenerator<string>;
};

type Handler = (req: IncomingMessage, body: string, res: ServerResponse) => void;

async function withServer(handler: Handler, run: (chat: Chat, seen: string[]) => Promise<void>) {
  const seen: string[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (part) => (body += part));
    req.on("end", () => {
      seen.push(`${req.method} ${req.url} ${body}`);
      handler(req, body, res);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no port");
  const storage = new Map<string, string>([["unsloth_backend_port", String(address.port)]]);
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (key: string) => storage.get(key) ?? null,
  };
  let base = "";
  const chat = loadWithStubs<Chat>(new URL("../src/ask/chat.ts", import.meta.url), {
    "@/features/auth/api": {
      authFetch: (path: string, init?: RequestInit) => fetch(`${base}${path}`, init),
    },
    "@/features/chat/api/padded-response": paddedResponse,
    "@/lib/api-base": {
      BACKEND_PORT_STORAGE_KEY: "unsloth_backend_port",
      setApiBase: (port: number) => {
        base = `http://127.0.0.1:${port}`;
      },
    },
  });
  try {
    assert.equal(chat.adoptBackendPort(), true);
    await run(chat, seen);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

const sse = (res: ServerResponse, frames: unknown[], done = true): void => {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  for (const frame of frames) res.write(`data: ${JSON.stringify(frame)}\n\n`);
  if (done) res.write("data: [DONE]\n\n");
  res.end();
};
const delta = (content: string, finish: string | null = null) => ({
  choices: [{ delta: { content }, finish_reason: finish }],
});

async function collect(chat: Chat): Promise<{ text: string; error: string | null }> {
  let text = "";
  try {
    for await (const part of chat.streamAnswer("m", [{ role: "user", content: "hi" }], new AbortController().signal)) {
      text += part;
    }
    return { text, error: null };
  } catch (error) {
    return { text, error: (error as { kind?: string }).kind ?? String(error) };
  }
}

test("a stream that ends with [DONE] is a whole answer", async () => {
  await withServer(
    (_req, _body, res) => sse(res, [delta("Hel"), delta("lo"), delta("", "stop")]),
    async (chat) => assert.deepEqual(await collect(chat), { text: "Hello", error: null }),
  );
});

test("a connection that drops before [DONE] is not an answer", async () => {
  await withServer(
    (_req, _body, res) => sse(res, [delta("Hel")], false),
    async (chat) => assert.deepEqual(await collect(chat), { text: "Hel", error: "failed" }),
  );
});

test("a clipped answer is reported, not shown as complete", async () => {
  await withServer(
    (_req, _body, res) => sse(res, [delta("Hel"), delta("", "length")]),
    async (chat) => assert.equal((await collect(chat)).error, "failed"),
  );
});

test("a reply with only reasoning shows the reasoning", async () => {
  await withServer(
    (_req, _body, res) =>
      sse(res, [{ choices: [{ delta: { reasoning_content: "think" }, finish_reason: null }] }, delta("", "stop")]),
    async (chat) => assert.deepEqual(await collect(chat), { text: "think", error: null }),
  );
});

test("the loaded model is used without loading anything", async () => {
  await withServer(
    (_req, _body, res) => res.end(JSON.stringify({ active_model: "org/loaded" })),
    async (chat, seen) => {
      assert.equal(await chat.resolveModel(new AbortController().signal, () => assert.fail()), "org/loaded");
      assert.deepEqual(seen.map((line) => line.split(" ")[1]), ["/api/inference/status"]);
    },
  );
});

test("with nothing loaded, the model last loaded in Chat is loaded", async () => {
  await withServer(
    (req, _body, res) => {
      if (req.url === "/api/inference/status") return res.end(JSON.stringify({ active_model: null }));
      if (req.url === "/api/settings/last-local-model") {
        return res.end(JSON.stringify({ id: "org/last-GGUF", kind: "gguf", gguf_variant: "Q4_K_M" }));
      }
      res.end(JSON.stringify({ status: "loaded" }));
    },
    async (chat, seen) => {
      const loading: string[] = [];
      assert.equal(await chat.resolveModel(new AbortController().signal, (m) => loading.push(m)), "org/last-GGUF");
      assert.deepEqual(loading, ["org/last-GGUF"]);
      assert.equal(seen[2], 'POST /api/inference/load {"model_path":"org/last-GGUF","gguf_variant":"Q4_K_M"}');
    },
  );
});

test("a load Chat already started is waited out, then used", async () => {
  let polls = 0;
  await withServer(
    (_req, _body, res) => {
      polls += 1;
      res.end(
        JSON.stringify(
          polls < 3 ? { active_model: null, loading: ["org/b"] } : { active_model: "org/b", loading: [] },
        ),
      );
    },
    async (chat, seen) => {
      assert.equal(await chat.resolveModel(new AbortController().signal, () => assert.fail()), "org/b");
      assert.ok(seen.every((line) => line.includes("/api/inference/status")), "nothing else was requested");
    },
  );
});

test("a load that fails after its 200 is a failure", async () => {
  await withServer(
    (req, _body, res) => {
      if (req.url === "/api/inference/status") return res.end(JSON.stringify({ active_model: null }));
      if (req.url === "/api/settings/last-local-model") return res.end(JSON.stringify({ id: "org/m" }));
      res.end(JSON.stringify({ _deferred_error: { status_code: 500 } }));
    },
    async (chat) => {
      await assert.rejects(chat.resolveModel(new AbortController().signal, () => {}), { kind: "failed" });
    },
  );
});

test("with nothing loaded and nothing remembered, it asks the user to load a model", async () => {
  await withServer(
    (req, _body, res) =>
      res.end(JSON.stringify(req.url === "/api/inference/status" ? { active_model: null } : {})),
    async (chat) => {
      await assert.rejects(chat.resolveModel(new AbortController().signal, () => {}), { kind: "noModel" });
    },
  );
});
