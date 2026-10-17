// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A load refused by "Keep multiple models loaded: Always" reads as that however it arrives: the
// X-Unsloth-Refusal header (shared with the memory refusal, so its VALUE decides), its dict in a
// padded body, or only the backend's wording (chat drops headers today).

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { register } from "node:module";
import test, { afterEach } from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const refusal = await import("../src/lib/multi-model-refusal.ts");
const { isMultiModelRefusal, refusalHeadersOf } = refusal;
const { loadDiffusionModel } = await import("../src/features/images/api.ts");
const { loadVideoModel } = await import("../src/features/video/api.ts");
const DETAIL = "Another model holds the GPU and this load would unload it.";
const REFUSED = { "x-unsloth-refusal": "multi-model-always" };
const refused = (err: unknown) =>
  isMultiModelRefusal(null, refusalHeadersOf(err), (err as Error).message);

const LOADS = [
  ["Images", () => loadDiffusionModel({ model_path: "FLUX.2-klein" })],
  ["Video", () => loadVideoModel({ model_path: "LTX-2" })],
] as const;

for (const [page, load] of LOADS) {
  test(`a ${page} load POST keeps the refusal header, and a plain failure reads as before`, async () => {
    const cases: [number, Record<string, string>, string, boolean][] = [
      [409, { ...REFUSED, "retry-after": "5" }, DETAIL, true],
      [500, {}, "CUDA out of memory", false],
    ];
    for (const [status, headers, detail, want] of cases) {
      const body = JSON.stringify({ detail });
      globalThis.fetch = (async () =>
        new Response(body, { status, headers })) as typeof fetch;
      await assert.rejects(load(), (err: unknown) => {
        assert.ok(err instanceof refusal.LoadRequestError);
        assert.deepEqual(
          [err.message, err.status, err.headers["retry-after"]],
          [detail, status, headers["retry-after"]],
        );
        return refused(err) === want;
      });
    }
  });
}

test("the matcher: header value first, then the backend's wording", () => {
  type Bag = Parameters<typeof isMultiModelRefusal>[1];
  type Row = [number | null, Bag, string, boolean];
  const memory = new Headers({ "X-Unsloth-Refusal": "memory-estimate" });
  const cases: Row[] = [
    [200, REFUSED, "x", true], // a padded body's dict, under a 200
    [null, { "X-Unsloth-Refusal": " Multi-Model-Always " }, "", true],
    ...refusal.MULTI_MODEL_REFUSAL_MARKERS.map(
      (m): Row => [null, undefined, m, true],
    ),
    [409, memory, "is set to Always", false], // the shared header's value wins
    [409, undefined, DETAIL, false],
    [409, {}, "", false],
  ];
  for (const [status, headers, message, want] of cases) {
    assert.equal(
      isMultiModelRefusal(status, headers, message),
      want,
      `${JSON.stringify(headers)} ${message}`,
    );
  }
  // The LoadModelError shape (an error with .headers) is read by them; other failures carry none.
  assert.equal(
    refused(Object.assign(new Error("Refused."), { headers: REFUSED })),
    true,
  );
  assert.equal(refusalHeadersOf(new Error("x")), undefined);
});

test("a chat load refused inside a padded body is recognised by its wording", async () => {
  const detail = "Keep multiple models loaded is set to Always.";
  const body = JSON.stringify({
    _deferred_error: { status_code: 409, detail, headers: REFUSED },
  });
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () =>
      res.writeHead(200, { "Content-Type": "application/json" }).end(body),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as { port: number };
  // Only what loadModel needs answered is spelled out; every other import is a no-op.
  const used: Record<string, unknown> = {
    "@/features/auth": {
      authFetch: (path: string, init?: RequestInit) =>
        originalFetch(`http://127.0.0.1:${port}${path}`, init),
    },
    "./padded-response": await import(
      "../src/features/chat/api/padded-response.ts"
    ),
    "@/lib/format-fastapi-error": await import(
      "../src/lib/format-fastapi-error.ts"
    ),
    "@/features/hf-auth": {
      prepareHfTokenForUse: async () => ({ proceed: true }),
    },
    "@/lib/model-lifecycle-events": {
      withModelLoadNotice: (_r: string, _m: string, run: () => unknown) =>
        run(),
    },
  };
  const stubs = new Proxy(used, {
    has: () => true,
    get: (t, k: string) =>
      t[k] ?? new Proxy({}, { get: () => () => undefined }),
  });
  const chat = loadWithStubs<{ loadModel: (p: object) => Promise<unknown> }>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    stubs,
  );
  try {
    await assert.rejects(
      chat.loadModel({ model_path: "unsloth/Qwen3-0.6B-GGUF" }),
      (err: unknown) => {
        assert.equal((err as Error).message, detail);
        // chat-api drops the padded headers until LoadModelError carries them; the wording still matches.
        return refused(err);
      },
    );
  } finally {
    server.close();
  }
});

test("Images, Video and chat raise the refusal under its title, with the owner's settings link", () => {
  const media = [
    "if (isMultiModelRefusal(null, refusalHeadersOf(failure), text)) {",
    "{ description: text || fallback, action: modelMemorySettingsAction() }",
    'reportLoadFailure(err, "Failed to start load");',
  ];
  const pins: [string, string[]][] = [
    ["features/images/images-page.tsx", media],
    ["features/video/video-page.tsx", media],
    [
      "features/chat/hooks/use-chat-model-runtime.ts",
      [
        "isMultiModelRefusal(null, refusalHeadersOf(err), message);",
        "const title = refusal ? MULTI_MODEL_REFUSAL_TITLE : summary;",
        "const action = refusal ? modelMemorySettingsAction() : logsAction;",
      ],
    ],
  ];
  for (const [path, required] of pins) {
    const source = readSrc(path);
    for (const pin of required)
      assert.ok(source.includes(pin), `${path} missing: ${pin}`);
  }
});

test("the wording matched here is the backend's", (t) => {
  const arbiter = new URL(
    "../../backend/core/inference/gpu_arbiter.py",
    import.meta.url,
  );
  const source = existsSync(arbiter) ? readFileSync(arbiter, "utf8") : "";
  if (!source.includes("MULTI_MODEL_REFUSAL_MARKERS ="))
    return t.skip("the backend exports no refusal markers yet");
  for (const marker of refusal.MULTI_MODEL_REFUSAL_MARKERS)
    assert.ok(source.includes(marker), marker);
});
