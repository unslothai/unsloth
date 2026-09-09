import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

test("new profiles use Auto and obsolete selections retain Required while clearing grants", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  try {
    for (const [saved, expected, reselect] of [[null, "auto", false], ["required", "required", false], ["os_isolation_required", "required", false], ["limited", "required", true], ["container_isolation", "required", true]] as const) {
      const removed: string[] = [];
      Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => saved, removeItem: (key: string) => removed.push(key) } });
      const module = loadWithStubs<{ useIsolationStore: { mode: string; reselect: boolean } }>(new URL("../src/features/chat/tool-isolation.ts", import.meta.url), {
        zustand: { create: (factory: () => unknown) => factory() }, "@/features/auth": {},
      });
      assert.equal(module.useIsolationStore.mode, expected);
      assert.equal(module.useIsolationStore.reselect, reselect);
      assert.equal(removed.length, 2);
    }
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor);
    else Reflect.deleteProperty(globalThis, "localStorage");
  }
});

test("serialized tool mode is refreshed after waits and an actual 401 refresh", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Record<string, unknown>[] = [];
  let mode = "auto";
  let calls = 0;
  const auth = loadWithStubs<{ authFetch: (...args: unknown[]) => Promise<Response> }>(
    new URL("../src/features/auth/api.ts", import.meta.url), {
      "@/lib/api-base": { apiUrl: (s: string) => s, isTauri: false },
      "./session": { getAuthToken: () => "access", getRefreshToken: () => "refresh", mustChangePassword: () => false, storeAuthTokens() {}, clearAuthTokens() {}, setMustChangePassword() {} },
    },
  );
  const api = loadWithStubs<{ streamChatCompletions: (payload: Record<string, unknown>, signal: AbortSignal, context: null, before: () => void) => AsyncGenerator<unknown> }>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url), {
      "@/features/auth": auth, "@/features/hf-auth": {},
      "@/features/hub/lib/abort-signals": {}, "@/features/hub/lib/hub-token-header": {},
      "@/features/hub/lib/network": {}, "@/features/native-intents/api": {},
      "@/lib/format-fastapi-error": {}, "@/lib/model-lifecycle-events": {},
      "../utils/chat-history-revision": {}, "./gguf-variants-request": {},
      "./padded-response": {}, "./generation-length.ts": {},
    },
  );
  globalThis.fetch = async (url, init) => {
    if (String(url).includes("refresh")) {
      mode = "required";
      return Response.json({ access_token: "new", refresh_token: "new-refresh" });
    }
    requests.push(JSON.parse(String(init?.body)));
    if (calls++ === 0) return new Response(null, { status: 401 });
    return new Response("data: [DONE]\n\n", { headers: { "Content-Type": "text/event-stream" } });
  };
  try {
    const payload: Record<string, unknown> = { model: "test", messages: [] };
    await Promise.resolve(); // The first-save/admission owner has yielded.
    for await (const _ of api.streamChatCompletions(payload, new AbortController().signal, null, () => { payload.tool_execution_mode = mode; })) { /* Drain SSE. */ }
    assert.deepEqual(requests.map(r => r.tool_execution_mode), ["auto", "required"]);
  } finally { globalThis.fetch = originalFetch; }
});
