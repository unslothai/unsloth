import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createPlatformChat,
  createPlatformSession,
  deleteAllPlatformChats,
  deletePlatformChat,
  deletePlatformSessionMessage,
  deletePlatformSessions,
  getPlatformChat,
  getPlatformRelatedQuestionsCompatibility,
  getPlatformSession,
  listPlatformChats,
  listPlatformSessions,
  replacePlatformChat,
  updatePlatformChat,
  updatePlatformSession,
  updatePlatformSessionCompatibility,
} from "../chat-api";
import {
  clearPlatformSession,
  storePlatformSessionToken,
} from "../auth-session";
import { platformTestServer } from "./test-server";

const chat = {
  id: "chat-1",
  name: "Project A",
  dataset_ids: ["dataset-1"],
  prompt_config: { system: "Use the selected dataset." },
  create_time: 1,
  update_time: 2,
};
const session = {
  id: "session-1",
  chat_id: "chat-1",
  name: "New Chat",
  messages: [],
  create_time: 3,
  update_time: 4,
};
const ok = (data: unknown) => HttpResponse.json({ code: 0, data });

describe("Rag Platform Phase 7 Chat/Session service", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => {
    clearPlatformSession();
    vi.unstubAllEnvs();
  });

  it("uses the active hybrid Chat collection and item contracts", async () => {
    const writes: Array<{ method: string; body: unknown }> = [];
    platformTestServer.use(
      http.get("http://platform.test/api/v1/chats", ({ request }) => {
        expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
          page: "2",
          page_size: "25",
          keywords: "Project",
          orderby: "name",
          desc: "false",
        });
        return ok({ chats: [chat], total: 1 });
      }),
      http.get("http://platform.test/api/v1/chats/chat-1", () => ok(chat)),
      http.post("http://platform.test/api/v1/chats", async ({ request }) => {
        writes.push({ method: "POST", body: await request.json() });
        return ok(chat);
      }),
      http.patch(
        "http://platform.test/api/v1/chats/chat-1",
        async ({ request }) => {
          writes.push({ method: "PATCH", body: await request.json() });
          return ok(chat);
        },
      ),
      http.put(
        "http://platform.test/api/v1/chats/chat-1",
        async ({ request }) => {
          writes.push({ method: "PUT", body: await request.json() });
          return ok(chat);
        },
      ),
      http.delete("http://platform.test/api/v1/chats", async ({ request }) => {
        writes.push({ method: "DELETE", body: await request.json() });
        return ok({ success_count: 1 });
      }),
      http.delete("http://platform.test/api/v1/chats/chat-1", () =>
        ok(true),
      ),
    );

    await expect(
      listPlatformChats({
        page: 2,
        pageSize: 25,
        keywords: " Project ",
        orderby: "name",
        desc: false,
      }),
    ).resolves.toEqual({ chats: [chat], total: 1 });
    await expect(getPlatformChat("chat-1")).resolves.toEqual(chat);
    await createPlatformChat({
      name: "Project A",
      dataset_ids: ["dataset-1"],
    });
    await updatePlatformChat("chat-1", {
      dataset_ids: ["dataset-2"],
    });
    await replacePlatformChat("chat-1", {
      name: "Project A",
      dataset_ids: ["dataset-2"],
    });
    await deletePlatformChat("chat-1");
    await deleteAllPlatformChats();

    expect(writes).toEqual([
      {
        method: "POST",
        body: { name: "Project A", dataset_ids: ["dataset-1"] },
      },
      { method: "PATCH", body: { dataset_ids: ["dataset-2"] } },
      {
        method: "PUT",
        body: { name: "Project A", dataset_ids: ["dataset-2"] },
      },
      { method: "DELETE", body: { delete_all: true } },
    ]);
  });

  it("uses Chat-scoped Session CRUD and turn-delete contracts", async () => {
    const calls: Array<{ method: string; path: string; body?: unknown }> = [];
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/chats/chat-1/sessions",
        ({ request }) => {
          calls.push({ method: "GET", path: new URL(request.url).pathname });
          expect(Object.fromEntries(new URL(request.url).searchParams)).toEqual({
            page: "1",
            page_size: "10",
            id: "session-1",
            orderby: "update_time",
            desc: "true",
          });
          return ok([session]);
        },
      ),
      http.post(
        "http://platform.test/api/v1/chats/chat-1/sessions",
        async ({ request }) => {
          calls.push({
            method: "POST",
            path: new URL(request.url).pathname,
            body: await request.json(),
          });
          return ok(session);
        },
      ),
      http.get(
        "http://platform.test/api/v1/chats/chat-1/sessions/session-1",
        () => ok(session),
      ),
      http.patch(
        "http://platform.test/api/v1/chats/chat-1/sessions/session-1",
        async ({ request }) => {
          calls.push({
            method: "PATCH",
            path: new URL(request.url).pathname,
            body: await request.json(),
          });
          return ok({ ...session, name: "Renamed" });
        },
      ),
      http.delete(
        "http://platform.test/api/v1/chats/chat-1/sessions",
        async ({ request }) => {
          calls.push({
            method: "DELETE",
            path: new URL(request.url).pathname,
            body: await request.json(),
          });
          return ok({ success_count: 1 });
        },
      ),
      http.delete(
        "http://platform.test/api/v1/chats/chat-1/sessions/session-1/messages/turn-1",
        () => ok(session),
      ),
    );

    await listPlatformSessions("chat-1", {
      page: 1,
      pageSize: 10,
      id: "session-1",
    });
    await createPlatformSession("chat-1", { name: "New Chat" });
    await expect(getPlatformSession("chat-1", "session-1")).resolves.toEqual(
      session,
    );
    await updatePlatformSession("chat-1", "session-1", { name: "Renamed" });
    await deletePlatformSessions("chat-1", ["session-1"]);
    await deletePlatformSessionMessage("chat-1", "session-1", "turn-1");

    expect(calls).toEqual([
      {
        method: "GET",
        path: "/api/v1/chats/chat-1/sessions",
      },
      {
        method: "POST",
        path: "/api/v1/chats/chat-1/sessions",
        body: { name: "New Chat" },
      },
      {
        method: "PATCH",
        path: "/api/v1/chats/chat-1/sessions/session-1",
        body: { name: "Renamed" },
      },
      {
        method: "DELETE",
        path: "/api/v1/chats/chat-1/sessions",
        body: { ids: ["session-1"] },
      },
    ]);
  });

  it("keeps deprecated aliases API-only and authenticated", async () => {
    storePlatformSessionToken("phase-7-token");
    const seen: Array<{ path: string; method: string; body: unknown }> = [];
    platformTestServer.use(
      http.put(
        "http://platform.test/api/v1/chats/chat-1/sessions/session-1",
        async ({ request }) => {
          expect(request.headers.get("authorization")).toBe(
            "Bearer phase-7-token",
          );
          seen.push({
            path: new URL(request.url).pathname,
            method: request.method,
            body: await request.json(),
          });
          return ok(session);
        },
      ),
      http.post(
        "http://platform.test/api/v1/sessions/related_questions",
        async ({ request }) => {
          expect(request.headers.get("authorization")).toBe(
            "Bearer phase-7-token",
          );
          seen.push({
            path: new URL(request.url).pathname,
            method: request.method,
            body: await request.json(),
          });
          return ok(["Related one"]);
        },
      ),
    );

    await updatePlatformSessionCompatibility("chat-1", "session-1", {
      name: "Legacy rename",
    });
    await expect(
      getPlatformRelatedQuestionsCompatibility("Question"),
    ).resolves.toEqual(["Related one"]);
    expect(seen).toEqual([
      {
        path: "/api/v1/chats/chat-1/sessions/session-1",
        method: "PUT",
        body: { name: "Legacy rename" },
      },
      {
        path: "/api/v1/sessions/related_questions",
        method: "POST",
        body: { question: "Question" },
      },
    ]);
  });
});
