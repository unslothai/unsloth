import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as memory from "../memory-api";
import * as search from "../search-api";
import { platformTestServer } from "./test-server";

interface Seen {
  method: string;
  path: string;
  query: string;
  body: unknown;
}
const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, message: "success", data });
const eventStream = () => {
  const encoder = new TextEncoder();
  const wire =
    'data: {"code":0,"message":"","data":{"answer":"Hello","reference":{},"final":false}}\n\n' +
    'data: {"code":0,"message":"","data":{"answer":" world[DONE]","reference":{},"final":false}}\n\n' +
    'data: {"code":0,"message":"","data":{"answer":"Hello world[DONE]","reference":{},"final":false}}\n\n' +
    'data: {"code":0,"message":"","data":{"answer":"","reference":{"chunks":[{"id":"chunk-1","doc_name":"Guide","kb_id":"dataset-1","content":"source"}]},"final":true}}\n\n' +
    'data: {"code":0,"message":"","data":true}\n\n';
  return new HttpResponse(
    new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(wire));
        controller.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
};
const memoryDto = {
  id: "memory-1",
  name: "Support",
  memory_type: ["raw", "semantic"],
  embd_id: "embed-1",
  llm_id: "chat-1",
  permissions: "me",
  memory_size: 1024,
  forgetting_policy: "FIFO",
};
const messageDto = {
  message_id: 7,
  memory_id: "memory-1",
  session_id: "session-1",
  agent_id: "agent-1",
  status: 1,
  content: "Remember this",
};
const searchDto = {
  id: "search-1",
  name: "Docs",
  description: "Search docs",
  created_by: "user-1",
  search_config: {
    kb_ids: ["dataset-1"],
    chat_id: "chat-1",
    similarity_threshold: 0.2,
    vector_similarity_weight: 0.3,
  },
};

describe("Rag Platform Phase 13 memory and search contracts", () => {
  const seen: Seen[] = [];
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    seen.length = 0;
    platformTestServer.use(
      http.all("http://platform.test/*", async ({ request }) => {
        const url = new URL(request.url);
        const body = ["GET", "HEAD"].includes(request.method)
          ? null
          : await request.json().catch(() => null);
        seen.push({
          method: request.method,
          path: url.pathname,
          query: url.search,
          body,
        });
        if (
          url.pathname.endsWith("/completions") ||
          url.pathname.endsWith("/completion")
        ) {
          return eventStream();
        }
        if (url.pathname === "/api/v1/memories" && request.method === "GET")
          return ok({ memory_list: [memoryDto], total_count: 1 });
        if (url.pathname === "/api/v1/memories" && request.method === "POST")
          return ok(memoryDto);
        if (url.pathname.endsWith("/config")) return ok(memoryDto);
        if (
          url.pathname === "/api/v1/memories/memory-1" &&
          request.method === "GET"
        )
          return ok({
            messages: { message_list: [messageDto], total_count: 1 },
            storage_type: "table",
          });
        if (url.pathname === "/api/v1/messages" && request.method === "GET")
          return ok([messageDto]);
        if (url.pathname === "/api/v1/messages/search") return ok([messageDto]);
        if (url.pathname.endsWith("/content")) return ok(messageDto);
        if (url.pathname === "/api/v1/searches" && request.method === "GET")
          return ok({ search_apps: [searchDto], total: 1 });
        if (url.pathname === "/api/v1/searches" && request.method === "POST")
          return ok({ search_id: "search-1" });
        if (
          url.pathname === "/api/v1/searches/search-1" &&
          request.method !== "DELETE"
        )
          return ok(searchDto);
        return ok(true);
      }),
    );
  });
  afterEach(() => vi.unstubAllEnvs());

  it("uses exact memory CRUD, config, message list/search/content/lifecycle contracts", async () => {
    expect(
      await memory.listPlatformMemories({
        page: 2,
        pageSize: 20,
        keywords: "support",
        memoryType: "semantic",
      }),
    ).toMatchObject({
      total: 1,
      items: [{ id: "memory-1", memoryTypes: ["raw", "semantic"] }],
    });
    await memory.createPlatformMemory({
      name: "Support",
      memoryTypes: ["raw", "semantic"],
      embeddingModelId: "embed-1",
      llmId: "chat-1",
    });
    await memory.getPlatformMemoryConfig("memory-1");
    await memory.updatePlatformMemory("memory-1", {
      permissions: "team",
      memorySize: 2048,
      forgettingPolicy: "FIFO",
    });
    await memory.listPlatformMemoryMessages("memory-1", {
      page: 1,
      pageSize: 20,
      keywords: "session",
    });
    await memory.listRecentPlatformMemoryMessages(["memory-1"], {
      limit: 10,
      sessionId: "session-1",
    });
    await memory.searchPlatformMemoryMessages(["memory-1"], "remember", {
      topN: 5,
    });
    await memory.addPlatformMemoryMessage({
      memoryIds: ["memory-1"],
      agentId: "agent-1",
      sessionId: "session-1",
      userInput: "Hi",
      agentResponse: "Hello",
    });
    await memory.getPlatformMemoryMessageContent("memory-1", "7");
    await memory.updatePlatformMemoryMessageStatus("memory-1", "7", false);
    await memory.forgetPlatformMemoryMessage("memory-1", "7");
    await memory.deletePlatformMemory("memory-1");
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/memories",
          query: "?page=2&page_size=20&keywords=support&memory_type=semantic",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/memories",
          body: {
            name: "Support",
            memory_type: ["raw", "semantic"],
            embd_id: "embed-1",
            llm_id: "chat-1",
          },
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/memories/memory-1/config",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/memories/memory-1",
          query: "?page=1&page_size=20&keywords=session",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/messages",
          query: "?memory_id=memory-1&limit=10&session_id=session-1",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/messages/search",
          query:
            "?memory_id=memory-1&query=remember&top_n=5&similarity_threshold=0.2&keywords_similarity_weight=0.7",
        }),
        expect.objectContaining({
          method: "PUT",
          path: "/api/v1/messages/memory-1:7",
          body: { status: false },
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/messages/memory-1:7",
        }),
      ]),
    );
  });

  it("uses exact search CRUD and both active SSE aliases while exposing sources", async () => {
    await search.listPlatformSearches({
      page: 1,
      pageSize: 20,
      keywords: "docs",
      ownerIds: ["owner-1"],
    });
    expect(
      await search.createPlatformSearch({
        name: "Docs",
        description: "Search docs",
      }),
    ).toBe("search-1");
    const app = await search.getPlatformSearch("search-1");
    await search.updatePlatformSearch("search-1", {
      name: app.name,
      description: app.description,
      config: app.config,
    });
    const plural = [];
    for await (const event of search.streamPlatformSearchCompletion(
      "search-1",
      "hello",
      ["dataset-1"],
    ))
      plural.push(event);
    const singular = [];
    for await (const event of search.streamPlatformSearchCompletionAlias(
      "search-1",
      "hello",
      ["dataset-1"],
    ))
      singular.push(event);
    await search.deletePlatformSearch("search-1");
    expect(plural).toEqual(
      expect.arrayContaining([
        { type: "answer", answer: "Hello" },
        { type: "answer", answer: " world" },
        expect.objectContaining({
          type: "reference",
          reference: expect.objectContaining({
            chunks: [
              expect.objectContaining({
                id: "chunk-1",
                datasetId: "dataset-1",
              }),
            ],
          }),
        }),
      ]),
    );
    expect(singular.some((event) => event.type === "done")).toBe(true);
    expect(
      plural
        .filter((event) => event.type === "answer")
        .map((event) => event.answer)
        .join(""),
    ).toBe("Hello world");
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/searches",
          query:
            "?page=1&page_size=20&keywords=docs&owner_ids=owner-1&orderby=update_time&desc=true",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/searches/search-1/completions",
          body: { question: "hello", kb_ids: ["dataset-1"] },
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/searches/search-1/completion",
          body: { question: "hello", kb_ids: ["dataset-1"] },
        }),
      ]),
    );
  });
});
