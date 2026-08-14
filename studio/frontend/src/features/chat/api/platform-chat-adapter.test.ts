import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import {
  GENERAL_CHAT_NAME,
  PLATFORM_CHAT_FANOUT_CONCURRENCY,
  ensureGeneralPlatformChat,
  getPlatformChatFanoutMetrics,
  listPlatformThreadsForChat,
  mapPlatformChatToProject,
  mapPlatformSessionMessages,
  mapPlatformSessionToThread,
  updatePlatformProjectForChat,
} from "./platform-chat-adapter";
import {
  setPlatformProjectOverlay,
  setPlatformThreadOverlay,
} from "./platform-chat-overlay";

const ok = (data: unknown) => HttpResponse.json({ code: 0, data });

describe("Rag Platform Phase 7 chat domain adapter", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
  });

  afterEach(() => vi.unstubAllEnvs());

  it("maps Chat and Session records without inventing backend fields", () => {
    setPlatformProjectOverlay("chat-1", { archived: true });
    setPlatformThreadOverlay("session-1", {
      archived: true,
      pairId: "pair-local",
      forkedFromThreadId: "parent-local",
    });
    const project = mapPlatformChatToProject({
      id: "chat-1",
      name: "Docs",
      dataset_ids: ["dataset-1"],
      llm_id: "model-1",
      prompt_config: { system: "Ground answers." },
      create_time: 10,
      update_time: 20,
    });
    expect(project).toMatchObject({
      id: "chat-1",
      name: "Docs",
      datasetIds: ["dataset-1"],
      platformLlmId: "model-1",
      instructions: "Ground answers.",
      archived: true,
    });

    expect(
      mapPlatformSessionToThread({
        id: "session-1",
        chat_id: "chat-1",
        name: "Question",
        create_time: 30,
      }),
    ).toMatchObject({
      id: "session-1",
      projectId: "chat-1",
      title: "Question",
      modelType: "base",
      archived: true,
      pairId: "pair-local",
      forkedFromThreadId: "parent-local",
    });
  });

  it("normalizes a shared backend turn id into unique Assistant UI ids", () => {
    const messages = mapPlatformSessionMessages({
      id: "session-1",
      chat_id: "chat-1",
      create_time: 100,
      messages: [
        { id: "turn-1", role: "user", content: "Question" },
        { id: "turn-1", role: "assistant", content: "Answer" },
        { role: "assistant", content: "Prologue" },
      ],
      reference: [{ chunks: ["chunk-1"] }, null],
    });

    expect(messages.map((message) => message.id)).toEqual([
      "turn-1:user:0",
      "turn-1:assistant:1",
      "session-1:assistant:2",
    ]);
    expect(messages[1]).toMatchObject({
      parentId: "turn-1:user:0",
      metadata: {
        platformMessageId: "turn-1",
        platformReference: { chunks: ["chunk-1"] },
      },
    });
    expect(messages[2].parentId).toBe("turn-1:assistant:1");
  });

  it("bounds and records the Chat-to-Session N+1 fan-out", async () => {
    const chats = Array.from({ length: 7 }, (_, index) => ({
      id: `chat-${index}`,
      name: `Project ${index}`,
      create_time: index + 1,
      update_time: index + 1,
    }));
    let active = 0;
    let peak = 0;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/chats", () =>
        ok({ chats, total: chats.length }),
      ),
      http.get(
        "http://platform.test/api/v1/chats/:chatId/sessions",
        async ({ params }) => {
          active += 1;
          peak = Math.max(peak, active);
          await delay(20);
          active -= 1;
          return ok([
            {
              id: `session-${params.chatId}`,
              chat_id: params.chatId,
              name: "History",
              messages: [{ role: "assistant", content: "Hello" }],
              create_time: 1,
              update_time: 2,
            },
          ]);
        },
      ),
    );

    await expect(listPlatformThreadsForChat()).resolves.toHaveLength(7);
    expect(peak).toBeLessThanOrEqual(PLATFORM_CHAT_FANOUT_CONCURRENCY);
    expect(getPlatformChatFanoutMetrics()).toMatchObject({
      chatCount: 7,
      sessionRequests: 7,
      peakConcurrency: peak,
    });
  });

  it("creates the reserved General Chat idempotently across repeated calls", async () => {
    let general: Record<string, unknown> | null = null;
    let creates = 0;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/chats", () =>
        ok({ chats: general ? [general] : [], total: general ? 1 : 0 }),
      ),
      http.post("http://platform.test/api/v1/chats", async ({ request }) => {
        creates += 1;
        expect(await request.json()).toEqual({
          name: GENERAL_CHAT_NAME,
          dataset_ids: [],
        });
        general = {
          id: "general-chat",
          name: GENERAL_CHAT_NAME,
          dataset_ids: [],
          create_time: 1,
          update_time: 1,
        };
        return ok(general);
      }),
    );

    await expect(ensureGeneralPlatformChat()).resolves.toMatchObject({
      id: "general-chat",
    });
    await expect(ensureGeneralPlatformChat()).resolves.toMatchObject({
      id: "general-chat",
    });
    expect(creates).toBe(1);
  });

  it("rejects renaming a project to the reserved General Chat name", async () => {
    await expect(
      updatePlatformProjectForChat("chat-1", { name: "general" }),
    ).rejects.toThrow("reserved for chats outside a project");
  });

  it("persists dataset scope while retaining the complete prompt config", async () => {
    let patch: unknown;
    const current = {
      id: "chat-1",
      name: "Docs",
      dataset_ids: ["dataset-1"],
      prompt_config: { system: "Old", prologue: "Welcome", quote: true },
      create_time: 1,
      update_time: 2,
    };
    platformTestServer.use(
      http.get("http://platform.test/api/v1/chats/chat-1", () => ok(current)),
      http.patch(
        "http://platform.test/api/v1/chats/chat-1",
        async ({ request }) => {
          patch = await request.json();
          return ok({
            ...current,
            dataset_ids: ["dataset-2"],
            prompt_config: { ...current.prompt_config, system: "New" },
          });
        },
      ),
    );

    await expect(
      updatePlatformProjectForChat("chat-1", {
        datasetIds: ["dataset-2"],
        instructions: "New",
      }),
    ).resolves.toMatchObject({
      datasetIds: ["dataset-2"],
      instructions: "New",
    });
    expect(patch).toEqual({
      dataset_ids: ["dataset-2"],
      prompt_config: { system: "New", prologue: "Welcome", quote: true },
    });
  });
});
