import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  buildBackendChatExport: vi.fn(),
  createPlatformChat: vi.fn(),
  createPlatformSession: vi.fn(),
  ensureGeneralPlatformChat: vi.fn(),
  listAllPlatformChats: vi.fn(),
  threadRows: [] as unknown[],
  messageRows: [] as unknown[],
}));

vi.mock("@/features/chat/api/chat-api", () => ({
  buildBackendChatExport: mocks.buildBackendChatExport,
}));
vi.mock("@/features/chat/api/platform-chat-adapter", () => ({
  ensureGeneralPlatformChat: mocks.ensureGeneralPlatformChat,
}));
vi.mock("@/features/chat/db", () => ({
  db: {
    threads: { toArray: vi.fn(() => Promise.resolve(mocks.threadRows)) },
    messages: { toArray: vi.fn(() => Promise.resolve(mocks.messageRows)) },
  },
}));
vi.mock("@/integrations/platform-backend", () => ({
  createPlatformChat: mocks.createPlatformChat,
  createPlatformSession: mocks.createPlatformSession,
  listAllPlatformChats: mocks.listAllPlatformChats,
}));

import {
  buildPlatformChatMigrationPlan,
  runPlatformChatMigration,
  serializePlatformChatMigrationExport,
  type LegacyChatMigrationSnapshot,
} from "./platform-chat-migration";

function snapshot(): LegacyChatMigrationSnapshot {
  return {
    exportedAt: "2026-08-18T00:00:00.000Z",
    sourceWarnings: [],
    projects: [
      {
        id: "legacy-project",
        name: "Araştırma",
        instructions: "Yalnızca kaynaklara dayan.",
        archived: true,
        createdAt: 1,
        updatedAt: 2,
        rootPath: "/private/legacy",
      },
    ],
    threads: [
      {
        id: "legacy-thread",
        title: "Eski sohbet",
        modelType: "base",
        projectId: "legacy-project",
        archived: false,
        createdAt: 3,
        forkedFromThreadId: "parent",
      },
    ],
    messages: [
      {
        id: "legacy-message",
        threadId: "legacy-thread",
        role: "user",
        content: [{ type: "text", text: "Merhaba" }],
        createdAt: 4,
      },
    ],
  };
}

describe("platform chat migration", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mocks.listAllPlatformChats.mockResolvedValue([]);
    mocks.createPlatformChat.mockResolvedValue({ id: "platform-chat" });
    mocks.createPlatformSession.mockResolvedValue({ id: "platform-session" });
    mocks.ensureGeneralPlatformChat.mockResolvedValue({ id: "general-chat" });
  });

  it("dry-run reports unsupported history and overlay fields without deleting data", () => {
    const plan = buildPlatformChatMigrationPlan(snapshot(), "user-1");

    expect(plan.totals).toEqual({
      projects: 1,
      threads: 1,
      messages: 1,
      alreadyMigrated: 0,
      pending: 2,
    });
    expect(plan.unsupported.map((item) => item.kind)).toEqual([
      "message",
      "project-field",
      "thread-field",
    ]);
    expect(serializePlatformChatMigrationExport(plan)).toContain(
      '"deletionPerformed": false',
    );
  });

  it("migrates supported fields and persists an idempotent resume ledger", async () => {
    const firstPlan = buildPlatformChatMigrationPlan(snapshot(), "user-1");
    const first = await runPlatformChatMigration(firstPlan);

    expect(first).toMatchObject({
      completedProjects: 1,
      completedThreads: 1,
      failures: [],
      aborted: false,
    });
    expect(mocks.createPlatformChat).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Araştırma",
        description: "[Rag Platform migration:v1:legacy-project]",
      }),
      undefined,
    );
    expect(mocks.createPlatformSession).toHaveBeenCalledWith(
      "platform-chat",
      { name: "Eski sohbet" },
      undefined,
    );

    const resumedPlan = buildPlatformChatMigrationPlan(snapshot(), "user-1");
    expect(resumedPlan.totals).toMatchObject({ alreadyMigrated: 2, pending: 0 });
    const resumed = await runPlatformChatMigration(resumedPlan);
    expect(resumed.skipped).toBe(2);
    expect(mocks.createPlatformChat).toHaveBeenCalledTimes(1);
    expect(mocks.createPlatformSession).toHaveBeenCalledTimes(1);
  });

  it("keeps completed records after a partial failure and resumes only the failed item", async () => {
    mocks.createPlatformSession.mockRejectedValueOnce(new Error("temporary"));
    const first = await runPlatformChatMigration(
      buildPlatformChatMigrationPlan(snapshot(), "user-2"),
    );
    expect(first.completedProjects).toBe(1);
    expect(first.failures).toHaveLength(1);

    mocks.createPlatformSession.mockResolvedValueOnce({ id: "session-resumed" });
    const resumedPlan = buildPlatformChatMigrationPlan(snapshot(), "user-2");
    expect(resumedPlan.projects[0]?.status).toBe("migrated");
    expect(resumedPlan.threads[0]?.status).toBe("pending");
    const resumed = await runPlatformChatMigration(resumedPlan);
    expect(resumed).toMatchObject({
      completedProjects: 0,
      completedThreads: 1,
      skipped: 1,
      failures: [],
    });
  });

  it("reuses a server-side project marker when the local ledger is missing", async () => {
    mocks.listAllPlatformChats.mockResolvedValue([
      {
        id: "already-on-server",
        description: "[Rag Platform migration:v1:legacy-project]",
      },
    ]);
    await runPlatformChatMigration(
      buildPlatformChatMigrationPlan(snapshot(), "user-3"),
    );

    expect(mocks.createPlatformChat).not.toHaveBeenCalled();
    expect(mocks.createPlatformSession).toHaveBeenCalledWith(
      "already-on-server",
      { name: "Eski sohbet" },
      undefined,
    );
  });
});
