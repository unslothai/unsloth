import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  platformEnabled: true,
  listPlatformProjects: vi.fn(),
  listLegacyProjects: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  isPlatformChatPersistenceEnabled: () => mocks.platformEnabled,
}));

vi.mock("../api/platform-chat-adapter", () => ({
  buildPlatformChatExportForChat: vi.fn(),
  clearPlatformChatsForChat: vi.fn(),
  createPlatformProjectForChat: vi.fn(),
  createPlatformThreadForChat: vi.fn(),
  deletePlatformProjectForChat: vi.fn(),
  deletePlatformThreadsForChat: vi.fn(),
  getPlatformMessageForChat: vi.fn(),
  getPlatformProjectForChat: vi.fn(),
  getPlatformThreadForChat: vi.fn(),
  listPlatformMessagesForChat: vi.fn(),
  listPlatformProjectsForChat: mocks.listPlatformProjects,
  listPlatformThreadsForChat: vi.fn(),
  updatePlatformProjectForChat: vi.fn(),
  updatePlatformThreadForChat: vi.fn(),
}));

vi.mock("../api/chat-api", () => ({
  ChatThreadDeletedError: class ChatThreadDeletedError extends Error {},
  batchListChatMessages: vi.fn(),
  buildBackendChatExport: vi.fn(),
  clearBackendChats: vi.fn(),
  deleteChatProject: vi.fn(),
  deleteChatThreads: vi.fn(),
  getChatMessage: vi.fn(),
  getChatProject: vi.fn(),
  getChatThread: vi.fn(),
  listChatImportLedger: vi.fn(),
  listChatMessages: vi.fn(),
  listChatProjects: mocks.listLegacyProjects,
  listChatThreads: vi.fn(),
  notifyChatHistoryUpdated: vi.fn(),
  notifyChatProjectsUpdated: vi.fn(),
  recordChatImportLedger: vi.fn(),
  saveChatMessage: vi.fn(),
  saveChatProject: vi.fn(),
  saveChatThread: vi.fn(),
  syncChatMessages: vi.fn(),
  updateChatProject: vi.fn(),
  updateChatThread: vi.fn(),
}));

import { listStoredChatProjects } from "./chat-history-storage";

describe("platform chat-history storage dispatch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists projects from Rag Platform without calling legacy chat storage", async () => {
    const projects = [
      {
        id: "platform-project",
        name: "Platform",
        archived: false,
        createdAt: 1,
        updatedAt: 1,
      },
    ];
    mocks.listPlatformProjects.mockResolvedValueOnce(projects);

    await expect(listStoredChatProjects()).resolves.toEqual(projects);
    expect(mocks.listPlatformProjects).toHaveBeenCalledWith({});
    expect(mocks.listLegacyProjects).not.toHaveBeenCalled();
  });
});
