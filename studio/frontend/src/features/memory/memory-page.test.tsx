import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PlatformMemory } from "@/integrations/platform-backend";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  models: vi.fn(async () => []),
  config: vi.fn(),
  messages: vi.fn(),
  consent: vi.fn(() => false),
  setConsent: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => {
  class PlatformApiError extends Error {
    httpStatus: number | null;
    code: string | number;
    constructor(
      message: string,
      options: { httpStatus: number | null; code: string | number },
    ) {
      super(message);
      this.httpStatus = options.httpStatus;
      this.code = options.code;
    }
    get isAbort() {
      return this.code === "CLIENT_ABORTED";
    }
    get isTimeout() {
      return this.code === "CLIENT_TIMEOUT";
    }
  }
  const getPlatformUiError = (error: unknown) => {
    const value = error as PlatformApiError;
    if (value.code === "CLIENT_ABORTED")
      return { kind: "aborted", message: "İstek iptal edildi." };
    if (value.code === "CLIENT_TIMEOUT")
      return {
        kind: "timeout",
        message: "Rag Platform isteği zaman aşımına uğradı.",
      };
    if (value.httpStatus === 403)
      return { kind: "permission", message: "Bu işlem için yetkiniz yok." };
    return { kind: "request", message: "İstek reddedildi." };
  };
  const noop = vi.fn(async () => undefined);
  return {
    PLATFORM_MEMORY_TYPES: ["raw", "semantic", "episodic", "procedural"],
    PlatformApiError,
    getPlatformUiError,
    listPlatformMemories: mocks.list,
    listTenantModels: mocks.models,
    platformModelReference: (model: {
      id: string;
      name: string;
      providerName: string;
      instanceName: string;
    }) =>
      model.providerName ? `${model.name}@${model.providerName}` : model.id,
    resolvePlatformModelReference: (reference: string) => reference,
    getPlatformMemoryConfig: mocks.config,
    listPlatformMemoryMessages: mocks.messages,
    hasPlatformMemoryConsent: mocks.consent,
    setPlatformMemoryConsent: mocks.setConsent,
    addPlatformMemoryMessage: noop,
    createPlatformMemory: noop,
    deletePlatformMemory: noop,
    forgetPlatformMemoryMessage: noop,
    getPlatformMemoryMessageContent: noop,
    listRecentPlatformMemoryMessages: vi.fn(async () => []),
    searchPlatformMemoryMessages: vi.fn(async () => []),
    updatePlatformMemory: noop,
    updatePlatformMemoryMessageStatus: noop,
  };
});

import { buildPlatformMemoryUpdateInput, MemoryPage } from "./memory-page";

const memory: PlatformMemory = {
  id: "memory-1",
  name: "Support",
  ownerName: "Owner",
  tenantId: "tenant",
  memoryTypes: ["raw"],
  storageType: "table",
  embeddingModelId: "embed",
  llmId: "chat",
  permissions: "me",
  description: "",
  memorySize: 1024,
  forgettingPolicy: "FIFO",
  temperature: 0.5,
  systemPrompt: "",
  userPrompt: "",
  createTime: null,
  updateTime: null,
};

describe("Phase 13 memory product route", () => {
  afterEach(() => vi.clearAllMocks());

  it("sends only changed memory fields and does not rewrite legacy model references", () => {
    expect(
      buildPlatformMemoryUpdateInput(
        { ...memory, description: "Changed", memorySize: 2048 },
        memory,
        {
          embedding: "embed@VLLM",
          chat: "chat@VLLM",
        },
      ),
    ).toEqual({ description: "Changed", memorySize: 2048 });
  });

  it("renders loading and explicit empty states", async () => {
    let resolveList: ((value: unknown) => void) | undefined;
    mocks.list.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveList = resolve;
        }),
    );
    render(<MemoryPage />);
    expect(screen.getByText("Yükleniyor")).toBeInTheDocument();
    resolveList?.({ items: [], total: 0 });
    expect(await screen.findByText("Henüz hafıza yok.")).toBeInTheDocument();
  });

  it("renders permission errors from the typed client", async () => {
    const { PlatformApiError } = await import(
      "@/integrations/platform-backend"
    );
    mocks.list.mockRejectedValue(
      new PlatformApiError("forbidden", {
        httpStatus: 403,
        code: 403,
        endpoint: "/memories",
      }),
    );
    render(<MemoryPage />);
    expect(
      await screen.findByText("Bu işlem için yetkiniz yok."),
    ).toBeInTheDocument();
  });

  it("requires explicit consent before the message form is accessible", async () => {
    mocks.list.mockResolvedValue({ items: [memory], total: 1 });
    mocks.config.mockResolvedValue(memory);
    mocks.messages.mockResolvedValue({
      items: [],
      total: 0,
      storageType: "table",
    });
    render(<MemoryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /Support/ }));
    expect(await screen.findByText("Kayıt kapalı")).toBeInTheDocument();
    expect(screen.queryByLabelText("Kullanıcı mesajı")).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("switch", { name: /Bu hafızaya sohbet kaydı/ }),
    );
    expect(
      await screen.findByLabelText("Kullanıcı mesajı"),
    ).toBeInTheDocument();
    expect(mocks.setConsent).toHaveBeenCalledWith("memory-1", true);
    await waitFor(() => expect(mocks.messages).toHaveBeenCalled());
  });
});
