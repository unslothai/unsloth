import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  datasets: vi.fn(async () => ({ items: [], total: 0 })),
  models: vi.fn(async () => []),
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
    if (value.httpStatus === 403)
      return { kind: "permission", message: "Bu işlem için yetkiniz yok." };
    return { kind: "request", message: "İstek reddedildi." };
  };
  const noop = vi.fn(async () => undefined);
  return {
    PlatformApiError,
    getPlatformUiError,
    listPlatformSearches: mocks.list,
    listPlatformDatasets: mocks.datasets,
    listTenantModels: mocks.models,
    platformModelReference: (model: {
      id: string;
      name: string;
      providerName: string;
      instanceName: string;
    }) =>
      model.providerName ? `${model.name}@${model.providerName}` : model.id,
    resolvePlatformModelReference: (reference: string) => reference,
    createPlatformSearch: noop,
    deletePlatformSearch: noop,
    getPlatformSearch: noop,
    updatePlatformSearch: noop,
    streamPlatformSearchCompletion: vi.fn(),
  };
});

import { SearchPage } from "./search-page";

describe("Phase 13 search product route", () => {
  afterEach(() => vi.clearAllMocks());

  it("renders loading and explicit empty states", async () => {
    let resolveList: ((value: unknown) => void) | undefined;
    mocks.list.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveList = resolve;
        }),
    );
    render(<SearchPage />);
    expect(screen.getByText("Yükleniyor")).toBeInTheDocument();
    resolveList?.({ items: [], total: 0 });
    expect(
      await screen.findByText("Henüz arama uygulaması yok."),
    ).toBeInTheDocument();
  });

  it("renders permission errors from the typed client", async () => {
    const { PlatformApiError } = await import(
      "@/integrations/platform-backend"
    );
    mocks.list.mockRejectedValue(
      new PlatformApiError("forbidden", {
        httpStatus: 403,
        code: 403,
        endpoint: "/searches",
      }),
    );
    render(<SearchPage />);
    expect(
      await screen.findByText("Bu işlem için yetkiniz yok."),
    ).toBeInTheDocument();
  });

  it("suppresses cleanup aborts", async () => {
    const { PlatformApiError } = await import(
      "@/integrations/platform-backend"
    );
    mocks.list.mockRejectedValue(
      new PlatformApiError("aborted", {
        httpStatus: null,
        code: "CLIENT_ABORTED",
        endpoint: "/searches",
      }),
    );
    render(<SearchPage />);
    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalled());
    expect(screen.queryByText("İstek iptal edildi.")).not.toBeInTheDocument();
  });
});
