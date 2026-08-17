import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listFiles: vi.fn(),
  listConnectors: vi.fn(async () => []),
  getConnector: vi.fn(async () => ({ id: "", name: "", source: "rest_api", refreshFrequency: 0, pruneFrequency: 0, timeoutSeconds: 30 })),
}));

vi.mock("@/integrations/platform-backend", () => {
  class PlatformApiError extends Error {
    httpStatus: number | null;
    code: string | number;
    constructor(message: string, options: { httpStatus: number | null; code: string | number }) {
      super(message);
      this.httpStatus = options.httpStatus;
      this.code = options.code;
    }
    get isTimeout() { return this.code === "CLIENT_TIMEOUT"; }
    get isAbort() { return this.code === "CLIENT_ABORTED"; }
  }
  const noop = vi.fn(async () => true);
  return {
    CONNECTOR_OAUTH_MESSAGE: "rag-platform-connector-oauth",
    PLATFORM_CONNECTOR_SOURCES: ["rest_api", "google_drive", "gmail", "box"],
    PlatformApiError,
    clearPendingConnectorOAuth: noop,
    connectorOAuthRedirectUri: () => "http://localhost/callback",
    createPlatformCommit: noop,
    createPlatformConnector: noop,
    createPlatformFolder: noop,
    deletePlatformConnector: noop,
    deletePlatformFiles: noop,
    diffPlatformCommits: noop,
    downloadPlatformFile: vi.fn(async () => new Blob(["text"], { type: "text/plain" })),
    getPlatformCommit: noop,
    getPlatformCommitFileContent: noop,
    getPlatformCommitTree: noop,
    getPlatformConnector: mocks.getConnector,
    getPlatformFileAncestors: vi.fn(async () => []),
    getPlatformFileParent: vi.fn(async () => null),
    getPlatformUncommittedChanges: vi.fn(async () => []),
    isPlatformConnectorsEnabled: () => true,
    linkPlatformFilesToDatasets: noop,
    listPlatformCommitFiles: vi.fn(async () => []),
    listPlatformCommits: vi.fn(async () => ({ total: 0, page: 1, pageSize: 50, commits: [] })),
    listPlatformConnectorLogs: vi.fn(async () => ({ total: 0, logs: [] })),
    listPlatformConnectors: mocks.listConnectors,
    listPlatformDatasets: mocks.listDatasets,
    listPlatformFiles: mocks.listFiles,
    listPlatformFileVersions: vi.fn(async () => []),
    movePlatformFiles: noop,
    openConnectorOAuthWindow: vi.fn(),
    readPendingConnectorOAuth: vi.fn(() => null),
    rebuildPlatformConnector: noop,
    redactConnectorSecrets: (value: unknown) => value,
    savePendingConnectorOAuth: noop,
    startBoxConnectorOAuth: noop,
    startGoogleConnectorOAuth: noop,
    testPlatformConnector: noop,
    updatePlatformConnector: noop,
    uploadPlatformFiles: vi.fn(async () => []),
    waitForConnectorOAuthResult: noop,
  };
});

import { coalesceCommitMetadataChanges, FilesPage } from "./files-page";

describe("Phase 12 files product route", () => {
  afterEach(() => vi.clearAllMocks());

  it("coalesces a rename and move for one file into one atomic commit item", () => {
    expect(coalesceCommitMetadataChanges([
      {
        fileId: "file-1",
        fileName: "renamed.txt",
        operation: "rename",
        oldHash: null,
        newHash: null,
        oldLocation: null,
        newLocation: null,
        oldName: "original.txt",
        newName: "renamed.txt",
        oldParentId: "folder-a",
        newParentId: "folder-b",
      },
      {
        fileId: "file-1",
        fileName: "renamed.txt",
        operation: "move",
        oldHash: null,
        newHash: null,
        oldLocation: null,
        newLocation: null,
        oldName: null,
        newName: null,
        oldParentId: "folder-a",
        newParentId: "folder-b",
      },
    ])).toEqual([
      {
        fileId: "file-1",
        fileName: "renamed.txt",
        operation: "move",
        oldHash: null,
        newHash: null,
        oldLocation: null,
        newLocation: null,
        oldName: "original.txt",
        newName: "renamed.txt",
        oldParentId: "folder-a",
        newParentId: "folder-b",
      },
    ]);
  });

  it("renders a loading state and the explicit empty file state", async () => {
    let resolveFiles: ((value: unknown) => void) | undefined;
    mocks.listDatasets.mockResolvedValue({ items: [], total: 0 });
    mocks.listFiles.mockImplementation(
      () => new Promise((resolve) => { resolveFiles = resolve; }),
    );
    render(<FilesPage />);
    expect(screen.getByText("Yükleniyor")).toBeInTheDocument();
    resolveFiles?.({ total: 0, files: [], parentFolder: null });
    expect(await screen.findByText("Bu klasör boş.")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Connector’lar" })).toBeEnabled();
    expect(mocks.listDatasets).toHaveBeenCalledWith(
      { page: 1, pageSize: 100 },
      expect.any(AbortSignal),
    );
  });

  it("does not present effect cleanup aborts as user-facing errors", async () => {
    const { PlatformApiError } = await import("@/integrations/platform-backend");
    mocks.listDatasets.mockRejectedValue(
      new PlatformApiError("aborted", { httpStatus: null, code: "CLIENT_ABORTED", endpoint: "/datasets" }),
    );
    mocks.listFiles.mockRejectedValue(
      new PlatformApiError("aborted", { httpStatus: null, code: "CLIENT_ABORTED", endpoint: "/files" }),
    );
    render(<FilesPage />);
    await waitFor(() => expect(mocks.listFiles).toHaveBeenCalled());
    expect(screen.queryByText("İşlem iptal edildi.")).not.toBeInTheDocument();
  });

  it("does not present connector cleanup aborts as user-facing errors", async () => {
    const { PlatformApiError } = await import("@/integrations/platform-backend");
    mocks.listDatasets.mockResolvedValue({ items: [], total: 0 });
    mocks.listFiles.mockResolvedValue({ total: 0, files: [], parentFolder: null });
    mocks.listConnectors.mockRejectedValueOnce(
      new PlatformApiError("aborted", { httpStatus: null, code: "CLIENT_ABORTED", endpoint: "/connectors" }),
    );
    render(<FilesPage />);
    const connectorTab = screen.getByRole("tab", { name: "Connector’lar" });
    connectorTab.focus();
    fireEvent.keyDown(connectorTab, { key: "Enter", code: "Enter" });
    await waitFor(() => expect(mocks.listConnectors).toHaveBeenCalled());
    expect(screen.queryByText("İşlem iptal edildi.")).not.toBeInTheDocument();
  });

  it("renders the permission state returned by the typed client", async () => {
    const { PlatformApiError } = await import("@/integrations/platform-backend");
    mocks.listDatasets.mockResolvedValue({ items: [], total: 0 });
    mocks.listFiles.mockRejectedValue(
      new PlatformApiError("forbidden", { httpStatus: 403, code: 403, endpoint: "/files" }),
    );
    render(<FilesPage />);
    await waitFor(() =>
      expect(screen.getByText("Bu işlem için yetkiniz yok.")).toBeInTheDocument(),
    );
  });
});
