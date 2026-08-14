import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";

const mocks = vi.hoisted(() => ({
  useLibrary: vi.fn(),
}));

vi.mock("./use-document-library", () => ({
  useDocumentLibrary: mocks.useLibrary,
}));

vi.mock("./document-asset-dialog", () => ({
  DocumentAssetDialog: () => null,
  DocumentInlinePreview: ({ document }: { document: { name: string } }) => (
    <div>İçerik: {document.name}</div>
  ),
}));

vi.mock("./dataset-quality-workspace", () => ({
  DatasetQualityWorkspace: ({ mode }: { mode: string }) => (
    <div>Kalite çalışma alanı: {mode}</div>
  ),
}));

import { DocumentLibraryPage } from "./document-library-page";

function renderPage() {
  return render(
    <TooltipProvider>
      <DocumentLibraryPage />
    </TooltipProvider>,
  );
}

const documents = [
  {
    id: "doc-1",
    datasetId: "dataset-1",
    name: "guide.pdf",
    thumbnail: null,
    sizeBytes: 2048,
    sourceType: "local",
    location: null,
    tokenCount: 421,
    chunkCount: 8,
    progress: 1,
    progressMessage: null,
    processDuration: 2,
    suffix: "pdf",
    run: "3",
    backendStatus: "3",
    status: "completed" as const,
    parserId: "naive",
    chunkMethod: "naive",
    pipelineId: null,
    pipelineName: null,
    createdAt: "2026-08-13T10:00:00Z",
    updatedAt: "2026-08-13T10:04:00Z",
  },
  {
    id: "doc-2",
    datasetId: "dataset-1",
    name: "notes.txt",
    thumbnail: null,
    sizeBytes: 1024,
    sourceType: "local",
    location: null,
    tokenCount: 88,
    chunkCount: 2,
    progress: 0.45,
    progressMessage: "Ayrıştırılıyor",
    processDuration: 1,
    suffix: "txt",
    run: "1",
    backendStatus: "1",
    status: "running" as const,
    parserId: "naive",
    chunkMethod: "naive",
    pipelineId: null,
    pipelineName: "Varsayılan",
    createdAt: "2026-08-13T11:00:00Z",
    updatedAt: "2026-08-13T11:01:00Z",
  },
];

function libraryState() {
  return {
    datasets: [{ id: "dataset-1", name: "Ürün belgeleri" }],
    datasetId: "dataset-1",
    setDatasetId: vi.fn(),
    selectedDataset: { id: "dataset-1", name: "Ürün belgeleri" },
    documents,
    totalDocuments: documents.length,
    page: 1,
    pageSize: 10,
    totalPages: 1,
    keywords: "",
    setPage: vi.fn(),
    setPageSize: vi.fn(),
    setKeywords: vi.fn(),
    loadingDatasets: false,
    loadingDocuments: false,
    mutating: false,
    error: null,
    refresh: vi.fn().mockResolvedValue(documents),
    upload: vi.fn(),
    parse: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn().mockResolvedValue(undefined),
    remove: vi.fn().mockResolvedValue(undefined),
    rename: vi.fn().mockResolvedValue(undefined),
  };
}

describe("Document Library Hub layouts", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    );
    mocks.useLibrary.mockReturnValue(libraryState());
  });

  it("opens the selected document in the right-hand detail panel by default", () => {
    renderPage();

    expect(screen.getByRole("button", { name: "Split view" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("list", { name: "Belgeler" })).toBeInTheDocument();
    expect(screen.getByLabelText("guide.pdf detayları")).toBeInTheDocument();
    expect(screen.getByText("İçerik: guide.pdf")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "notes.txt ayrıntılarını aç" }),
    );
    expect(screen.getByLabelText("notes.txt detayları")).toBeInTheDocument();
    expect(screen.getByText("İçerik: notes.txt")).toBeInTheDocument();
  });

  it("switches between the same two-column, split, and compact result modes as Hub", () => {
    renderPage();
    const layout = screen.getByLabelText("Results layout");

    fireEvent.click(
      within(layout).getByRole("button", { name: "Two columns" }),
    );
    expect(
      screen.getByRole("list", { name: "Belgeler iki sütun görünümü" }),
    ).toBeInTheDocument();
    expect(
      within(layout).getByRole("button", { name: "Two columns" }),
    ).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(within(layout).getByRole("button", { name: "Compact" }));
    expect(
      screen.getByRole("columnheader", { name: "Belge" }),
    ).toBeInTheDocument();
    expect(
      within(layout).getByRole("button", { name: "Compact" }),
    ).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "guide.pdf" }));
    expect(screen.getByLabelText("guide.pdf detayları")).toBeInTheDocument();
    expect(
      within(layout).getByRole("button", { name: "Split view" }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("uses the Hub scope toggle and keeps upload actions beside search", () => {
    renderPage();
    const scope = screen.getByRole("radiogroup", { name: "Belge kapsamı" });
    const toolbar = scope.parentElement;

    expect(toolbar).not.toBeNull();
    expect(
      within(scope).getByRole("radio", { name: "Dataset belgeleri" }),
    ).toHaveAttribute("aria-checked", "true");
    expect(
      within(toolbar as HTMLElement).getByRole("searchbox"),
    ).toHaveAttribute("placeholder", "Belgelerde ara");
    const autoProcess = within(toolbar as HTMLElement).getByRole("checkbox", {
      name: "Otomatik işle",
    });
    expect(autoProcess).toHaveAttribute("aria-checked", "true");
    expect(
      within(toolbar as HTMLElement).getByRole("button", {
        name: "Dosya seç",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "Dataset değiştir" }),
    ).toBeInTheDocument();

    fireEvent.click(autoProcess);
    expect(autoProcess).toHaveAttribute("aria-checked", "false");
    fireEvent.click(
      within(scope).getByRole("radio", { name: "Genel belgeler" }),
    );
    expect(screen.getByText("Bağımsız dosya inceleme")).toBeInTheDocument();
  });

  it("opens chunk and retrieval tools inside the Documents dataset scope", () => {
    renderPage();
    const workspace = screen.getByRole("radiogroup", {
      name: "Dataset çalışma alanı",
    });

    fireEvent.click(within(workspace).getByRole("radio", { name: "Chunks" }));
    expect(
      screen.getByText("Kalite çalışma alanı: chunks"),
    ).toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Belgelerde ara"),
    ).not.toBeInTheDocument();

    fireEvent.click(
      within(workspace).getByRole("radio", { name: "Retrieval" }),
    );
    expect(
      screen.getByText("Kalite çalışma alanı: retrieval"),
    ).toBeInTheDocument();
    expect(
      within(workspace).getByRole("radio", { name: "Retrieval" }),
    ).toHaveAttribute("aria-checked", "true");
  });
});
