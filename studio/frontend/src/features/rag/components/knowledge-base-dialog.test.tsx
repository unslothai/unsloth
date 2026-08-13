import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { KnowledgeBaseDialog } from "./knowledge-base-dialog";

const mocks = vi.hoisted(() => ({
  create: vi.fn(),
  delete: vi.fn(),
  get: vi.fn(),
  listPage: vi.fn(),
  readiness: vi.fn(),
  update: vi.fn(),
}));

vi.mock("../api/platform-dataset-adapter", () => ({
  datasetEmbeddingModelReference: (model: {
    id: string;
    instanceName: string;
    name: string;
    providerName: string;
  }) =>
    /^[0-9a-f]{32}$/i.test(model.id)
      ? model.id
      : `${model.name}@${model.instanceName}@${model.providerName}`,
  getKnowledgeBase: mocks.get,
  listKnowledgeBasePage: mocks.listPage,
}));

vi.mock("../api/rag-api", () => ({
  createKnowledgeBase: mocks.create,
  deleteKnowledgeBase: mocks.delete,
  updateKnowledgeBase: mocks.update,
}));

vi.mock("@/integrations/platform-backend", () => ({
  getPlatformModelReadiness: mocks.readiness,
  isPlatformApiError: (error: unknown) =>
    typeof error === "object" && error !== null && "__platform" in error,
}));

vi.mock("./platform-pipeline-select", () => ({
  PlatformPipelineSelect: ({
    onChange,
    value,
  }: {
    onChange: (value: string) => void;
    value: string;
  }) => (
    <label>
      Pipeline
      <select
        aria-label="Pipeline"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Yerleşik parser</option>
        <option value="pipeline-1">Pipeline 1</option>
      </select>
    </label>
  ),
}));

vi.mock("@/lib/toast", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

const dataset = {
  id: "dataset-1",
  name: "Product docs",
  description: "Reference",
  documentCount: 4,
  embeddingModel: "embedding-1",
  permission: "me" as const,
  chunkMethod: "naive",
  parserConfig: { chunk_token_num: 512 },
  pipelineId: null,
};

describe("Phase 4 KnowledgeBaseDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.listPage.mockResolvedValue({
      items: [dataset],
      page: 1,
      pageSize: 8,
      total: 9,
    });
    mocks.get.mockResolvedValue(dataset);
    mocks.create.mockResolvedValue(dataset);
    mocks.update.mockResolvedValue(dataset);
    mocks.delete.mockResolvedValue(undefined);
    mocks.readiness.mockResolvedValue({
      ready: true,
      missing: [],
      defaults: [
        {
          capability: "embedding",
          enabled: true,
          modelId: "embedding-1",
          modelName: "Embed",
        },
      ],
      models: [
        {
          id: "embedding-1",
          name: "Embed",
          providerId: "provider-1",
          providerName: "Provider",
          instanceId: "instance-1",
          instanceName: "Primary",
          capabilities: ["embedding"],
          status: "active",
          maxTokens: null,
        },
      ],
    });
  });

  it("exposes paginated, searchable and sortable dataset reads", async () => {
    render(<KnowledgeBaseDialog open={true} onOpenChange={vi.fn()} />);

    expect(await screen.findByText("Product docs")).toBeInTheDocument();
    expect(screen.getByText("9 knowledge base")).toBeInTheDocument();
    expect(mocks.listPage).toHaveBeenCalledWith(
      {
        page: 1,
        pageSize: 8,
        name: "",
        orderBy: "update_time",
        desc: true,
      },
      expect.any(AbortSignal),
    );

    fireEvent.change(screen.getByLabelText("Knowledge base ara"), {
      target: { value: "manual" },
    });
    await waitFor(() =>
      expect(mocks.listPage).toHaveBeenLastCalledWith(
        expect.objectContaining({ name: "manual", page: 1 }),
        expect.any(AbortSignal),
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: "Sonraki sayfa" }));
    await waitFor(() =>
      expect(mocks.listPage).toHaveBeenLastCalledWith(
        expect.objectContaining({ page: 2 }),
        expect.any(AbortSignal),
      ),
    );
  });

  it("creates a dataset with required and advanced contract fields", async () => {
    render(<KnowledgeBaseDialog open={true} onOpenChange={vi.fn()} />);
    await screen.findByText("Product docs");
    fireEvent.click(screen.getByRole("button", { name: "Yeni" }));

    await screen.findByRole("option", { name: "Embed · Primary" });
    fireEvent.change(screen.getByLabelText("Ad"), {
      target: { value: " Manuals " },
    });
    fireEvent.change(screen.getByLabelText("Açıklama"), {
      target: { value: " Support docs " },
    });
    fireEvent.change(screen.getByLabelText("Erişim"), {
      target: { value: "team" },
    });
    fireEvent.change(screen.getByLabelText("Chunk yöntemi"), {
      target: { value: "book" },
    });
    fireEvent.change(screen.getByLabelText("Parser config (JSON)"), {
      target: { value: '{"chunk_token_num":256}' },
    });
    fireEvent.change(screen.getByLabelText("Pipeline"), {
      target: { value: "pipeline-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Oluştur" }));

    await waitFor(() =>
      expect(mocks.create).toHaveBeenCalledWith(
        {
          name: "Manuals",
          description: "Support docs",
          embeddingModel: "Embed@Primary@Provider",
          permission: "team",
          chunkMethod: "book",
          parserConfig: { chunk_token_num: 256 },
          pipelineId: "pipeline-1",
        },
        expect.any(AbortSignal),
      ),
    );
  });

  it("loads the active Go detail contract before update and confirms destructive delete counts", async () => {
    render(<KnowledgeBaseDialog open={true} onOpenChange={vi.fn()} />);
    await screen.findByText("Product docs");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Product docs ayarlarını düzenle",
      }),
    );

    await waitFor(() =>
      expect(mocks.get).toHaveBeenCalledWith(
        "dataset-1",
        expect.any(AbortSignal),
      ),
    );
    await screen.findByText("Knowledge base ayarları");
    fireEvent.change(screen.getByLabelText("Açıklama"), {
      target: { value: "" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Değişiklikleri kaydet" }),
    );
    await waitFor(() =>
      expect(mocks.update).toHaveBeenCalledWith(
        "dataset-1",
        expect.objectContaining({ description: undefined }),
        expect.any(AbortSignal),
      ),
    );

    await screen.findByText("Product docs");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Product docs knowledge base’ini sil",
      }),
    );
    expect(screen.getByText("4 belge")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Sil" }));
    await waitFor(() =>
      expect(mocks.delete).toHaveBeenCalledWith(
        "dataset-1",
        expect.any(AbortSignal),
      ),
    );
  });

  it("renders permission failures inline and offers deterministic retry", async () => {
    mocks.listPage.mockRejectedValueOnce({
      __platform: true,
      isAbort: false,
      isPermissionError: true,
    });
    render(<KnowledgeBaseDialog open={true} onOpenChange={vi.fn()} />);

    expect(
      await screen.findByText("Bu dataset işlemi için yetkiniz yok."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Yeniden dene" }));
    expect(await screen.findByText("Product docs")).toBeInTheDocument();
  });
});
