import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

const mocks = vi.hoisted(() => ({
  getScope: vi.fn(),
  updateScope: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/integrations/platform-backend")>();
  return { ...actual, isPlatformChatPersistenceEnabled: () => true };
});

vi.mock("../api/platform-chat-adapter", () => ({
  getPlatformChatDatasetScope: mocks.getScope,
  updatePlatformChatDatasetScope: mocks.updateScope,
}));

vi.mock("@/features/rag/components/dataset-scope-selector", () => ({
  DatasetScopeSelector: ({
    selectedIds,
    onChange,
    disabled,
  }: ComponentProps<"button"> & {
    selectedIds: string[];
    onChange: (ids: string[]) => void;
    disabled?: boolean;
  }) => (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onChange(["dataset-1"])}
    >
      {selectedIds.includes("dataset-1") ? "baran selected" : "Select baran"}
    </button>
  ),
}));

import { PlatformChatSourcesButton } from "./platform-chat-sources-button";

describe("PlatformChatSourcesButton", () => {
  beforeEach(() => {
    mocks.getScope.mockReset();
    mocks.updateScope.mockReset();
    mocks.getScope.mockResolvedValue({
      id: "general-chat",
      datasetIds: [],
    });
    mocks.updateScope.mockResolvedValue({
      id: "general-chat",
      datasetIds: ["dataset-1"],
    });
  });

  it("loads and persists General Chat dataset scope", async () => {
    render(
      <TooltipProvider>
        <PlatformChatSourcesButton projectId={null} />
      </TooltipProvider>,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Manage chat sources" }),
    );
    expect(await screen.findByText("No sources active")).toBeInTheDocument();
    expect(mocks.getScope).toHaveBeenCalledWith(null, expect.any(AbortSignal));

    fireEvent.click(screen.getByRole("button", { name: "Select baran" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(mocks.updateScope).toHaveBeenCalledWith(
        null,
        ["dataset-1"],
        expect.any(AbortSignal),
      ),
    );
    expect(await screen.findByText("1 source active")).toBeInTheDocument();
  });
});
