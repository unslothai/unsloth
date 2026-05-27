import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockOpenPreview } = vi.hoisted(() => ({
  mockOpenPreview: vi.fn(),
}));

vi.mock("@/features/rag/stores/preview-store", () => ({
  usePreviewStore: (
    selector?: (state: { open: typeof mockOpenPreview }) => unknown,
  ) => {
    const state = { open: mockOpenPreview };
    return typeof selector === "function" ? selector(state) : state;
  },
}));

vi.mock("@assistant-ui/react", () => ({
  useAuiState: (
    selector: (state: { message: { content: unknown[] } }) => unknown,
  ) =>
    selector({
      message: { content: [{ type: "text", text: "Answer ready." }] },
    }),
}));

import { SearchKnowledgeBaseToolUI } from "@/components/assistant-ui/tool-ui-search-knowledge-base";

const TOOL_UI = SearchKnowledgeBaseToolUI as React.ComponentType<
  Record<string, unknown>
>;
const SEARCHED_DOCS_BUTTON_RE = /searched docs/i;
const MAIN_PREVIEW_BUTTON_RE = /open preview of main\.pdf/i;
const LEGACY_PREVIEW_BUTTON_RE = /open preview of legacy\.pdf/i;

function renderTool(result: string) {
  return render(
    React.createElement(TOOL_UI, {
      args: { query: "what is interior modeling?" },
      result,
      status: { type: "complete" },
    }),
  );
}

beforeEach(() => {
  mockOpenPreview.mockClear();
});

describe("SearchKnowledgeBaseToolUI preview routing", () => {
  it("opens preview from a retrieved chunk source label", async () => {
    renderTool(
      '<chunk id="1" source="main.pdf" page="3" document_id="doc-abc" chunk_id="chunk-xyz">The paper objectives.</chunk>',
    );

    await userEvent.click(
      screen.getByRole("button", { name: SEARCHED_DOCS_BUTTON_RE }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: MAIN_PREVIEW_BUTTON_RE }),
    );

    expect(mockOpenPreview).toHaveBeenCalledWith({
      documentId: "doc-abc",
      backendChunkId: "chunk-xyz",
    });
  });

  it("keeps legacy chunk labels non-clickable without durable IDs", async () => {
    renderTool(
      '<chunk id="1" source="legacy.pdf" page="3">Legacy chunk text.</chunk>',
    );

    await userEvent.click(
      screen.getByRole("button", { name: SEARCHED_DOCS_BUTTON_RE }),
    );

    expect(
      screen.queryByRole("button", { name: LEGACY_PREVIEW_BUTTON_RE }),
    ).toBeNull();
    expect(mockOpenPreview).not.toHaveBeenCalled();
  });
});
