import type { RagDocument } from "@/features/rag/api/rag-api";
import { DocumentRow } from "@/features/rag/components/document-row";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

function makeDoc(overrides: Partial<RagDocument> = {}): RagDocument {
  return {
    id: "doc-abc",
    kb_id: "kb-1",
    thread_id: null,
    filename: "report.pdf",
    content_type: "application/pdf",
    status: "completed",
    num_chunks: 5,
    byte_size: 10240,
    error: null,
    created_at: 1_700_000_000,
    ...overrides,
  };
}

let onPreview: ReturnType<typeof vi.fn>;
let onDelete: ReturnType<typeof vi.fn>;

beforeEach(() => {
  onPreview = vi.fn();
  onDelete = vi.fn();
});

describe("DocumentRow preview event propagation", () => {
  it("clicking a previewable row opens document-level preview", async () => {
    render(
      React.createElement(DocumentRow, {
        doc: makeDoc(),
        onPreview: onPreview as () => void,
        onDelete: onDelete as () => void,
      }),
    );

    await userEvent.click(
      screen.getByRole("button", { name: /open preview of report.pdf/i }),
    );

    expect(onPreview).toHaveBeenCalledTimes(1);
  });

  it("Enter and Space open a previewable row", () => {
    render(
      React.createElement(DocumentRow, {
        doc: makeDoc(),
        onPreview: onPreview as () => void,
        onDelete: onDelete as () => void,
      }),
    );
    const row = screen.getByRole("button", {
      name: /open preview of report.pdf/i,
    });

    fireEvent.keyDown(row, { key: "Enter" });
    fireEvent.keyDown(row, { key: " " });

    expect(onPreview).toHaveBeenCalledTimes(2);
  });

  it("clicking delete does not open preview", async () => {
    render(
      React.createElement(DocumentRow, {
        doc: makeDoc(),
        onPreview: onPreview as () => void,
        onDelete: onDelete as () => void,
      }),
    );

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));

    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onPreview).not.toHaveBeenCalled();
  });

  it("non-previewable rows have no row button semantics", () => {
    render(
      React.createElement(DocumentRow, {
        doc: makeDoc({ status: "pending" }),
        onDelete: onDelete as () => void,
      }),
    );

    expect(screen.queryByRole("button", { name: /open preview/i })).toBeNull();
  });
});
