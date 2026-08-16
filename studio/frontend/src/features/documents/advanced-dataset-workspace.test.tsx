import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./advanced-dataset/metadata-panel", () => ({
  default: () => <div>metadata-panel</div>,
}));
vi.mock("./advanced-dataset/tags-panel", () => ({
  default: () => <div>tags-panel</div>,
}));
vi.mock("./advanced-dataset/graph-panel", () => ({
  default: () => <div>graph-panel</div>,
}));
vi.mock("./advanced-dataset/artifacts-panel", () => ({
  default: () => <div>artifacts-panel</div>,
}));
vi.mock("./advanced-dataset/indexing-panel", () => ({
  default: () => <div>indexing-panel</div>,
}));
vi.mock("./advanced-dataset/skills-panel", () => ({
  default: () => <div>skills-panel</div>,
}));

import AdvancedDatasetWorkspace from "./advanced-dataset-workspace";

describe("Phase 10 advanced dataset workspace", () => {
  it("keeps the workspace inside its parent and scrolls only the active panel", async () => {
    render(
      <AdvancedDatasetWorkspace
        datasetId="dataset-1"
        datasetName="Çok uzun dataset adı dar ekranlarda taşmamalı"
      />,
    );

    expect(screen.getByTestId("advanced-dataset-workspace")).toHaveClass(
      "h-full",
      "min-w-0",
      "overflow-hidden",
    );
    expect(
      screen.getByRole("tablist", { name: "Gelişmiş dataset alanları" }),
    ).toHaveClass("grid", "grid-cols-2", "sm:grid-cols-3", "xl:grid-cols-6");
    expect(screen.getByRole("tabpanel")).toHaveClass(
      "min-w-0",
      "overflow-x-hidden",
      "overflow-y-auto",
    );
    expect(await screen.findByText("metadata-panel")).toBeInTheDocument();
  });

  it("keeps every capability in a separate lazy tab and labels experimental surfaces", async () => {
    render(
      <AdvancedDatasetWorkspace
        datasetId="dataset-1"
        datasetName="Ürün belgeleri"
      />,
    );
    expect(await screen.findByText("metadata-panel")).toBeInTheDocument();
    const tabs = screen.getByRole("tablist", {
      name: "Gelişmiş dataset alanları",
    });
    for (const [label, content] of [
      ["Etiketler", "tags-panel"],
      ["Grafikdeneysel", "graph-panel"],
      ["Artifactdeneysel", "artifacts-panel"],
      ["İndeks & ingestion", "indexing-panel"],
      ["Becerilerdeneysel", "skills-panel"],
    ]) {
      fireEvent.click(screen.getByRole("tab", { name: label }));
      expect(await screen.findByText(content)).toBeInTheDocument();
    }
    expect(tabs).toBeInTheDocument();
    expect(screen.getAllByText("deneysel")).toHaveLength(3);
  });
});
