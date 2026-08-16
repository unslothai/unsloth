import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  hasDatasetSkills: vi.fn(),
  getDatasetSkillTree: vi.fn(),
  listGlobalSkillSpaces: vi.fn(),
  searchGlobalSkills: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/integrations/platform-backend")>()),
  hasDatasetSkills: mocks.hasDatasetSkills,
  getDatasetSkillTree: mocks.getDatasetSkillTree,
  listGlobalSkillSpaces: mocks.listGlobalSkillSpaces,
  searchGlobalSkills: mocks.searchGlobalSkills,
}));

import SkillsPanel from "./skills-panel";

describe("Phase 10 skill ownership/runtime boundary", () => {
  beforeEach(() => {
    mocks.hasDatasetSkills.mockResolvedValue({ has: false });
    mocks.getDatasetSkillTree.mockResolvedValue(null);
    mocks.listGlobalSkillSpaces.mockRejectedValue(
      new Error("Table 'rag_platform.skill_spaces' doesn't exist"),
    );
    mocks.searchGlobalSkills.mockRejectedValue(
      new Error("connect: connection refused"),
    );
  });

  it("keeps dataset skills usable and hides global mutations when runtime prerequisites are absent", async () => {
    render(<SkillsPanel datasetId="dataset-1" datasetName="Docs" />);
    expect(
      await screen.findByText("Bu dataset için derlenmiş beceri yok."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Global skill space" }));
    expect(
      await screen.findByText(
        "Global skill space bu runtime'da kullanılamıyor.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Kaydet" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "İndeksle" }),
    ).not.toBeInTheDocument();
  });
});
