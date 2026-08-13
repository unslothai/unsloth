import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatProvidersSettings } from "./chat-providers-dialog";

vi.mock("./api/providers-api", () => ({
  createProviderConfig: vi.fn(),
  deleteProviderConfig: vi.fn(),
  listProviderModels: vi.fn().mockResolvedValue([]),
  listProviderRegistry: vi.fn().mockResolvedValue([]),
  testProviderConnection: vi.fn(),
  updateProviderConfig: vi.fn(),
}));

vi.mock("./stores/external-providers-store", () => ({
  useExternalProvidersStore: (
    selector: (state: {
      connectionsEnabled: boolean;
      setConnectionsEnabled: () => void;
    }) => unknown,
  ) =>
    selector({
      connectionsEnabled: true,
      setConnectionsEnabled: vi.fn(),
    }),
}));

vi.mock("./sync-external-providers", () => ({
  pruneProviderModelIds: vi.fn((models: string[]) => models),
  syncExternalProvidersFromBackend: vi.fn().mockResolvedValue([]),
}));

describe("ChatProvidersSettings connection flow", () => {
  it("opens platform provider management from Add connection", () => {
    render(
      <ChatProvidersSettings
        providers={[]}
        onProvidersChange={vi.fn()}
        platformConnection={<div>Platform connection management</div>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Add connection" }));

    expect(screen.getByText("Platform connection management")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Back to connections" }),
    ).toBeVisible();
  });

  it("keeps configured platform providers in the main connection list", () => {
    render(
      <ChatProvidersSettings
        providers={[]}
        onProvidersChange={vi.fn()}
        platformConnectionCount={1}
        platformModelCount={3}
        platformConnections={<div>Configured Rag Platform provider</div>}
      />,
    );

    expect(screen.getByText("1 connections · 3 models")).toBeVisible();
    expect(screen.getByText("Configured Rag Platform provider")).toBeVisible();
    expect(screen.queryByText("No connections yet")).not.toBeInTheDocument();
  });
});
