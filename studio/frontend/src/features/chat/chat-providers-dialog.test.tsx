import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChatProvidersSettings } from "./chat-providers-dialog";

const providerApiMocks = vi.hoisted(() => ({
  listProviderRegistry: vi.fn().mockResolvedValue([]),
  syncExternalProvidersFromBackend: vi.fn().mockResolvedValue([]),
}));

vi.mock("./api/providers-api", () => ({
  createProviderConfig: vi.fn(),
  deleteProviderConfig: vi.fn(),
  listProviderModels: vi.fn().mockResolvedValue([]),
  listProviderRegistry: providerApiMocks.listProviderRegistry,
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
  syncExternalProvidersFromBackend:
    providerApiMocks.syncExternalProvidersFromBackend,
}));

describe("ChatProvidersSettings connection flow", () => {
  beforeEach(() => vi.clearAllMocks());

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

  it("does not call unsupported legacy provider endpoints in platform mode", () => {
    render(
      <ChatProvidersSettings
        providers={[]}
        onProvidersChange={vi.fn()}
        legacyBackendSyncEnabled={false}
        platformConnection={<div>Platform connection management</div>}
      />,
    );

    expect(providerApiMocks.listProviderRegistry).not.toHaveBeenCalled();
    expect(
      providerApiMocks.syncExternalProvidersFromBackend,
    ).not.toHaveBeenCalled();
  });
});
