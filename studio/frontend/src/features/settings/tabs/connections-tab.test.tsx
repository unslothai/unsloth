import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { ConnectionsTab } from "./connections-tab";

vi.mock("@/features/chat/chat-providers-dialog", () => ({
  ChatProvidersSettings: ({
    platformConnection,
    platformConnections,
  }: {
    platformConnection?:
      | ReactNode
      | ((actions: { close: () => void }) => ReactNode);
    platformConnections?: ReactNode;
  }) => (
    <div>
      Legacy connection list
      <div data-testid="platform-connection-slot">
        {typeof platformConnection === "function"
          ? platformConnection({ close: vi.fn() })
          : platformConnection}
      </div>
      <div data-testid="platform-connections-slot">{platformConnections}</div>
    </div>
  ),
}));

vi.mock("@/features/chat/stores/external-providers-store", () => ({
  useExternalProvidersStore: (
    selector: (state: {
      providers: never[];
      setProviders: () => void;
    }) => unknown,
  ) => selector({ providers: [], setProviders: vi.fn() }),
}));

vi.mock("@/integrations/platform-backend", () => ({
  isPlatformAuthEnabled: () => true,
  isPlatformModelToolsEnabled: () => true,
}));

vi.mock("../components/platform-models-settings", () => ({
  PlatformModelsSettings: ({ mode }: { mode?: string }) => (
    <div>Platform model management: {mode}</div>
  ),
}));

describe("ConnectionsTab", () => {
  it("provides Rag Platform management to the existing connection flow", () => {
    render(<ConnectionsTab />);

    expect(screen.getByText("Legacy connection list")).toBeVisible();
    expect(screen.getByTestId("platform-connection-slot")).toBeVisible();
    expect(screen.getByText("Platform model management: create")).toBeVisible();
    expect(screen.getByText("Platform model management: manage")).toBeVisible();
    expect(screen.getByTestId("platform-connections-slot")).toBeVisible();
    expect(screen.getByTestId("connections-tab-content")).toHaveClass(
      "pb-8",
      "sm:pb-10",
    );
  });
});
