import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  getStatus: vi.fn(),
  getStats: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  getPlatformOperationsStatus: mocks.getStatus,
  getPlatformUsageStats: mocks.getStats,
  getPlatformUiError: () => ({ message: "Güvenli operasyon hatası" }),
}));

import { PlatformOperationsPanel } from "./platform-operations-panel";

describe("PlatformOperationsPanel", () => {
  beforeEach(() => {
    mocks.getStatus.mockResolvedValue({
      overall: "healthy",
      services: [
        {
          id: "database",
          label: "database",
          status: "healthy",
          type: "mysql",
          latencyMs: 2.5,
        },
      ],
      taskExecutorCount: 2,
    });
    mocks.getStats.mockResolvedValue({
      pageViews: [{ at: "2026-08-16", value: 12 }],
      uniqueVisitors: [{ at: "2026-08-16", value: 5 }],
      speed: [],
      tokensThousands: [],
      rounds: [],
      thumbsUp: [],
    });
  });

  it("renders sanitized dependency and usage summaries and supports refresh", async () => {
    render(<PlatformOperationsPanel />);

    expect(await screen.findByText("mysql · 2.5 ms")).toBeVisible();
    expect(screen.getByText("2 etkin yürütücü")).toBeVisible();
    expect(screen.getByText("12")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Yenile" }));
    await waitFor(() => expect(mocks.getStatus).toHaveBeenCalledTimes(2));
  });

  it("keeps successful partial data visible when the other request fails", async () => {
    mocks.getStats.mockRejectedValueOnce(new Error("secret backend detail"));
    render(<PlatformOperationsPanel />);

    expect(await screen.findByText("mysql · 2.5 ms")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Güvenli operasyon hatası",
    );
    expect(screen.queryByText("secret backend detail")).not.toBeInTheDocument();
  });
});
