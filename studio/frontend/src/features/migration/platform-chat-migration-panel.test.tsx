import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  dryRun: vi.fn(),
  run: vi.fn(),
  download: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  usePlatformSessionStore: (selector: (state: unknown) => unknown) =>
    selector({ user: { id: "user-1", email: "user@example.test" } }),
}));
vi.mock("@/lib/native-files", () => ({ downloadFile: mocks.download }));
vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), info: vi.fn() },
}));
vi.mock("./platform-chat-migration", async (importOriginal) => {
  const original = await importOriginal<typeof import("./platform-chat-migration")>();
  return {
    ...original,
    dryRunPlatformChatMigration: mocks.dryRun,
    runPlatformChatMigration: mocks.run,
  };
});

import { PlatformChatMigrationPanel } from "./platform-chat-migration-panel";

const plan = {
  version: 1 as const,
  generatedAt: "2026-08-18T00:00:00.000Z",
  ownerId: "user-1",
  snapshot: {
    exportedAt: "2026-08-18T00:00:00.000Z",
    projects: [],
    threads: [],
    messages: [],
    sourceWarnings: [],
  },
  projects: [{ legacyId: "p1", label: "Proje", status: "pending" as const }],
  threads: [{ legacyId: "t1", label: "Sohbet", status: "pending" as const }],
  unsupported: [],
  totals: { projects: 1, threads: 1, messages: 0, alreadyMigrated: 0, pending: 2 },
};

describe("PlatformChatMigrationPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.dryRun.mockResolvedValue(plan);
    mocks.run.mockResolvedValue({
      completedProjects: 1,
      completedThreads: 1,
      skipped: 0,
      failures: [],
      aborted: false,
    });
    mocks.download.mockResolvedValue(undefined);
  });

  it("exposes dry-run, export and migration through the settings UI", async () => {
    render(<PlatformChatMigrationPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Dry-run" }));

    expect(await screen.findByText("Bekleyen: 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Export al" }));
    await waitFor(() => expect(mocks.download).toHaveBeenCalledOnce());

    fireEvent.click(screen.getByRole("button", { name: "Migration başlat" }));
    await waitFor(() => expect(mocks.run).toHaveBeenCalledOnce());
    expect(mocks.run.mock.calls[0]?.[0]).toEqual(plan);
  });
});
