import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const systemMocks = vi.hoisted(() => ({
  useSystemInfo: vi.fn(),
  loadHuggingFaceCacheSettings: vi.fn(),
  checkConnection: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/hooks/use-system", () => ({
  aggregateGpuMemoryTotalGb: vi.fn(() => 0),
  useSystemInfo: systemMocks.useSystemInfo,
}));

vi.mock("../api/hugging-face-cache", () => ({
  loadHuggingFaceCacheSettings: systemMocks.loadHuggingFaceCacheSettings,
  updateHuggingFaceCacheSettings: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  isPlatformAuthEnabled: () => true,
  usePlatformConnectionStore: (
    selector: (state: Record<string, unknown>) => unknown,
  ) =>
    selector({
      status: "connected",
      version: "v0.26.4",
      health: { status: "ok", db: "ok", redis: "ok" },
      error: null,
      lastCheckedAt: "2026-08-16T12:00:00.000Z",
      checkConnection: systemMocks.checkConnection,
    }),
}));

vi.mock("@/i18n", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/i18n")>()),
  useT: () => (key: string) => key,
}));

import { ResourcesTab } from "./resources-tab";

describe("ResourcesTab in Rag Platform mode", () => {
  it("shows verified platform health without calling unsupported legacy telemetry", async () => {
    render(<ResourcesTab />);

    expect(screen.getByText("v0.26.4")).toBeVisible();
    expect(
      screen.getByText(
        "CPU, RAM, disk, GPU ve model belleği telemetrisi bu Rag Platform dağıtımında sunulmuyor. Yanıltıcı sıfır değerler yerine yalnızca backend tarafından doğrulanan servis durumu gösterilir.",
      ),
    ).toBeVisible();
    expect(systemMocks.useSystemInfo).not.toHaveBeenCalled();
    expect(systemMocks.loadHuggingFaceCacheSettings).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(systemMocks.checkConnection).toHaveBeenCalledTimes(1),
    );
  });
});
