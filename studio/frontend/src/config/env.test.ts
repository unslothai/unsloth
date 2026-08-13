import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/platform-backend/config", () => ({
  isPlatformAuthEnabled: () => true,
}));

import { fetchDeviceType, usePlatformStore } from "./env";

describe("Rag Platform hardware capability compatibility", () => {
  beforeEach(() => {
    usePlatformStore.setState({
      chatOnly: true,
      chatOnlyReason: "mlx_unavailable",
      detectionDeferred: true,
      fetched: false,
    });
  });

  it("settles legacy capabilities without polling /api/health", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(fetchDeviceType({ force: true })).resolves.toBeTruthy();

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(usePlatformStore.getState()).toMatchObject({
      chatOnly: false,
      chatOnlyReason: null,
      detectionDeferred: false,
      fetched: true,
    });
  });
});
