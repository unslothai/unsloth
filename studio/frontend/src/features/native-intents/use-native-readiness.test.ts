import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/platform-backend/config", () => ({
  isPlatformAuthEnabled: () => true,
}));

vi.mock("@/lib/api-base", () => ({
  apiUrl: (path: string) => path,
  isTauri: true,
}));

import { useNativePathLeasesSupported } from "./use-native-readiness";

describe("useNativePathLeasesSupported in Rag Platform mode", () => {
  it("does not start the legacy health polling loop", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const { result } = renderHook(() => useNativePathLeasesSupported());

    expect(result.current).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
