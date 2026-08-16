import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  status: "disconnected",
  checkConnection: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/config/platform-capabilities", () => ({
  isPlatformOnlyMode: () => true,
}));

vi.mock("@/integrations/platform-backend", () => ({
  usePlatformConnectionStore: (
    selector: (state: Record<string, unknown>) => unknown,
  ) =>
    selector({
      status: mocks.status,
      checkConnection: mocks.checkConnection,
    }),
}));

import { PlatformBackendBanner } from "./platform-backend-banner";

describe("PlatformBackendBanner", () => {
  beforeEach(() => {
    mocks.status = "disconnected";
    mocks.checkConnection.mockClear();
  });

  it("distinguishes a disconnected backend from an empty product state", () => {
    render(<PlatformBackendBanner />);

    expect(screen.getByRole("status")).toHaveTextContent(
      "Veriler boş değil; şu anda yüklenemiyor.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Yeniden dene" }));
    expect(mocks.checkConnection).toHaveBeenCalledTimes(2);
  });

  it("shows an authorization-specific recovery message", () => {
    mocks.status = "unauthorized";
    render(<PlatformBackendBanner />);

    expect(screen.getByRole("status")).toHaveTextContent(
      "Rag Platform oturumu doğrulanamadı. Yeniden giriş yapın.",
    );
    expect(
      screen.queryByRole("button", { name: "Yeniden dene" }),
    ).not.toBeInTheDocument();
  });
});
